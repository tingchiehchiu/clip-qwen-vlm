"""A CLIP vision tower connected to Qwen by a new linear or MLP adapter."""

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPImageProcessor, CLIPVisionModel

from .common import path, read_json, sha256, write_json
from .data import PLACEHOLDER_ID


def build_adapter(vision_width, language_width, adapter_type):
    if adapter_type == "linear":
        return nn.Linear(vision_width, language_width)
    if adapter_type == "mlp":
        return nn.Sequential(
            nn.Linear(vision_width, language_width),
            nn.GELU(),
            nn.Linear(language_width, language_width),
        )
    raise ValueError(f"Unsupported adapter_type: {adapter_type}")


class ConstructedLVLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config_values = config
        vision_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
        clip_path = path(config["clip_model_path"])
        qwen_path = path(config["qwen_model_path"])
        self.image_processor = CLIPImageProcessor.from_pretrained(clip_path, local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_path, local_files_only=True, trust_remote_code=False)
        self.vision = CLIPVisionModel.from_pretrained(clip_path, torch_dtype=vision_dtype, local_files_only=True)
        self.language = AutoModelForCausalLM.from_pretrained(qwen_path, torch_dtype=torch.float32, local_files_only=True, trust_remote_code=False)
        vision_width = self.vision.config.hidden_size
        language_width = self.language.config.hidden_size
        if (vision_width, language_width) != (1024, 896):
            raise ValueError(f"Unexpected CLIP/Qwen widths: {(vision_width, language_width)}")
        self.adapter = build_adapter(vision_width, language_width, config["adapter_type"])
        self.vision_layer = config["vision_layer"]
        self.drop_cls = config["drop_cls"]
        self.visual_tokens = (self.vision.config.image_size // self.vision.config.patch_size) ** 2
        if not self.drop_cls:
            self.visual_tokens += 1
        self.freeze("random")

    def freeze(self, stage):
        if stage not in ("random", "align", "instruct"):
            raise ValueError("stage must be random, align, or instruct")
        self.vision.requires_grad_(False)
        self.language.requires_grad_(stage == "instruct")
        self.adapter.requires_grad_(stage in ("align", "instruct"))
        self.vision.eval()
        if stage != "instruct":
            self.language.eval()
        self.stage = stage
        return {name: sum(p.numel() for p in module.parameters() if p.requires_grad) for name, module in
                (("vision", self.vision), ("language", self.language), ("adapter", self.adapter))}

    def image_features(self, pixel_values):
        with torch.no_grad():
            vision_dtype = next(self.vision.parameters()).dtype
            hidden = self.vision(pixel_values=pixel_values.to(vision_dtype), output_hidden_states=True).hidden_states[self.vision_layer]
        if self.drop_cls:
            hidden = hidden[:, 1:, :]
        if hidden.shape[1] != self.visual_tokens:
            raise ValueError(f"Unexpected visual token count: {hidden.shape[1]} != {self.visual_tokens}")
        adapter_dtype = next(self.adapter.parameters()).dtype
        return self.adapter(hidden.to(adapter_dtype))

    def assemble(self, input_ids, labels, visual):
        """Replace one -1 sentinel with all projected patch embeddings, then right-pad."""
        embedding = self.language.get_input_embeddings()
        vectors, targets = [], []
        for ids, gold, patches in zip(input_ids, labels, visual):
            if ids.count(PLACEHOLDER_ID) != 1 or len(ids) != len(gold):
                raise ValueError("Invalid image marker or label alignment")
            position = ids.index(PLACEHOLDER_ID)
            patches = patches.to(embedding.weight.dtype)
            before = torch.tensor(ids[:position], device=patches.device, dtype=torch.long)
            after = torch.tensor(ids[position + 1:], device=patches.device, dtype=torch.long)
            vectors.append(torch.cat((embedding(before), patches, embedding(after)), dim=0))
            targets.append(torch.tensor(gold[:position] + [-100] * len(patches) + gold[position + 1:],
                                        device=patches.device, dtype=torch.long))
        width = max(len(row) for row in vectors)
        dim = vectors[0].shape[-1]
        embeds = vectors[0].new_zeros((len(vectors), width, dim))
        attention = torch.zeros((len(vectors), width), device=embeds.device, dtype=torch.long)
        padded_labels = torch.full((len(vectors), width), -100, device=embeds.device, dtype=torch.long)
        for index, (row, target) in enumerate(zip(vectors, targets)):
            size = len(row)
            embeds[index, :size] = row
            attention[index, :size] = 1
            padded_labels[index, :size] = target
        return embeds, attention, padded_labels

    def forward(self, pixel_values, input_ids, labels):
        visual = self.image_features(pixel_values)
        embeds, attention, targets = self.assemble(input_ids, labels, visual)
        if embeds.shape[1] > self.config_values["max_sequence_length"]:
            raise ValueError("Expanded sequence exceeds max_sequence_length")
        return self.language(inputs_embeds=embeds, attention_mask=attention, labels=targets, use_cache=False)

    @torch.inference_mode()
    def answer(self, pixel_values, input_ids, max_new_tokens):
        self.eval()
        visual = self.image_features(pixel_values)
        embeds, attention, _ = self.assemble([input_ids], [[-100] * len(input_ids)], visual)
        if embeds.shape[1] > self.config_values["max_sequence_length"]:
            raise ValueError("Prompt exceeds max_sequence_length")
        output = self.language(inputs_embeds=embeds, attention_mask=attention, use_cache=True)
        generated = []
        stop_ids = {self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|im_end|>")}
        for _ in range(max_new_tokens):
            token = int(output.logits[:, -1, :].argmax(dim=-1).item())
            if token in stop_ids:
                break
            generated.append(token)
            attention = torch.cat((attention, attention.new_ones((1, 1))), dim=1)
            output = self.language(input_ids=torch.tensor([[token]], device=attention.device),
                                   attention_mask=attention, past_key_values=output.past_key_values,
                                   use_cache=True)
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip(), generated

    def save_state(self, directory, *, include_language, metadata):
        directory = path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        torch.save(self.adapter.state_dict(), directory / "adapter.pt")
        if include_language:
            torch.save(self.language.state_dict(), directory / "language.pt")
        files = ["adapter.pt"] + (["language.pt"] if include_language else [])
        write_json(directory / "metadata.json", {
            **metadata, "files_sha256": {name: sha256(directory / name) for name in files}
        })

    def load_state(self, directory, *, include_language):
        directory = path(directory)
        metadata = read_json(directory / "metadata.json")
        if metadata.get("config", {}).get("adapter_type") != self.config_values["adapter_type"]:
            raise ValueError("Checkpoint adapter type differs from model config")
        expected = {"adapter.pt"} | ({"language.pt"} if include_language else set())
        if set(metadata.get("files_sha256", {})) != expected:
            raise ValueError("Checkpoint file set differs from requested state")
        for name, digest in metadata["files_sha256"].items():
            if sha256(directory / name) != digest:
                raise ValueError(f"Checkpoint hash mismatch: {name}")
        self.adapter.load_state_dict(torch.load(directory / "adapter.pt", map_location="cpu", weights_only=True))
        if include_language:
            self.language.load_state_dict(torch.load(directory / "language.pt", map_location="cpu", weights_only=True))
        return metadata
