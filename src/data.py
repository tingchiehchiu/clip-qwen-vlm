"""LLaVA conversation parsing and assistant-only Qwen token labels."""

import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from .common import path

IMAGE_MARKER = "<image>"
PLACEHOLDER_ID = -1


def conversations(record):
    messages = record.get("conversations")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Expected a nonempty LLaVA conversations list")
    result = []
    for message in messages:
        role = {"human": "user", "gpt": "assistant"}.get(message.get("from"))
        if role is None or not isinstance(message.get("value"), str):
            raise ValueError("Unexpected conversation role or content")
        result.append((role, message["value"]))
    if result[0][0] != "user" or sum(text.count(IMAGE_MARKER) for _, text in result) != 1:
        raise ValueError("Each record must start with a user turn and have one image marker")
    if any(role == "assistant" and IMAGE_MARKER in text for role, text in result):
        raise ValueError("Image marker belongs in a user turn")
    if not any(role == "assistant" for role, _ in result):
        raise ValueError("No assistant target")
    return result


def encode_messages(tokenizer, messages, *, assistant_prefix=False):
    """Use Qwen chat delimiters; supervise assistant content and end token only."""
    ids, labels = [], []

    def append(text, supervise):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        ids.extend(tokens)
        labels.extend(tokens if supervise else [-100] * len(tokens))

    for role, content in messages:
        if role not in ("user", "assistant"):
            raise ValueError(f"Unsupported role: {role}")
        append(f"<|im_start|>{role}\n", False)
        parts = content.split(IMAGE_MARKER)
        if len(parts) > 2:
            raise ValueError("More than one image marker in a turn")
        append(parts[0], role == "assistant")
        if len(parts) == 2:
            ids.append(PLACEHOLDER_ID)
            labels.append(-100)
            append(parts[1], role == "assistant")
        append("<|im_end|>\n", role == "assistant")
    if assistant_prefix:
        append("<|im_start|>assistant\n", False)
    if ids.count(PLACEHOLDER_ID) != 1 or len(ids) != len(labels):
        raise ValueError("Expected exactly one aligned image placeholder")
    if not assistant_prefix and not any(label != -100 for label in labels):
        raise ValueError("No supervised assistant tokens")
    return ids, labels


class ConversationDataset(Dataset):
    def __init__(self, json_file, image_root, tokenizer, image_processor, max_sequence_length, visual_tokens):
        records = json.loads(path(json_file).read_text())
        if not isinstance(records, list) or not records:
            raise ValueError("Dataset JSON must be a nonempty list")
        self.image_root = path(image_root)
        self.image_processor = image_processor
        self.items = []
        self.skipped_length = 0
        for row in records:
            messages = conversations(row)
            ids, labels = encode_messages(tokenizer, messages)
            if len(ids) - 1 + visual_tokens > max_sequence_length:
                self.skipped_length += 1
                continue
            image = self.image_root / row["image"]
            if not image.is_file():
                raise FileNotFoundError(f"Missing referenced image: {image}")
            self.items.append((row["id"], image, ids, labels))
        if not self.items:
            raise ValueError("No usable records after sequence-length check")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        identifier, image, ids, labels = self.items[index]
        with Image.open(image) as handle:
            pixels = self.image_processor(images=handle.convert("RGB"), return_tensors="pt")["pixel_values"][0]
        return {"id": identifier, "input_ids": ids, "labels": labels, "pixel_values": pixels}


def collate(items):
    import torch

    return {
        "input_ids": [item["input_ids"] for item in items],
        "labels": [item["labels"] for item in items],
        "pixel_values": torch.stack([item["pixel_values"] for item in items]),
        "ids": [item["id"] for item in items],
    }
