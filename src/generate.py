"""Generate identical-question outputs from random, aligned, or tuned states."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from .common import path, read_json, sha256
from .data import IMAGE_MARKER, encode_messages
from .model import ConstructedLVLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/construction.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--state", choices=("random", "align", "instruct"), required=True)
    parser.add_argument("--manifest", required=True, help="JSONL with id, image_path, question")
    parser.add_argument("--output", required=True, help="Fresh raw-answer JSONL")
    args = parser.parse_args()
    config = read_json(args.config)
    checkpoint = path(args.checkpoint)
    metadata = read_json(checkpoint / "metadata.json")
    architecture = ("clip_model_path", "qwen_model_path", "vision_layer", "drop_cls", "adapter_type")
    if metadata.get("stage") != args.state or any(
        metadata.get("config", {}).get(key) != config.get(key) for key in architecture
    ):
        raise ValueError("Checkpoint state or component config mismatch")
    manifest = path(args.manifest)
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Manifest must have unique IDs")
    output = path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if not torch.cuda.is_available():
        raise RuntimeError("Generation needs a CUDA GPU")
    model = ConstructedLVLM(config).to("cuda:0")
    model.load_state(checkpoint, include_language=args.state == "instruct")
    model.freeze("random")
    model.eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        for row in rows:
            image = (manifest.parent / row["image_path"]).resolve()
            with Image.open(image) as picture:
                pixels = model.image_processor(images=picture.convert("RGB"), return_tensors="pt")["pixel_values"].to("cuda:0")
            ids, _ = encode_messages(model.tokenizer, [("user", IMAGE_MARKER + "\n" + row["question"])],
                                      assistant_prefix=True)
            raw, tokens = model.answer(pixels, ids, config["max_new_tokens"])
            record = {"id": row["id"], "image_path": str(image), "image_sha256": sha256(image),
                      "question": row["question"], "state": args.state, "checkpoint": str(checkpoint),
                      "raw_prediction": raw, "generation_tokens": len(tokens),
                      "at_cap": len(tokens) == config["max_new_tokens"]}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(row["id"], raw)
    (output.with_suffix(".provenance.json")).write_text(json.dumps({"config": config, "checkpoint_metadata": metadata,
        "manifest_sha256": sha256(manifest), "state": args.state, "checkpoint": str(checkpoint)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
