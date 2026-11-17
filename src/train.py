"""Two-stage training: adapter alignment, then full Qwen + adapter tuning."""

import argparse
import json
import math
import time

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .common import path, read_json, require_recipe, seed_all, sha256, write_json
from .data import ConversationDataset, collate
from .model import ConstructedLVLM


def schedule(optimizer, name, total_updates, warmup_ratio):
    warmup = round(total_updates * warmup_ratio)

    def multiplier(step):
        if warmup and step < warmup:
            return max((step + 1) / warmup, 1e-8)
        progress = (step - warmup) / max(total_updates - warmup, 1)
        if name == "cosine":
            return 0.5 * (1 + math.cos(math.pi * min(progress, 1)))
        if name == "linear":
            return max(1 - progress, 0)
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def gradient_evidence(model):
    evidence = {}
    for name, module in (("vision", model.vision), ("language", model.language), ("adapter", model.adapter)):
        norms = [float(parameter.grad.norm().item()) for parameter in module.parameters()
                 if parameter.grad is not None]
        if any(not math.isfinite(value) for value in norms):
            raise RuntimeError(f"Nonfinite {name} gradient")
        evidence[name] = max(norms, default=0.0)
    if evidence["vision"] != 0 or evidence["adapter"] <= 0:
        raise RuntimeError(f"Unexpected vision/adapter gradients: {evidence}")
    if model.stage == "align" and evidence["language"] != 0:
        raise RuntimeError("Frozen language model received a gradient")
    if model.stage == "instruct" and evidence["language"] <= 0:
        raise RuntimeError("Instruction-tuned language model received no gradient")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/construction.json")
    parser.add_argument("--stage", choices=("align", "instruct"), required=True)
    parser.add_argument("--output", required=True, help="Fresh run directory")
    parser.add_argument("--aligned-checkpoint", help="Stage-1 epoch directory; required for instruction tuning")
    parser.add_argument("--max-updates", type=int, help="Short diagnostic run; do not report as full training")
    args = parser.parse_args()
    if args.max_updates is not None and args.max_updates < 1:
        parser.error("--max-updates must be positive")
    if args.stage == "instruct" and not args.aligned_checkpoint:
        parser.error("--aligned-checkpoint is required for instruction tuning")
    if args.stage == "align" and args.aligned_checkpoint:
        parser.error("Alignment starts from a new random adapter")
    if not torch.cuda.is_available():
        raise RuntimeError("Training needs a CUDA GPU")
    config = read_json(args.config)
    recipe = require_recipe(config, args.stage)
    output = path(args.output)
    if output.exists():
        raise FileExistsError(output)
    seed_all(config["seed"])
    device = torch.device("cuda:0")
    model = ConstructedLVLM(config).to(device)
    parent = None
    if args.stage == "instruct":
        parent = model.load_state(args.aligned_checkpoint, include_language=False)
        if parent.get("stage") != "align" or not parent.get("complete_epoch"):
            raise ValueError("Stage 2 requires a completed aligned adapter checkpoint")
        architecture = ("clip_model_path", "qwen_model_path", "vision_layer", "drop_cls", "adapter_type")
        if any(parent.get("config", {}).get(key) != config.get(key) for key in architecture):
            raise ValueError("Aligned adapter was trained with different components or visual features")
    counts = model.freeze(args.stage)
    if counts["vision"] != 0 or counts["adapter"] <= 0 or (args.stage == "align") != (counts["language"] == 0):
        raise RuntimeError(f"Trainable scope mismatch: {counts}")
    if args.stage == "instruct" and recipe.get("gradient_checkpointing", True):
        model.language.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.language.config.use_cache = False
    print(f"Preparing {args.stage} dataset and checking image paths...", flush=True)
    dataset = ConversationDataset(recipe["data_json"], recipe["image_root"], model.tokenizer,
                                  model.image_processor, config["max_sequence_length"], model.visual_tokens)
    print(f"Ready: {len(dataset)} records; {dataset.skipped_length} skipped for length", flush=True)
    output.mkdir(parents=True, exist_ok=False)
    if args.stage == "align":
        model.save_state(output / "random", include_language=False,
                         metadata={"stage": "random", "seed": config["seed"], "config": config})
    loader = DataLoader(dataset, batch_size=recipe["batch_size"], shuffle=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(config["seed"]), num_workers=0)
    accumulation = recipe["gradient_accumulation"]
    updates_per_epoch = math.ceil(len(loader) / accumulation)
    total_updates = updates_per_epoch * recipe["epochs"]
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
    scheduler = schedule(optimizer, recipe["scheduler"], total_updates, recipe["warmup_ratio"])
    manifest = {
        "stage": args.stage, "config": config, "recipe": recipe, "trainable_by_part": counts,
        "records": len(dataset), "skipped_overlength": dataset.skipped_length,
        "data_json_sha256": sha256(recipe["data_json"]), "parent_checkpoint": args.aligned_checkpoint,
        "diagnostic_max_updates": args.max_updates,
    }
    write_json(output / "run.json", manifest)
    log = (output / "steps.jsonl").open("x")
    optimizer.zero_grad(set_to_none=True)
    update = 0
    window_loss_sum = 0.0
    started = time.monotonic()
    use_bf16 = torch.cuda.is_bf16_supported()
    target_updates = min(total_updates, args.max_updates) if args.max_updates else total_updates
    progress = tqdm(total=target_updates, desc=f"{args.stage} {config['adapter_type']}",
                    unit="update", dynamic_ncols=True)
    try:
        for epoch in range(1, recipe["epochs"] + 1):
            model.adapter.train()
            model.vision.eval()
            model.language.train(args.stage == "instruct")
            epoch_evidence = None
            for batch_index, batch in enumerate(loader):
                pixels = batch["pixel_values"].to(device)
                window_start = (batch_index // accumulation) * accumulation
                divisor = min(accumulation, len(loader) - window_start)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    loss = model(pixels, batch["input_ids"], batch["labels"]).loss
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                window_loss_sum += float(loss.item())
                (loss / divisor).backward()
                if (batch_index + 1) % accumulation != 0 and batch_index + 1 != len(loader):
                    continue
                if epoch_evidence is None:
                    epoch_evidence = gradient_evidence(model)
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),
                                               recipe.get("max_grad_norm", 1.0))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1
                row = {"epoch": epoch, "update": update, "loss": window_loss_sum / divisor,
                       "learning_rate": scheduler.get_last_lr()[0], "elapsed_seconds": time.monotonic() - started}
                window_loss_sum = 0.0
                log.write(json.dumps(row) + "\n")
                log.flush()
                progress.set_postfix(epoch=epoch, loss=f"{row['loss']:.4f}",
                                     lr=f"{row['learning_rate']:.2e}", refresh=False)
                progress.update(1)
                if args.max_updates and update >= args.max_updates:
                    model.save_state(output / f"smoke-step-{update}", include_language=args.stage == "instruct",
                                     metadata={**manifest, "completed_updates": update, "stage": args.stage,
                                               "complete_epoch": False, "gradient_evidence": epoch_evidence})
                    tqdm.write(json.dumps(row))
                    return
            model.save_state(output / f"epoch-{epoch}", include_language=args.stage == "instruct",
                             metadata={**manifest, "completed_updates": update, "epoch": epoch,
                                       "complete_epoch": True, "gradient_evidence": epoch_evidence})
            tqdm.write(f"Saved epoch {epoch} after {update} updates")
    finally:
        progress.close()
        log.close()


if __name__ == "__main__":
    main()
