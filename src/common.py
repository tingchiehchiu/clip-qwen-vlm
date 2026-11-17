"""Configuration and reproducibility helpers for the construction pipeline."""

import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def path(value):
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def read_json(file):
    return json.loads(path(file).read_text())


def write_json(file, value):
    target = path(file)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def sha256(file):
    digest = hashlib.sha256()
    with path(file).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_all(seed):
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_recipe(config, stage):
    recipe = config.get(stage)
    needed = ("epochs", "batch_size", "gradient_accumulation", "learning_rate", "warmup_ratio", "weight_decay", "scheduler", "data_json", "image_root")
    if not isinstance(recipe, dict) or any(recipe.get(key) is None for key in needed):
        raise ValueError(f"Fill all {stage} recipe fields before training: {needed}")
    if not isinstance(recipe["epochs"], int) or recipe["epochs"] < 1:
        raise ValueError("epochs must be a positive integer")
    if not isinstance(recipe["batch_size"], int) or recipe["batch_size"] < 1:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(recipe["gradient_accumulation"], int) or recipe["gradient_accumulation"] < 1:
        raise ValueError("gradient_accumulation must be a positive integer")
    if recipe["learning_rate"] <= 0 or not 0 <= recipe["warmup_ratio"] < 1:
        raise ValueError("Invalid learning rate or warmup ratio")
    if recipe["scheduler"] not in ("cosine", "linear", "constant"):
        raise ValueError("scheduler must be cosine, linear, or constant")
    return recipe
