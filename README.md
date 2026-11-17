# CLIP–Qwen vision-language model

A small LLaVA-style model assembled from a frozen CLIP vision encoder, Qwen2.5-0.5B-Instruct, and a newly initialized image adapter. The adapter can be a single linear layer or a two-layer MLP. It maps 576 CLIP image patches into Qwen's embedding space.

Training has two stages: align the adapter on LLaVA-CC3M-Pretrain-595K while CLIP and Qwen are frozen, then instruction-tune the adapter and Qwen on LLaVA-Instruct-150K while CLIP remains frozen. The code saves a random state, an aligned state, and checkpoints after each instruction-tuning epoch for comparison. This repository contains source and a small example question manifest. It does not contain training data, component weights, trained checkpoints, or personal images.

## Setup

Use Linux with a CUDA GPU and Python 3.12. From this directory:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
source ./env.sh
"$MAIN_PY" -m unittest discover -s tests -v
```

The pinned requirements were used for the original training runs. A CPU environment can run the contract tests; training and generation require CUDA.

## Prepare components and data

The configured files live under `storage/`, which Git ignores. The downloads are large: the [CC3M image ZIP](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K/tree/main) is about 6.46 GB, and [COCO train2017](https://github.com/haotian-liu/LLaVA/blob/main/README.md) is a separate image archive. Check free space before downloading. Run the commands below from the repository root after `source ./env.sh`.

First record the full commit revisions of the two component models and two dataset repositories. This makes a fresh run reproducible even if their default branches move:

```bash
"$MAIN_PY" - <<'PY'
from huggingface_hub import HfApi

api = HfApi()
for label, repo in (
    ("CLIP_COMMIT", "openai/clip-vit-large-patch14-336"),
    ("QWEN_COMMIT", "Qwen/Qwen2.5-0.5B-Instruct"),
):
    print(label, api.model_info(repo).sha)
for label, repo in (
    ("CC3M_COMMIT", "liuhaotian/LLaVA-CC3M-Pretrain-595K"),
    ("INSTRUCT_COMMIT", "liuhaotian/LLaVA-Instruct-150K"),
):
    print(label, api.dataset_info(repo).sha)
PY
```

Copy those four full hashes into shell variables in the current terminal. Run `scripts/fetch_components.sh` to download the [CLIP](https://huggingface.co/openai/clip-vit-large-patch14-336) and [Qwen](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) models. The script places them in the model directories configured for training and records their revisions in `storage/models/component_revisions.txt`:

```bash
export CLIP_COMMIT=PASTE_FULL_CLIP_HASH
export QWEN_COMMIT=PASTE_FULL_QWEN_HASH
export CC3M_COMMIT=PASTE_FULL_CC3M_HASH
export INSTRUCT_COMMIT=PASTE_FULL_INSTRUCT_HASH
bash scripts/fetch_components.sh "$CLIP_COMMIT" "$QWEN_COMMIT"
```

Download the [CC3M pretraining conversations and images](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K) and [LLaVA instruction conversations](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K):

```bash
"$HF_BIN" download liuhaotian/LLaVA-CC3M-Pretrain-595K chat.json images.zip \
  --repo-type dataset --revision "$CC3M_COMMIT" --local-dir "$VLM_STORAGE_ROOT/data/cc3m"
"$HF_BIN" download liuhaotian/LLaVA-Instruct-150K llava_instruct_150k.json \
  --repo-type dataset --revision "$INSTRUCT_COMMIT" --local-dir "$VLM_STORAGE_ROOT/data/instruct"
```

Inspect the CC3M ZIP layout, then extract it. The configured image root must directly contain the filenames referenced by `chat.json`:

```bash
unzip -Z -1 "$VLM_STORAGE_ROOT/data/cc3m/images.zip" | head
mkdir -p "$VLM_STORAGE_ROOT/data/cc3m/images"
unzip -q "$VLM_STORAGE_ROOT/data/cc3m/images.zip" -d "$VLM_STORAGE_ROOT/data/cc3m/images"
```

The instruction JSON references COCO train2017 images. Download the archive from the [COCO link in LLaVA's data instructions](https://github.com/haotian-liu/LLaVA/blob/main/README.md), check it, and extract it so the `train2017/` directory sits under `storage/data/instruct/images/`:

```bash
wget -c http://images.cocodataset.org/zips/train2017.zip \
  -O "$VLM_STORAGE_ROOT/data/instruct/train2017.zip"
unzip -tq "$VLM_STORAGE_ROOT/data/instruct/train2017.zip"
mkdir -p "$VLM_STORAGE_ROOT/data/instruct/images"
unzip -q "$VLM_STORAGE_ROOT/data/instruct/train2017.zip" \
  -d "$VLM_STORAGE_ROOT/data/instruct/images"
```

Before training, verify that both annotation files load and all referenced images exist at the configured paths:

```bash
"$MAIN_PY" - <<'PY'
import json
from pathlib import Path

config = json.loads(Path("configs/construction.json").read_text())
for stage in ("align", "instruct"):
    recipe = config[stage]
    records = json.loads(Path(recipe["data_json"]).read_text())
    root = Path(recipe["image_root"])
    missing = [str(root / row["image"]) for row in records
               if not (root / row["image"]).is_file()]
    print(f"{stage}: {len(records)} records, {len(missing)} missing images")
    if missing:
        raise SystemExit(f"First missing image: {missing[0]}")
PY
"$MAIN_PY" -m src.inspect --config configs/construction.json
```

If the archive extracts into another layout, change `image_root` in **both** configs to the directory containing the referenced filenames. Keep the exact downloaded revisions in your run notes. Follow the model and dataset terms linked above; do not add their files to Git.

## Train

The two configs differ by `adapter_type` only. These commands show the linear run; substitute `configs/construction_mlp.json` and fresh output paths for the MLP run.

```bash
"$MAIN_PY" -m src.train --config configs/construction.json --stage align --output outputs/linear-align
"$MAIN_PY" -m src.train --config configs/construction.json --stage instruct --aligned-checkpoint outputs/linear-align/epoch-1 --output outputs/linear-instruct
```

Stage 1 uses one epoch, learning rate `1e-3`, and gradient accumulation of 8. Stage 2 uses three epochs, learning rate `2e-5`, and the same accumulation. Both use batch size 1 and seed 17. `--max-updates` is available for diagnostics; it is not a full training run. Use a new output directory for every run.

## Generate answers

Copy `studies/demo_questions.example.jsonl` to `studies/demo_questions.jsonl`, put your own image under `studies/images/`, and update the manifest's relative `image_path`. Then run generation with a saved checkpoint:

```bash
"$MAIN_PY" -m src.generate --config configs/construction.json --checkpoint outputs/linear-instruct/epoch-3 --state instruct --manifest studies/demo_questions.jsonl --output outputs/linear-answers.jsonl
```

The checkpoint metadata checks architecture and configuration identity, so existing checkpoints made with a different config path layout need their original config. Raw answers and provenance are saved alongside each other.

## Layout and provenance

- `src/model.py`: CLIP, Qwen, adapter, image-token insertion, checkpoint loading and saving.
- `src/data.py`: LLaVA conversation parsing and assistant-only training labels.
- `src/train.py`: two-stage training and per-update logs.
- `src/generate.py`: generation with checkpoint provenance.
- `configs/`: linear and MLP recipes.
- `tests/`: CPU contract checks.

This code was developed for a course model-construction study. CLIP, Qwen, LLaVA training datasets, and COCO images are third-party resources with their own terms. No third-party weights or image assets are redistributed here.
