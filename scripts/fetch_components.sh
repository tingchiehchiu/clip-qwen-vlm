#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: scripts/fetch_components.sh CLIP_COMMIT QWEN_COMMIT" >&2
  exit 2
fi

if [[ -z ${VLM_STORAGE_ROOT:-} || -z ${HF_BIN:-} ]]; then
  source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
fi
storage_root=${VLM_STORAGE_ROOT:?VLM_STORAGE_ROOT is required}
hf_bin=${HF_BIN:?HF_BIN is required}
if [[ ! -x "$hf_bin" ]]; then
  echo "Hugging Face CLI is missing: $hf_bin (set HF_BIN to its path)" >&2
  exit 2
fi
mkdir -p "$storage_root/models/clip-vit-large-patch14-336" "$storage_root/models/Qwen2.5-0.5B-Instruct" "$storage_root/cache/hf" "$storage_root/tmp"
export HF_HOME="$storage_root/cache/hf" TMPDIR="$storage_root/tmp"
"$hf_bin" download openai/clip-vit-large-patch14-336 --revision "$1" --local-dir "$storage_root/models/clip-vit-large-patch14-336"
"$hf_bin" download Qwen/Qwen2.5-0.5B-Instruct --revision "$2" --local-dir "$storage_root/models/Qwen2.5-0.5B-Instruct"
printf 'CLIP=%s\nQWEN=%s\n' "$1" "$2" > "$storage_root/models/component_revisions.txt"
