#!/usr/bin/env bash
# Source from the repository root after creating .venv.
_vlm_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VLM_STORAGE_ROOT="${VLM_STORAGE_ROOT:-$_vlm_repo_root/storage}"
export MAIN_PY="${MAIN_PY:-$_vlm_repo_root/.venv/bin/python}"
export HF_BIN="${HF_BIN:-$_vlm_repo_root/.venv/bin/hf}"
export HF_HOME="$VLM_STORAGE_ROOT/cache/hf"
export TMPDIR="$VLM_STORAGE_ROOT/tmp"
mkdir -p "$VLM_STORAGE_ROOT"/{data,models,checkpoints,outputs,cache/hf,tmp}
unset _vlm_repo_root
