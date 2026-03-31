#!/usr/bin/env bash
# Download challenge FineWeb shards + tokenizer into ./data (HF Hub).
# Run once per volume / fresh pod before training.
#
# Env:
#   TRAIN_SHARDS   default 80
#   VARIANT        default sp1024  (matches train_gpt defaults)
#   HF_TOKEN       optional
#   MATCHED_FINEWEB_REPO_ID  optional override
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SHARDS="${TRAIN_SHARDS:-80}"
VARIANT="${VARIANT:-sp1024}"

echo "runpod_bootstrap_data: TRAIN_SHARDS=${SHARDS} VARIANT=${VARIANT}"
python data/cached_challenge_fineweb.py --train-shards "${SHARDS}" --variant "${VARIANT}"
