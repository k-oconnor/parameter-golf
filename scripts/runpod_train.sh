#!/usr/bin/env bash
# Single-node multi-GPU launcher for RunPod (or any host with torchrun).
# Usage (inside repo root, or from anywhere):
#   bash scripts/runpod_train.sh
#
# Common env (optional):
#   NPROC_PER_NODE=8          # GPUs on this machine (train_gpt requires 8 % NPROC == 0)
#   TRAIN_BATCH_TOKENS=524288 # global tokens per optimizer step (larger on 8xH100)
#   ATTENTION_BACKEND=rbl|flash
#   TORCH_COMPILE=1           # Linux + Triton usually OK; set 0 to debug
#   HF_TOKEN=...              # faster HF downloads for data/bootstrap
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

NPROC="${NPROC_PER_NODE:-${NPROC:-8}}"
if (( 8 % NPROC != 0 )); then
  echo "train_gpt.py requires 8 % NPROC_PER_NODE == 0 (got NPROC_PER_NODE=${NPROC})" >&2
  exit 1
fi

MASTER_PORT="${MASTER_PORT:-29500}"

echo "runpod_train: ROOT=${ROOT} nproc_per_node=${NPROC} MASTER_PORT=${MASTER_PORT}"
echo "runpod_train: TRAIN_BATCH_TOKENS=${TRAIN_BATCH_TOKENS:-<default>} ATTENTION_BACKEND=${ATTENTION_BACKEND:-<default>}"

exec torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC}" \
  --master_port="${MASTER_PORT}" \
  train_gpt.py
