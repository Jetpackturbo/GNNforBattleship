#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_benchmark_checkpoints.sh <gnn_ckpt> <attn_ckpt> [n_games] [seed] [output_json]
#
# Examples:
#   ./run_benchmark_checkpoints.sh checkpoints/gnn-policy.pt checkpoints/attn-policy.pt
#   ./run_benchmark_checkpoints.sh checkpoints/gnn-policy.pt checkpoints/attn-policy.pt 500

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <gnn_ckpt> <attn_ckpt> [n_games] [seed] [output_json]" >&2
  exit 1
fi

GNN_CKPT="$1"
ATTN_CKPT="$2"
N_GAMES="${3:-200}"
SEED="${4:-100}"
OUTPUT_JSON="${5:-results/benchmark-checkpoints-${N_GAMES}.json}"

python test_model.py \
  --checkpoint "${GNN_CKPT}" \
  --checkpoint "${ATTN_CKPT}" \
  --n-games "${N_GAMES}" \
  --seed "${SEED}" \
  --save-json "${OUTPUT_JSON}" \
  "${@:6}"
