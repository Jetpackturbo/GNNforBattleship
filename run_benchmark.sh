#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_benchmark.sh [n_games] [seed] [output_json]
#
# Examples:
#   ./run_benchmark.sh
#   ./run_benchmark.sh 500
#   ./run_benchmark.sh 500 123 results/benchmark-500.json

N_GAMES="${1:-200}"
SEED="${2:-100}"
OUTPUT_JSON="${3:-results/benchmark-${N_GAMES}.json}"

python test_model.py \
  --n-games "${N_GAMES}" \
  --seed "${SEED}" \
  --save-json "${OUTPUT_JSON}" \
  "${@:4}"
