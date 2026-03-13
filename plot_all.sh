#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DEVICE="${DEVICE:-cpu}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/requested_suite}"
RESULTS_DIR="${RESULTS_DIR:-results/requested_suite}"
PLOTS_DIR="${PLOTS_DIR:-plots/requested_suite}"
GIFS_DIR="${GIFS_DIR:-gifs/requested_suite}"

N_BENCHMARK_GAMES="${N_BENCHMARK_GAMES:-50}"
SURPRISE_GAMES="${SURPRISE_GAMES:-20}"
SURPRISE_STEPS="${SURPRISE_STEPS:-50}"
POSTERIOR_SAMPLES="${POSTERIOR_SAMPLES:-16}"
GIF_STEPS="${GIF_STEPS:-50}"
GIF_BOARD_SEED="${GIF_BOARD_SEED:-42}"

GUIDED_MCTS_CHECKPOINT="${GUIDED_MCTS_CHECKPOINT:-$CHECKPOINT_DIR/attn_mcts_finetune.pt}"

required_checkpoints=(
  "$CHECKPOINT_DIR/gnn_pdf_teacher.pt"
  "$CHECKPOINT_DIR/attn_pdf_teacher.pt"
  "$CHECKPOINT_DIR/gnn_mcts_finetune.pt"
  "$CHECKPOINT_DIR/attn_mcts_finetune.pt"
  "$CHECKPOINT_DIR/gnn_mcts_finetune_uct.pt"
  "$CHECKPOINT_DIR/attn_mcts_finetune_uct.pt"
)

for checkpoint in "${required_checkpoints[@]}"; do
  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing checkpoint: $checkpoint" >&2
    echo "Run ./train.sh first or set CHECKPOINT_DIR appropriately." >&2
    exit 1
  fi
done

mkdir -p "$RESULTS_DIR" "$PLOTS_DIR" "$GIFS_DIR"

python experiment_suite.py \
  --device "$DEVICE" \
  --n-benchmark-games "$N_BENCHMARK_GAMES" \
  --surprise-games "$SURPRISE_GAMES" \
  --surprise-steps "$SURPRISE_STEPS" \
  --posterior-samples "$POSTERIOR_SAMPLES" \
  --gif-steps "$GIF_STEPS" \
  --gif-board-seed "$GIF_BOARD_SEED" \
  --results-dir "$RESULTS_DIR" \
  --plots-dir "$PLOTS_DIR" \
  --gifs-dir "$GIFS_DIR" \
  --gnn-pdf "$CHECKPOINT_DIR/gnn_pdf_teacher.pt" \
  --attn-pdf "$CHECKPOINT_DIR/attn_pdf_teacher.pt" \
  --gnn-mcts "$CHECKPOINT_DIR/gnn_mcts_finetune.pt" \
  --attn-mcts "$CHECKPOINT_DIR/attn_mcts_finetune.pt" \
  --gnn-mcts-uct "$CHECKPOINT_DIR/gnn_mcts_finetune_uct.pt" \
  --attn-mcts-uct "$CHECKPOINT_DIR/attn_mcts_finetune_uct.pt" \
  --guided-mcts-checkpoint "$GUIDED_MCTS_CHECKPOINT"

echo
echo "Saved full plot suite under $PLOTS_DIR"
echo "Saved GIFs under $GIFS_DIR"
echo "Saved result JSON files under $RESULTS_DIR"
