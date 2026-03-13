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

GUIDED_MCTS_CHECKPOINT="${GUIDED_MCTS_CHECKPOINT:-$CHECKPOINT_DIR/attn-policy-large.pt}"

required_checkpoints=(
  "$CHECKPOINT_DIR/attn_pdf_teacher.pt"
  "$CHECKPOINT_DIR/gnn_pdf_teacher.pt"
  "$CHECKPOINT_DIR/gnn-mcts-smoke.pt"
  "$CHECKPOINT_DIR/gnn-policy-large.pt"
  "$CHECKPOINT_DIR/attn-policy-large.pt"
  "$CHECKPOINT_DIR/smoke-gnn.pt"
  "$CHECKPOINT_DIR/smoke-attn.pt"
)

for checkpoint in "${required_checkpoints[@]}"; do
  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing checkpoint: $checkpoint" >&2
    echo "Run ./train.sh first or set CHECKPOINT_DIR appropriately." >&2
    exit 1
  fi
done

# If results already exist and SKIP_SIMULATION is set, avoid re-running the
# full experiment suite (which is the expensive part).
if [[ "${SKIP_SIMULATION:-0}" == "1" ]] && ls "$RESULTS_DIR"/*.json >/dev/null 2>&1; then
  echo "Found existing results in $RESULTS_DIR and SKIP_SIMULATION=1."
  echo "Skipping simulation; leaving existing plots/GIFs as-is."
  exit 0
fi

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
  --gnn-mcts "$CHECKPOINT_DIR/gnn-mcts-smoke.pt" \
  --attn-mcts "$CHECKPOINT_DIR/attn-policy-large.pt" \
  --gnn-mcts-uct "$CHECKPOINT_DIR/gnn-policy-large.pt" \
  --attn-mcts-uct "$CHECKPOINT_DIR/smoke-attn.pt" \
  --guided-mcts-checkpoint "$GUIDED_MCTS_CHECKPOINT"

# Also generate a standalone temporal heatmap PDF for the GNN policy.
python make_plots.py heatmaps \
  --trajectory-agent gnn \
  --checkpoint "$CHECKPOINT_DIR/gnn-policy-large.pt" \
  --device "$DEVICE" \
  --board-seed "$GIF_BOARD_SEED" \
  --interval 5 \
  --max-steps "$GIF_STEPS" \
  --posterior-samples "$POSTERIOR_SAMPLES" \
  --output "$PLOTS_DIR/temporal-heatmaps-gnn.pdf"

echo
echo "Saved full plot suite under $PLOTS_DIR"
echo "Saved GIFs under $GIFS_DIR"
echo "Saved result JSON files under $RESULTS_DIR"
