#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DEVICE="${DEVICE:-cuda:0}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/requested_suite}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-dataset_cache/requested_suite}"
USE_WANDB="${USE_WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-GNNforBattleship}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_API_KEY="${WANDB_API_KEY:-}"
WANDB_RUN_PREFIX="${WANDB_RUN_PREFIX:-}"

PDF_EPOCHS="${PDF_EPOCHS:-12}"
PDF_N_TRAIN="${PDF_N_TRAIN:-6000}"
PDF_N_VAL="${PDF_N_VAL:-1200}"

MCTS_EPOCHS="${MCTS_EPOCHS:-10}"
MCTS_N_TRAIN="${MCTS_N_TRAIN:-2500}"
MCTS_N_VAL="${MCTS_N_VAL:-500}"

BATCH_SIZE="${BATCH_SIZE:-128}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
NUM_LAYERS="${NUM_LAYERS:-6}"
NUM_HEADS="${NUM_HEADS:-4}"
LR="${LR:-3e-4}"
MCTS_SIMULATIONS="${MCTS_SIMULATIONS:-8}"
MCTS_ROLLOUT_DEPTH="${MCTS_ROLLOUT_DEPTH:-3}"
MCTS_EXPLORATION="${MCTS_EXPLORATION:-0.5}"
MCTS_PRIOR_SOURCE="${MCTS_PRIOR_SOURCE:-heuristic}"
MCTS_LEAF_EVALUATOR="${MCTS_LEAF_EVALUATOR:-heuristic}"
MCTS_LEAF_SAMPLES="${MCTS_LEAF_SAMPLES:-16}"

mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$DATASET_CACHE_DIR"

run_train() {
  local label="$1"
  shift
  local run_name
  run_name="$(printf '%s' "$label" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/_/g; s/_\+/_/g; s/^_//; s/_$//')"
  if [[ -n "$WANDB_RUN_PREFIX" ]]; then
    run_name="${WANDB_RUN_PREFIX}_${run_name}"
  fi
  echo
  echo "============================================================"
  echo "Training: $label"
  echo "============================================================"
  if [[ "$USE_WANDB" == "1" ]]; then
    local wandb_args=(
      --use-wandb
      --wandb-project "$WANDB_PROJECT"
      --wandb-run-name "$run_name"
      --wandb-mode "$WANDB_MODE"
    )
    if [[ -n "$WANDB_ENTITY" ]]; then
      wandb_args+=(--wandb-entity "$WANDB_ENTITY")
    fi
    if [[ -n "$WANDB_API_KEY" ]]; then
      wandb_args+=(--wandb-api-key "$WANDB_API_KEY")
    fi
    python train_model.py "${wandb_args[@]}" "$@"
  else
    python train_model.py "$@"
  fi
}

run_train "GNN on PDF Teacher" \
  --model gnn \
  --output "$CHECKPOINT_DIR/gnn_pdf_teacher.pt" \
  --device "$DEVICE" \
  --epochs "$PDF_EPOCHS" \
  --n-train "$PDF_N_TRAIN" \
  --n-val "$PDF_N_VAL" \
  --batch-size "$BATCH_SIZE" \
  --hidden-dim "$HIDDEN_DIM" \
  --num-layers "$NUM_LAYERS" \
  --lr "$LR" \
  --dataset-cache-dir "$DATASET_CACHE_DIR" \
  --teacher-policy probability_density \
  # --no-tqdm

run_train "GNN-ATTN on PDF Teacher" \
  --model attn \
  --output "$CHECKPOINT_DIR/attn_pdf_teacher.pt" \
  --device "$DEVICE" \
  --epochs "$PDF_EPOCHS" \
  --n-train "$PDF_N_TRAIN" \
  --n-val "$PDF_N_VAL" \
  --batch-size "$BATCH_SIZE" \
  --hidden-dim "$HIDDEN_DIM" \
  --num-layers "$NUM_LAYERS" \
  --num-heads "$NUM_HEADS" \
  --lr "$LR" \
  --dataset-cache-dir "$DATASET_CACHE_DIR" \
  --teacher-policy probability_density \
  # --no-tqdm

run_train "GNN finetuned with MCTS" \
  --model gnn \
  --output "$CHECKPOINT_DIR/gnn_mcts_finetune.pt" \
  --device "$DEVICE" \
  --epochs "$MCTS_EPOCHS" \
  --n-train "$MCTS_N_TRAIN" \
  --n-val "$MCTS_N_VAL" \
  --batch-size "$BATCH_SIZE" \
  --hidden-dim "$HIDDEN_DIM" \
  --num-layers "$NUM_LAYERS" \
  --lr "$LR" \
  --dataset-cache-dir "$DATASET_CACHE_DIR" \
  --init-from "$CHECKPOINT_DIR/gnn_pdf_teacher.pt" \
  --teacher-policy mcts \
  --teacher-mcts-simulations "$MCTS_SIMULATIONS" \
  --teacher-mcts-rollout-depth "$MCTS_ROLLOUT_DEPTH" \
  --teacher-mcts-exploration "$MCTS_EXPLORATION" \
  --teacher-mcts-tree-policy puct \
  --teacher-mcts-prior-source "$MCTS_PRIOR_SOURCE" \
  --teacher-mcts-leaf-evaluator "$MCTS_LEAF_EVALUATOR" \
  --teacher-mcts-leaf-samples "$MCTS_LEAF_SAMPLES" \
  # --no-tqdm

run_train "GNN-ATTN finetuned with MCTS" \
  --model attn \
  --output "$CHECKPOINT_DIR/attn_mcts_finetune.pt" \
  --device "$DEVICE" \
  --epochs "$MCTS_EPOCHS" \
  --n-train "$MCTS_N_TRAIN" \
  --n-val "$MCTS_N_VAL" \
  --batch-size "$BATCH_SIZE" \
  --hidden-dim "$HIDDEN_DIM" \
  --num-layers "$NUM_LAYERS" \
  --num-heads "$NUM_HEADS" \
  --lr "$LR" \
  --dataset-cache-dir "$DATASET_CACHE_DIR" \
  --init-from "$CHECKPOINT_DIR/attn_pdf_teacher.pt" \
  --teacher-policy mcts \
  --teacher-mcts-simulations "$MCTS_SIMULATIONS" \
  --teacher-mcts-rollout-depth "$MCTS_ROLLOUT_DEPTH" \
  --teacher-mcts-exploration "$MCTS_EXPLORATION" \
  --teacher-mcts-tree-policy puct \
  --teacher-mcts-prior-source "$MCTS_PRIOR_SOURCE" \
  --teacher-mcts-leaf-evaluator "$MCTS_LEAF_EVALUATOR" \
  --teacher-mcts-leaf-samples "$MCTS_LEAF_SAMPLES" \
  # --no-tqdm

run_train "GNN finetuned with MCTS (UCT)" \
  --model gnn \
  --output "$CHECKPOINT_DIR/gnn_mcts_finetune_uct.pt" \
  --device "$DEVICE" \
  --epochs "$MCTS_EPOCHS" \
  --n-train "$MCTS_N_TRAIN" \
  --n-val "$MCTS_N_VAL" \
  --batch-size "$BATCH_SIZE" \
  --hidden-dim "$HIDDEN_DIM" \
  --num-layers "$NUM_LAYERS" \
  --lr "$LR" \
  --dataset-cache-dir "$DATASET_CACHE_DIR" \
  --init-from "$CHECKPOINT_DIR/gnn_pdf_teacher.pt" \
  --teacher-policy mcts \
  --teacher-mcts-simulations "$MCTS_SIMULATIONS" \
  --teacher-mcts-rollout-depth "$MCTS_ROLLOUT_DEPTH" \
  --teacher-mcts-exploration "$MCTS_EXPLORATION" \
  --teacher-mcts-tree-policy uct \
  --teacher-mcts-prior-source "$MCTS_PRIOR_SOURCE" \
  --teacher-mcts-leaf-evaluator "$MCTS_LEAF_EVALUATOR" \
  --teacher-mcts-leaf-samples "$MCTS_LEAF_SAMPLES" \
  # --no-tqdm

run_train "GNN-ATTN finetuned with MCTS (UCT)" \
  --model attn \
  --output "$CHECKPOINT_DIR/attn_mcts_finetune_uct.pt" \
  --device "$DEVICE" \
  --epochs "$MCTS_EPOCHS" \
  --n-train "$MCTS_N_TRAIN" \
  --n-val "$MCTS_N_VAL" \
  --batch-size "$BATCH_SIZE" \
  --hidden-dim "$HIDDEN_DIM" \
  --num-layers "$NUM_LAYERS" \
  --num-heads "$NUM_HEADS" \
  --lr "$LR" \
  --dataset-cache-dir "$DATASET_CACHE_DIR" \
  --init-from "$CHECKPOINT_DIR/attn_pdf_teacher.pt" \
  --teacher-policy mcts \
  --teacher-mcts-simulations "$MCTS_SIMULATIONS" \
  --teacher-mcts-rollout-depth "$MCTS_ROLLOUT_DEPTH" \
  --teacher-mcts-exploration "$MCTS_EXPLORATION" \
  --teacher-mcts-tree-policy uct \
  --teacher-mcts-prior-source "$MCTS_PRIOR_SOURCE" \
  --teacher-mcts-leaf-evaluator "$MCTS_LEAF_EVALUATOR" \
  --teacher-mcts-leaf-samples "$MCTS_LEAF_SAMPLES" \
  # --no-tqdm

echo
echo "Finished training requested checkpoint suite in $CHECKPOINT_DIR"
echo "Dataset cache stored in $DATASET_CACHE_DIR"
