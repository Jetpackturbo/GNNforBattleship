#!/usr/bin/env python
# coding: utf-8
"""CLI for training Battleship policy models and saving checkpoints."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

import torch

from gnn import GRID_SIZE, SHIP_LENGTHS, BattleshipGNN, benchmark_reference, train_gnn


ROOT = Path(__file__).resolve().parent


def _load_attention_module():
    module_path = ROOT / "gnn-attn.py"
    spec = importlib.util.spec_from_file_location("gnn_attn_file", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load attention module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _init_wandb_run(args: argparse.Namespace, config: dict):
    if not args.use_wandb:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb logging was requested but the package is not installed. "
            "Install it with `pip install wandb` or `pip install -r requirements.txt`."
        ) from exc

    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
        wandb.login(key=args.wandb_api_key, relogin=True)
    else:
        wandb.login()

    wandb_root = ROOT / ".wandb"
    wandb_cache = wandb_root / "cache"
    wandb_config = wandb_root / "config"
    wandb_root.mkdir(exist_ok=True)
    wandb_cache.mkdir(exist_ok=True)
    wandb_config.mkdir(exist_ok=True)

    os.environ.setdefault("WANDB_DIR", str(wandb_root))
    os.environ.setdefault("WANDB_CACHE_DIR", str(wandb_cache))
    os.environ.setdefault("WANDB_CONFIG_DIR", str(wandb_config))

    settings = wandb.Settings(start_method="thread")
    init_kwargs = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "name": args.wandb_run_name,
        "mode": args.wandb_mode,
        "config": config,
        "settings": settings,
    }

    try:
        return wandb.init(**init_kwargs)
    except Exception:
        if args.wandb_mode == "online":
            print("wandb online init failed; falling back to offline mode.")
            init_kwargs["mode"] = "offline"
            return wandb.init(**init_kwargs)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["gnn", "attn"], required=True)
    parser.add_argument("--output", required=True, help="Checkpoint path to write.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--n-train", type=int, default=4000)
    parser.add_argument("--n-val", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-context-shots", type=int, default=40)
    parser.add_argument(
        "--teacher-policy",
        choices=["probability_density", "mcts"],
        default="probability_density",
    )
    parser.add_argument(
        "--teacher-mcts-simulations",
        type=int,
        default=32,
        help="Number of simulations per MCTS teacher query.",
    )
    parser.add_argument(
        "--teacher-mcts-rollout-depth",
        type=int,
        default=10,
        help="Rollout depth for the MCTS teacher.",
    )
    parser.add_argument(
        "--teacher-mcts-exploration",
        type=float,
        default=1.4,
        help="UCT exploration constant for the MCTS teacher.",
    )
    parser.add_argument(
        "--teacher-mcts-tree-policy",
        choices=["uct", "puct", "uct_hybrid"],
        default="puct",
        help="Tree policy used by the MCTS teacher.",
    )
    parser.add_argument(
        "--teacher-mcts-prior-source",
        choices=["heuristic", "neural", "blend"],
        default="blend",
        help="Action-prior source used by the MCTS teacher.",
    )
    parser.add_argument(
        "--teacher-mcts-leaf-evaluator",
        choices=["heuristic", "rollout", "hybrid"],
        default="heuristic",
        help="How the MCTS teacher evaluates newly expanded leaves.",
    )
    parser.add_argument(
        "--teacher-mcts-leaf-samples",
        type=int,
        default=16,
        help="Posterior samples used by heuristic MCTS leaf evaluation.",
    )
    parser.add_argument(
        "--surprise-augmentation",
        action="store_true",
        help="Bias sampled training states toward high Bayesian surprise observations.",
    )
    parser.add_argument(
        "--surprise-samples",
        type=int,
        default=8,
        help="Posterior samples used to estimate Bayesian surprise.",
    )
    parser.add_argument(
        "--surprise-alpha",
        type=float,
        default=1.0,
        help="Exponent controlling how strongly surprise biases state selection.",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
        help="Only used for --model attn.",
    )
    parser.add_argument(
        "--use-pyg",
        action="store_true",
        help="Use torch-geometric for the plain GNN model.",
    )
    parser.add_argument(
        "--no-tqdm",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    parser.add_argument(
        "--dataset-cache-dir",
        default=str(ROOT / "dataset_cache"),
        help="Directory used to cache generated training/validation datasets.",
    )
    parser.add_argument(
        "--no-dataset-cache",
        action="store_true",
        help="Disable on-disk dataset caching for generated teacher data.",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument("--wandb-project", default="GNNforBattleship")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-api-key")
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="online",
    )
    parser.add_argument(
        "--init-from",
        help="Optional checkpoint to initialize from (for MCTS finetuning).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    common_train_kwargs = {
        "n_epochs": args.epochs,
        "n_train": args.n_train,
        "n_val": args.n_val,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "max_context_shots": args.max_context_shots,
        "seed": args.seed,
        "device": args.device,
    }
    teacher_kwargs = None
    if args.teacher_policy == "mcts":
        teacher_kwargs = {
            "n_simulations": args.teacher_mcts_simulations,
            "rollout_depth": args.teacher_mcts_rollout_depth,
            "exploration": args.teacher_mcts_exploration,
            "tree_policy": args.teacher_mcts_tree_policy,
            "prior_source": args.teacher_mcts_prior_source,
            "leaf_evaluator": args.teacher_mcts_leaf_evaluator,
            "leaf_samples": args.teacher_mcts_leaf_samples,
        }

    init_model = None
    if args.init_from:
        init_path = Path(args.init_from).expanduser().resolve()
        checkpoint: dict = torch.load(init_path, map_location=args.device)
        ckpt_type = checkpoint.get("model_type")
        ckpt_kwargs = checkpoint.get("model_kwargs", {})
        if ckpt_type != args.model:
            raise ValueError(
                f"--init-from checkpoint has model_type={ckpt_type!r}, "
                f"but --model={args.model!r} was requested."
            )
        if ckpt_type == "gnn":
            init_model = BattleshipGNN(**ckpt_kwargs)
        elif ckpt_type == "attn":
            attn_module = _load_attention_module()
            init_model = attn_module.BattleshipAttentionGNN(**ckpt_kwargs)
        else:
            raise ValueError(f"Unsupported checkpoint model type in --init-from: {ckpt_type!r}")
        init_model.load_state_dict(checkpoint["state_dict"])

    config = {
        "model": args.model,
        "output": str(output_path),
        "task": "battleship_next_move_prediction",
        "training_objective": f"imitate_{args.teacher_policy}_teacher",
        "teacher_policy": args.teacher_policy,
        "board_size": GRID_SIZE,
        "num_cells": GRID_SIZE * GRID_SIZE,
        "ship_lengths": list(SHIP_LENGTHS),
        "observation_features": [
            "is_hit",
            "is_miss",
            "is_unknown",
            "row_normalized",
            "col_normalized",
        ],
        "input_feature_dim": 5,
        "label_type": "policy_distribution_over_unrevealed_cells",
        "benchmark_reference": benchmark_reference(),
        **common_train_kwargs,
        "num_heads": args.num_heads,
        "use_pyg": args.use_pyg,
        "teacher_kwargs": teacher_kwargs,
        "surprise_augmentation": args.surprise_augmentation,
        "surprise_samples": args.surprise_samples,
        "surprise_alpha": args.surprise_alpha,
        "dataset_cache_dir": None if args.no_dataset_cache else args.dataset_cache_dir,
        "init_from": None if not args.init_from else str(Path(args.init_from).expanduser().resolve()),
    }
    wandb_run = _init_wandb_run(args, config)

    try:
        if args.model == "gnn":
            model_kwargs = {
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "use_pyg": args.use_pyg,
            }
            model, history = train_gnn(
                use_pyg=args.use_pyg,
                show_progress=not args.no_tqdm,
                wandb_run=wandb_run,
                teacher_policy=args.teacher_policy,
                teacher_kwargs=teacher_kwargs,
                surprise_augmentation=args.surprise_augmentation,
                surprise_samples=args.surprise_samples,
                surprise_alpha=args.surprise_alpha,
                dataset_cache_dir=None if args.no_dataset_cache else args.dataset_cache_dir,
                init_model=init_model if args.model == "gnn" else None,
                **common_train_kwargs,
            )
        else:
            attn_module = _load_attention_module()
            model_kwargs = {
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "num_heads": args.num_heads,
            }
            model, history = attn_module.train_attention_gnn(
                num_heads=args.num_heads,
                show_progress=not args.no_tqdm,
                wandb_run=wandb_run,
                teacher_policy=args.teacher_policy,
                teacher_kwargs=teacher_kwargs,
                surprise_augmentation=args.surprise_augmentation,
                surprise_samples=args.surprise_samples,
                surprise_alpha=args.surprise_alpha,
                dataset_cache_dir=None if args.no_dataset_cache else args.dataset_cache_dir,
                init_model=init_model if args.model == "attn" else None,
                **common_train_kwargs,
            )

        checkpoint = {
            "model_type": args.model,
            "model_kwargs": model_kwargs,
            "train_kwargs": common_train_kwargs,
            "teacher_policy": args.teacher_policy,
            "teacher_kwargs": teacher_kwargs,
            "surprise_augmentation": args.surprise_augmentation,
            "surprise_samples": args.surprise_samples,
            "surprise_alpha": args.surprise_alpha,
            "dataset_cache_dir": None if args.no_dataset_cache else args.dataset_cache_dir,
            "history": history,
            "state_dict": model.state_dict(),
            "init_from": None if not args.init_from else str(Path(args.init_from).expanduser().resolve()),
        }
        torch.save(checkpoint, output_path)

        print(f"Saved {args.model} checkpoint to {output_path}")
        if "val_top1" in history and history["val_top1"]:
            print(f"Final val_top1: {history['val_top1'][-1]:.3f}")
            if wandb_run is not None:
                wandb_run.summary["final_val_top1"] = history["val_top1"][-1]
        if wandb_run is not None:
            wandb_run.summary["checkpoint_path"] = str(output_path)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
