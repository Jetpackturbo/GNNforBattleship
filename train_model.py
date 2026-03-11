#!/usr/bin/env python
# coding: utf-8
"""CLI for training Battleship policy models and saving checkpoints."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

import torch

from gnn import train_gnn


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

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        config=config,
    )


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

    config = {
        "model": args.model,
        "output": str(output_path),
        **common_train_kwargs,
        "num_heads": args.num_heads,
        "use_pyg": args.use_pyg,
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
                **common_train_kwargs,
            )

        checkpoint = {
            "model_type": args.model,
            "model_kwargs": model_kwargs,
            "train_kwargs": common_train_kwargs,
            "history": history,
            "state_dict": model.state_dict(),
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
