#!/usr/bin/env python
# coding: utf-8
"""CLI for benchmarking saved Battleship policy checkpoints."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import torch

from gnn import BattleshipGNN, benchmark_reference, compare_all_agents, summarize_results


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


def _load_checkpoint(path: Path, device: str) -> tuple[str, object]:
    checkpoint: dict[str, Any] = torch.load(path, map_location=device)
    model_type = checkpoint["model_type"]
    model_kwargs = checkpoint["model_kwargs"]

    if model_type == "gnn":
        model = BattleshipGNN(**model_kwargs)
    elif model_type == "attn":
        attn_module = _load_attention_module()
        model = attn_module.BattleshipAttentionGNN(**model_kwargs)
    else:
        raise ValueError(f"Unsupported model type in checkpoint: {model_type}")

    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model_type, model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Checkpoint path. Can be passed multiple times.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument(
        "--include-mcts",
        action="store_true",
        help="Include the MCTS planning baseline in the benchmark.",
    )
    parser.add_argument("--mcts-simulations", type=int, default=96)
    parser.add_argument("--mcts-rollout-depth", type=int, default=18)
    parser.add_argument("--mcts-exploration", type=float, default=1.4)
    parser.add_argument(
        "--save-json",
        help="Optional path to save benchmark results as JSON.",
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

    config = {
        "n_games": args.n_games,
        "seed": args.seed,
        "device": args.device,
        "checkpoints": list(args.checkpoint),
        "include_mcts": args.include_mcts,
    }
    wandb_run = _init_wandb_run(args, config)

    try:
        gnn_model = None
        extra_agents: dict[str, object] = {}

        for ckpt_str in args.checkpoint:
            ckpt_path = Path(ckpt_str).expanduser().resolve()
            model_type, model = _load_checkpoint(ckpt_path, args.device)
            label = ckpt_path.stem

            if model_type == "gnn":
                if gnn_model is None:
                    gnn_model = model
                else:
                    from gnn import GNNAgent

                    extra_agents[f"GNN {label}"] = GNNAgent(model, device=args.device)
            else:
                attn_module = _load_attention_module()
                extra_agents[f"ATTN {label}"] = attn_module.AttentionGNNAgent(
                    model, device=args.device
                )

        print(benchmark_reference())
        results = compare_all_agents(
            n_games=args.n_games,
            gnn_model=gnn_model,
            seed=args.seed,
            device=args.device,
            verbose=True,
            extra_agents=extra_agents if extra_agents else None,
            show_progress=not args.no_tqdm,
            wandb_run=wandb_run,
            include_mcts=args.include_mcts,
            mcts_kwargs={
                "n_simulations": args.mcts_simulations,
                "rollout_depth": args.mcts_rollout_depth,
                "exploration": args.mcts_exploration,
            },
        )

        if args.save_json:
            save_path = Path(args.save_json).expanduser().resolve()
            save_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "benchmark_reference": benchmark_reference(),
                "n_games": args.n_games,
                "seed": args.seed,
                "raw_results": results,
                "summary": summarize_results(results, seed=args.seed),
            }
            with save_path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            print(f"Saved benchmark results to {save_path}")
            if wandb_run is not None:
                wandb_run.summary["results_json"] = str(save_path)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
