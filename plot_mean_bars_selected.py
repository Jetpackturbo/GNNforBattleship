#!/usr/bin/env python
# coding: utf-8
"""Generate mean-shot bar graph for a selected set of Battleship agents."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from battleship_ising import BattleshipGame
from gnn import BattleshipGNN, GNNAgent, IsingBPAgent, ProbabilityDensityAgent, RandomAgent, play_game


ROOT = Path(__file__).resolve().parent


def _load_attention_module():
    module_path = ROOT / "gnn-attn.py"
    spec = importlib.util.spec_from_file_location("gnn_attn_file", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load attention module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
        raise ValueError(f"Unsupported checkpoint model type: {model_type}")

    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model_type, model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gnn-pdf-checkpoint", required=True)
    parser.add_argument("--attn-pdf-checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--output", required=True, help="Output bar-plot path.")
    parser.add_argument(
        "--output-json",
        help="Optional JSON output with raw results and summary stats.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    gnn_path = Path(args.gnn_pdf_checkpoint).expanduser().resolve()
    attn_path = Path(args.attn_pdf_checkpoint).expanduser().resolve()

    gnn_type, gnn_model = _load_checkpoint(gnn_path, args.device)
    attn_type, attn_model = _load_checkpoint(attn_path, args.device)
    if gnn_type != "gnn":
        raise ValueError(f"--gnn-pdf-checkpoint must be model_type='gnn', got {gnn_type!r}")
    if attn_type != "attn":
        raise ValueError(f"--attn-pdf-checkpoint must be model_type='attn', got {attn_type!r}")

    attn_module = _load_attention_module()

    # Requested names:
    # - "Ising BP no Prior" but labeled as "Ising BP"
    # - "GNN PDF Teacher"
    # - "GNN with ATTN with PDF teacher"
    # - "random"
    # - "PDF teacher"
    agents: dict[str, object] = {
        "Ising BP": IsingBPAgent(h_prior=0.0),
        "GNN PDF Teacher": GNNAgent(gnn_model, device=args.device),
        "GNN with ATTN with PDF teacher": attn_module.AttentionGNNAgent(attn_model, device=args.device),
        "random": RandomAgent(seed=args.seed),
        "PDF teacher": ProbabilityDensityAgent(seed=args.seed),
    }

    results: dict[str, list[int]] = {name: [] for name in agents}
    for game_idx in range(args.n_games):
        game_seed = args.seed + game_idx
        game = BattleshipGame(seed=game_seed)
        for name, agent in agents.items():
            outcome = play_game(agent, game=game, seed=game_seed)
            results[name].append(int(outcome["n_shots"]))

    ordered_names = [
        "Ising BP",
        "GNN PDF Teacher",
        "GNN with ATTN with PDF teacher",
        "random",
        "PDF teacher",
    ]
    means = np.array([np.mean(results[name]) for name in ordered_names], dtype=float)
    stds = np.array([np.std(results[name]) for name in ordered_names], dtype=float)
    x = np.arange(len(ordered_names))

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(ordered_names)))
    ax.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        color=colors,
        edgecolor="black",
        linewidth=0.7,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(ordered_names, rotation=18, ha="right")
    ax.set_ylabel("Mean shots to win")
    ax.set_title("Mean Battleship game length by agent")
    ax.grid(axis="y", alpha=0.3)

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved bar graph to {output_path}")

    if args.output_json:
        payload = {
            "ordered_labels": ordered_names,
            "raw_results": results,
            "summary": {
                name: {
                    "n_games": int(len(results[name])),
                    "mean": float(np.mean(results[name])),
                    "median": float(np.median(results[name])),
                    "std": float(np.std(results[name])),
                    "min": float(np.min(results[name])),
                    "max": float(np.max(results[name])),
                }
                for name in ordered_names
            },
        }
        out_json = Path(args.output_json).expanduser().resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Saved JSON to {out_json}")


if __name__ == "__main__":
    main()

