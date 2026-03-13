#!/usr/bin/env python
# coding: utf-8
"""Plot Bayesian surprise curves for GNN, GNN-ATTN, Ising BP, and Probability Density."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from battleship_ising import BattleshipGame
from gnn import (
    BattleshipGNN,
    GNNAgent,
    GRID_SIZE,
    IsingBPAgent,
    ProbabilityDensityAgent,
)
from mcts import bayesian_surprise, estimate_posterior_occupancy


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
    checkpoint: dict = torch.load(path, map_location=device)
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
    parser.add_argument("--gnn-checkpoint", required=True)
    parser.add_argument("--attn-checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=400)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--posterior-samples", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    gnn_path = Path(args.gnn_checkpoint).expanduser().resolve()
    attn_path = Path(args.attn_checkpoint).expanduser().resolve()
    gnn_type, gnn_model = _load_checkpoint(gnn_path, args.device)
    attn_type, attn_model = _load_checkpoint(attn_path, args.device)

    if gnn_type != "gnn":
        raise ValueError(f"--gnn-checkpoint must be model_type='gnn', got {gnn_type!r}")
    if attn_type != "attn":
        raise ValueError(f"--attn-checkpoint must be model_type='attn', got {attn_type!r}")

    attn_module = _load_attention_module()

    factories: dict[str, callable] = {
        f"GNN {gnn_path.stem}": lambda: GNNAgent(gnn_model, device=args.device),
        f"GNN-ATTN {attn_path.stem}": lambda: attn_module.AttentionGNNAgent(attn_model, device=args.device),
        "Ising BP": lambda: IsingBPAgent(),
        "Probability Density": lambda: ProbabilityDensityAgent(),
    }

    surprise_by_agent: dict[str, list[list[float]]] = {name: [] for name in factories}

    for game_idx in range(args.n_games):
        board_seed = args.seed + game_idx
        game = BattleshipGame(seed=board_seed)
        posterior_rng = np.random.default_rng(board_seed + 1000)
        base_revealed = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        base_hit_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        initial_posterior = estimate_posterior_occupancy(
            base_revealed,
            base_hit_mask,
            n_samples=args.posterior_samples,
            rng=posterior_rng,
        )

        for name, factory in factories.items():
            agent = factory()
            agent.reset()
            revealed = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
            hit_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
            true_hits = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
            prev_posterior = initial_posterior.copy()
            surprises: list[float] = []

            for _ in range(args.max_steps):
                row, col = agent.best_guess()
                is_hit = game.shoot(row, col)
                agent.observe(row, col, is_hit)
                revealed[row, col] = True
                if is_hit:
                    hit_mask[row, col] = True
                    true_hits[row, col] = True

                current_posterior = estimate_posterior_occupancy(
                    revealed,
                    hit_mask,
                    n_samples=args.posterior_samples,
                    rng=posterior_rng,
                )
                surprise_value, _ = bayesian_surprise(prev_posterior, current_posterior)
                surprises.append(float(surprise_value))
                prev_posterior = current_posterior

                if game.all_sunk(true_hits):
                    break

            surprise_by_agent[name].append(surprises)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(surprise_by_agent)))

    for color, (name, trajectories) in zip(colors, surprise_by_agent.items()):
        max_len = max(len(traj) for traj in trajectories)
        values = np.full((len(trajectories), max_len), np.nan, dtype=float)
        for idx, traj in enumerate(trajectories):
            values[idx, : len(traj)] = traj

        mean_curve = np.nanmean(values, axis=0)
        counts = np.sum(~np.isnan(values), axis=0)
        sem = np.nanstd(values, axis=0) / np.sqrt(np.maximum(counts, 1))
        ci = 1.96 * sem
        steps = np.arange(1, max_len + 1)

        ax.plot(steps, mean_curve, label=name, color=color, linewidth=2)
        ax.fill_between(steps, mean_curve - ci, mean_curve + ci, color=color, alpha=0.2)

    ax.set_xlabel("Time step")
    ax.set_ylabel("Average Bayesian surprise")
    ax.set_title("Bayesian surprise: GNN vs GNN-ATTN vs Ising BP vs Probability Density")
    ax.grid(True, alpha=0.3)
    ax.legend()

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Bayesian surprise plot to {output}")


if __name__ == "__main__":
    main()

