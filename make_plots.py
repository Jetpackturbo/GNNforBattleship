#!/usr/bin/env python
# coding: utf-8
"""Generate benchmark summary plots and temporal Battleship heatmaps."""

from __future__ import annotations

import argparse
import importlib.util
import json
from math import ceil
from pathlib import Path
from typing import Any

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
    RandomAgent,
    summarize_results,
)
from mcts import MCTSAgent, bayesian_surprise, estimate_posterior_occupancy, sample_consistent_board


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


def _load_results_payload(path: Path) -> tuple[dict[str, list[int]], dict[str, dict[str, float]]]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    if payload and all(isinstance(v, list) for v in payload.values()):
        raw_results = payload
        summary = summarize_results(raw_results)
        return raw_results, summary

    raw_results = payload["raw_results"]
    summary = payload.get("summary") or summarize_results(raw_results)
    return raw_results, summary


def _plot_benchmark_bars(results_json: Path, output: Path, title: str) -> None:
    _, summary = _load_results_payload(results_json)
    # Optionally filter out legacy baselines that should not appear in new plots.
    names = [name for name in summary.keys() if name != "Hunt Target"]
    means = np.array([summary[name]["mean"] for name in names], dtype=float)
    stds = np.array([summary[name]["std"] for name in names], dtype=float)

    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        means,
        yerr=stds,
        capsize=5,
        label="Mean shots (±1 std)",
        color="#5DA5DA",
        edgecolor="black",
        linewidth=0.7,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Moves to finish 10x10 Battleship")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _build_trajectory_agent(
    agent_name: str,
    device: str,
    checkpoint: Path | None,
    mcts_simulations: int,
    mcts_rollout_depth: int,
) -> object:
    if agent_name == "random":
        return RandomAgent()
    if agent_name == "probability_density":
        return ProbabilityDensityAgent()
    if agent_name == "ising_bp":
        return IsingBPAgent()
    if agent_name == "mcts":
        return MCTSAgent(
            n_simulations=mcts_simulations,
            rollout_depth=mcts_rollout_depth,
            tree_policy="puct",
            prior_source="blend",
            leaf_evaluator="heuristic",
        )
    if agent_name in {"gnn", "attn"}:
        if checkpoint is None:
            raise ValueError(f"--checkpoint is required for trajectory agent '{agent_name}'.")
        model_type, model = _load_checkpoint(checkpoint, device)
        if model_type == "gnn":
            return GNNAgent(model, device=device)
        attn_module = _load_attention_module()
        return attn_module.AttentionGNNAgent(model, device=device)
    raise ValueError(f"Unsupported trajectory agent: {agent_name}")


def _estimate_posterior_heatmap(
    revealed: np.ndarray,
    hit_mask: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    posterior = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
    for _ in range(n_samples):
        board = sample_consistent_board(revealed, hit_mask, rng=rng)
        posterior += board
    posterior /= max(n_samples, 1)
    posterior[revealed & hit_mask] = 1.0
    posterior[revealed & ~hit_mask] = 0.0
    return posterior


def _overlay_observations(ax, revealed: np.ndarray, hit_mask: np.ndarray) -> None:
    miss_mask = revealed & ~hit_mask
    for row, col in np.argwhere(hit_mask):
        ax.text(col, row, "x", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    for row, col in np.argwhere(miss_mask):
        ax.text(col, row, ".", ha="center", va="center", color="cyan", fontsize=9, fontweight="bold")


def _plot_temporal_heatmaps(
    trajectory_agent_name: str,
    output: Path,
    board_seed: int,
    interval: int,
    max_steps: int,
    posterior_samples: int,
    checkpoint: Path | None,
    device: str,
    mcts_simulations: int,
    mcts_rollout_depth: int,
) -> None:
    agent = _build_trajectory_agent(
        trajectory_agent_name,
        device=device,
        checkpoint=checkpoint,
        mcts_simulations=mcts_simulations,
        mcts_rollout_depth=mcts_rollout_depth,
    )
    game = BattleshipGame(seed=board_seed)
    agent.reset()

    revealed = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    hit_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    true_hits = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    snapshots: list[tuple[int, np.ndarray, np.ndarray]] = [(0, revealed.copy(), hit_mask.copy())]

    for step in range(1, max_steps + 1):
        row, col = agent.best_guess()
        is_hit = game.shoot(row, col)
        agent.observe(row, col, is_hit)

        revealed[row, col] = True
        if is_hit:
            hit_mask[row, col] = True
            true_hits[row, col] = True

        if step % interval == 0 or game.all_sunk(true_hits):
            snapshots.append((step, revealed.copy(), hit_mask.copy()))
        if game.all_sunk(true_hits):
            break

    rng = np.random.default_rng(board_seed)
    heatmaps = [
        (step, _estimate_posterior_heatmap(snapshot_revealed, snapshot_hits, posterior_samples, rng), snapshot_revealed, snapshot_hits)
        for step, snapshot_revealed, snapshot_hits in snapshots
    ]

    ncols = min(5, len(heatmaps))
    nrows = ceil(len(heatmaps) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.2 * nrows))
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for ax in axes.flat:
        ax.axis("off")

    for ax, (step, heatmap, snapshot_revealed, snapshot_hits) in zip(axes.flat, heatmaps):
        ax.axis("on")
        im = ax.imshow(heatmap, cmap="hot", vmin=0.0, vmax=1.0)
        ax.set_title(f"Step {step}")
        ax.set_xticks(range(GRID_SIZE))
        ax.set_yticks(range(GRID_SIZE))
        ax.set_xticklabels(list("ABCDEFGHIJ"))
        _overlay_observations(ax, snapshot_revealed, snapshot_hits)
        fig.colorbar(im, ax=ax, fraction=0.046)

    model_label = None
    if checkpoint is not None and trajectory_agent_name in {"gnn", "attn"}:
        model_label = checkpoint.stem

    fig.suptitle(
        f"Estimated boat probability every {interval} steps\n"
        f"trajectory={trajectory_agent_name}"
        + ("" if model_label is None else f", model={model_label}")
        + f", board_seed={board_seed}, posterior_samples={posterior_samples}",
        y=0.98,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _build_agent_factories(
    device: str,
    gnn_checkpoint: Path | None,
    attn_checkpoint: Path | None,
    include_mcts: bool,
    mcts_simulations: int,
    mcts_rollout_depth: int,
) -> dict[str, callable]:
    factories: dict[str, callable] = {
        "Random": lambda: RandomAgent(),
        "Probability Density": lambda: ProbabilityDensityAgent(),
        "Ising BP": lambda: IsingBPAgent(),
    }
    if include_mcts:
        factories["MCTS"] = lambda: MCTSAgent(
            n_simulations=mcts_simulations,
            rollout_depth=mcts_rollout_depth,
            tree_policy="puct",
            prior_source="blend",
            leaf_evaluator="heuristic",
        )
    if gnn_checkpoint is not None:
        _, gnn_model = _load_checkpoint(gnn_checkpoint, device)
        factories[f"GNN {gnn_checkpoint.stem}"] = lambda model=gnn_model: GNNAgent(model, device=device)
    if attn_checkpoint is not None:
        _, attn_model = _load_checkpoint(attn_checkpoint, device)
        attn_module = _load_attention_module()
        factories[f"ATTN {attn_checkpoint.stem}"] = (
            lambda model=attn_model: attn_module.AttentionGNNAgent(model, device=device)
        )
    return factories


def _plot_surprise_curves(
    output: Path,
    n_games: int,
    seed: int,
    posterior_samples: int,
    max_steps: int,
    device: str,
    gnn_checkpoint: Path | None,
    attn_checkpoint: Path | None,
    include_mcts: bool,
    mcts_simulations: int,
    mcts_rollout_depth: int,
) -> None:
    factories = _build_agent_factories(
        device=device,
        gnn_checkpoint=gnn_checkpoint,
        attn_checkpoint=attn_checkpoint,
        include_mcts=include_mcts,
        mcts_simulations=mcts_simulations,
        mcts_rollout_depth=mcts_rollout_depth,
    )

    surprise_by_agent: dict[str, list[list[float]]] = {name: [] for name in factories}

    for game_idx in range(n_games):
        board_seed = seed + game_idx
        game = BattleshipGame(seed=board_seed)
        posterior_rng = np.random.default_rng(board_seed + 1000)
        base_revealed = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        base_hit_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        initial_posterior = estimate_posterior_occupancy(
            base_revealed,
            base_hit_mask,
            n_samples=posterior_samples,
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

            for _ in range(max_steps):
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
                    n_samples=posterior_samples,
                    rng=posterior_rng,
                )
                surprise_value, _ = bayesian_surprise(prev_posterior, current_posterior)
                surprises.append(surprise_value)
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
    ax.set_title("Average Bayesian surprise per time step")
    ax.grid(True, alpha=0.3)
    ax.legend()

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark_parser = subparsers.add_parser("benchmark-bars")
    benchmark_parser.add_argument("--results-json", required=True)
    benchmark_parser.add_argument("--output", default="plots/benchmark-bars.png")
    benchmark_parser.add_argument(
        "--title",
        default="10x10 Battleship benchmark: mean and median with confidence bounds",
    )

    heatmap_parser = subparsers.add_parser("heatmaps")
    heatmap_parser.add_argument(
        "--trajectory-agent",
        choices=["random", "probability_density", "ising_bp", "mcts", "gnn", "attn"],
        default="probability_density",
    )
    heatmap_parser.add_argument("--checkpoint")
    heatmap_parser.add_argument("--device", default="cpu")
    heatmap_parser.add_argument("--board-seed", type=int, default=42)
    heatmap_parser.add_argument("--interval", type=int, default=10)
    heatmap_parser.add_argument("--max-steps", type=int, default=100)
    heatmap_parser.add_argument("--posterior-samples", type=int, default=64)
    heatmap_parser.add_argument("--mcts-simulations", type=int, default=64)
    heatmap_parser.add_argument("--mcts-rollout-depth", type=int, default=12)
    heatmap_parser.add_argument("--output", default="plots/temporal-heatmaps.png")

    surprise_parser = subparsers.add_parser("surprise-curves")
    surprise_parser.add_argument("--output", default="plots/bayesian-surprise-curves.png")
    surprise_parser.add_argument("--n-games", type=int, default=20)
    surprise_parser.add_argument("--seed", type=int, default=100)
    surprise_parser.add_argument("--posterior-samples", type=int, default=32)
    surprise_parser.add_argument("--max-steps", type=int, default=100)
    surprise_parser.add_argument("--device", default="cpu")
    surprise_parser.add_argument("--gnn-checkpoint")
    surprise_parser.add_argument("--attn-checkpoint")
    surprise_parser.add_argument("--include-mcts", action="store_true")
    surprise_parser.add_argument("--mcts-simulations", type=int, default=32)
    surprise_parser.add_argument("--mcts-rollout-depth", type=int, default=10)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "benchmark-bars":
        _plot_benchmark_bars(
            results_json=Path(args.results_json).expanduser().resolve(),
            output=Path(args.output).expanduser().resolve(),
            title=args.title,
        )
        print(f"Saved benchmark bar plot to {Path(args.output).expanduser().resolve()}")
        return

    if args.command == "heatmaps":
        checkpoint = None if args.checkpoint is None else Path(args.checkpoint).expanduser().resolve()
        _plot_temporal_heatmaps(
            trajectory_agent_name=args.trajectory_agent,
            output=Path(args.output).expanduser().resolve(),
            board_seed=args.board_seed,
            interval=args.interval,
            max_steps=args.max_steps,
            posterior_samples=args.posterior_samples,
            checkpoint=checkpoint,
            device=args.device,
            mcts_simulations=args.mcts_simulations,
            mcts_rollout_depth=args.mcts_rollout_depth,
        )
        print(f"Saved temporal heatmaps to {Path(args.output).expanduser().resolve()}")
        return

    _plot_surprise_curves(
        output=Path(args.output).expanduser().resolve(),
        n_games=args.n_games,
        seed=args.seed,
        posterior_samples=args.posterior_samples,
        max_steps=args.max_steps,
        device=args.device,
        gnn_checkpoint=None
        if args.gnn_checkpoint is None
        else Path(args.gnn_checkpoint).expanduser().resolve(),
        attn_checkpoint=None
        if args.attn_checkpoint is None
        else Path(args.attn_checkpoint).expanduser().resolve(),
        include_mcts=args.include_mcts,
        mcts_simulations=args.mcts_simulations,
        mcts_rollout_depth=args.mcts_rollout_depth,
    )
    print(f"Saved Bayesian surprise curves to {Path(args.output).expanduser().resolve()}")


if __name__ == "__main__":
    main()
