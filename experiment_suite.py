#!/usr/bin/env python
# coding: utf-8
"""Run the requested Battleship experiment suite, plots, and GIF exports."""

from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import torch
from PIL import Image, ImageDraw

from battleship_ising import BattleshipGame
from gnn import (
    BattleshipGNN,
    GNNAgent,
    IsingBPAgent,
    ProbabilityDensityAgent,
    RandomAgent,
    benchmark_reference,
    play_game,
    summarize_results,
)
from mcts import MCTSAgent, bayesian_surprise, estimate_posterior_occupancy


ROOT = Path(__file__).resolve().parent
GRID_SIZE = 10


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


def _slugify(label: str) -> str:
    lowered = label.lower().replace("+", "plus")
    chars = [ch if ch.isalnum() else "_" for ch in lowered]
    slug = "".join(chars)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def _grouped(items: list[str], size: int) -> list[list[str]]:
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


def _binary_entropy(p: np.ndarray) -> np.ndarray:
    """Elementwise Bernoulli entropy H(p) in nats."""
    eps = 1e-12
    p_clip = np.clip(p, eps, 1.0 - eps)
    return -(p_clip * np.log(p_clip) + (1.0 - p_clip) * np.log(1.0 - p_clip))


def _posterior_entropy(posterior: np.ndarray, revealed: np.ndarray) -> float:
    """Total posterior entropy of per-cell ship occupancy (sum of Bernoulli entropies)."""
    # Posterior is forced to 0/1 on revealed cells; masking avoids tiny numeric noise.
    h = _binary_entropy(posterior)
    h = np.where(revealed, 0.0, h)
    return float(np.sum(h))


def _policy_prior_fn(model: object, device: str) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    def _fn(hit_mask: np.ndarray, revealed: np.ndarray) -> np.ndarray:
        if hasattr(model, "predict_from_masks"):
            return np.asarray(model.predict_from_masks(hit_mask, revealed, device=device), dtype=np.float64)
        raise TypeError("Loaded model does not expose predict_from_masks().")

    return _fn


@dataclass(frozen=True)
class AgentSpec:
    label: str
    factory: Callable[[int], object]


def _make_model_agent_factory(model: object, model_type: str, device: str) -> Callable[[int], object]:
    if model_type == "gnn":
        return lambda _: GNNAgent(model, device=device)

    attn_module = _load_attention_module()
    return lambda _: attn_module.AttentionGNNAgent(model, device=device)


def _benchmark_specs(
    learned_specs: list[tuple[str, Path]],
    guided_mcts_label: str,
    guided_mcts_checkpoint: Path,
    device: str,
) -> list[AgentSpec]:
    specs: list[AgentSpec] = [
        AgentSpec("Random", lambda seed: RandomAgent(seed=seed)),
        AgentSpec("Probability Density", lambda seed: ProbabilityDensityAgent(seed=seed)),
        AgentSpec(
            "MCTS solution",
            lambda seed: MCTSAgent(
                seed=seed,
                n_simulations=32,
                rollout_depth=10,
                exploration=1.4,
                tree_policy="puct",
                prior_source="blend",
                leaf_evaluator="heuristic",
                leaf_samples=16,
            ),
        ),
    ]

    guided_type, guided_model = _load_checkpoint(guided_mcts_checkpoint, device)
    guided_prior = _policy_prior_fn(guided_model, device)
    specs.append(
        AgentSpec(
            guided_mcts_label,
            lambda seed: MCTSAgent(
                seed=seed,
                n_simulations=32,
                rollout_depth=10,
                exploration=1.4,
                tree_policy="puct",
                prior_source="blend",
                leaf_evaluator="heuristic",
                leaf_samples=16,
                policy_prior_fn=guided_prior,
            ),
        )
    )

    for label, checkpoint in learned_specs:
        model_type, model = _load_checkpoint(checkpoint, device)
        specs.append(AgentSpec(label, _make_model_agent_factory(model, model_type, device)))

    return specs


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _run_benchmarks(agent_specs: list[AgentSpec], n_games: int, seed: int) -> dict[str, list[int]]:
    results: dict[str, list[int]] = {spec.label: [] for spec in agent_specs}
    for game_idx in range(n_games):
        game_seed = seed + game_idx
        game = BattleshipGame(seed=game_seed)
        for spec in agent_specs:
            agent = spec.factory(game_seed)
            outcome = play_game(agent, game=game, seed=game_seed)
            results[spec.label].append(int(outcome["n_shots"]))
        if (game_idx + 1) % 10 == 0 or game_idx + 1 == n_games:
            print(f"Benchmarks: completed {game_idx + 1}/{n_games} games")
    return results


def _plot_benchmark_groups(
    summary: dict[str, dict[str, float]],
    ordered_labels: list[str],
    output_dir: Path,
    group_size: int = 6,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for group_idx, labels in enumerate(_grouped(ordered_labels, group_size), start=1):
        means = np.array([summary[label]["mean"] for label in labels], dtype=float)
        stds = np.array([summary[label]["std"] for label in labels], dtype=float)
        x = np.arange(len(labels))

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(
            x,
            means,
            yerr=stds,
            capsize=5,
            label="Mean shots (±1 std)",
            color="#4C78A8",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_ylabel("Moves to finish 10x10 Battleship")
        ax.set_title(f"Benchmark comparison group {group_idx}")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
        plt.tight_layout()

        out_path = output_dir / f"benchmark_group_{group_idx}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out_path)
    return outputs


def _run_surprise_trajectories(
    agent_specs: list[AgentSpec],
    n_games: int,
    seed: int,
    max_steps: int,
    posterior_samples: int,
) -> tuple[dict[str, list[list[float]]], dict[str, list[list[float]]]]:
    surprise_by_agent: dict[str, list[list[float]]] = {spec.label: [] for spec in agent_specs}
    entropy_by_agent: dict[str, list[list[float]]] = {spec.label: [] for spec in agent_specs}
    for game_idx in range(n_games):
        board_seed = seed + game_idx
        base_revealed = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        base_hit_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        posterior_rng = np.random.default_rng(board_seed + 1000)
        initial_posterior = estimate_posterior_occupancy(
            base_revealed,
            base_hit_mask,
            n_samples=posterior_samples,
            rng=posterior_rng,
        )

        for spec in agent_specs:
            game = BattleshipGame(seed=board_seed)
            agent = spec.factory(board_seed)
            agent.reset()
            revealed = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
            hit_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
            true_hits = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
            prev_posterior = initial_posterior.copy()
            trajectory: list[float] = []
            entropies: list[float] = [_posterior_entropy(prev_posterior, revealed)]

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
                trajectory.append(float(surprise_value))
                entropies.append(_posterior_entropy(current_posterior, revealed))
                prev_posterior = current_posterior

                if game.all_sunk(true_hits):
                    break

            surprise_by_agent[spec.label].append(trajectory)
            entropy_by_agent[spec.label].append(entropies)
        print(f"Surprise curves: completed {game_idx + 1}/{n_games} boards")
    return surprise_by_agent, entropy_by_agent


def _plot_surprise_groups(
    surprise_by_agent: dict[str, list[list[float]]],
    ordered_labels: list[str],
    output_dir: Path,
    max_steps: int,
    group_size: int = 6,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for group_idx, labels in enumerate(_grouped(ordered_labels, group_size), start=1):
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))

        for color, label in zip(colors, labels):
            trajectories = surprise_by_agent[label]
            values = np.full((len(trajectories), max_steps), np.nan, dtype=float)
            for idx, trajectory in enumerate(trajectories):
                limit = min(len(trajectory), max_steps)
                values[idx, :limit] = trajectory[:limit]
            mean_curve = np.nanmean(values, axis=0)
            counts = np.sum(~np.isnan(values), axis=0)
            sem = np.nanstd(values, axis=0) / np.sqrt(np.maximum(counts, 1))
            ci = 1.96 * sem
            steps = np.arange(1, max_steps + 1)
            ax.plot(steps, mean_curve, color=color, linewidth=2, label=label)
            ax.fill_between(steps, mean_curve - ci, mean_curve + ci, color=color, alpha=0.2)

        ax.set_xlabel("Time step")
        ax.set_ylabel("Average Bayesian surprise")
        ax.set_title(f"Average Bayesian surprise over first {max_steps} steps, group {group_idx}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()

        out_path = output_dir / f"surprise_group_{group_idx}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out_path)
    return outputs


def _plot_entropy_groups(
    entropy_by_agent: dict[str, list[list[float]]],
    ordered_labels: list[str],
    output_dir: Path,
    max_steps: int,
    group_size: int = 6,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    # Entropy trajectories include step 0, so plot steps 0..max_steps.
    steps = np.arange(0, max_steps + 1)
    for group_idx, labels in enumerate(_grouped(ordered_labels, group_size), start=1):
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))

        for color, label in zip(colors, labels):
            trajectories = entropy_by_agent[label]
            values = np.full((len(trajectories), max_steps + 1), np.nan, dtype=float)
            for idx, trajectory in enumerate(trajectories):
                limit = min(len(trajectory), max_steps + 1)
                values[idx, :limit] = trajectory[:limit]
            mean_curve = np.nanmean(values, axis=0)
            counts = np.sum(~np.isnan(values), axis=0)
            sem = np.nanstd(values, axis=0) / np.sqrt(np.maximum(counts, 1))
            ci = 1.96 * sem
            ax.plot(steps, mean_curve, color=color, linewidth=2, label=label)
            ax.fill_between(steps, mean_curve - ci, mean_curve + ci, color=color, alpha=0.2)

        ax.set_xlabel("Time step")
        ax.set_ylabel("Posterior entropy (nats, summed over cells)")
        ax.set_title(f"Estimated-boat posterior entropy over time, group {group_idx}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()

        out_path = output_dir / f"entropy_group_{group_idx}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out_path)
    return outputs


def _plot_cumulative_surprise_groups(
    surprise_by_agent: dict[str, list[list[float]]],
    ordered_labels: list[str],
    output_dir: Path,
    max_steps: int,
    group_size: int = 6,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    steps = np.arange(1, max_steps + 1)
    for group_idx, labels in enumerate(_grouped(ordered_labels, group_size), start=1):
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))

        for color, label in zip(colors, labels):
            trajectories = surprise_by_agent[label]
            values = np.full((len(trajectories), max_steps), np.nan, dtype=float)
            for idx, trajectory in enumerate(trajectories):
                limit = min(len(trajectory), max_steps)
                if limit:
                    values[idx, :limit] = np.cumsum(np.asarray(trajectory[:limit], dtype=float))
            mean_curve = np.nanmean(values, axis=0)
            counts = np.sum(~np.isnan(values), axis=0)
            sem = np.nanstd(values, axis=0) / np.sqrt(np.maximum(counts, 1))
            ci = 1.96 * sem
            ax.plot(steps, mean_curve, color=color, linewidth=2, label=label)
            ax.fill_between(steps, mean_curve - ci, mean_curve + ci, color=color, alpha=0.2)

        ax.set_xlabel("Time step")
        ax.set_ylabel("Cumulative Bayesian surprise (reward)")
        ax.set_title(f"Cumulative Bayesian-surprise reward over first {max_steps} steps, group {group_idx}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()

        out_path = output_dir / f"cumulative_surprise_group_{group_idx}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out_path)
    return outputs


def _render_gif_frame(
    label: str,
    step: int,
    heatmap: np.ndarray,
    revealed: np.ndarray,
    hit_mask: np.ndarray,
    cell_size: int = 28,
) -> Image.Image:
    cmap = cm.get_cmap("hot")
    rgb = (cmap(np.clip(heatmap, 0.0, 1.0))[..., :3] * 255).astype(np.uint8)
    image = Image.fromarray(rgb, mode="RGB").resize(
        (GRID_SIZE * cell_size, GRID_SIZE * cell_size),
        resample=Image.Resampling.NEAREST,
    )
    canvas = Image.new("RGB", (image.width, image.height + 40), color=(20, 20, 20))
    canvas.paste(image, (0, 40))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), f"{label} | Step {step}", fill=(255, 255, 255))

    for idx in range(GRID_SIZE + 1):
        offset = idx * cell_size
        draw.line((0, 40 + offset, image.width, 40 + offset), fill=(70, 70, 70))
        draw.line((offset, 40, offset, 40 + image.height), fill=(70, 70, 70))

    for row, col in np.argwhere(revealed & ~hit_mask):
        x0 = int(col * cell_size)
        y0 = int(40 + row * cell_size)
        draw.text((x0 + cell_size // 2 - 3, y0 + cell_size // 2 - 6), ".", fill=(80, 255, 255))

    for row, col in np.argwhere(hit_mask):
        x0 = int(col * cell_size)
        y0 = int(40 + row * cell_size)
        draw.text((x0 + cell_size // 2 - 5, y0 + cell_size // 2 - 8), "X", fill=(255, 255, 255))

    return canvas


def _export_agent_gif(
    spec: AgentSpec,
    output_path: Path,
    board_seed: int,
    max_steps: int,
    posterior_samples: int,
) -> None:
    game = BattleshipGame(seed=board_seed)
    agent = spec.factory(board_seed)
    agent.reset()
    revealed = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    hit_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    true_hits = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    posterior_rng = np.random.default_rng(board_seed + 7000)
    frames: list[Image.Image] = []

    initial = estimate_posterior_occupancy(
        revealed,
        hit_mask,
        n_samples=posterior_samples,
        rng=posterior_rng,
    )
    frames.append(_render_gif_frame(spec.label, 0, initial, revealed, hit_mask))

    for step in range(1, max_steps + 1):
        if not game.all_sunk(true_hits):
            row, col = agent.best_guess()
            is_hit = game.shoot(row, col)
            agent.observe(row, col, is_hit)
            revealed[row, col] = True
            if is_hit:
                hit_mask[row, col] = True
                true_hits[row, col] = True
        posterior = estimate_posterior_occupancy(
            revealed,
            hit_mask,
            n_samples=posterior_samples,
            rng=posterior_rng,
        )
        frames.append(_render_gif_frame(spec.label, step, posterior, revealed, hit_mask))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=220,
        loop=0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-benchmark-games", type=int, default=50)
    parser.add_argument("--surprise-games", type=int, default=20)
    parser.add_argument("--surprise-steps", type=int, default=50)
    parser.add_argument("--posterior-samples", type=int, default=16)
    parser.add_argument("--gif-steps", type=int, default=50)
    parser.add_argument("--gif-board-seed", type=int, default=42)
    parser.add_argument("--results-dir", default="results/requested_suite")
    parser.add_argument("--plots-dir", default="plots/requested_suite")
    parser.add_argument("--gifs-dir", default="gifs/requested_suite")
    parser.add_argument("--gnn-pdf", required=True)
    parser.add_argument("--attn-pdf", required=True)
    parser.add_argument("--gnn-mcts", required=True)
    parser.add_argument("--attn-mcts", required=True)
    parser.add_argument("--gnn-mcts-uct", required=True)
    parser.add_argument("--attn-mcts-uct", required=True)
    parser.add_argument(
        "--guided-mcts-checkpoint",
        help="Checkpoint used as the neural prior for 'MCTS with teacher'. Defaults to --attn-mcts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir).expanduser().resolve()
    plots_dir = Path(args.plots_dir).expanduser().resolve()
    gifs_dir = Path(args.gifs_dir).expanduser().resolve()

    learned_specs = [
        ("GNN on PDF Teacher", Path(args.gnn_pdf).expanduser().resolve()),
        ("GNN-ATTN on PDF Teacher", Path(args.attn_pdf).expanduser().resolve()),
        ("GNN finetuned with MCTS", Path(args.gnn_mcts).expanduser().resolve()),
        ("GNN-ATTN finetuned with MCTS", Path(args.attn_mcts).expanduser().resolve()),
        ("GNN finetuned with MCTS (UCT)", Path(args.gnn_mcts_uct).expanduser().resolve()),
        ("GNN-ATTN finetuned with MCTS (UCT)", Path(args.attn_mcts_uct).expanduser().resolve()),
    ]

    guided_mcts_checkpoint = (
        Path(args.guided_mcts_checkpoint).expanduser().resolve()
        if args.guided_mcts_checkpoint is not None
        else Path(args.attn_mcts).expanduser().resolve()
    )
    guided_mcts_label = "MCTS with neural prior"

    agent_specs = _benchmark_specs(
        learned_specs=learned_specs,
        guided_mcts_label=guided_mcts_label,
        guided_mcts_checkpoint=guided_mcts_checkpoint,
        device=args.device,
    )
    ordered_labels = [spec.label for spec in agent_specs]

    print(benchmark_reference())
    benchmark_results = _run_benchmarks(agent_specs, n_games=args.n_benchmark_games, seed=100)
    benchmark_summary = summarize_results(benchmark_results, seed=100)
    _save_json(
        results_dir / "benchmark_results.json",
        {
            "benchmark_reference": benchmark_reference(),
            "n_games": args.n_benchmark_games,
            "ordered_labels": ordered_labels,
            "raw_results": benchmark_results,
            "summary": benchmark_summary,
        },
    )
    benchmark_plot_paths = _plot_benchmark_groups(
        benchmark_summary,
        ordered_labels,
        plots_dir / "benchmarks",
        group_size=6,
    )

    surprise_data, entropy_data = _run_surprise_trajectories(
        agent_specs=agent_specs,
        n_games=args.surprise_games,
        seed=400,
        max_steps=args.surprise_steps,
        posterior_samples=args.posterior_samples,
    )
    _save_json(
        results_dir / "surprise_results.json",
        {
            "n_games": args.surprise_games,
            "max_steps": args.surprise_steps,
            "ordered_labels": ordered_labels,
            "surprise_by_agent": surprise_data,
        },
    )
    _save_json(
        results_dir / "entropy_results.json",
        {
            "n_games": args.surprise_games,
            "max_steps": args.surprise_steps,
            "ordered_labels": ordered_labels,
            "entropy_by_agent": entropy_data,
            "entropy_definition": "sum over cells of Bernoulli entropy of posterior ship occupancy, masked to unrevealed cells",
            "units": "nats",
        },
    )
    surprise_plot_paths = _plot_surprise_groups(
        surprise_data,
        ordered_labels,
        plots_dir / "surprise",
        max_steps=args.surprise_steps,
        group_size=6,
    )
    cumulative_surprise_plot_paths = _plot_cumulative_surprise_groups(
        surprise_data,
        ordered_labels,
        plots_dir / "cumulative_surprise",
        max_steps=args.surprise_steps,
        group_size=6,
    )
    entropy_plot_paths = _plot_entropy_groups(
        entropy_data,
        ordered_labels,
        plots_dir / "entropy",
        max_steps=args.surprise_steps,
        group_size=6,
    )

    gif_labels = [
        "GNN on PDF Teacher",
        "GNN-ATTN on PDF Teacher",
        "GNN finetuned with MCTS",
        "GNN-ATTN finetuned with MCTS",
        "GNN finetuned with MCTS (UCT)",
        "GNN-ATTN finetuned with MCTS (UCT)",
        "MCTS solution",
        "MCTS with neural prior",
    ]
    gif_paths: dict[str, str] = {}
    for spec in agent_specs:
        if spec.label not in gif_labels:
            continue
        output_path = gifs_dir / f"{_slugify(spec.label)}.gif"
        print(f"Generating GIF for {spec.label}")
        _export_agent_gif(
            spec,
            output_path=output_path,
            board_seed=args.gif_board_seed,
            max_steps=args.gif_steps,
            posterior_samples=args.posterior_samples,
        )
        gif_paths[spec.label] = str(output_path)

    _save_json(
        results_dir / "artifact_manifest.json",
        {
            "benchmark_plots": [str(path) for path in benchmark_plot_paths],
            "surprise_plots": [str(path) for path in surprise_plot_paths],
            "cumulative_surprise_plots": [str(path) for path in cumulative_surprise_plot_paths],
            "entropy_plots": [str(path) for path in entropy_plot_paths],
            "gifs": gif_paths,
        },
    )
    print("Suite complete.")


if __name__ == "__main__":
    main()
