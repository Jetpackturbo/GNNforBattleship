#!/usr/bin/env python
# coding: utf-8
"""Benchmark Battleship Ising/BP with a symmetric (no-bias) prior.

"No prior model" here means we set the external-field prior to h_prior=0, which
corresponds to P(ship)=0.5 before seeing any evidence (symmetric between ship
and water). This lets you compare against the default fleet-density prior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from battleship_ising import BattleshipGame, BattleshipIsing
from gnn import summarize_results


class IsingBPAgentCustom:
    """Minimal agent wrapper around BattleshipIsing for benchmarking."""

    def __init__(self, *, grid_size: int, J: float, bp_iters: int, h_prior: float | None) -> None:
        self.model = BattleshipIsing(grid_size=grid_size, J=J, h_prior=h_prior)
        self.bp_iters = int(bp_iters)

    def reset(self) -> None:
        self.model.reset()

    def observe(self, row: int, col: int, is_hit: bool) -> None:
        self.model.observe(row, col, is_hit)

    def best_guess(self) -> tuple[int, int]:
        self.model.run_bp(num_iter=self.bp_iters)
        return self.model.best_guess()


def _play_game_count_shots(*, seed: int, grid_size: int, agent: IsingBPAgentCustom, max_shots: int) -> int:
    game = BattleshipGame(grid_size=grid_size, seed=seed)
    agent.reset()
    hit_mask = np.zeros((grid_size, grid_size), dtype=bool)
    shots = 0
    for _ in range(max_shots):
        shots += 1
        row, col = agent.best_guess()
        is_hit = game.shoot(row, col)
        agent.observe(row, col, is_hit)
        if is_hit:
            hit_mask[row, col] = True
        if game.all_sunk(hit_mask):
            break
    return shots


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-games", type=int, default=200)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--grid-size", type=int, default=10)
    p.add_argument("--J", type=float, default=0.5)
    p.add_argument("--bp-iters", type=int, default=60)
    p.add_argument("--max-shots", type=int, default=100)
    p.add_argument(
        "--no-prior",
        action="store_true",
        help="Use symmetric prior h_prior=0 (P(ship)=0.5) instead of fleet-density prior.",
    )
    p.add_argument(
        "--output-json",
        help="Optional path to save raw results as JSON.",
    )
    p.add_argument(
        "--include-default",
        action="store_true",
        help="When saving JSON, include both default-prior and no-prior Ising results in benchmark-bars format.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    runs: list[tuple[str, float | None]] = []
    if args.include_default and args.no_prior:
        runs.append(("Ising BP", None))
        runs.append(("Ising BP No Prior", 0.0))
    else:
        label = "Ising BP No Prior" if args.no_prior else "Ising BP"
        runs.append((label, 0.0 if args.no_prior else None))

    raw_results: dict[str, list[int]] = {}
    run_summaries: dict[str, dict[str, float | int | None]] = {}
    for label, h_prior in runs:
        agent = IsingBPAgentCustom(
            grid_size=args.grid_size,
            J=args.J,
            bp_iters=args.bp_iters,
            h_prior=h_prior,
        )
        shots = []
        for i in range(args.n_games):
            shots.append(
                _play_game_count_shots(
                    seed=args.seed + i,
                    grid_size=args.grid_size,
                    agent=agent,
                    max_shots=args.max_shots,
                )
            )
        raw_results[label] = [int(x) for x in shots]
        arr = np.asarray(shots, dtype=float)
        run_summaries[label] = {
            "n_games": int(arr.size),
            "mean_shots": float(arr.mean()),
            "median_shots": float(np.median(arr)),
            "std_shots": float(arr.std()),
            "min_shots": float(arr.min()),
            "max_shots": float(arr.max()),
            "J": float(args.J),
            "bp_iters": int(args.bp_iters),
            "grid_size": int(args.grid_size),
            "seed": int(args.seed),
            "h_prior": None if h_prior is None else float(h_prior),
            "max_shots": int(args.max_shots),
        }

    printable = run_summaries if len(run_summaries) > 1 else next(iter(run_summaries.values()))
    print(json.dumps(printable, indent=2))

    if args.output_json:
        out_path = Path(args.output_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if len(raw_results) > 1:
            payload = {
                "raw_results": raw_results,
                "summary": summarize_results(raw_results, seed=args.seed),
                "ordered_labels": list(raw_results.keys()),
                "config": {
                    "n_games": args.n_games,
                    "J": args.J,
                    "bp_iters": args.bp_iters,
                    "grid_size": args.grid_size,
                    "seed": args.seed,
                    "max_shots": args.max_shots,
                },
            }
        else:
            only_label = next(iter(raw_results))
            payload = {
                "agent": only_label,
                "summary": run_summaries[only_label],
                "raw_shots": raw_results[only_label],
            }
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote results to {out_path}")


if __name__ == "__main__":
    main()

