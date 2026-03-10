#!/usr/bin/env python
# coding: utf-8
"""
Demo: Battleship inference with the Ising model and Belief Propagation.

Runs two experiments:
  1. Prior belief map  – show P(ship) before any shots (should reflect the
     uniform 17 % prior modulated by ferromagnetic correlations).
  2. Full game simulation  – the BP agent repeatedly picks the unrevealed
     cell with the highest posterior P(ship), fires, records the result, and
     re-runs BP until all ships are sunk.  Snapshot belief maps are shown
     every SNAPSHOT_EVERY shots.

Usage
-----
    python battleship_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from battleship_ising import BattleshipIsing, BattleshipGame

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GRID_SIZE      = 10
J              = 0.5      # Ising coupling strength
BP_ITERS       = 60       # BP sweeps per move
SNAPSHOT_EVERY = 5        # print a belief-map snapshot every N shots
MAX_SHOTS      = 100      # safety cap on the number of shots per game
SEED           = 42       # reproducible board layout
COL_LABELS     = list("ABCDEFGHIJ")


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _annotate_board(ax, model: BattleshipIsing) -> None:
    """Overlay X (hit) and · (miss) markers on a matplotlib axes."""
    for r in range(model.N):
        for c in range(model.N):
            if model.revealed[r, c]:
                is_hit = model.h[r, c] > 0
                ax.text(
                    c, r,
                    "✕" if is_hit else "·",
                    ha="center", va="center",
                    color="white" if is_hit else "deepskyblue",
                    fontsize=11, fontweight="bold",
                )


def plot_belief_map(
    model: BattleshipIsing,
    game:  BattleshipGame,
    title: str = "Belief map",
    show_true: bool = True,
) -> None:
    """Side-by-side: true board (left) and BP posterior P(ship) (right)."""
    ncols = 2 if show_true else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5.5))

    if not show_true:
        axes = [axes]

    if show_true:
        ax = axes[0]
        ship_map = game.grid.astype(float)
        im0 = ax.imshow(ship_map, cmap="Greens", vmin=0, vmax=1.4)
        ax.set_title("True board (hidden from agent)", fontsize=11)
        ax.set_xticks(range(model.N))
        ax.set_xticklabels(COL_LABELS)
        ax.set_yticks(range(model.N))
        _annotate_board(ax, model)

    ax = axes[-1]
    p = model.beliefs()
    im1 = ax.imshow(p, cmap="hot", vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(range(model.N))
    ax.set_xticklabels(COL_LABELS)
    ax.set_yticks(range(model.N))
    _annotate_board(ax, model)
    plt.colorbar(im1, ax=ax, label="P(ship)")

    plt.tight_layout()
    plt.show()


def plot_convergence(history: list[float], shot_num: int) -> None:
    """Log-scale plot of per-iteration message change after a shot."""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.semilogy(history)
    ax.set_xlabel("BP iteration")
    ax.set_ylabel("mean |Δm|")
    ax.set_title(f"BP convergence after shot {shot_num}")
    ax.grid(True)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Experiment 1: prior belief map
# ---------------------------------------------------------------------------

def demo_prior() -> None:
    """Show the belief map before any shots are taken."""
    print("=" * 60)
    print("Experiment 1: prior belief map (no shots yet)")
    print("=" * 60)

    model = BattleshipIsing(grid_size=GRID_SIZE, J=J)
    game  = BattleshipGame(grid_size=GRID_SIZE, seed=SEED)

    conv = model.run_bp(num_iter=BP_ITERS)
    p    = model.beliefs()

    print(f"  BP converged in {len(conv)} iterations "
          f"(final Δm = {conv[-1]:.2e})")
    print(f"  Prior P(ship): min={p.min():.3f}  mean={p.mean():.3f}  "
          f"max={p.max():.3f}")
    print(f"  Expected mean ≈ {17/100:.3f} (17 ship cells / 100 total)\n")

    plot_belief_map(model, game, title="Prior P(ship | no observations)")


# ---------------------------------------------------------------------------
# Experiment 2: full game simulation
# ---------------------------------------------------------------------------

def run_game(
    seed:          int  = SEED,
    J:             float = J,
    bp_iters:      int   = BP_ITERS,
    snapshot_every: int  = SNAPSHOT_EVERY,
    verbose:       bool  = True,
) -> dict:
    """Run one complete game and return statistics.

    The agent policy is purely greedy: shoot at argmax P(ship | evidence).

    Returns
    -------
    dict with keys:
        shots_to_finish : int
        shots : list of (row, col)
        hits  : list of bool
    """
    game  = BattleshipGame(grid_size=GRID_SIZE, seed=seed)
    model = BattleshipIsing(grid_size=GRID_SIZE, J=J)

    shots:  list[tuple[int, int]] = []
    hits:   list[bool]            = []
    hit_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)

    if verbose:
        print("=" * 60)
        print(f"Experiment 2: full game  (seed={seed}, J={J})")
        print("=" * 60)
        print(f"  Fleet occupies {game.total_ship_cells} / "
              f"{GRID_SIZE**2} cells ({game.total_ship_cells}%).")
        print()

    for shot_num in range(1, MAX_SHOTS + 1):
        conv = model.run_bp(num_iter=bp_iters)

        # Greedy policy: highest posterior ship probability
        row, col = model.best_guess()
        is_hit   = game.shoot(row, col)

        model.observe(row, col, is_hit)

        shots.append((row, col))
        hits.append(is_hit)
        if is_hit:
            hit_mask[row, col] = True

        if verbose:
            result = "HIT  ●" if is_hit else "miss ○"
            bp_iters_taken = len(conv)
            print(f"  Shot {shot_num:3d}  {COL_LABELS[col]}{row+1:2d}  →  "
                  f"{result}   (BP: {bp_iters_taken} iters, "
                  f"Δm={conv[-1]:.1e})")

        # Snapshot belief maps at regular intervals
        if shot_num % snapshot_every == 0 or game.all_sunk(hit_mask):
            n_hits   = int(hit_mask.sum())
            n_misses = shot_num - n_hits
            plot_belief_map(
                model, game,
                title=(f"After shot {shot_num}  "
                       f"({n_hits} hits, {n_misses} misses)"),
            )

        if game.all_sunk(hit_mask):
            if verbose:
                print(f"\n  ✓  All ships sunk in {shot_num} shots!")
            return {
                "shots_to_finish": shot_num,
                "shots": shots,
                "hits":  hits,
            }

    if verbose:
        print(f"\n  Game not finished in {MAX_SHOTS} shots.")
    return {
        "shots_to_finish": MAX_SHOTS,
        "shots": shots,
        "hits":  hits,
    }


# ---------------------------------------------------------------------------
# Experiment 3: convergence comparison across J values
# ---------------------------------------------------------------------------

def demo_coupling_sweep() -> None:
    """Show how belief maps change with the coupling strength J."""
    J_vals  = [0.0, 0.3, 0.5, 0.8]
    seed    = SEED

    fig, axes = plt.subplots(2, len(J_vals), figsize=(5 * len(J_vals), 9))

    # Inject a known hit at (5, 4) to visualise how it propagates
    hit_row, hit_col = 5, 4

    for col_idx, j_val in enumerate(J_vals):
        model = BattleshipIsing(grid_size=GRID_SIZE, J=j_val)
        game  = BattleshipGame(grid_size=GRID_SIZE, seed=seed)

        # Prior beliefs (no observations)
        model.run_bp(num_iter=BP_ITERS)
        p_prior = model.beliefs()

        # Beliefs after one hit
        model.observe(hit_row, hit_col, is_hit=True)
        model.run_bp(num_iter=BP_ITERS)
        p_hit = model.beliefs()

        for row_idx, (p, label) in enumerate(
            [(p_prior, "prior"), (p_hit, f"after hit at {COL_LABELS[hit_col]}{hit_row+1}")]
        ):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(p, cmap="hot", vmin=0, vmax=1)
            ax.set_title(f"J={j_val:.1f}\n{label}", fontsize=9)
            ax.set_xticks(range(GRID_SIZE))
            ax.set_xticklabels(COL_LABELS, fontsize=7)
            ax.set_yticks(range(GRID_SIZE))
            if row_idx == 1:
                # Mark the hit cell
                ax.text(hit_col, hit_row, "✕", ha="center", va="center",
                        color="cyan", fontsize=12, fontweight="bold")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Effect of coupling strength J on BP beliefs\n"
        "(top: prior, bottom: after one hit)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_prior()
    results = run_game()
    demo_coupling_sweep()
