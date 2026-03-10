#!/usr/bin/env python
# coding: utf-8
"""
Battleship inference via the Ising model and Belief Propagation.

Each cell of the opponent's 10×10 board is modeled as a binary Ising spin:
    x_{i,j} = +1  →  cell contains part of a ship
    x_{i,j} = -1  →  cell is water

The joint distribution is:
    P(x) ∝ exp( J · Σ_{<s,t>} x_s x_t  +  Σ_{i,j} h_{i,j} x_{i,j} )

where the sum over <s,t> runs over all horizontally or vertically adjacent
cell pairs and the external fields h_{i,j} encode hit/miss evidence.

Belief Propagation (BP) is run in the half-LLR message convention used
throughout this codebase (see bpsol.py).  After convergence the marginal
    P(x_{i,j} = +1 | evidence) = σ(2 b_{i,j})
is used to rank unrevealed cells: the agent always shoots at the cell with
the highest posterior ship probability.
"""

from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# Ising / BP model
# ---------------------------------------------------------------------------

class BattleshipIsing:
    """Ising model + Belief Propagation inference for Battleship.

    Attributes
    ----------
    N : int
        Grid side length (default 10).
    J : float
        Ferromagnetic coupling strength (J > 0).  Adjacent cells tend to share
        the same state, capturing the spatial contiguity of ships.
    h_prior : float
        External field for cells whose state is not yet known.
        Set to  ½ log(p/(1−p))  where p is the prior ship-cell density.
        For the standard fleet (17 ship cells / 100 total), p ≈ 0.17 so
        h_prior ≈ −0.793.  Negative because most cells are water.
    H_OBS : float  (class-level constant)
        Field magnitude clamped to confirmed hits (+H_OBS) or misses (−H_OBS).
        Should be large enough that the posterior is nearly deterministic but
        small enough to avoid arctanh overflow.

    Message arrays (half-LLR convention, matching bpsol.py)
    --------------------------------------------------------
    All messages  m = ½ log(ν(+1)/ν(−1)).

    hhorright[i, j]  message from cell (i, j)   → (i, j+1)   shape (N, N−1)
    hhorleft [i, j]  message from cell (i, j+1) → (i, j)     shape (N, N−1)
    hvertdown[i, j]  message from cell (i, j)   → (i+1, j)   shape (N−1, N)
    hvertup  [i, j]  message from cell (i+1, j) → (i, j)     shape (N−1, N)

    BP update rule
    --------------
    The message from node s to neighbouring node t is:

        m_{s→t}^{new} = h_s + Σ_{u ∈ N(s) \\ {t}}  f_J( m_{u→s}^{current} )

    where  f_J(m) = arctanh( tanh(J) · tanh(m) )  is the Ising edge factor
    applied in the half-LLR domain.

    Belief
    ------
        b_{i,j} = h_{i,j} + Σ_{k ∈ N(i,j)}  f_J( m_{k→(i,j)} )
        P(ship at (i,j)) = exp(2 b) / (1 + exp(2 b))
    """

    H_OBS: float = 10.0

    def __init__(
        self,
        grid_size: int = 10,
        J: float = 0.5,
        h_prior: float | None = None,
    ) -> None:
        """
        Parameters
        ----------
        grid_size : int
            Side length of the square grid.
        J : float
            Ising coupling constant.  Larger values mean stronger ship-cluster
            correlations; the 2-D Ising critical point is J_c ≈ 0.44, so
            values slightly above that yield strong but not diverging
            correlations.
        h_prior : float or None
            Prior external field for unknown cells.  If None, computed from
            the standard Battleship fleet density:
                h_prior = ½ log( 17/83 ) ≈ −0.793
        """
        self.N = grid_size
        self.J = float(J)

        if h_prior is None:
            p = 17.0 / 100.0        # 17 ship cells on a 10×10 board
            self.h_prior = 0.5 * np.log(p / (1.0 - p))
        else:
            self.h_prior = float(h_prior)

        # External fields: start at the prior, updated on each observation.
        self.h = np.full((grid_size, grid_size), self.h_prior, dtype=float)

        # Boolean mask: True where the cell has been shot at.
        self.revealed = np.zeros((grid_size, grid_size), dtype=bool)

        self._reset_messages()

    # ------------------------------------------------------------------
    # Message initialisation
    # ------------------------------------------------------------------

    def _reset_messages(self) -> None:
        """Set all BP messages to zero (uninformative)."""
        N = self.N
        self.hhorright = np.zeros((N, N - 1))
        self.hhorleft  = np.zeros((N, N - 1))
        self.hvertdown = np.zeros((N - 1, N))
        self.hvertup   = np.zeros((N - 1, N))

    # ------------------------------------------------------------------
    # Core BP primitives
    # ------------------------------------------------------------------

    def _f(self, m: np.ndarray) -> np.ndarray:
        """Ising edge factor in the half-LLR domain.

            f_J(m) = arctanh( tanh(J) · tanh(m) )

        Messages are clipped to ±15 before the inner tanh to prevent the
        argument of arctanh from reaching ±1 due to floating-point saturation.
        """
        return np.arctanh(np.tanh(self.J) * np.tanh(np.clip(m, -15.0, 15.0)))

    def _bp_step(self) -> float:
        """Execute one parallel BP sweep and return the mean message change.

        All four message arrays are updated simultaneously (flooded schedule),
        matching the parallel update used in bpsol.py.

        For each directed edge (s → t) the update sums f_J of every incoming
        message at s except the one arriving from t:

            hhorright_new[i, j]  (s=(i,j),   t=(i,j+1))
                = h[i, j]
                + f_J( hhorright[i, j−1] )  if j > 0   (from left)
                + f_J( hvertdown[i−1, j] )  if i > 0   (from above)
                + f_J( hvertup  [i,   j] )  if i < N−1 (from below)

        and symmetrically for the other three directions.  The boundary
        conditions are handled automatically by the slicing arithmetic: terms
        that would require out-of-bounds indices simply have zero contributions
        because the corresponding slice is empty.
        """
        N = self.N

        frr = self._f(self.hhorright)   # shape (N, N−1)
        frl = self._f(self.hhorleft)    # shape (N, N−1)
        frd = self._f(self.hvertdown)   # shape (N−1, N)
        fru = self._f(self.hvertup)     # shape (N−1, N)

        # ---- hhorright[i, j] : (i,j) → (i,j+1), excludes hhorleft[i,j] ----
        # base term h[i, j]  for j = 0 … N−2
        new_rr = self.h[:, :N - 1].copy()
        new_rr[:, 1:]   += frr[:, :N - 2]    # from left:  f(hhorright[i, j−1])
        new_rr[1:, :]   += frd[:, :N - 1]    # from above: f(hvertdown[i−1, j])
        new_rr[:N - 1, :] += fru[:, :N - 1]  # from below: f(hvertup[i, j])

        # ---- hhorleft[i, j] : (i,j+1) → (i,j), excludes hhorright[i,j] ----
        # base term h[i, j+1]  for j = 0 … N−2
        new_rl = self.h[:, 1:].copy()
        new_rl[:, :N - 2] += frl[:, 1:]      # from right: f(hhorleft[i, j+1])
        new_rl[1:, :]     += frd[:, 1:]       # from above: f(hvertdown[i−1, j+1])
        new_rl[:N - 1, :] += fru[:, 1:]       # from below: f(hvertup[i, j+1])

        # ---- hvertdown[i, j] : (i,j) → (i+1,j), excludes hvertup[i,j] ----
        # base term h[i, j]  for i = 0 … N−2
        new_rd = self.h[:N - 1, :].copy()
        new_rd[:, 1:]     += frr[:N - 1, :]   # from left:  f(hhorright[i, j−1])
        new_rd[:, :N - 1] += frl[:N - 1, :]   # from right: f(hhorleft[i, j])
        new_rd[1:, :]     += frd[:N - 2, :]   # from above: f(hvertdown[i−1, j])

        # ---- hvertup[i, j] : (i+1,j) → (i,j), excludes hvertdown[i,j] ----
        # base term h[i+1, j]  for i = 0 … N−2
        new_ru = self.h[1:, :].copy()
        new_ru[:, 1:]     += frr[1:, :]        # from left:  f(hhorright[i+1, j−1])
        new_ru[:, :N - 1] += frl[1:, :]        # from right: f(hhorleft[i+1, j])
        new_ru[:N - 2, :] += fru[1:, :]        # from below: f(hvertup[i+1, j])

        delta = float(np.mean(np.abs(new_rr - self.hhorright)))

        self.hhorright = new_rr
        self.hhorleft  = new_rl
        self.hvertdown = new_rd
        self.hvertup   = new_ru

        return delta

    # ------------------------------------------------------------------
    # Public inference API
    # ------------------------------------------------------------------

    def run_bp(self, num_iter: int = 50, tol: float = 1e-6) -> list[float]:
        """Run parallel BP for up to *num_iter* iterations.

        Parameters
        ----------
        num_iter : int
            Maximum number of sweeps.
        tol : float
            Stop early when the mean absolute change in hhorright drops below
            this threshold (convergence criterion).

        Returns
        -------
        list of float
            Per-iteration mean message change, useful for diagnostics.
        """
        history: list[float] = []
        for _ in range(num_iter):
            delta = self._bp_step()
            history.append(delta)
            if delta < tol:
                break
        return history

    def observe(self, row: int, col: int, is_hit: bool) -> None:
        """Record a shot result and reset messages for re-convergence.

        Parameters
        ----------
        row, col : int
            Zero-indexed cell coordinates.
        is_hit : bool
            True if the shot was a hit; False if a miss.
        """
        self.h[row, col] = self.H_OBS if is_hit else -self.H_OBS
        self.revealed[row, col] = True
        self._reset_messages()

    def beliefs(self) -> np.ndarray:
        """Compute  P(x_{i,j} = +1 | evidence)  for every cell.

        Using the current message state (call run_bp first):

            b[i, j]  = h[i, j] + Σ_{k ∈ N(i,j)} f_J( m_{k→(i,j)} )
            P(ship)  = exp(2 b) / (1 + exp(2 b))

        Returns
        -------
        np.ndarray, shape (N, N)
            Posterior probability that each cell contains a ship, in [0, 1].
        """
        N = self.N
        frr = self._f(self.hhorright)
        frl = self._f(self.hhorleft)
        frd = self._f(self.hvertdown)
        fru = self._f(self.hvertup)

        b = self.h.copy()
        b[:, 1:]    += frr    # incoming from left  (frr shape (N, N−1))
        b[:, :N - 1] += frl   # incoming from right
        b[1:, :]    += frd    # incoming from above (frd shape (N−1, N))
        b[:N - 1, :] += fru   # incoming from below

        # half-LLR → probability: P(+1) = exp(2b) / (1 + exp(2b))
        return np.exp(2.0 * b) / (1.0 + np.exp(2.0 * b))

    def best_guess(self) -> tuple[int, int]:
        """Return the unrevealed cell with the highest posterior P(ship).

        Returns
        -------
        (row, col) : tuple[int, int]
        """
        p = self.beliefs()
        masked = np.where(self.revealed, -1.0, p)
        idx = int(np.argmax(masked))
        return divmod(idx, self.N)

    def reset(self) -> None:
        """Clear all observations and messages, returning to the prior."""
        self.h[:] = self.h_prior
        self.revealed[:] = False
        self._reset_messages()


# ---------------------------------------------------------------------------
# Battleship game environment
# ---------------------------------------------------------------------------

class BattleshipGame:
    """Simulates a fixed opponent board for testing the Ising inference agent.

    Standard fleet
    --------------
    Carrier      5 cells
    Battleship   4 cells
    Cruiser      3 cells
    Submarine    3 cells
    Destroyer    2 cells
    Total:      17 ship cells on a 10×10 board  (density ≈ 0.17)
    """

    SHIP_LENGTHS: list[int] = [5, 4, 3, 3, 2]
    SHIP_NAMES:   list[str] = [
        "Carrier", "Battleship", "Cruiser", "Submarine", "Destroyer"
    ]

    def __init__(self, grid_size: int = 10, seed: int | None = None) -> None:
        """
        Parameters
        ----------
        grid_size : int
            Side length of the board.
        seed : int or None
            Random seed for reproducible ship placement.
        """
        self.N = grid_size
        self.rng = np.random.default_rng(seed)
        self.grid = np.zeros((grid_size, grid_size), dtype=int)
        self._place_ships()

    def _place_ships(self) -> None:
        """Randomly place all ships with no overlaps."""
        for length, name in zip(self.SHIP_LENGTHS, self.SHIP_NAMES):
            placed = False
            for _ in range(100_000):
                horizontal = bool(self.rng.integers(0, 2))
                if horizontal:
                    r = int(self.rng.integers(0, self.N))
                    c = int(self.rng.integers(0, self.N - length + 1))
                    cells = [(r, c + k) for k in range(length)]
                else:
                    r = int(self.rng.integers(0, self.N - length + 1))
                    c = int(self.rng.integers(0, self.N))
                    cells = [(r + k, c) for k in range(length)]

                if all(self.grid[rr, cc] == 0 for rr, cc in cells):
                    for rr, cc in cells:
                        self.grid[rr, cc] = 1
                    placed = True
                    break

            if not placed:
                raise RuntimeError(
                    f"Could not place {name} (length {length}) after many attempts"
                )

    def shoot(self, row: int, col: int) -> bool:
        """Fire at (row, col).  Returns True on a hit."""
        return bool(self.grid[row, col])

    @property
    def total_ship_cells(self) -> int:
        return int(self.grid.sum())

    def all_sunk(self, hits: np.ndarray) -> bool:
        """Return True when every ship cell has been revealed as a hit.

        Parameters
        ----------
        hits : np.ndarray of bool, shape (N, N)
            Boolean mask where True indicates a confirmed hit.
        """
        return bool((self.grid & hits.astype(int)).sum() == self.total_ship_cells)
