#!/usr/bin/env python
# coding: utf-8
"""
mcts.py — Monte Carlo Tree Search baseline for Battleship.

This implementation uses determinized planning over hidden ship layouts that are
sampled to be consistent with current hit/miss observations. The search tree is
built over observation states, while the sampled hidden board provides concrete
hit/miss outcomes during each simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from battleship_ising import BattleshipGame


GRID_SIZE = 10
SHIP_LENGTHS = list(BattleshipGame.SHIP_LENGTHS)

_PLACEMENTS_CACHE: dict[int, dict[int, list[tuple[tuple[int, int], ...]]]] = {}


def _sample_argmax(score: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    max_score = float(score.max())
    choices = np.argwhere(score == max_score)
    row, col = choices[int(rng.integers(len(choices)))]
    return int(row), int(col)


def _build_placements(grid_size: int) -> dict[int, list[tuple[tuple[int, int], ...]]]:
    placements_by_length: dict[int, list[tuple[tuple[int, int], ...]]] = {}
    for length in sorted(set(SHIP_LENGTHS)):
        placements: list[tuple[tuple[int, int], ...]] = []
        for row in range(grid_size):
            for col in range(grid_size - length + 1):
                placements.append(tuple((row, col + offset) for offset in range(length)))
        for col in range(grid_size):
            for row in range(grid_size - length + 1):
                placements.append(tuple((row + offset, col) for offset in range(length)))
        placements_by_length[length] = placements
    return placements_by_length


def _get_placements(grid_size: int) -> dict[int, list[tuple[tuple[int, int], ...]]]:
    if grid_size not in _PLACEMENTS_CACHE:
        _PLACEMENTS_CACHE[grid_size] = _build_placements(grid_size)
    return _PLACEMENTS_CACHE[grid_size]


def _probability_density_scores(
    revealed: np.ndarray,
    hit_mask: np.ndarray,
    ship_lengths: list[int],
    hit_bonus: float = 12.0,
) -> np.ndarray:
    """Placement-count heuristic used as the default rollout policy."""
    grid_size = revealed.shape[0]
    score = np.zeros((grid_size, grid_size), dtype=np.float64)

    def _accumulate(require_hit_overlap: bool) -> np.ndarray:
        out = np.zeros((grid_size, grid_size), dtype=np.float64)
        for length in ship_lengths:
            for row in range(grid_size):
                for col in range(grid_size - length + 1):
                    rev = revealed[row, col : col + length]
                    hits = hit_mask[row, col : col + length]
                    misses = rev & ~hits
                    if np.any(misses):
                        continue
                    hit_count = int(hits.sum())
                    if require_hit_overlap and hit_count == 0:
                        continue
                    weight = 1.0 + hit_bonus * hit_count
                    for offset in range(length):
                        rr, cc = row, col + offset
                        if not revealed[rr, cc]:
                            out[rr, cc] += weight

            for col in range(grid_size):
                for row in range(grid_size - length + 1):
                    rev = revealed[row : row + length, col]
                    hits = hit_mask[row : row + length, col]
                    misses = rev & ~hits
                    if np.any(misses):
                        continue
                    hit_count = int(hits.sum())
                    if require_hit_overlap and hit_count == 0:
                        continue
                    weight = 1.0 + hit_bonus * hit_count
                    for offset in range(length):
                        rr, cc = row + offset, col
                        if not revealed[rr, cc]:
                            out[rr, cc] += weight
        return out

    active_hits = bool(hit_mask.any())
    score = _accumulate(require_hit_overlap=active_hits)
    if active_hits and score.max() == 0:
        score = _accumulate(require_hit_overlap=False)
    if score.max() == 0:
        score[~revealed] = 1.0
    return score


def estimate_posterior_occupancy(
    revealed: np.ndarray,
    hit_mask: np.ndarray,
    n_samples: int = 32,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Approximate the posterior occupancy map by consistent-board sampling."""
    if rng is None:
        rng = np.random.default_rng()

    posterior = np.zeros(revealed.shape, dtype=np.float64)
    for _ in range(max(n_samples, 1)):
        posterior += sample_consistent_board(revealed, hit_mask, rng=rng)
    posterior /= max(n_samples, 1)
    posterior[revealed & hit_mask] = 1.0
    posterior[revealed & ~hit_mask] = 0.0
    return posterior


def bayesian_surprise(
    prior_posterior: np.ndarray,
    posterior: np.ndarray,
    eps: float = 1e-6,
) -> tuple[float, np.ndarray]:
    """Return mean Bernoulli KL surprise and its per-cell contribution map."""
    p = np.clip(prior_posterior.astype(np.float64), eps, 1.0 - eps)
    q = np.clip(posterior.astype(np.float64), eps, 1.0 - eps)
    kl_map = q * np.log(q / p) + (1.0 - q) * np.log((1.0 - q) / (1.0 - p))
    return float(np.mean(kl_map)), kl_map


def sample_consistent_board(
    revealed: np.ndarray,
    hit_mask: np.ndarray,
    ship_lengths: Optional[list[int]] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample a plausible hidden board consistent with current observations.

    The sampler greedily assigns ships to cover known hits, then places the
    remaining ships uniformly over placements that do not contradict misses or
    already occupied cells. This is an approximate posterior sampler, chosen for
    speed so MCTS can be used as a practical planning baseline.
    """
    if rng is None:
        rng = np.random.default_rng()

    grid_size = revealed.shape[0]
    miss_mask = revealed & ~hit_mask
    placements_by_length = _get_placements(grid_size)
    lengths_template = sorted(SHIP_LENGTHS if ship_lengths is None else ship_lengths, reverse=True)
    hit_positions = {tuple(pos) for pos in np.argwhere(hit_mask)}

    for _ in range(64):
        occupied = np.zeros((grid_size, grid_size), dtype=bool)
        remaining_lengths = list(lengths_template)
        uncovered_hits = set(hit_positions)

        # Greedily explain observed hits first.
        while uncovered_hits:
            seed = next(iter(uncovered_hits))
            candidates: list[tuple[int, tuple[tuple[int, int], ...], int, int]] = []

            for length in sorted(set(remaining_lengths), reverse=True):
                for placement in placements_by_length[length]:
                    if seed not in placement:
                        continue
                    if any(miss_mask[row, col] or occupied[row, col] for row, col in placement):
                        continue
                    covered_hits = sum((row, col) in uncovered_hits for row, col in placement)
                    if covered_hits == 0:
                        continue
                    extra_cells = length - covered_hits
                    candidates.append((length, placement, covered_hits, extra_cells))

            if not candidates:
                break

            best_hits = max(item[2] for item in candidates)
            filtered = [item for item in candidates if item[2] == best_hits]
            min_extra = min(item[3] for item in filtered)
            filtered = [item for item in filtered if item[3] == min_extra]
            length, placement, _, _ = filtered[int(rng.integers(len(filtered)))]

            remaining_lengths.remove(length)
            for row, col in placement:
                occupied[row, col] = True
                uncovered_hits.discard((row, col))
        else:
            # Place remaining ships at random among legal placements.
            placement_failed = False
            for length in remaining_lengths:
                candidates = [
                    placement
                    for placement in placements_by_length[length]
                    if not any(miss_mask[row, col] or occupied[row, col] for row, col in placement)
                ]
                if not candidates:
                    placement_failed = True
                    break
                placement = candidates[int(rng.integers(len(candidates)))]
                for row, col in placement:
                    occupied[row, col] = True

            if not placement_failed and all(occupied[row, col] for row, col in hit_positions):
                return occupied.astype(np.int8)

    return _sample_consistent_board_exact(
        revealed,
        hit_mask,
        ship_lengths=ship_lengths,
        rng=rng,
    )


def _sample_consistent_board_exact(
    revealed: np.ndarray,
    hit_mask: np.ndarray,
    ship_lengths: Optional[list[int]] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Exact backtracking sampler used as a robustness fallback."""
    if rng is None:
        rng = np.random.default_rng()

    grid_size = revealed.shape[0]
    miss_mask = revealed & ~hit_mask
    placements_by_length = _get_placements(grid_size)
    lengths = sorted(SHIP_LENGTHS if ship_lengths is None else ship_lengths, reverse=True)
    occupied = np.zeros((grid_size, grid_size), dtype=bool)

    hit_positions = [tuple(pos) for pos in np.argwhere(hit_mask)]
    unique_remaining_lengths = sorted(set(lengths), reverse=True)

    def _placement_valid(placement: tuple[tuple[int, int], ...]) -> bool:
        return not any(miss_mask[row, col] or occupied[row, col] for row, col in placement)

    def _hits_coverable(next_idx: int) -> bool:
        uncovered_hits = [(row, col) for row, col in hit_positions if not occupied[row, col]]
        if not uncovered_hits:
            return True

        remaining_lengths = lengths[next_idx:]
        if not remaining_lengths:
            return False

        candidate_lengths = [length for length in unique_remaining_lengths if length in remaining_lengths]
        for hit_row, hit_col in uncovered_hits:
            coverable = False
            for length in candidate_lengths:
                for placement in placements_by_length[length]:
                    if (hit_row, hit_col) not in placement:
                        continue
                    if _placement_valid(placement):
                        coverable = True
                        break
                if coverable:
                    break
            if not coverable:
                return False
        return True

    if not _hits_coverable(0):
        raise RuntimeError("Observations are inconsistent with the standard fleet.")

    def _backtrack(idx: int) -> bool:
        if idx == len(lengths):
            return all(occupied[row, col] for row, col in hit_positions)

        length = lengths[idx]
        candidates = [p for p in placements_by_length[length] if _placement_valid(p)]
        if not candidates:
            return False

        order = rng.permutation(len(candidates))
        for candidate_idx in order:
            placement = candidates[int(candidate_idx)]
            for row, col in placement:
                occupied[row, col] = True

            if _hits_coverable(idx + 1) and _backtrack(idx + 1):
                return True

            for row, col in placement:
                occupied[row, col] = False
        return False

    if not _backtrack(0):
        raise RuntimeError("Could not sample a board consistent with observations.")

    return occupied.astype(np.int8)


def _step(
    revealed: np.ndarray,
    hit_mask: np.ndarray,
    hidden_board: np.ndarray,
    action: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, bool]:
    row, col = action
    next_revealed = revealed.copy()
    next_hit_mask = hit_mask.copy()
    next_revealed[row, col] = True
    is_hit = bool(hidden_board[row, col])
    if is_hit:
        next_hit_mask[row, col] = True
    return next_revealed, next_hit_mask, is_hit


def _terminal(
    revealed: np.ndarray,
    hit_mask: np.ndarray,
    hidden_board: np.ndarray,
) -> bool:
    ship_mask = hidden_board.astype(bool)
    return bool(np.all(hit_mask[ship_mask]) or np.all(revealed))


@dataclass
class _ActionStats:
    visits: int = 0
    total_value: float = 0.0
    children: dict[bool, "_SearchNode"] = field(default_factory=dict)


class _SearchNode:
    def __init__(self, revealed: np.ndarray, hit_mask: np.ndarray) -> None:
        self.revealed = revealed
        self.hit_mask = hit_mask
        self.visits = 0
        self.action_stats: dict[tuple[int, int], _ActionStats] = {}
        self.unexpanded_actions = [tuple(pos) for pos in np.argwhere(~revealed)]


class MCTSAgent:
    """Determinized MCTS baseline for Battleship planning."""

    def __init__(
        self,
        grid_size: int = GRID_SIZE,
        ship_lengths: Optional[list[int]] = None,
        n_simulations: int = 96,
        rollout_depth: int = 18,
        exploration: float = 1.4,
        rollout_hit_bonus: float = 12.0,
        seed: Optional[int] = 0,
    ) -> None:
        self.N = grid_size
        self.ship_lengths = SHIP_LENGTHS if ship_lengths is None else ship_lengths
        self.n_simulations = int(n_simulations)
        self.rollout_depth = int(rollout_depth)
        self.exploration = float(exploration)
        self.rollout_hit_bonus = float(rollout_hit_bonus)
        self.rng = np.random.default_rng(seed)

        self.revealed = np.zeros((grid_size, grid_size), dtype=bool)
        self.hit_mask = np.zeros((grid_size, grid_size), dtype=bool)

    def reset(self) -> None:
        self.revealed[:] = False
        self.hit_mask[:] = False

    def observe(self, row: int, col: int, is_hit: bool) -> None:
        self.revealed[row, col] = True
        self.hit_mask[row, col] = bool(is_hit)

    def _select_action(self, node: _SearchNode) -> tuple[int, int]:
        best_action: Optional[tuple[int, int]] = None
        best_value = -np.inf

        for action, stats in node.action_stats.items():
            if stats.visits == 0:
                return action
            exploit = stats.total_value / stats.visits
            explore = self.exploration * np.sqrt(np.log(node.visits + 1.0) / stats.visits)
            score = exploit + explore
            if score > best_value:
                best_value = score
                best_action = action

        if best_action is None:
            unknown = np.argwhere(~node.revealed)
            row, col = unknown[int(self.rng.integers(len(unknown)))]
            return int(row), int(col)
        return best_action

    def _rollout(
        self,
        revealed: np.ndarray,
        hit_mask: np.ndarray,
        hidden_board: np.ndarray,
    ) -> int:
        rollout_steps = 0
        current_revealed = revealed.copy()
        current_hit_mask = hit_mask.copy()

        while rollout_steps < self.rollout_depth and not _terminal(
            current_revealed, current_hit_mask, hidden_board
        ):
            score = _probability_density_scores(
                current_revealed,
                current_hit_mask,
                self.ship_lengths,
                hit_bonus=self.rollout_hit_bonus,
            )
            action = _sample_argmax(score, self.rng)
            current_revealed, current_hit_mask, _ = _step(
                current_revealed, current_hit_mask, hidden_board, action
            )
            rollout_steps += 1

        while not _terminal(current_revealed, current_hit_mask, hidden_board):
            unknown = np.argwhere(~current_revealed)
            row, col = unknown[int(self.rng.integers(len(unknown)))]
            current_revealed, current_hit_mask, _ = _step(
                current_revealed, current_hit_mask, hidden_board, (int(row), int(col))
            )
            rollout_steps += 1

        return rollout_steps

    def _run_search(self) -> _SearchNode:
        root = _SearchNode(self.revealed.copy(), self.hit_mask.copy())
        if not root.unexpanded_actions:
            return root

        for _ in range(self.n_simulations):
            hidden_board = sample_consistent_board(
                self.revealed,
                self.hit_mask,
                ship_lengths=self.ship_lengths,
                rng=self.rng,
            )

            node = root
            path_nodes = [root]
            path_edges: list[tuple[_SearchNode, tuple[int, int]]] = []
            total_steps = 0

            while not _terminal(node.revealed, node.hit_mask, hidden_board):
                if node.unexpanded_actions:
                    action_idx = int(self.rng.integers(len(node.unexpanded_actions)))
                    action = node.unexpanded_actions.pop(action_idx)
                    stats = node.action_stats.setdefault(action, _ActionStats())

                    next_revealed, next_hit_mask, is_hit = _step(
                        node.revealed, node.hit_mask, hidden_board, action
                    )
                    child = stats.children.get(is_hit)
                    if child is None:
                        child = _SearchNode(next_revealed, next_hit_mask)
                        stats.children[is_hit] = child

                    path_edges.append((node, action))
                    node = child
                    path_nodes.append(node)
                    total_steps += 1
                    break

                action = self._select_action(node)
                stats = node.action_stats[action]
                next_revealed, next_hit_mask, is_hit = _step(
                    node.revealed, node.hit_mask, hidden_board, action
                )
                child = stats.children.get(is_hit)
                if child is None:
                    child = _SearchNode(next_revealed, next_hit_mask)
                    stats.children[is_hit] = child

                path_edges.append((node, action))
                node = child
                path_nodes.append(node)
                total_steps += 1

                if stats.visits == 0:
                    break

            if not _terminal(node.revealed, node.hit_mask, hidden_board):
                total_steps += self._rollout(node.revealed, node.hit_mask, hidden_board)

            value = 1.0 / max(total_steps, 1)

            for visited_node in path_nodes:
                visited_node.visits += 1
            for parent, action in path_edges:
                stats = parent.action_stats.setdefault(action, _ActionStats())
                stats.visits += 1
                stats.total_value += value

        return root

    def beliefs(self) -> np.ndarray:
        root = self._run_search()
        score = np.zeros((self.N, self.N), dtype=np.float64)

        if root.action_stats:
            for action, stats in root.action_stats.items():
                row, col = action
                score[row, col] = float(stats.visits)
        else:
            score[~self.revealed] = 1.0

        total = score.sum()
        return score / total if total > 0 else score

    def best_guess(self) -> tuple[int, int]:
        beliefs = self.beliefs()
        return divmod(int(np.argmax(beliefs)), self.N)
