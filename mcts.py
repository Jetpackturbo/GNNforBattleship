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
from typing import Callable, Optional

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


def _normalize_action_probs(score: np.ndarray, revealed: np.ndarray) -> np.ndarray:
    """Normalize nonnegative action scores over unrevealed cells."""
    probs = np.zeros_like(score, dtype=np.float64)
    unknown = ~revealed
    if not np.any(unknown):
        return probs
    masked = np.clip(score[unknown].astype(np.float64), 0.0, None)
    total = float(masked.sum())
    if total <= 0.0:
        probs[unknown] = 1.0 / float(masked.size)
    else:
        probs[unknown] = masked / total
    return probs


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
    prior: float = 0.0
    visits: int = 0
    total_value: float = 0.0
    children: dict[bool, "_SearchNode"] = field(default_factory=dict)


class _SearchNode:
    def __init__(
        self,
        revealed: np.ndarray,
        hit_mask: np.ndarray,
        action_priors: np.ndarray,
    ) -> None:
        self.revealed = revealed
        self.hit_mask = hit_mask
        self.visits = 0
        self.action_stats: dict[tuple[int, int], _ActionStats] = {}
        unknown = np.argwhere(~revealed)
        uniform_prior = 1.0 / max(len(unknown), 1)
        for row, col in unknown:
            row_i, col_i = int(row), int(col)
            prior = float(action_priors[row_i, col_i]) if action_priors.size else uniform_prior
            if prior <= 0.0:
                prior = uniform_prior
            self.action_stats[(row_i, col_i)] = _ActionStats(prior=prior)


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
        gamma: float = 0.97,
        tree_policy: str = "uct_hybrid",
        prior_source: str = "blend",
        leaf_evaluator: str = "heuristic",
        leaf_samples: int = 16,
        policy_prior_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
        value_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        seed: Optional[int] = 0,
    ) -> None:
        self.N = grid_size
        self.ship_lengths = SHIP_LENGTHS if ship_lengths is None else ship_lengths
        self.n_simulations = int(n_simulations)
        self.rollout_depth = int(rollout_depth)
        self.exploration = float(exploration)
        self.rollout_hit_bonus = float(rollout_hit_bonus)
        self.gamma = float(gamma)
        self.tree_policy = str(tree_policy)
        self.prior_source = str(prior_source)
        self.leaf_evaluator = str(leaf_evaluator)
        self.leaf_samples = int(leaf_samples)
        self.policy_prior_fn = policy_prior_fn
        self.value_fn = value_fn
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
            exploit = (stats.total_value / stats.visits) if stats.visits > 0 else 0.0
            if stats.visits == 0:
                if self.tree_policy == "uct":
                    score = float("inf")
                elif self.tree_policy == "puct":
                    score = (
                        exploit
                        + self.exploration
                        * stats.prior
                        * np.sqrt(node.visits + 1.0)
                    )
                elif self.tree_policy == "uct_hybrid":
                    score = float("inf")
                else:
                    raise ValueError(f"Unknown tree_policy: {self.tree_policy}")
            else:
                if self.tree_policy == "uct":
                    explore = self.exploration * np.sqrt(
                        np.log(node.visits + 1.0) / stats.visits
                    )
                    score = exploit + explore
                elif self.tree_policy == "puct":
                    explore = (
                        self.exploration
                        * stats.prior
                        * np.sqrt(node.visits + 1.0)
                        / (1.0 + stats.visits)
                    )
                    score = exploit + explore
                elif self.tree_policy == "uct_hybrid":
                    uct_explore = self.exploration * np.sqrt(
                        np.log(node.visits + 1.0) / stats.visits
                    )
                    prior_bonus = (
                        0.25
                        * self.exploration
                        * stats.prior
                        * np.sqrt(node.visits + 1.0)
                        / (1.0 + stats.visits)
                    )
                    score = exploit + uct_explore + prior_bonus
                else:
                    raise ValueError(f"Unknown tree_policy: {self.tree_policy}")
            if score > best_value:
                best_value = score
                best_action = action

        if best_action is None:
            unknown = np.argwhere(~node.revealed)
            row, col = unknown[int(self.rng.integers(len(unknown)))]
            return int(row), int(col)
        return best_action

    def _heuristic_priors(self, revealed: np.ndarray, hit_mask: np.ndarray) -> np.ndarray:
        score = _probability_density_scores(
            revealed,
            hit_mask,
            self.ship_lengths,
            hit_bonus=self.rollout_hit_bonus,
        )
        return _normalize_action_probs(score, revealed)

    def _neural_priors(self, revealed: np.ndarray, hit_mask: np.ndarray) -> np.ndarray:
        if self.policy_prior_fn is None:
            return np.zeros(revealed.shape, dtype=np.float64)
        raw = self.policy_prior_fn(hit_mask.copy(), revealed.copy())
        return _normalize_action_probs(np.asarray(raw, dtype=np.float64), revealed)

    def _compute_action_priors(self, revealed: np.ndarray, hit_mask: np.ndarray) -> np.ndarray:
        heuristic = self._heuristic_priors(revealed, hit_mask)
        neural = self._neural_priors(revealed, hit_mask)

        if self.prior_source == "heuristic":
            return heuristic
        if self.prior_source == "neural":
            return neural if neural.sum() > 0 else heuristic
        if self.prior_source == "blend":
            if neural.sum() <= 0:
                return heuristic
            mixed = 0.5 * heuristic + 0.5 * neural
            return _normalize_action_probs(mixed, revealed)
        raise ValueError(f"Unknown prior_source: {self.prior_source}")

    def _heuristic_leaf_value(self, revealed: np.ndarray, hit_mask: np.ndarray) -> float:
        posterior = estimate_posterior_occupancy(
            revealed,
            hit_mask,
            n_samples=self.leaf_samples,
            rng=self.rng,
        )
        total_ship_cells = float(sum(self.ship_lengths))
        progress = float(hit_mask.sum()) / max(total_ship_cells, 1.0)
        unknown = ~revealed
        if np.any(unknown):
            next_hit = float(posterior[unknown].max())
            p = np.clip(posterior[unknown], 1e-6, 1.0 - 1e-6)
            entropy = -p * np.log(p) - (1.0 - p) * np.log(1.0 - p)
            certainty = 1.0 - float(entropy.mean() / np.log(2.0))
        else:
            next_hit = 0.0
            certainty = 1.0
        value = 0.55 * progress + 0.30 * next_hit + 0.15 * certainty
        return float(np.clip(value, 0.0, 1.0))

    def _evaluate_leaf(
        self,
        revealed: np.ndarray,
        hit_mask: np.ndarray,
        hidden_board: np.ndarray,
        discount: float,
    ) -> tuple[float, int]:
        if self.value_fn is not None:
            guided_value = float(self.value_fn(hit_mask.copy(), revealed.copy()))
            return discount * float(np.clip(guided_value, 0.0, 1.0)), 0

        if self.leaf_evaluator == "heuristic":
            return discount * self._heuristic_leaf_value(revealed, hit_mask), 0
        if self.leaf_evaluator == "rollout":
            return self._rollout(revealed, hit_mask, hidden_board, discount)
        if self.leaf_evaluator == "hybrid":
            heuristic_value = discount * self._heuristic_leaf_value(revealed, hit_mask)
            rollout_value, rollout_steps = self._rollout(revealed, hit_mask, hidden_board, discount)
            return 0.5 * (heuristic_value + rollout_value), rollout_steps
        raise ValueError(f"Unknown leaf_evaluator: {self.leaf_evaluator}")

    def _rollout(
        self,
        revealed: np.ndarray,
        hit_mask: np.ndarray,
        hidden_board: np.ndarray,
        discount: float,
    ) -> tuple[float, int]:
        rollout_steps = 0
        current_revealed = revealed.copy()
        current_hit_mask = hit_mask.copy()
        rollout_value = 0.0

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
            current_revealed, current_hit_mask, is_hit = _step(
                current_revealed, current_hit_mask, hidden_board, action
            )
            if is_hit:
                rollout_value += discount
            discount *= self.gamma
            rollout_steps += 1

        while not _terminal(current_revealed, current_hit_mask, hidden_board):
            unknown = np.argwhere(~current_revealed)
            row, col = unknown[int(self.rng.integers(len(unknown)))]
            current_revealed, current_hit_mask, is_hit = _step(
                current_revealed, current_hit_mask, hidden_board, (int(row), int(col))
            )
            if is_hit:
                rollout_value += discount
            discount *= self.gamma
            rollout_steps += 1

        rollout_value += 0.05 * discount
        return rollout_value, rollout_steps

    def _run_search(self) -> _SearchNode:
        root_priors = self._compute_action_priors(self.revealed, self.hit_mask)
        root = _SearchNode(
            self.revealed.copy(),
            self.hit_mask.copy(),
            root_priors,
        )
        if not root.action_stats:
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
            value = 0.0
            discount = 1.0

            while not _terminal(node.revealed, node.hit_mask, hidden_board):
                action = self._select_action(node)
                stats = node.action_stats[action]
                next_revealed, next_hit_mask, is_hit = _step(
                    node.revealed, node.hit_mask, hidden_board, action
                )
                if is_hit:
                    value += discount
                discount *= self.gamma
                child = stats.children.get(is_hit)
                if child is None:
                    child_priors = self._compute_action_priors(next_revealed, next_hit_mask)
                    child = _SearchNode(
                        next_revealed,
                        next_hit_mask,
                        child_priors,
                    )
                    stats.children[is_hit] = child

                path_edges.append((node, action))
                node = child
                path_nodes.append(node)
                total_steps += 1

                if stats.visits == 0:
                    break

            if _terminal(node.revealed, node.hit_mask, hidden_board):
                value += discount
            else:
                leaf_value, rollout_steps = self._evaluate_leaf(
                    node.revealed,
                    node.hit_mask,
                    hidden_board,
                    discount,
                )
                value += leaf_value
                total_steps += rollout_steps

            value += 0.05 / max(total_steps, 1)

            for visited_node in path_nodes:
                visited_node.visits += 1
            for parent, action in path_edges:
                stats = parent.action_stats[action]
                stats.visits += 1
                stats.total_value += value

        return root

    def beliefs(self) -> np.ndarray:
        if not self.hit_mask.any():
            return self._compute_action_priors(self.revealed, self.hit_mask)

        root = self._run_search()
        score = np.zeros((self.N, self.N), dtype=np.float64)

        if root.action_stats:
            for action, stats in root.action_stats.items():
                row, col = action
                q_value = (stats.total_value / stats.visits) if stats.visits > 0 else 0.0
                score[row, col] = float(stats.visits) + 5.0 * max(q_value, 0.0) + stats.prior
        else:
            score[~self.revealed] = 1.0

        total = score.sum()
        return score / total if total > 0 else score

    def best_guess(self) -> tuple[int, int]:
        beliefs = self.beliefs()
        return divmod(int(np.argmax(beliefs)), self.N)
