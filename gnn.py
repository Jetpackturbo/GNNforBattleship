#!/usr/bin/env python
# coding: utf-8
"""
gnn.py — policy-learning GNN for Battleship move selection.

This module keeps the Ising/BP solver as an analytic baseline, but the learned
model is now trained to predict the *next move policy* rather than hidden ship
occupancy. Supervision comes from a DataGenetics-style probability-density
teacher: on every partial board, we count valid placements of the standard
fleet and fire at the highest-scoring unrevealed cell.

Benchmark derivation
--------------------
The online benchmark in this file is derived from Nick Berry's DataGenetics
"Battleship" analysis:
https://www.datagenetics.com/blog/december32011/

Berry compares Random, Hunt/Target, parity-aware search, and probability
density play. This code mirrors the same family of baselines, but adapts them
to this repo's simpler environment, which reports only hit/miss feedback and
does not announce when a ship has been sunk.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from battleship_ising import BattleshipGame, BattleshipIsing
from mcts import MCTSAgent, bayesian_surprise, estimate_posterior_occupancy

try:
    from torch_geometric.nn import MessagePassing

    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    MessagePassing = None  # type: ignore

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


GRID_SIZE = 10
N_CELLS = GRID_SIZE * GRID_SIZE
NODE_FEATURES = 5
EDGE_FEATURES = 1
SHIP_LENGTHS = [5, 4, 3, 3, 2]


def _maybe_tqdm(iterable, enabled: bool, **kwargs):
    """Wrap an iterable in tqdm when available and requested."""
    if enabled and tqdm is not None:
        return tqdm(iterable, **kwargs)
    return iterable


def _wandb_log(run: Any, payload: dict[str, float], step: Optional[int] = None) -> None:
    """Log metrics to wandb when a run is active."""
    if run is not None:
        run.log(payload, step=step)


def _json_ready(value: Any) -> Any:
    """Convert nested config values into a stable JSON-serializable structure."""
    if isinstance(value, dict):
        return {
            str(k): _json_ready(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _dataset_cache_path(
    cache_dir: Path,
    *,
    n_samples: int,
    max_context_shots: int,
    seed: int,
    teacher_policy: str,
    teacher_kwargs: Optional[dict[str, Any]],
    surprise_augmentation: bool,
    surprise_samples: int,
    surprise_alpha: float,
) -> tuple[Path, dict[str, Any]]:
    """Build a deterministic cache path for a generated dataset."""
    metadata = {
        "n_samples": int(n_samples),
        "max_context_shots": int(max_context_shots),
        "seed": int(seed),
        "teacher_policy": str(teacher_policy),
        "teacher_kwargs": _json_ready({} if teacher_kwargs is None else teacher_kwargs),
        "surprise_augmentation": bool(surprise_augmentation),
        "surprise_samples": int(surprise_samples),
        "surprise_alpha": float(surprise_alpha),
        "grid_size": GRID_SIZE,
        "ship_lengths": list(SHIP_LENGTHS),
        "node_features": NODE_FEATURES,
    }
    digest = hashlib.sha256(
        json.dumps(metadata, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return cache_dir / f"policy_dataset_{digest}.npz", metadata


def _load_cached_dataset(cache_path: Path) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load a cached dataset saved by `generate_dataset`."""
    with np.load(cache_path, allow_pickle=False) as payload:
        features = payload["features"]
        policies = payload["policies"]
        masks = payload["masks"].astype(bool, copy=False)
    return [(features[i], policies[i], masks[i]) for i in range(features.shape[0])]


def _save_cached_dataset(
    cache_path: Path,
    metadata: dict[str, Any],
    data: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    """Persist a generated dataset to a compressed cache file."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    features = np.stack([item[0] for item in data]).astype(np.float32, copy=False)
    policies = np.stack([item[1] for item in data]).astype(np.float32, copy=False)
    masks = np.stack([item[2] for item in data]).astype(bool, copy=False)
    np.savez_compressed(
        cache_path,
        features=features,
        policies=policies,
        masks=masks,
        metadata=np.array(json.dumps(metadata, sort_keys=True)),
    )


def build_grid_edges(grid_size: int = GRID_SIZE):
    """Build directed 4-neighbour grid edges."""
    src_list, dst_list, dir_list = [], [], []
    for r in range(grid_size):
        for c in range(grid_size):
            node = r * grid_size + c
            if c + 1 < grid_size:
                src_list += [node, node + 1]
                dst_list += [node + 1, node]
                dir_list += [1.0, 1.0]
            if r + 1 < grid_size:
                src_list += [node, node + grid_size]
                dst_list += [node + grid_size, node]
                dir_list += [0.0, 0.0]

    src = torch.tensor(src_list, dtype=torch.long)
    dst = torch.tensor(dst_list, dtype=torch.long)
    edge_dir = torch.tensor(dir_list, dtype=torch.float32).unsqueeze(-1)
    edge_index = torch.stack([src, dst], dim=0)
    return src, dst, edge_dir, edge_index


_SRC, _DST, _EDGE_DIR, _EDGE_INDEX = build_grid_edges(GRID_SIZE)


def observation_masks_to_features(
    hit_mask: np.ndarray,
    revealed: np.ndarray,
    grid_size: int = GRID_SIZE,
) -> torch.Tensor:
    """Convert hit/miss observations into node features."""
    N = grid_size
    flat_hit = hit_mask.flatten().astype(bool)
    flat_rev = revealed.flatten().astype(bool)

    is_hit = (flat_rev & flat_hit).astype(np.float32)
    is_miss = (flat_rev & ~flat_hit).astype(np.float32)
    is_unknown = (~flat_rev).astype(np.float32)
    rows = np.repeat(np.arange(N), N).astype(np.float32) / max(N - 1, 1)
    cols = np.tile(np.arange(N), N).astype(np.float32) / max(N - 1, 1)

    feats = np.stack([is_hit, is_miss, is_unknown, rows, cols], axis=-1)
    return torch.from_numpy(feats)


def state_to_features(
    h: np.ndarray,
    revealed: np.ndarray,
    grid_size: int = GRID_SIZE,
) -> torch.Tensor:
    """Backward-compatible wrapper from Ising external fields to features."""
    hit_mask = revealed & (h > 0)
    return observation_masks_to_features(hit_mask, revealed, grid_size)


def masked_policy_distribution(scores: np.ndarray, revealed: np.ndarray) -> np.ndarray:
    """Softmax scores over unrevealed cells only."""
    probs = np.zeros_like(scores, dtype=np.float64)
    unknown = ~revealed
    if not np.any(unknown):
        return probs
    masked_scores = scores[unknown]
    shifted = masked_scores - masked_scores.max()
    exp_scores = np.exp(shifted)
    probs[unknown] = exp_scores / exp_scores.sum()
    return probs


def _sample_argmax(score: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    """Choose uniformly among the cells with maximal score."""
    max_score = float(score.max())
    choices = np.argwhere(score == max_score)
    row, col = choices[int(rng.integers(len(choices)))]
    return int(row), int(col)


def _policy_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Cross-entropy against a masked policy target distribution."""
    masked_logits = logits.masked_fill(~mask, -1e9)
    log_probs = F.log_softmax(masked_logits, dim=-1)
    return -(targets * log_probs).sum(dim=-1).mean()


class GridMPNNLayer(nn.Module):
    """One message-passing sweep on the fixed Battleship grid."""

    def __init__(self, hidden_dim: int, edge_dim: int = EDGE_FEATURES):
        super().__init__()
        self.register_buffer("src_idx", _SRC.clone())
        self.register_buffer("dst_idx", _DST.clone())
        self.register_buffer("edge_dir", _EDGE_DIR.clone())
        self.n_edges = int(_SRC.shape[0])

        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.upd_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeezed = x.dim() == 2
        if squeezed:
            x = x.unsqueeze(0)

        batch_size, n_nodes, hidden_dim = x.shape
        edge_count = self.n_edges

        x_src = x[:, self.src_idx, :]
        edge_feats = self.edge_dir.unsqueeze(0).expand(batch_size, -1, -1)
        messages = self.msg_mlp(torch.cat([x_src, edge_feats], dim=-1))

        idx = self.dst_idx.view(1, edge_count, 1).expand(batch_size, edge_count, hidden_dim)
        agg = torch.zeros(batch_size, n_nodes, hidden_dim, device=x.device, dtype=x.dtype)
        agg = agg.scatter_add(1, idx, messages)

        out = self.norm(x + self.upd_mlp(torch.cat([x, agg], dim=-1)))
        return out.squeeze(0) if squeezed else out


if HAS_PYG:
    class PyGMPNNLayer(MessagePassing):
        """PyG message-passing layer for single-graph inference/training."""

        def __init__(self, hidden_dim: int, edge_dim: int = EDGE_FEATURES):
            super().__init__(aggr="add")
            self.msg_mlp = nn.Sequential(
                nn.Linear(hidden_dim + edge_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.upd_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.norm = nn.LayerNorm(hidden_dim)

        def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_attr: torch.Tensor,
        ) -> torch.Tensor:
            return self.norm(x + self.propagate(edge_index, x=x, edge_attr=edge_attr))

        def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
            return self.msg_mlp(torch.cat([x_j, edge_attr], dim=-1))

        def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            return self.upd_mlp(torch.cat([x, aggr_out], dim=-1))


class BattleshipGNN(nn.Module):
    """Message-passing policy network over the Battleship grid."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 6,
        use_pyg: bool = False,
    ) -> None:
        super().__init__()
        if use_pyg and not HAS_PYG:
            raise RuntimeError("use_pyg=True but torch-geometric is not installed.")

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_pyg = use_pyg

        self.encoder = nn.Sequential(
            nn.Linear(NODE_FEATURES, hidden_dim),
            nn.ReLU(),
        )
        if self.use_pyg:
            self.layers = nn.ModuleList([PyGMPNNLayer(hidden_dim) for _ in range(num_layers)])
        else:
            self.layers = nn.ModuleList([GridMPNNLayer(hidden_dim) for _ in range(num_layers)])
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return per-cell policy logits."""
        if self.use_pyg:
            if x.dim() != 2:
                raise ValueError("PyG mode expects unbatched input of shape (N, F).")
            if edge_index is None:
                edge_index = _EDGE_INDEX.to(x.device)
            if edge_attr is None:
                edge_attr = _EDGE_DIR.to(x.device)
            h = self.encoder(x)
            for layer in self.layers:
                h = layer(h, edge_index, edge_attr)
            return self.decoder(h).squeeze(-1)

        squeezed = x.dim() == 2
        if squeezed:
            x = x.unsqueeze(0)
        h = self.encoder(x)
        for layer in self.layers:
            h = layer(h)
        logits = self.decoder(h).squeeze(-1)
        return logits.squeeze(0) if squeezed else logits

    def predict(
        self,
        observed_state: np.ndarray,
        revealed: np.ndarray,
        device: str = "cpu",
    ) -> np.ndarray:
        """Predict a next-move distribution over the board."""
        x = state_to_features(observed_state, revealed).to(device)
        self.eval()
        with torch.no_grad():
            logits = self.forward(x).detach().cpu().numpy().reshape(GRID_SIZE, GRID_SIZE)
        return masked_policy_distribution(logits, revealed)

    def predict_from_masks(
        self,
        hit_mask: np.ndarray,
        revealed: np.ndarray,
        device: str = "cpu",
    ) -> np.ndarray:
        """Predict a next-move distribution from hit/miss masks."""
        x = observation_masks_to_features(hit_mask, revealed).to(device)
        self.eval()
        with torch.no_grad():
            logits = self.forward(x).detach().cpu().numpy().reshape(GRID_SIZE, GRID_SIZE)
        return masked_policy_distribution(logits, revealed)


class GNNAgent:
    """Policy agent backed by a trained `BattleshipGNN`."""

    def __init__(
        self,
        model: BattleshipGNN,
        grid_size: int = GRID_SIZE,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.grid_size = grid_size
        self.revealed = np.zeros((grid_size, grid_size), dtype=bool)
        self.hit_mask = np.zeros((grid_size, grid_size), dtype=bool)

    def reset(self) -> None:
        self.revealed[:] = False
        self.hit_mask[:] = False

    def observe(self, row: int, col: int, is_hit: bool) -> None:
        self.revealed[row, col] = True
        self.hit_mask[row, col] = bool(is_hit)

    def best_guess(self) -> tuple[int, int]:
        p = self.beliefs()
        return divmod(int(np.argmax(p)), self.grid_size)

    def beliefs(self) -> np.ndarray:
        return self.model.predict_from_masks(self.hit_mask, self.revealed, self.device)


class RandomAgent:
    """Shoots uniformly at random over unrevealed cells."""

    def __init__(self, grid_size: int = GRID_SIZE, seed: Optional[int] = None) -> None:
        self.grid_size = grid_size
        self.rng = np.random.default_rng(seed)
        self.revealed = np.zeros((grid_size, grid_size), dtype=bool)

    def reset(self) -> None:
        self.revealed[:] = False

    def observe(self, row: int, col: int, is_hit: bool) -> None:
        self.revealed[row, col] = True

    def best_guess(self) -> tuple[int, int]:
        unknown = np.argwhere(~self.revealed)
        row, col = unknown[int(self.rng.integers(len(unknown)))]
        return int(row), int(col)

    def beliefs(self) -> np.ndarray:
        score = np.where(self.revealed, 0.0, 1.0)
        total = score.sum()
        return score / total if total > 0 else score


class HuntTargetAgent:
    """DataGenetics-style hunt/target baseline with parity hunt mode."""

    def __init__(self, grid_size: int = GRID_SIZE, seed: Optional[int] = 0) -> None:
        self.N = grid_size
        self.rng = np.random.default_rng(seed)
        self.revealed = np.zeros((grid_size, grid_size), dtype=bool)
        self.hit_mask = np.zeros((grid_size, grid_size), dtype=bool)

    def reset(self) -> None:
        self.revealed[:] = False
        self.hit_mask[:] = False

    def observe(self, row: int, col: int, is_hit: bool) -> None:
        self.revealed[row, col] = True
        self.hit_mask[row, col] = bool(is_hit)

    def _score(self) -> np.ndarray:
        score = np.zeros((self.N, self.N), dtype=np.float64)
        hit_cells = np.argwhere(self.hit_mask)

        for row, col in hit_cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = int(row + dr), int(col + dc)
                if 0 <= nr < self.N and 0 <= nc < self.N and not self.revealed[nr, nc]:
                    score[nr, nc] += 1.0

        if score.max() > 0:
            return score

        parity = (np.indices((self.N, self.N)).sum(axis=0) % 2) == 0
        score[(~self.revealed) & parity] = 1.0
        if score.max() == 0:
            score[~self.revealed] = 1.0
        return score

    def best_guess(self) -> tuple[int, int]:
        return _sample_argmax(self._score(), self.rng)

    def beliefs(self) -> np.ndarray:
        score = self._score()
        total = score.sum()
        return score / total if total > 0 else score


class ProbabilityDensityAgent:
    """Placement-count baseline adapted from Berry's probability-density policy.

    Because this environment does not announce "sunk ship" events, the agent
    cannot remove lengths from the fleet mid-game. Instead, it counts placements
    for the full standard fleet and heavily upweights placements that pass
    through already-confirmed hits.
    """

    def __init__(
        self,
        grid_size: int = GRID_SIZE,
        ship_lengths: Optional[list[int]] = None,
        hit_bonus: float = 12.0,
        seed: Optional[int] = 0,
    ) -> None:
        self.N = grid_size
        self.ship_lengths = SHIP_LENGTHS if ship_lengths is None else ship_lengths
        self.hit_bonus = float(hit_bonus)
        self.rng = np.random.default_rng(seed)
        self.revealed = np.zeros((grid_size, grid_size), dtype=bool)
        self.hit_mask = np.zeros((grid_size, grid_size), dtype=bool)

    def reset(self) -> None:
        self.revealed[:] = False
        self.hit_mask[:] = False

    def observe(self, row: int, col: int, is_hit: bool) -> None:
        self.revealed[row, col] = True
        self.hit_mask[row, col] = bool(is_hit)

    @property
    def miss_mask(self) -> np.ndarray:
        return self.revealed & ~self.hit_mask

    def _score(self, require_hit_overlap: bool) -> np.ndarray:
        score = np.zeros((self.N, self.N), dtype=np.float64)

        for length in self.ship_lengths:
            for row in range(self.N):
                for col in range(self.N - length + 1):
                    rev = self.revealed[row, col : col + length]
                    hits = self.hit_mask[row, col : col + length]
                    misses = rev & ~hits
                    if np.any(misses):
                        continue
                    hit_count = int(hits.sum())
                    if require_hit_overlap and hit_count == 0:
                        continue
                    weight = 1.0 + self.hit_bonus * hit_count
                    for offset in range(length):
                        rr, cc = row, col + offset
                        if not self.revealed[rr, cc]:
                            score[rr, cc] += weight

            for col in range(self.N):
                for row in range(self.N - length + 1):
                    rev = self.revealed[row : row + length, col]
                    hits = self.hit_mask[row : row + length, col]
                    misses = rev & ~hits
                    if np.any(misses):
                        continue
                    hit_count = int(hits.sum())
                    if require_hit_overlap and hit_count == 0:
                        continue
                    weight = 1.0 + self.hit_bonus * hit_count
                    for offset in range(length):
                        rr, cc = row + offset, col
                        if not self.revealed[rr, cc]:
                            score[rr, cc] += weight

        return score

    def raw_scores(self) -> np.ndarray:
        active_hits = bool(self.hit_mask.any())
        score = self._score(require_hit_overlap=active_hits)
        if active_hits and score.max() == 0:
            score = self._score(require_hit_overlap=False)
        if score.max() == 0:
            score[~self.revealed] = 1.0
        return score

    def best_guess(self) -> tuple[int, int]:
        return _sample_argmax(self.raw_scores(), self.rng)

    def beliefs(self) -> np.ndarray:
        score = self.raw_scores()
        total = score.sum()
        return score / total if total > 0 else score


class IsingBPAgent:
    """Greedy Battleship agent using the Ising/BP posterior."""

    def __init__(self, grid_size: int = GRID_SIZE, J: float = 0.5, bp_iters: int = 60) -> None:
        self._model = BattleshipIsing(grid_size=grid_size, J=J)
        self.bp_iters = bp_iters

    def reset(self) -> None:
        self._model.reset()

    def observe(self, row: int, col: int, is_hit: bool) -> None:
        self._model.observe(row, col, is_hit)

    def best_guess(self) -> tuple[int, int]:
        self._model.run_bp(num_iter=self.bp_iters)
        return self._model.best_guess()

    def beliefs(self) -> np.ndarray:
        self._model.run_bp(num_iter=self.bp_iters)
        return self._model.beliefs()


def _generate_policy_sample(
    rng: np.random.Generator,
    max_context_shots: int = 40,
    teacher_policy: str = "probability_density",
    teacher_kwargs: Optional[dict[str, Any]] = None,
    surprise_augmentation: bool = False,
    surprise_samples: int = 8,
    surprise_alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate one imitation-learning sample from the chosen teacher policy."""
    teacher_kwargs = {} if teacher_kwargs is None else dict(teacher_kwargs)
    while True:
        game_seed = int(rng.integers(0, 2**31))
        game = BattleshipGame(grid_size=GRID_SIZE, seed=game_seed)
        trajectory_agent = ProbabilityDensityAgent(grid_size=GRID_SIZE, seed=game_seed)
        n_steps = int(rng.integers(0, max_context_shots + 1))
        stored_states: list[tuple[np.ndarray, np.ndarray, float]] = []
        posterior_rng = np.random.default_rng(game_seed + 17)
        prev_posterior = None

        if surprise_augmentation:
            empty_revealed = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
            empty_hits = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
            prev_posterior = estimate_posterior_occupancy(
                empty_revealed,
                empty_hits,
                n_samples=surprise_samples,
                rng=posterior_rng,
            )

        solved = False
        for _ in range(n_steps):
            row, col = trajectory_agent.best_guess()
            is_hit = game.shoot(row, col)
            trajectory_agent.observe(row, col, is_hit)
            if surprise_augmentation:
                current_posterior = estimate_posterior_occupancy(
                    trajectory_agent.revealed,
                    trajectory_agent.hit_mask,
                    n_samples=surprise_samples,
                    rng=posterior_rng,
                )
                assert prev_posterior is not None
                surprise_value, _ = bayesian_surprise(prev_posterior, current_posterior)
                stored_states.append(
                    (
                        trajectory_agent.revealed.copy(),
                        trajectory_agent.hit_mask.copy(),
                        surprise_value,
                    )
                )
                prev_posterior = current_posterior
            if game.all_sunk(trajectory_agent.hit_mask):
                solved = True
                break

        if not solved and np.any(~trajectory_agent.revealed):
            if surprise_augmentation and stored_states:
                weights = np.array(
                    [0.1 + max(state_surprise, 0.0) ** surprise_alpha for _, _, state_surprise in stored_states],
                    dtype=np.float64,
                )
                weights /= weights.sum()
                chosen_idx = int(rng.choice(len(stored_states), p=weights))
                selected_revealed, selected_hit_mask, _ = stored_states[chosen_idx]
            else:
                selected_revealed = trajectory_agent.revealed.copy()
                selected_hit_mask = trajectory_agent.hit_mask.copy()

            if teacher_policy == "probability_density":
                planner = ProbabilityDensityAgent(grid_size=GRID_SIZE, seed=game_seed)
                for row, col in np.argwhere(selected_hit_mask):
                    planner.observe(int(row), int(col), True)
                for row, col in np.argwhere(selected_revealed & ~selected_hit_mask):
                    planner.observe(int(row), int(col), False)
            elif teacher_policy == "mcts":
                planner = MCTSAgent(grid_size=GRID_SIZE, seed=game_seed, **teacher_kwargs)
                observed_hits = np.argwhere(selected_hit_mask)
                observed_misses = np.argwhere(selected_revealed & ~selected_hit_mask)
                for row, col in observed_hits:
                    planner.observe(int(row), int(col), True)
                for row, col in observed_misses:
                    planner.observe(int(row), int(col), False)
            else:
                raise ValueError(f"Unknown teacher_policy: {teacher_policy}")

            feats = observation_masks_to_features(
                selected_hit_mask, selected_revealed
            ).numpy()
            policy = planner.beliefs().astype(np.float32).flatten()
            mask = (~selected_revealed).flatten().astype(bool)
            return feats, policy, mask


def generate_dataset(
    n_samples: int = 6000,
    max_context_shots: int = 40,
    seed: int = 0,
    show_progress: bool = True,
    teacher_policy: str = "probability_density",
    teacher_kwargs: Optional[dict[str, Any]] = None,
    surprise_augmentation: bool = False,
    surprise_samples: int = 8,
    surprise_alpha: float = 1.0,
    cache_dir: Optional[str | Path] = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Generate `(features, target_policy, unknown_mask)` samples."""
    cache_path: Optional[Path] = None
    metadata: Optional[dict[str, Any]] = None
    if cache_dir is not None:
        cache_root = Path(cache_dir).expanduser().resolve()
        cache_path, metadata = _dataset_cache_path(
            cache_root,
            n_samples=n_samples,
            max_context_shots=max_context_shots,
            seed=seed,
            teacher_policy=teacher_policy,
            teacher_kwargs=teacher_kwargs,
            surprise_augmentation=surprise_augmentation,
            surprise_samples=surprise_samples,
            surprise_alpha=surprise_alpha,
        )
        if cache_path.exists():
            print(f"Loading cached dataset from {cache_path}")
            return _load_cached_dataset(cache_path)

    rng = np.random.default_rng(seed)
    data = []
    iterator = _maybe_tqdm(
        range(n_samples),
        show_progress,
        total=n_samples,
        desc="Generating policy states",
        leave=False,
    )
    for i in iterator:
        data.append(
            _generate_policy_sample(
                rng,
                max_context_shots=max_context_shots,
                teacher_policy=teacher_policy,
                teacher_kwargs=teacher_kwargs,
                surprise_augmentation=surprise_augmentation,
                surprise_samples=surprise_samples,
                surprise_alpha=surprise_alpha,
            )
        )
        if not show_progress and (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{n_samples} policy states ...")
    if cache_path is not None and metadata is not None:
        print(f"Saving dataset cache to {cache_path}")
        _save_cached_dataset(cache_path, metadata, data)
    return data


def train_gnn(
    n_epochs: int = 30,
    n_train: int = 6000,
    n_val: int = 1000,
    hidden_dim: int = 64,
    num_layers: int = 6,
    lr: float = 3e-4,
    batch_size: int = 64,
    max_context_shots: int = 40,
    seed: int = 0,
    device: str = "cpu",
    use_pyg: bool = False,
    show_progress: bool = True,
    wandb_run: Any = None,
    teacher_policy: str = "probability_density",
    teacher_kwargs: Optional[dict[str, Any]] = None,
    surprise_augmentation: bool = False,
    surprise_samples: int = 8,
    surprise_alpha: float = 1.0,
    dataset_cache_dir: Optional[str | Path] = None,
) -> tuple["BattleshipGNN", dict]:
    """Train the move-selection GNN by imitating the chosen teacher policy."""
    print("Generating policy-training data ...")
    train_data = generate_dataset(
        n_train,
        max_context_shots=max_context_shots,
        seed=seed,
        show_progress=show_progress,
        teacher_policy=teacher_policy,
        teacher_kwargs=teacher_kwargs,
        surprise_augmentation=surprise_augmentation,
        surprise_samples=surprise_samples,
        surprise_alpha=surprise_alpha,
        cache_dir=dataset_cache_dir,
    )
    val_data = generate_dataset(
        n_val,
        max_context_shots=max_context_shots,
        seed=seed + 1,
        show_progress=show_progress,
        teacher_policy=teacher_policy,
        teacher_kwargs=teacher_kwargs,
        surprise_augmentation=surprise_augmentation,
        surprise_samples=surprise_samples,
        surprise_alpha=surprise_alpha,
        cache_dir=dataset_cache_dir,
    )

    model = BattleshipGNN(hidden_dim=hidden_dim, num_layers=num_layers, use_pyg=use_pyg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = defaultdict(list)

    def _run_epoch(dataset, train: bool) -> tuple[float, float]:
        model.train(train)
        rng = np.random.default_rng(seed + (0 if train else 9999))
        indices = np.arange(len(dataset))
        if train:
            rng.shuffle(indices)

        total_loss = 0.0
        total_acc = 0.0
        total_batches = 0

        iterator = _maybe_tqdm(
            range(0, len(dataset), batch_size),
            show_progress,
            total=(len(dataset) + batch_size - 1) // batch_size,
            desc="Train batches" if train else "Val batches",
            leave=False,
        )
        for start in iterator:
            batch_idx = indices[start : start + batch_size]
            if train:
                optimizer.zero_grad()

            if model.use_pyg:
                losses = []
                batch_correct = 0.0
                for i in batch_idx:
                    feats, target, mask = dataset[i]
                    x_t = torch.tensor(feats, dtype=torch.float32, device=device)
                    y_t = torch.tensor(target, dtype=torch.float32, device=device)
                    m_t = torch.tensor(mask, dtype=torch.bool, device=device)
                    logits = model(x_t)
                    loss_i = _policy_loss(logits.unsqueeze(0), y_t.unsqueeze(0), m_t.unsqueeze(0))
                    losses.append(loss_i)

                    pred_idx = int(logits.masked_fill(~m_t, -1e9).argmax().item())
                    target_idx = int(y_t.argmax().item())
                    batch_correct += float(pred_idx == target_idx)

                loss = torch.stack(losses).mean()
                batch_acc = batch_correct / max(len(batch_idx), 1)
            else:
                feats, targets, masks = [], [], []
                for i in batch_idx:
                    f, t, m = dataset[i]
                    feats.append(f)
                    targets.append(t)
                    masks.append(m)

                x_t = torch.tensor(np.stack(feats), dtype=torch.float32, device=device)
                y_t = torch.tensor(np.stack(targets), dtype=torch.float32, device=device)
                m_t = torch.tensor(np.stack(masks), dtype=torch.bool, device=device)

                logits = model(x_t)
                loss = _policy_loss(logits, y_t, m_t)

                pred_idx = logits.masked_fill(~m_t, -1e9).argmax(dim=-1)
                target_idx = y_t.argmax(dim=-1)
                batch_acc = float((pred_idx == target_idx).float().mean().item())

            if train:
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item())
            total_acc += batch_acc
            total_batches += 1

        denom = max(total_batches, 1)
        return total_loss / denom, total_acc / denom

    print(
        f"Training policy GNN ({n_train} train / {n_val} val / {n_epochs} epochs) ..."
    )
    epoch_iterator = _maybe_tqdm(
        range(1, n_epochs + 1),
        show_progress,
        total=n_epochs,
        desc="Epochs",
    )
    for epoch in epoch_iterator:
        train_loss, train_top1 = _run_epoch(train_data, train=True)
        val_loss, val_top1 = _run_epoch(val_data, train=False)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_top1"].append(train_top1)
        history["val_top1"].append(val_top1)
        _wandb_log(
            wandb_run,
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_top1": train_top1,
                "val_top1": val_top1,
            },
            step=epoch,
        )

        if hasattr(epoch_iterator, "set_postfix"):
            epoch_iterator.set_postfix(val_loss=f"{val_loss:.4f}", val_top1=f"{val_top1:.3f}")

        if (not show_progress) and (epoch % 5 == 0 or epoch == 1):
            print(
                f"  Epoch {epoch:3d}/{n_epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"val_top1={val_top1:.3f}"
            )

    return model, dict(history)


def play_game(
    agent,
    game: Optional[BattleshipGame] = None,
    seed: Optional[int] = None,
    max_shots: int = 100,
) -> dict:
    """Simulate a full Battleship game with any compatible agent."""
    if game is None:
        game = BattleshipGame(grid_size=GRID_SIZE, seed=seed)

    agent.reset()
    shots, hits = [], []
    hit_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)

    for _ in range(max_shots):
        row, col = agent.best_guess()
        is_hit = game.shoot(row, col)
        agent.observe(row, col, is_hit)

        shots.append((row, col))
        hits.append(is_hit)
        if is_hit:
            hit_mask[row, col] = True
        if game.all_sunk(hit_mask):
            break

    return {
        "n_shots": len(shots),
        "hit_rate": sum(hits) / len(hits) if hits else 0.0,
        "shots": shots,
        "hits": hits,
    }


def benchmark_reference() -> str:
    """Return the online benchmark source used for the baseline design."""
    return (
        "Derived from Nick Berry's DataGenetics Battleship analysis "
        "(Random, Hunt/Target, Probability Density), adapted to hit/miss-only "
        "feedback: https://www.datagenetics.com/blog/december32011/"
    )


def summarize_results(
    results: dict[str, list[int]],
    ci_level: float = 0.95,
    bootstrap_samples: int = 1000,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Compute summary statistics, confidence bounds, and ranges."""
    rng = np.random.default_rng(seed)
    alpha = (1.0 - ci_level) / 2.0
    summary: dict[str, dict[str, float]] = {}

    for name, shots in results.items():
        arr = np.array(shots, dtype=np.float64)
        if arr.size == 0:
            summary[name] = {
                "n_games": 0.0,
                "mean": float("nan"),
                "median": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "range": float("nan"),
                "ci_level": ci_level,
                "mean_ci_low": float("nan"),
                "mean_ci_high": float("nan"),
                "median_ci_low": float("nan"),
                "median_ci_high": float("nan"),
                "mean_err_low": float("nan"),
                "mean_err_high": float("nan"),
                "median_err_low": float("nan"),
                "median_err_high": float("nan"),
            }
            continue

        boot_idx = rng.integers(0, arr.size, size=(bootstrap_samples, arr.size))
        boot_samples = arr[boot_idx]
        boot_means = boot_samples.mean(axis=1)
        boot_medians = np.median(boot_samples, axis=1)

        mean = float(arr.mean())
        median = float(np.median(arr))
        mean_ci_low, mean_ci_high = np.quantile(boot_means, [alpha, 1.0 - alpha])
        median_ci_low, median_ci_high = np.quantile(boot_medians, [alpha, 1.0 - alpha])

        summary[name] = {
            "n_games": float(arr.size),
            "mean": mean,
            "median": median,
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "range": float(arr.max() - arr.min()),
            "ci_level": ci_level,
            "mean_ci_low": float(mean_ci_low),
            "mean_ci_high": float(mean_ci_high),
            "median_ci_low": float(median_ci_low),
            "median_ci_high": float(median_ci_high),
            "mean_err_low": float(mean - mean_ci_low),
            "mean_err_high": float(mean_ci_high - mean),
            "median_err_low": float(median - median_ci_low),
            "median_err_high": float(median_ci_high - median),
        }

    return summary


def compare_all_agents(
    n_games: int = 200,
    gnn_model: Optional[BattleshipGNN] = None,
    seed: int = 100,
    device: str = "cpu",
    verbose: bool = True,
    extra_agents: Optional[dict[str, object]] = None,
    show_progress: bool = True,
    wandb_run: Any = None,
    include_mcts: bool = False,
    mcts_kwargs: Optional[dict[str, Any]] = None,
) -> dict[str, list[int]]:
    """Run the DataGenetics-style benchmark suite on shared game instances."""
    agents: dict[str, object] = {
        "Random": RandomAgent(seed=seed),
        "Hunt Target": HuntTargetAgent(seed=seed),
        "Probability Density": ProbabilityDensityAgent(seed=seed),
        "Ising BP": IsingBPAgent(),
    }
    if include_mcts:
        agents["MCTS"] = MCTSAgent(seed=seed, **({} if mcts_kwargs is None else mcts_kwargs))
    if gnn_model is not None:
        agents["GNN Policy"] = GNNAgent(gnn_model, device=device)
    if extra_agents:
        agents.update(extra_agents)

    results: dict[str, list[int]] = {name: [] for name in agents}

    game_iterator = _maybe_tqdm(
        range(n_games),
        show_progress,
        total=n_games,
        desc="Benchmark games",
    )
    for game_idx in game_iterator:
        game_seed = seed + game_idx
        game = BattleshipGame(grid_size=GRID_SIZE, seed=game_seed)
        for name, agent in agents.items():
            result = play_game(agent, game=game, seed=game_seed)
            results[name].append(int(result["n_shots"]))

        if verbose and (not show_progress) and (game_idx + 1) % 50 == 0:
            print(f"  Completed {game_idx + 1}/{n_games} games ...")

    if verbose:
        summary = summarize_results(results, seed=seed)
        print("\nBenchmark source:")
        print(f"  {benchmark_reference()}")
        print("\nResults")
        print(
            f"  {'Agent':<20} {'Mean':>6} {'Median':>7} {'Std':>6} "
            f"{'Mean 95% CI':>22} {'Median 95% CI':>24} {'Range':>11}"
        )
        print(
            f"  {'-' * 20} {'-' * 6} {'-' * 7} {'-' * 6} "
            f"{'-' * 22} {'-' * 24} {'-' * 11}"
        )
        for name in results:
            stats = summary[name]
            mean_ci = f"[{stats['mean_ci_low']:.1f}, {stats['mean_ci_high']:.1f}]"
            median_ci = f"[{stats['median_ci_low']:.1f}, {stats['median_ci_high']:.1f}]"
            value_range = f"[{stats['min']:.0f}, {stats['max']:.0f}]"
            print(
                f"  {name:<20} {stats['mean']:6.1f} {stats['median']:7.1f} {stats['std']:6.1f} "
                f"{mean_ci:>22} {median_ci:>24} {value_range:>11}"
            )
            _wandb_log(
                wandb_run,
                {
                    f"benchmark/{name}/mean_shots": stats["mean"],
                    f"benchmark/{name}/median_shots": stats["median"],
                    f"benchmark/{name}/std_shots": stats["std"],
                    f"benchmark/{name}/mean_ci_low": stats["mean_ci_low"],
                    f"benchmark/{name}/mean_ci_high": stats["mean_ci_high"],
                    f"benchmark/{name}/median_ci_low": stats["median_ci_low"],
                    f"benchmark/{name}/median_ci_high": stats["median_ci_high"],
                    f"benchmark/{name}/min_shots": stats["min"],
                    f"benchmark/{name}/max_shots": stats["max"],
                },
            )
        print()

    return results


def plot_training_history(history: dict) -> None:
    """Plot loss and top-1 teacher agreement curves."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Policy loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(history["train_top1"], label="train")
    axes[1].plot(history["val_top1"], label="val")
    axes[1].set_title("Teacher top-1 agreement")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_comparison(
    results: dict[str, list[int]],
    title: str = "Battleship benchmark comparison",
) -> None:
    """Visualise the benchmark outcome across agents."""
    import matplotlib.pyplot as plt

    names = list(results.keys())
    arrays = [np.array(results[name], dtype=np.float64) for name in names]
    means = [arr.mean() for arr in arrays]
    stds = [arr.std() for arr in arrays]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    axes[0].bar(names, means, color=colors, edgecolor="black", linewidth=0.7)
    axes[0].errorbar(names, means, yerr=stds, fmt="none", color="black", capsize=5)
    axes[0].set_ylabel("Mean shots to win")
    axes[0].set_title(title)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].tick_params(axis="x", rotation=20)

    box = axes[1].boxplot(
        [results[name] for name in names],
        labels=names,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
    axes[1].set_ylabel("Shots to win")
    axes[1].set_title("Game-length distribution")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.show()


def plot_belief_comparison(
    game: BattleshipGame,
    gnn_agent: Optional[GNNAgent] = None,
    ising_agent: Optional[IsingBPAgent] = None,
    density_agent: Optional[ProbabilityDensityAgent] = None,
    n_shots: int = 10,
) -> None:
    """Compare next-move maps after identical observations."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)
    shot_cells = list(np.ndindex(GRID_SIZE, GRID_SIZE))
    rng.shuffle(shot_cells)
    shot_cells = shot_cells[:n_shots]

    agents_to_plot = {}
    if gnn_agent is not None:
        gnn_agent.reset()
        agents_to_plot["GNN Policy"] = gnn_agent
    if ising_agent is not None:
        ising_agent.reset()
        agents_to_plot["Ising BP"] = ising_agent
    if density_agent is not None:
        density_agent.reset()
        agents_to_plot["Probability Density"] = density_agent

    for row, col in shot_cells:
        is_hit = game.shoot(row, col)
        for agent in agents_to_plot.values():
            agent.observe(row, col, is_hit)

    ncols = len(agents_to_plot) + 1
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    axes[0].imshow(game.grid, cmap="Greens", vmin=0, vmax=1.4)
    axes[0].set_title("True board")
    _annotate_shots(axes[0], shot_cells, game)

    for ax, (name, agent) in zip(axes[1:], agents_to_plot.items()):
        beliefs = agent.beliefs()
        im = ax.imshow(beliefs, cmap="hot", vmin=0, vmax=max(float(beliefs.max()), 1e-6))
        ax.set_title(f"{name}\n(after {n_shots} shots)")
        _annotate_shots(ax, shot_cells, game)
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.show()


def _annotate_shots(ax, shot_cells, game) -> None:
    """Draw shot markers on an axes."""
    ax.set_xticks(range(GRID_SIZE))
    ax.set_yticks(range(GRID_SIZE))
    ax.set_xticklabels(list("ABCDEFGHIJ"))
    for row, col in shot_cells:
        is_hit = bool(game.grid[row, col])
        ax.text(
            col,
            row,
            "x" if is_hit else ".",
            ha="center",
            va="center",
            color="white" if is_hit else "deepskyblue",
            fontsize=10,
            fontweight="bold",
        )


if __name__ == "__main__":
    print("=" * 70)
    print("  Battleship policy GNN")
    print(f"  PyTorch Geometric available: {'yes' if HAS_PYG else 'no'}")
    print("=" * 70)
    print(benchmark_reference())
    print()

    model, history = train_gnn(
        n_epochs=20,
        n_train=4000,
        n_val=800,
        hidden_dim=64,
        num_layers=6,
        lr=3e-4,
        batch_size=64,
        use_pyg=False,
    )
    plot_training_history(history)

    print("\nRunning benchmark over 200 games ...")
    results = compare_all_agents(n_games=200, gnn_model=model)
    plot_comparison(results)

    print("\nPlotting next-move comparison ...")
    demo_game = BattleshipGame(seed=42)
    plot_belief_comparison(
        demo_game,
        gnn_agent=GNNAgent(model),
        ising_agent=IsingBPAgent(),
        density_agent=ProbabilityDensityAgent(),
        n_shots=15,
    )
