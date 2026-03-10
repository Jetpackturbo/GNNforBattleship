#!/usr/bin/env python
# coding: utf-8
"""
gnn.py — GNN-based Battleship move predictor built on the Ising model.

Architecture overview
---------------------
The Battleship board is represented as a graph:
  Nodes  : 100 cells of a 10×10 grid
  Edges  : 360 directed edges (4-connected: left / right / up / down)
  Features per node: [is_hit, is_miss, is_unknown, row/N, col/N]  (5-D)
  Features per edge: [is_horizontal]                               (1-D)

A k-layer Message-Passing Neural Network (MPNN) is trained to predict
P(ship | observations) at every unrevealed cell.

Connection to Ising BP
----------------------
The Ising BP update (bpsol.py convention) is:

    m_{j→i}^{new}  =  h_j  +  Σ_{k≠i} arctanh( tanh(J) · tanh(m_{k→j}) )

The GNN generalises this: the arctanh(tanh(J)·tanh(·)) kernel becomes a
learned MLP, and h_j becomes a learned node embedding.  When the GNN
recovers this exact kernel it reproduces Ising BP; with more expressive
layers it can capture ship-length constraints and boundary effects.

PyTorch Geometric integration
------------------------------
If torch-geometric is installed the model can also be run in PyG mode
(uses the MessagePassing base class and DataLoader batching).

Install a compatible stack:

    conda create -n battleship python=3.10
    conda install pytorch=2.1 -c pytorch
    pip install torch-geometric

If torch-geometric is NOT available the code falls back to an equivalent
pure-PyTorch implementation that precomputes the fixed grid edges as
plain index tensors and uses scatter_add for aggregation.

Usage
-----
    from gnn import train_gnn, compare_all_agents, plot_comparison

    model, history = train_gnn(n_epochs=30, n_train=6000)
    results = compare_all_agents(n_games=200, gnn_model=model)
    plot_comparison(results)

Or run directly:
    python gnn.py
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Optional

from battleship_ising import BattleshipGame, BattleshipIsing

# ---------------------------------------------------------------------------
# Optional PyTorch Geometric import
# ---------------------------------------------------------------------------

try:
    from torch_geometric.data import Data, DataLoader as PyGDataLoader
    from torch_geometric.nn import MessagePassing
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    MessagePassing = None  # type: ignore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRID_SIZE  = 10
N_CELLS    = GRID_SIZE * GRID_SIZE
NODE_FEATURES = 5   # is_hit, is_miss, is_unknown, row/N, col/N
EDGE_FEATURES = 1   # is_horizontal (1) vs is_vertical (0)


# ---------------------------------------------------------------------------
# Graph structure helpers
# ---------------------------------------------------------------------------

def build_grid_edges(grid_size: int = GRID_SIZE):
    """Build directed edge arrays for a 4-connected grid.

    Returns
    -------
    src : LongTensor  shape (E,)   – source node index for each edge
    dst : LongTensor  shape (E,)   – destination node index
    edge_dir : FloatTensor (E, 1)  – 1.0 = horizontal, 0.0 = vertical
    edge_index : LongTensor (2, E) – PyG-style [src; dst] matrix

    For a 10×10 grid: E = 2 × (9×10 + 10×9) = 360 directed edges.
    """
    N = grid_size
    src_list, dst_list, dir_list = [], [], []

    for r in range(N):
        for c in range(N):
            node = r * N + c
            # right neighbour
            if c + 1 < N:
                src_list += [node, node + 1]
                dst_list += [node + 1, node]
                dir_list += [1.0, 1.0]          # horizontal
            # down neighbour
            if r + 1 < N:
                src_list += [node, node + N]
                dst_list += [node + N, node]
                dir_list += [0.0, 0.0]          # vertical

    src      = torch.tensor(src_list,  dtype=torch.long)
    dst      = torch.tensor(dst_list,  dtype=torch.long)
    edge_dir = torch.tensor(dir_list,  dtype=torch.float32).unsqueeze(-1)
    edge_index = torch.stack([src, dst], dim=0)
    return src, dst, edge_dir, edge_index


# Precompute once at module load
_SRC, _DST, _EDGE_DIR, _EDGE_INDEX = build_grid_edges(GRID_SIZE)


def state_to_features(
    h: np.ndarray,
    revealed: np.ndarray,
    grid_size: int = GRID_SIZE,
) -> torch.Tensor:
    """Convert Ising-model external-field arrays to node feature matrix.

    Parameters
    ----------
    h        : (N, N) external-field array from BattleshipIsing
    revealed : (N, N) bool mask – True where the cell has been shot

    Returns
    -------
    FloatTensor of shape (N*N, 5):
        col 0  is_hit      – revealed and h > 0
        col 1  is_miss     – revealed and h < 0
        col 2  is_unknown  – not yet revealed
        col 3  row / (N-1) – normalised row position
        col 4  col / (N-1) – normalised column position
    """
    N = grid_size
    flat_h  = h.flatten().astype(np.float32)
    flat_rv = revealed.flatten().astype(bool)

    is_hit     = (flat_rv & (flat_h > 0)).astype(np.float32)
    is_miss    = (flat_rv & (flat_h < 0)).astype(np.float32)
    is_unknown = (~flat_rv).astype(np.float32)

    rows = np.repeat(np.arange(N), N).astype(np.float32) / max(N - 1, 1)
    cols = np.tile(np.arange(N), N).astype(np.float32) / max(N - 1, 1)

    feats = np.stack([is_hit, is_miss, is_unknown, rows, cols], axis=-1)
    return torch.from_numpy(feats)   # (N*N, 5)


def make_pyg_data(
    h: np.ndarray,
    revealed: np.ndarray,
    labels: Optional[np.ndarray] = None,
) -> "Data":
    """Build a PyG Data object from a board state (requires PyG)."""
    if not HAS_PYG:
        raise RuntimeError("torch-geometric is not installed.")
    x = state_to_features(h, revealed)
    data = Data(
        x          = x,
        edge_index = _EDGE_INDEX.clone(),
        edge_attr  = _EDGE_DIR.clone(),
    )
    if labels is not None:
        data.y    = torch.from_numpy(labels.flatten().astype(np.float32))
        data.mask = torch.from_numpy((~revealed).flatten())   # loss mask
    return data


# ---------------------------------------------------------------------------
# Pure-PyTorch MPNN layer  (works without PyG)
# ---------------------------------------------------------------------------

class GridMPNNLayer(nn.Module):
    """One MPNN sweep on a fixed 4-connected grid.

    Implements the same computation as a PyG MessagePassing layer but uses
    precomputed index tensors and scatter_add for aggregation, requiring
    only standard PyTorch.

    Update rule (mirrors Ising BP structure):

        msg_{j→i}  =  MLP_msg( [x_j , e_{ji}] )
        agg_i      =  Σ_{j→i} msg_{j→i}
        x_i^{new}  =  LayerNorm( x_i + MLP_upd( [x_i , agg_i] ) )

    The residual + LayerNorm follows the Transformer convention and
    stabilises training.
    """

    def __init__(self, hidden_dim: int, edge_dim: int = EDGE_FEATURES):
        super().__init__()
        E = _SRC.shape[0]

        # Buffers: edge structure (no grad, moves with model.to(device))
        self.register_buffer("src_idx",  _SRC.clone())        # (E,)
        self.register_buffer("dst_idx",  _DST.clone())        # (E,)
        self.register_buffer("edge_dir", _EDGE_DIR.clone())   # (E, 1)

        self.n_nodes = N_CELLS
        self.n_edges = E
        self.hidden_dim = hidden_dim

        # Message MLP:  [x_j (H)  |  e_{ji} (D)]  →  H
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Update MLP:  [x_i (H)  |  agg_i (H)]  →  H
        self.upd_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, N, H) or (N, H)   node embeddings

        Returns
        -------
        (B, N, H)  updated node embeddings
        """
        squeezed = x.dim() == 2
        if squeezed:
            x = x.unsqueeze(0)

        B, N, H = x.shape
        E = self.n_edges

        # Gather sender features for every edge  →  (B, E, H)
        x_src = x[:, self.src_idx, :]

        # Append edge direction  →  (B, E, H+1)
        edge_feats = self.edge_dir.unsqueeze(0).expand(B, -1, -1)
        msg_in     = torch.cat([x_src, edge_feats], dim=-1)

        # Compute messages  →  (B, E, H)
        messages = self.msg_mlp(msg_in)

        # Aggregate (sum) messages per destination node  →  (B, N, H)
        idx = self.dst_idx.view(1, E, 1).expand(B, E, H)
        agg = torch.zeros(B, N, H, device=x.device, dtype=x.dtype)
        agg = agg.scatter_add(1, idx, messages)

        # Update with residual + LayerNorm
        out = self.norm(x + self.upd_mlp(torch.cat([x, agg], dim=-1)))

        return out.squeeze(0) if squeezed else out


# ---------------------------------------------------------------------------
# PyG MPNN layer  (only defined when torch-geometric is available)
# ---------------------------------------------------------------------------

if HAS_PYG:
    class PyGMPNNLayer(MessagePassing):
        """MessagePassing layer using the PyG API.

        Functionally identical to GridMPNNLayer but uses PyG's propagation
        pipeline, enabling automatic batching via PyG's DataLoader and
        support for arbitrary graph topologies (not just 10×10 grids).
        """

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
            out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
            return self.norm(x + out)

        def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
            return self.msg_mlp(torch.cat([x_j, edge_attr], dim=-1))

        def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            return self.upd_mlp(torch.cat([x, aggr_out], dim=-1))


# ---------------------------------------------------------------------------
# Full GNN model
# ---------------------------------------------------------------------------

class BattleshipGNN(nn.Module):
    """Graph Neural Network for Battleship ship-probability prediction.

    Takes a partial board observation (which cells are hit / miss) and
    outputs P(ship) ∈ (0, 1) for every cell.

    The architecture mirrors k iterations of Ising Belief Propagation:
    - the node encoder maps the 5-D observation features into a hidden
      space that plays the role of the Ising external field h_i;
    - each MPNN layer performs one round of message passing analogous
      to one BP sweep;
    - the decoder maps the final node embedding to a probability.

    Parameters
    ----------
    hidden_dim : int   Width of all intermediate layers.
    num_layers : int   Number of MPNN sweeps (≥ number of BP iterations
                       needed for convergence, typically 6–10).
    use_pyg    : bool  Force PyG (True) or GridMPNN (False).  None = auto.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 6,
        use_pyg: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_pyg    = HAS_PYG if use_pyg is None else use_pyg

        if self.use_pyg and not HAS_PYG:
            raise RuntimeError(
                "use_pyg=True but torch-geometric is not installed.\n"
                "Install with:  pip install torch-geometric"
            )

        # Node feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(NODE_FEATURES, hidden_dim),
            nn.ReLU(),
        )

        # MPNN layers
        if self.use_pyg:
            self.layers = nn.ModuleList(
                [PyGMPNNLayer(hidden_dim) for _ in range(num_layers)]
            )
        else:
            self.layers = nn.ModuleList(
                [GridMPNNLayer(hidden_dim) for _ in range(num_layers)]
            )

        # Output head: hidden → P(ship)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict P(ship) for every cell.

        Parameters
        ----------
        x : (B, N, 5) or (N, 5)   node feature matrix
        edge_index : (2, E)        only used in PyG mode
        edge_attr  : (E, 1)        only used in PyG mode

        Returns
        -------
        (B, N) or (N,)   ship probability for each cell
        """
        if self.use_pyg:
            # PyG path: x must be (N, 5); single graph
            assert x.dim() == 2, "PyG mode expects (N, F) input."
            if edge_index is None:
                edge_index = _EDGE_INDEX.to(x.device)
            if edge_attr is None:
                edge_attr = _EDGE_DIR.to(x.device)
            h = self.encoder(x)
            for layer in self.layers:
                h = layer(h, edge_index, edge_attr)
            return torch.sigmoid(self.decoder(h)).squeeze(-1)   # (N,)

        # Pure-PyTorch path: supports batched (B, N, F) input
        squeezed = x.dim() == 2
        if squeezed:
            x = x.unsqueeze(0)
        h = self.encoder(x)             # (B, N, hidden)
        for layer in self.layers:
            h = layer(h)                # (B, N, hidden)
        out = torch.sigmoid(self.decoder(h)).squeeze(-1)  # (B, N)
        return out.squeeze(0) if squeezed else out

    # ------------------------------------------------------------------
    # Convenience prediction from NumPy Ising state
    # ------------------------------------------------------------------

    def predict(
        self,
        h_field: np.ndarray,
        revealed: np.ndarray,
        device: str = "cpu",
    ) -> np.ndarray:
        """Predict P(ship) from a BattleshipIsing state.

        Returns an (N, N) NumPy array of probabilities.
        """
        x = state_to_features(h_field, revealed).to(device)
        self.eval()
        with torch.no_grad():
            p = self.forward(x).cpu().numpy()
        return p.reshape(GRID_SIZE, GRID_SIZE)


# ---------------------------------------------------------------------------
# GNN-based Battleship agent
# ---------------------------------------------------------------------------

class GNNAgent:
    """Battleship agent that uses a trained BattleshipGNN to pick moves.

    The agent maintains a BattleshipIsing model internally to track the
    board state (hit/miss fields), but uses the GNN — rather than BP —
    to compute the posterior P(ship) for move selection.
    """

    def __init__(
        self,
        model: BattleshipGNN,
        grid_size: int = GRID_SIZE,
        device: str = "cpu",
    ) -> None:
        self.model     = model.to(device).eval()
        self.device    = device
        self.grid_size = grid_size
        self._ising    = BattleshipIsing(grid_size=grid_size)

    def reset(self) -> None:
        self._ising.reset()

    def observe(self, row: int, col: int, is_hit: bool) -> None:
        self._ising.observe(row, col, is_hit)

    def best_guess(self) -> tuple[int, int]:
        """Return the unrevealed cell with the highest GNN-predicted P(ship)."""
        p = self.model.predict(self._ising.h, self._ising.revealed, self.device)
        masked = np.where(self._ising.revealed, -1.0, p)
        idx = int(np.argmax(masked))
        return divmod(idx, self.grid_size)

    def beliefs(self) -> np.ndarray:
        return self.model.predict(self._ising.h, self._ising.revealed, self.device)


# ---------------------------------------------------------------------------
# Baseline agents
# ---------------------------------------------------------------------------

class RandomAgent:
    """Shoots at a uniformly random unrevealed cell."""

    def __init__(self, grid_size: int = GRID_SIZE, seed: Optional[int] = None) -> None:
        self.grid_size = grid_size
        self.rng       = np.random.default_rng(seed)
        self.revealed  = np.zeros((grid_size, grid_size), dtype=bool)

    def reset(self) -> None:
        self.revealed[:] = False

    def observe(self, row: int, col: int, is_hit: bool) -> None:
        self.revealed[row, col] = True

    def best_guess(self) -> tuple[int, int]:
        unknown = list(zip(*np.where(~self.revealed)))
        return unknown[self.rng.integers(len(unknown))]


class FrequencyAgent:
    """Counts how many valid same-length ship placements cover each cell.

    For each unrevealed cell, accumulates the number of horizontal and
    vertical contiguous spans (of lengths 2–5, matching the standard fleet)
    that fit entirely in unrevealed territory.  Shoots at the cell with
    the highest count — a strong combinatorial heuristic.
    """

    LENGTHS = [5, 4, 3, 3, 2]

    def __init__(self, grid_size: int = GRID_SIZE) -> None:
        self.N        = grid_size
        self.revealed = np.zeros((grid_size, grid_size), dtype=bool)

    def reset(self) -> None:
        self.revealed[:] = False

    def observe(self, row: int, col: int, is_hit: bool) -> None:
        self.revealed[row, col] = True

    def _score(self) -> np.ndarray:
        N   = self.N
        score = np.zeros((N, N), dtype=float)
        free  = (~self.revealed).astype(int)   # 1 = can shoot here

        for length in self.LENGTHS:
            # Horizontal spans
            for r in range(N):
                for c in range(N - length + 1):
                    if free[r, c:c + length].sum() == length:
                        score[r, c:c + length] += 1.0

            # Vertical spans
            for c in range(N):
                for r in range(N - length + 1):
                    if free[r:r + length, c].sum() == length:
                        score[r:r + length, c] += 1.0

        return score

    def best_guess(self) -> tuple[int, int]:
        score  = self._score()
        masked = np.where(self.revealed, -1.0, score)
        idx    = int(np.argmax(masked))
        return divmod(idx, self.N)

    def beliefs(self) -> np.ndarray:
        s = self._score()
        mx = s.max()
        return (s / mx) if mx > 0 else s


class IsingBPAgent:
    """Agent using Ising Belief Propagation (from battleship_ising.py)."""

    def __init__(self, grid_size: int = GRID_SIZE, J: float = 0.5, bp_iters: int = 60) -> None:
        self._model   = BattleshipIsing(grid_size=grid_size, J=J)
        self.bp_iters = bp_iters

    def reset(self) -> None:
        self._model.reset()

    def observe(self, row: int, col: int, is_hit: bool) -> None:
        self._model.observe(row, col, is_hit)

    def best_guess(self) -> tuple[int, int]:
        self._model.run_bp(num_iter=self.bp_iters)
        return self._model.best_guess()

    def beliefs(self) -> np.ndarray:
        return self._model.beliefs()


# ---------------------------------------------------------------------------
# Training dataset generation
# ---------------------------------------------------------------------------

def _generate_sample(
    rng: np.random.Generator,
    max_shots: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate one (features, labels, unknown_mask) training triple.

    1. Place ships randomly on a 10×10 board.
    2. Reveal a random subset of cells (0 … max_shots).
    3. Return node features, ground-truth labels, and a mask selecting
       only unrevealed cells (the ones the model should learn to predict).
    """
    seed = int(rng.integers(0, 2**31))
    game = BattleshipGame(grid_size=GRID_SIZE, seed=seed)

    n_shots   = int(rng.integers(0, max_shots + 1))
    cells     = list(np.ndindex(GRID_SIZE, GRID_SIZE))
    rng.shuffle(cells)
    shot_cells = cells[:n_shots]

    # Build the BattleshipIsing external-field array from shots
    ising = BattleshipIsing(grid_size=GRID_SIZE)
    for r, c in shot_cells:
        ising.observe(r, c, bool(game.grid[r, c]))

    feats   = state_to_features(ising.h, ising.revealed).numpy()   # (N*N, 5)
    labels  = game.grid.flatten().astype(np.float32)               # (N*N,)
    unknown = (~ising.revealed).flatten().astype(bool)             # (N*N,)

    return feats, labels, unknown


def generate_dataset(
    n_samples: int = 6000,
    max_shots: int = 50,
    seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Generate a list of (features, labels, mask) training samples."""
    rng  = np.random.default_rng(seed)
    data = []
    for i in range(n_samples):
        data.append(_generate_sample(rng, max_shots))
        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{n_samples} samples …")
    return data


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_gnn(
    n_epochs:   int   = 30,
    n_train:    int   = 6000,
    n_val:      int   = 1000,
    hidden_dim: int   = 64,
    num_layers: int   = 6,
    lr:         float = 3e-4,
    batch_size: int   = 64,
    seed:       int   = 0,
    device:     str   = "cpu",
) -> tuple["BattleshipGNN", dict]:
    """Train a BattleshipGNN and return (model, training_history).

    The model is trained to minimise binary cross-entropy on UNKNOWN cells
    only (revealed cells already have known ground truth).

    Parameters
    ----------
    n_epochs   : training epochs
    n_train    : training samples
    n_val      : validation samples
    hidden_dim : width of MPNN layers
    num_layers : number of MPNN sweeps (analogous to BP iterations)
    lr         : Adam learning rate
    batch_size : mini-batch size
    seed       : RNG seed for reproducibility
    device     : "cpu" or "cuda"

    Returns
    -------
    model   : trained BattleshipGNN
    history : dict with keys "train_loss", "val_loss", "val_auc"
    """
    print("Generating training data …")
    train_data = generate_dataset(n_train, seed=seed)
    val_data   = generate_dataset(n_val,   seed=seed + 1)

    model     = BattleshipGNN(hidden_dim=hidden_dim, num_layers=num_layers)
    model     = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history   = defaultdict(list)

    def _run_epoch(dataset, train=True):
        model.train(train)
        rng     = np.random.default_rng(seed + (0 if train else 9999))
        indices = np.arange(len(dataset))
        if train:
            rng.shuffle(indices)

        total_loss, total_n = 0.0, 0
        for start in range(0, len(dataset), batch_size):
            batch_idx = indices[start : start + batch_size]
            feats, labels, masks = [], [], []
            for i in batch_idx:
                f, l, m = dataset[i]
                feats.append(f)
                labels.append(l)
                masks.append(m)

            # Stack into (B, N, 5), (B, N), (B, N)
            x_t = torch.tensor(np.stack(feats),  dtype=torch.float32, device=device)
            y_t = torch.tensor(np.stack(labels), dtype=torch.float32, device=device)
            m_t = torch.tensor(np.stack(masks),  dtype=torch.bool,    device=device)

            if train:
                optimizer.zero_grad()

            preds = model(x_t)      # (B, N)

            # BCE only on unknown cells
            loss = F.binary_cross_entropy(
                preds[m_t], y_t[m_t], reduction="mean"
            )

            if train:
                loss.backward()
                optimizer.step()

            n = int(m_t.sum().item())
            total_loss += loss.item() * n
            total_n    += n

        return total_loss / max(total_n, 1)

    print(f"Training GNN  ({n_train} train / {n_val} val / {n_epochs} epochs) …")
    for epoch in range(1, n_epochs + 1):
        t_loss = _run_epoch(train_data, train=True)
        v_loss = _run_epoch(val_data,   train=False)
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{n_epochs}  "
                  f"train_loss={t_loss:.4f}  val_loss={v_loss:.4f}")

    return model, dict(history)


# ---------------------------------------------------------------------------
# Game simulation  (agent-agnostic)
# ---------------------------------------------------------------------------

def play_game(
    agent,
    game: Optional[BattleshipGame] = None,
    seed: Optional[int] = None,
    max_shots: int = 100,
) -> dict:
    """Simulate one complete game with the given agent.

    Parameters
    ----------
    agent : any object with .reset(), .observe(r,c,hit), .best_guess()
    game  : pre-built BattleshipGame; if None a new one is created
    seed  : seed for BattleshipGame (ignored if game is provided)

    Returns
    -------
    dict with:
        n_shots  : shots needed to sink all ships
        hit_rate : fraction of shots that were hits
        shots    : list of (row, col)
        hits     : list of bool
    """
    if game is None:
        game = BattleshipGame(grid_size=GRID_SIZE, seed=seed)

    agent.reset()

    shots, hits = [], []
    hit_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)

    for _ in range(max_shots):
        row, col = agent.best_guess()
        is_hit   = game.shoot(row, col)
        agent.observe(row, col, is_hit)

        shots.append((row, col))
        hits.append(is_hit)
        if is_hit:
            hit_mask[row, col] = True

        if game.all_sunk(hit_mask):
            break

    return {
        "n_shots":  len(shots),
        "hit_rate": sum(hits) / len(hits) if hits else 0.0,
        "shots":    shots,
        "hits":     hits,
    }


# ---------------------------------------------------------------------------
# Agent comparison
# ---------------------------------------------------------------------------

def compare_all_agents(
    n_games:   int = 200,
    gnn_model: Optional[BattleshipGNN] = None,
    seed:      int = 100,
    device:    str = "cpu",
    verbose:   bool = True,
) -> dict[str, list]:
    """Run every agent for n_games and collect statistics.

    Agents benchmarked
    ------------------
    Random       : uniform random unrevealed cell
    Frequency    : valid-placement count heuristic
    Ising BP     : Belief Propagation on the Ising model
    GNN          : trained BattleshipGNN (only if gnn_model is provided)

    All agents face the SAME sequence of boards (same seeds) for a fair
    comparison.

    Returns
    -------
    dict mapping agent name → list of per-game shot counts
    """
    # Build agent factories (called fresh for each game)
    agents: dict[str, object] = {
        "Random":    RandomAgent(),
        "Frequency": FrequencyAgent(),
        "Ising BP":  IsingBPAgent(),
    }
    if gnn_model is not None:
        agents["GNN"] = GNNAgent(gnn_model, device=device)

    results: dict[str, list] = {name: [] for name in agents}

    for game_idx in range(n_games):
        game_seed = seed + game_idx
        game      = BattleshipGame(grid_size=GRID_SIZE, seed=game_seed)

        for name, agent in agents.items():
            r = play_game(agent, game=game, seed=game_seed)
            results[name].append(r["n_shots"])

        if verbose and (game_idx + 1) % 50 == 0:
            print(f"  Completed {game_idx + 1}/{n_games} games …")

    if verbose:
        print("\n── Results ──────────────────────────────────")
        print(f"  {'Agent':<12}  {'Mean':>6}  {'Median':>7}  {'Std':>6}")
        print(f"  {'─'*12}  {'─'*6}  {'─'*7}  {'─'*6}")
        for name, shots in results.items():
            a = np.array(shots)
            print(f"  {name:<12}  {a.mean():6.1f}  {np.median(a):7.1f}  {a.std():6.1f}")
        print()

    return results


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_training_history(history: dict) -> None:
    """Plot train / validation loss curves."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["train_loss"], label="train BCE")
    ax.plot(history["val_loss"],   label="val BCE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary cross-entropy")
    ax.set_title("GNN training history")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def plot_comparison(
    results:    dict[str, list],
    title:      str = "Agent comparison — shots to sink all ships",
    show_lines: bool = True,
) -> None:
    """Bar chart + box plot comparing agents by shots needed to win."""
    names  = list(results.keys())
    arrays = [np.array(results[n]) for n in names]
    means  = [a.mean() for a in arrays]
    stds   = [a.std()  for a in arrays]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: bar chart of means ──
    ax = axes[0]
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    bars   = ax.bar(names, means, color=colors, edgecolor="black", linewidth=0.7)
    ax.errorbar(names, means, yerr=stds, fmt="none", color="black",
                capsize=5, linewidth=1.5)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.5,
                f"{m:.1f}", ha="center", va="bottom", fontsize=9)
    if show_lines:
        # Mark random baseline
        ax.axhline(means[0], color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_ylabel("Mean shots to win")
    ax.set_title(title)
    ax.set_ylim(0, max(means) * 1.25)
    ax.grid(axis="y", alpha=0.3)

    # ── Right: box plot of distribution ──
    ax = axes[1]
    bp = ax.boxplot(
        [results[n] for n in names],
        labels=names,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_ylabel("Shots to win")
    ax.set_title("Distribution of game lengths")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_belief_comparison(
    game:     BattleshipGame,
    gnn_agent:  Optional[GNNAgent]    = None,
    ising_agent: Optional[IsingBPAgent] = None,
    n_shots:  int = 10,
) -> None:
    """Side-by-side GNN vs Ising-BP belief maps after n_shots random shots."""
    # Take n_shots random shots against the board
    rng = np.random.default_rng(42)
    cells = list(np.ndindex(GRID_SIZE, GRID_SIZE))
    rng.shuffle(cells)
    shot_cells = cells[:n_shots]

    agents_to_plot = {}
    if gnn_agent  is not None:
        gnn_agent.reset()
        agents_to_plot["GNN"] = gnn_agent
    if ising_agent is not None:
        ising_agent.reset()
        agents_to_plot["Ising BP"] = ising_agent

    # Feed identical shots to every agent
    for r, c in shot_cells:
        is_hit = game.shoot(r, c)
        for agent in agents_to_plot.values():
            agent.observe(r, c, is_hit)

    # Also run BP for Ising agent to get beliefs
    if ising_agent is not None:
        ising_agent.best_guess()   # triggers internal run_bp

    ncols = len(agents_to_plot) + 1
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    # True board
    ax = axes[0]
    ax.imshow(game.grid, cmap="Greens", vmin=0, vmax=1.4)
    ax.set_title("True board")
    _annotate_shots(ax, shot_cells, game)

    for ax, (name, agent) in zip(axes[1:], agents_to_plot.items()):
        p = agent.beliefs()
        im = ax.imshow(p, cmap="hot", vmin=0, vmax=1)
        ax.set_title(f"{name} beliefs\n(after {n_shots} shots)")
        _annotate_shots(ax, shot_cells, game)
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.show()


def _annotate_shots(ax, shot_cells, game):
    """Helper: add X/· markers on an axes."""
    COL_LABELS = list("ABCDEFGHIJ")
    ax.set_xticks(range(GRID_SIZE))
    ax.set_xticklabels(COL_LABELS)
    ax.set_yticks(range(GRID_SIZE))
    for r, c in shot_cells:
        is_hit = bool(game.grid[r, c])
        ax.text(c, r, "✕" if is_hit else "·",
                ha="center", va="center",
                color="white" if is_hit else "deepskyblue",
                fontsize=11, fontweight="bold")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 62)
    print("  Battleship GNN  —  Ising-model-inspired move predictor")
    print(f"  PyTorch Geometric: {'available' if HAS_PYG else 'NOT installed (using PyTorch fallback)'}")
    print("=" * 62)
    print()

    # ── 1. Train ─────────────────────────────────────────────────────
    model, history = train_gnn(
        n_epochs   = 30,
        n_train    = 6000,
        n_val      = 1000,
        hidden_dim = 64,
        num_layers = 6,
        lr         = 3e-4,
        batch_size = 64,
    )
    plot_training_history(history)

    # ── 2. Compare all agents ────────────────────────────────────────
    print("\nComparing agents over 200 games …")
    results = compare_all_agents(n_games=200, gnn_model=model)
    plot_comparison(results)

    # ── 3. Belief map side-by-side ───────────────────────────────────
    print("\nPlotting belief map comparison …")
    demo_game   = BattleshipGame(seed=42)
    gnn_agent   = GNNAgent(model)
    ising_agent = IsingBPAgent()
    plot_belief_comparison(demo_game, gnn_agent, ising_agent, n_shots=15)
