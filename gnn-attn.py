#!/usr/bin/env python
# coding: utf-8
"""
gnn-attn.py — attention-based Battleship policy network.

This file implements a graph-attention alternative to `gnn.py`. It uses the
same node features, the same DataGenetics-style probability-density teacher,
and the same benchmark harness so the models are directly comparable.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from gnn import (
    GRID_SIZE,
    NODE_FEATURES,
    BattleshipGame,
    GNNAgent,
    IsingBPAgent,
    ProbabilityDensityAgent,
    _maybe_tqdm,
    _wandb_log,
    benchmark_reference,
    compare_all_agents,
    generate_dataset,
    observation_masks_to_features,
    plot_comparison,
    plot_training_history,
)


def _policy_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked_logits = logits.masked_fill(~mask, -1e9)
    log_probs = torch.log_softmax(masked_logits, dim=-1)
    return -(targets * log_probs).sum(dim=-1).mean()


def build_incoming_neighborhood(grid_size: int = GRID_SIZE):
    """Precompute incoming neighbors for every node on the 4-connected grid."""
    n_nodes = grid_size * grid_size
    max_degree = 4
    incoming_idx = np.zeros((n_nodes, max_degree), dtype=np.int64)
    incoming_dir = np.zeros((n_nodes, max_degree, 1), dtype=np.float32)
    incoming_mask = np.zeros((n_nodes, max_degree), dtype=bool)

    for row in range(grid_size):
        for col in range(grid_size):
            node = row * grid_size + col
            neighbors = []
            if col - 1 >= 0:
                neighbors.append((node - 1, 1.0))
            if col + 1 < grid_size:
                neighbors.append((node + 1, 1.0))
            if row - 1 >= 0:
                neighbors.append((node - grid_size, 0.0))
            if row + 1 < grid_size:
                neighbors.append((node + grid_size, 0.0))

            for slot, (src, direction) in enumerate(neighbors):
                incoming_idx[node, slot] = src
                incoming_dir[node, slot, 0] = direction
                incoming_mask[node, slot] = True

    return (
        torch.from_numpy(incoming_idx),
        torch.from_numpy(incoming_dir),
        torch.from_numpy(incoming_mask),
    )


_INCOMING_IDX, _INCOMING_DIR, _INCOMING_MASK = build_incoming_neighborhood(GRID_SIZE)


class GridAttentionLayer(nn.Module):
    """Multi-head attention restricted to the local Battleship grid neighborhood."""

    def __init__(self, hidden_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.register_buffer("incoming_idx", _INCOMING_IDX.clone())
        self.register_buffer("incoming_dir", _INCOMING_DIR.clone())
        self.register_buffer("incoming_mask", _INCOMING_MASK.clone())

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.edge_proj = nn.Linear(1, num_heads)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeezed = x.dim() == 2
        if squeezed:
            x = x.unsqueeze(0)

        batch_size, n_nodes, _ = x.shape
        degree = self.incoming_idx.shape[1]

        neighbor_x = x[:, self.incoming_idx, :]

        q = self.q_proj(x).view(batch_size, n_nodes, self.num_heads, self.head_dim)
        k = self.k_proj(neighbor_x).view(
            batch_size, n_nodes, degree, self.num_heads, self.head_dim
        )
        v = self.v_proj(neighbor_x).view(
            batch_size, n_nodes, degree, self.num_heads, self.head_dim
        )

        logits = (q.unsqueeze(2) * k).sum(dim=-1) / np.sqrt(self.head_dim)
        logits = logits + self.edge_proj(self.incoming_dir).unsqueeze(0)

        mask = self.incoming_mask.unsqueeze(0).unsqueeze(-1)
        logits = logits.masked_fill(~mask, -1e9)
        attn = torch.softmax(logits, dim=2) * mask

        agg = (attn.unsqueeze(-1) * v).sum(dim=2).reshape(batch_size, n_nodes, self.hidden_dim)
        h = self.norm1(x + self.out_proj(agg))
        out = self.norm2(h + self.ffn(h))

        return out.squeeze(0) if squeezed else out


class BattleshipAttentionGNN(nn.Module):
    """Attention-based policy network for Battleship move selection."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(NODE_FEATURES, hidden_dim),
            nn.ReLU(),
        )
        self.layers = nn.ModuleList(
            [GridAttentionLayer(hidden_dim, num_heads=num_heads) for _ in range(num_layers)]
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeezed = x.dim() == 2
        if squeezed:
            x = x.unsqueeze(0)

        h = self.encoder(x)
        for layer in self.layers:
            h = layer(h)

        logits = self.decoder(h).squeeze(-1)
        return logits.squeeze(0) if squeezed else logits

    def predict_from_masks(
        self,
        hit_mask: np.ndarray,
        revealed: np.ndarray,
        device: str = "cpu",
    ) -> np.ndarray:
        x = observation_masks_to_features(hit_mask, revealed).to(device)
        self.eval()
        with torch.no_grad():
            logits = self.forward(x).detach().cpu().numpy().reshape(GRID_SIZE, GRID_SIZE)

        probs = np.zeros_like(logits, dtype=np.float64)
        unknown = ~revealed
        if np.any(unknown):
            masked = logits[unknown]
            masked = masked - masked.max()
            exp_scores = np.exp(masked)
            probs[unknown] = exp_scores / exp_scores.sum()
        return probs


class AttentionGNNAgent:
    """Battleship policy agent backed by `BattleshipAttentionGNN`."""

    def __init__(
        self,
        model: BattleshipAttentionGNN,
        grid_size: int = GRID_SIZE,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.grid_size = grid_size
        self.device = device
        self.revealed = np.zeros((grid_size, grid_size), dtype=bool)
        self.hit_mask = np.zeros((grid_size, grid_size), dtype=bool)

    def reset(self) -> None:
        self.revealed[:] = False
        self.hit_mask[:] = False

    def observe(self, row: int, col: int, is_hit: bool) -> None:
        self.revealed[row, col] = True
        self.hit_mask[row, col] = bool(is_hit)

    def best_guess(self) -> tuple[int, int]:
        return divmod(int(np.argmax(self.beliefs())), self.grid_size)

    def beliefs(self) -> np.ndarray:
        return self.model.predict_from_masks(self.hit_mask, self.revealed, self.device)


def train_attention_gnn(
    n_epochs: int = 30,
    n_train: int = 6000,
    n_val: int = 1000,
    hidden_dim: int = 64,
    num_layers: int = 4,
    num_heads: int = 4,
    lr: float = 3e-4,
    batch_size: int = 64,
    max_context_shots: int = 40,
    seed: int = 0,
    device: str = "cpu",
    show_progress: bool = True,
    wandb_run: Any = None,
) -> tuple["BattleshipAttentionGNN", dict]:
    """Train the attention-based policy network against the same teacher."""
    print("Generating policy-training data for attention GNN ...")
    train_data = generate_dataset(
        n_train,
        max_context_shots=max_context_shots,
        seed=seed,
        show_progress=show_progress,
    )
    val_data = generate_dataset(
        n_val,
        max_context_shots=max_context_shots,
        seed=seed + 1,
        show_progress=show_progress,
    )

    model = BattleshipAttentionGNN(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
    ).to(device)
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
            feats, targets, masks = [], [], []
            for i in batch_idx:
                feat, target, mask = dataset[i]
                feats.append(feat)
                targets.append(target)
                masks.append(mask)

            x_t = torch.tensor(np.stack(feats), dtype=torch.float32, device=device)
            y_t = torch.tensor(np.stack(targets), dtype=torch.float32, device=device)
            m_t = torch.tensor(np.stack(masks), dtype=torch.bool, device=device)

            if train:
                optimizer.zero_grad()

            logits = model(x_t)
            loss = _policy_loss(logits, y_t, m_t)

            if train:
                loss.backward()
                optimizer.step()

            pred_idx = logits.masked_fill(~m_t, -1e9).argmax(dim=-1)
            target_idx = y_t.argmax(dim=-1)
            batch_acc = float((pred_idx == target_idx).float().mean().item())

            total_loss += float(loss.item())
            total_acc += batch_acc
            total_batches += 1

        denom = max(total_batches, 1)
        return total_loss / denom, total_acc / denom

    print(
        f"Training attention GNN ({n_train} train / {n_val} val / {n_epochs} epochs) ..."
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


def compare_with_attention(
    attn_model: BattleshipAttentionGNN,
    n_games: int = 200,
    seed: int = 100,
    device: str = "cpu",
    gnn_model: Optional[object] = None,
    show_progress: bool = True,
    wandb_run: Any = None,
) -> dict[str, list[int]]:
    """Run the shared benchmark with the attention model included."""
    extra_agents = {"GNN Attention": AttentionGNNAgent(attn_model, device=device)}
    return compare_all_agents(
        n_games=n_games,
        gnn_model=gnn_model,
        seed=seed,
        device=device,
        verbose=True,
        extra_agents=extra_agents,
        show_progress=show_progress,
        wandb_run=wandb_run,
    )


if __name__ == "__main__":
    print("=" * 72)
    print("  Battleship attention GNN")
    print("=" * 72)
    print(benchmark_reference())
    print()

    model, history = train_attention_gnn(
        n_epochs=20,
        n_train=4000,
        n_val=800,
        hidden_dim=64,
        num_layers=4,
        num_heads=4,
        lr=3e-4,
        batch_size=64,
    )
    plot_training_history(history)

    print("\nRunning benchmark with attention model over 200 games ...")
    results = compare_with_attention(model, n_games=200)
    plot_comparison(results, title="Battleship benchmark with attention GNN")
