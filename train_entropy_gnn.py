#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import torch

from gnn import BattleshipGNN, train_gnn_entropy_gain, GRID_SIZE, SHIP_LENGTHS, benchmark_reference

def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, history = train_gnn_entropy_gain(
        n_epochs=20,
        n_train=4000,
        n_val=800,
        hidden_dim=64,
        num_layers=6,
        lr=3e-4,
        batch_size=64,
        max_context_shots=40,
        posterior_samples=16,
        seed=0,
        device=device,
        use_pyg=False,
        show_progress=True,
        wandb_run=None,
        init_model=None,
    )

    checkpoint = {
        "model_type": "gnn",
        "model_kwargs": {
            "hidden_dim": 64,
            "num_layers": 6,
            "use_pyg": False,
        },
        "train_kwargs": {
            "n_epochs": 20,
            "n_train": 4000,
            "n_val": 800,
            "hidden_dim": 64,
            "num_layers": 6,
            "lr": 3e-4,
            "batch_size": 64,
            "max_context_shots": 40,
            "posterior_samples": 16,
            "seed": 0,
            "device": device,
        },
        "training_objective": "entropy_gain",
        "grid_size": GRID_SIZE,
        "ship_lengths": list(SHIP_LENGTHS),
        "benchmark_reference": benchmark_reference(),
        "history": history,
        "state_dict": model.state_dict(),
    }

    out_path = Path("checkpoints/gnn_entropy_gain.pt").expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out_path)
    print(f"Saved entropy-gain GNN checkpoint to {out_path}")

if __name__ == "__main__":
    main()