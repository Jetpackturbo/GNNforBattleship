# GNNforBattleship
# OR
## Battleship via the Ising Model and Belief Propagation

This project models the **targeting problem in Battleship** — deciding which cell to shoot next — as inference in an **Ising graphical model**, with **Belief Propagation (BP)** as the inference engine.

### Problem Setup

A player observes the opponent's 10 × 10 grid through a sequence of shots.  Each shot reveals one cell as a **hit** (ship) or **miss** (water).  The goal is to locate and sink all ships in as few shots as possible.

Standard fleet: Carrier (5), Battleship (4), Cruiser (3), Submarine (3), Destroyer (2) — a total of **17 ship cells** out of 100.

---

## 1. Ising Model Formulation

### 1.1 Variables

Each cell (i, j) is assigned a binary spin:

```
x_{i,j} ∈ { −1, +1 }
     +1  →  cell contains part of a ship
     −1  →  cell is water
```

### 1.2 Energy Function

The joint distribution over all spins is the **ferromagnetic Ising model** on a 2-D grid:

```
P(x) ∝ exp( −E(x) )

E(x) = −J · Σ_{<s,t>} x_s x_t  −  Σ_{i,j} h_{i,j} x_{i,j}
```

where the first sum runs over all **horizontally and vertically adjacent** cell pairs.

**Parameters:**

| Symbol | Role |
|--------|------|
| J > 0  | Ferromagnetic coupling.  Adjacent cells prefer the same state, capturing the spatial *contiguity* of ships. |
| h_{i,j} | Cell-specific external field encoding evidence (see §1.3). |

### 1.3 External Fields — Encoding Evidence

The external field at each cell encodes what we know about it:

| Cell state | h_{i,j} | Effect on posterior |
|------------|---------|---------------------|
| Unknown (prior) | h_prior = ½ log(p / (1−p)) ≈ −0.793 | Reflects prior ship density p = 17/100 ≈ 0.17 |
| Confirmed **hit** | +H_obs = +10 | Pins x ≈ +1 with near certainty |
| Confirmed **miss** | −H_obs = −10 | Pins x ≈ −1 with near certainty |

The prior `h_prior ≈ −0.793` ensures that an isolated cell with no neighbour evidence has a marginal probability close to 17 % of containing a ship:

```
P(x = +1)  =  exp(2 h_prior) / (1 + exp(2 h_prior))
           =  sigmoid(2 × (−0.793))  ≈  0.17
```

### 1.4 Why a Ferromagnetic Coupling?

Ships are *contiguous* line segments, so if a cell contains a ship, its horizontal or vertical neighbours are more likely to also contain a ship.  A positive coupling J models exactly this:

- **J = 0**: cells are independent — no spatial structure is used.
- **J > 0**: adjacent cells are correlated, propagating hit evidence to neighbours.
- **J → ∞**: cells are completely locked together (too rigid for a grid with mixed ship/water boundaries).

The 2-D Ising critical point is J_c ≈ 0.44.  A value of J ≈ 0.5 (slightly super-critical) gives strong local correlations without making the entire board behave as a single cluster.

---

## 2. Belief Propagation Update Rules

### 2.1 Message Convention

Following the **half-LLR** convention used throughout this codebase (see `bpsol.py`), every directed edge carries a scalar message:

```
m_{s→t}  =  ½ log( ν_{s→t}(+1) / ν_{s→t}(−1) )
```

Four message arrays are maintained for the grid (notation matches `bpsol.py`):

| Array | Shape | Semantics |
|-------|-------|-----------|
| `hhorright[i, j]` | (N, N−1) | message from (i, j) → (i, j+1) |
| `hhorleft [i, j]` | (N, N−1) | message from (i, j+1) → (i, j) |
| `hvertdown[i, j]` | (N−1, N) | message from (i, j) → (i+1, j) |
| `hvertup  [i, j]` | (N−1, N) | message from (i+1, j) → (i, j) |

All messages are initialised to **zero** (uniform message prior) and updated iteratively.

### 2.2 BP Update Rule

The parallel (flooded) update for the message from node s to neighbour t is:

```
m_{s→t}^{new}  =  h_s  +  Σ_{u ∈ N(s) \ {t}}  f_J( m_{u→s}^{current} )
```

where the **edge factor** in the half-LLR domain is:

```
f_J(m)  =  arctanh( tanh(J) · tanh(m) )
```

This is the Ising sum-product rule: the tanh kernel contracts the message through the pairwise factor `ψ(x_s, x_t) = exp(J x_s x_t)`.

### 2.3 Explicit Expansions for the Grid

For an **interior** cell (i, j) sending to its right neighbour (i, j+1):

```
hhorright_new[i, j]  =  h[i, j]
    +  f_J( hhorright[i, j−1] )   ← from left   (i, j−1)
    +  f_J( hvertdown[i−1, j] )   ← from above  (i−1, j)
    +  f_J( hvertup  [i,   j] )   ← from below  (i+1, j)
```

The three other directions are handled symmetrically.  **Boundary terms are absent** when the sending cell lies on the corresponding edge of the grid; no special-casing is needed because the slicing arithmetic produces zero-length arrays for those terms.

### 2.4 Vectorised Implementation

All four message arrays are updated simultaneously in `BattleshipIsing._bp_step()` using NumPy broadcasting and strided slices, avoiding explicit loops over cells.  For example:

```python
# hhorright_new[i, j] = h[i, j] + (f from left) + (f from above) + (f from below)
new_rr = h[:, :N-1].copy()          # base:  h[i, j]       for j = 0…N−2
new_rr[:, 1:]    += f(rr[:, :N-2])  # from left:  j > 0
new_rr[1:, :]    += f(rd[:, :N-1])  # from above: i > 0
new_rr[:N-1, :]  += f(ru[:, :N-1])  # from below: i < N−1
```

This matches the flooded parallel schedule used in `bpsol.py`.

### 2.5 Belief Computation

After BP converges, the marginal belief at each cell is:

```
b[i, j]  =  h[i, j]  +  Σ_{k ∈ N(i,j)}  f_J( m_{k→(i,j)} )

P( x_{i,j} = +1 | evidence )  =  exp(2 b[i,j]) / (1 + exp(2 b[i,j]))
```

The conversion `exp(2b) / (1 + exp(2b))` matches `bpsol.py`'s convention for the half-LLR parameterisation.

---

## 3. Agent Strategy

After each shot the agent:

1. Calls `model.observe(row, col, is_hit)` — updates h_{row,col} and resets messages.
2. Calls `model.run_bp(num_iter=60)` — re-runs BP to convergence.
3. Calls `model.best_guess()` — returns `argmax_{unrevealed (i,j)} P(ship | evidence)`.
4. Fires at that cell.

This is a **greedy maximum-a-posteriori** policy: each shot maximises the immediate probability of a hit given all past observations.

---

## 4. Relationship to Existing Code

| File | Role |
|------|------|
| `bpsol.py` | Parallel BP on a random Ising grid; convergence study vs. β.  Establishes the half-LLR convention and flooded schedule used here. |
| `gridbpsol.py` | BP on an image grid with Gaussian unary potentials (foreground/background segmentation).  Demonstrates the generalised F-update including unary terms. |
| `comb_sum_product.py` | Exact sum-product on a comb-structured subgraph; used by the Gibbs sampler. |
| `comb_gibbs_step.py` | One Gibbs step using the comb decomposition for block sampling. |
| `ising_gibbs_comb.py` | Driver: runs the comb Gibbs sampler on a 60 × 60 Ising model. |
| **`battleship_ising.py`** | **Ising + BP inference for Battleship** (`BattleshipIsing`, `BattleshipGame`). |
| **`battleship_demo.py`** | **Demo**: prior beliefs, full game simulation, coupling-strength sweep. |

---

## 5. Usage

```python
from battleship_ising import BattleshipIsing, BattleshipGame

game  = BattleshipGame(seed=42)          # random opponent board
model = BattleshipIsing(J=0.5)           # Ising model with J=0.5

while True:
    model.run_bp(num_iter=60)            # propagate beliefs
    row, col = model.best_guess()        # greedy MAP action
    is_hit   = game.shoot(row, col)
    model.observe(row, col, is_hit)

    if game.all_sunk(model.revealed & (model.h > 0)):
        break
```

Run the full demo (3 experiments with visualisations):

```bash
python battleship_demo.py
```

---

## 6. Learned Move Policies

The repository now includes two learned Battleship move-selection models:

- `gnn.py`: message-passing GNN policy trained to imitate a strong
  probability-density Battleship heuristic.
- `gnn-attn.py`: attention-based graph policy trained on the same target.

These models **do not** predict hidden ship occupancy directly.  Instead, they
learn a **next-move distribution** over unrevealed cells, which better matches
the gameplay objective of choosing the next shot.

### Training target

The supervision target is a **DataGenetics-style probability-density policy**:
for each partial board, enumerate valid horizontal and vertical placements of
ships with lengths 5, 4, 3, 3, and 2, discard placements inconsistent with
known misses, and upweight placements that pass through known hits.  The next
move is the unrevealed cell with the highest aggregate placement count.

### Benchmark source

The benchmark suite in `gnn.py` is derived from Nick Berry's online Battleship
analysis:

- [DataGenetics: Battleship](https://www.datagenetics.com/blog/december32011/)

That article compares `Random`, `Hunt/Target`, and `Probability Density`
strategies over large simulation runs.  This repo mirrors the same family of
baselines, but adapts them to the simpler environment used here, which exposes
only **hit/miss** feedback and does **not** announce when a ship has been sunk.

### Training and testing scripts

Train a plain message-passing policy model:

```bash
python train_model.py \
  --model gnn \
  --output checkpoints/gnn-policy.pt \
  --epochs 20 \
  --n-train 4000 \
  --n-val 800
```

Train the attention-based policy model:

```bash
python train_model.py \
  --model attn \
  --output checkpoints/attn-policy.pt \
  --epochs 20 \
  --n-train 4000 \
  --n-val 800 \
  --num-layers 4 \
  --num-heads 4
```

Enable `tqdm` progress bars by default, or disable them with:

```bash
python train_model.py --model gnn --output checkpoints/gnn-policy.pt --no-tqdm
```

Enable Weights & Biases logging:

```bash
export WANDB_API_KEY=your_key_here
python train_model.py \
  --model gnn \
  --output checkpoints/gnn-policy.pt \
  --use-wandb \
  --wandb-project GNNforBattleship \
  --wandb-run-name gnn-policy-run
```

Or pass the API key directly:

```bash
python train_model.py \
  --model attn \
  --output checkpoints/attn-policy.pt \
  --use-wandb \
  --wandb-api-key your_key_here \
  --wandb-project GNNforBattleship \
  --wandb-run-name attn-policy-run
```

Benchmark one or more saved checkpoints against the built-in baselines:

```bash
python test_model.py \
  --checkpoint checkpoints/gnn-policy.pt \
  --checkpoint checkpoints/attn-policy.pt \
  --n-games 200 \
  --save-json results/benchmark.json
```

You can also log benchmark metrics to `wandb`:

```bash
python test_model.py \
  --checkpoint checkpoints/gnn-policy.pt \
  --checkpoint checkpoints/attn-policy.pt \
  --n-games 200 \
  --use-wandb \
  --wandb-project GNNforBattleship \
  --wandb-run-name benchmark-run
```

---

## 7. Design Notes and Limitations

**Why not encode exact ship-length constraints?**  The Ising model captures *local* pair correlations but not the global constraint that each ship has a specific fixed length.  Exact inference with length constraints would require a factor graph with chain factors along each row and column — significantly more complex.  The Ising model is a tractable, physically motivated approximation that still leverages the key spatial structure.

**BP on a loopy graph.**  A 2-D grid contains cycles, so BP is not exact.  It is known to converge for small J (high temperature) and may oscillate near the critical point.  In practice, 40–60 iterations are sufficient for the Battleship grid.

**Coupling strength J and prior calibration.**  The formula `h_prior = ½ log(p/(1−p))` gives the correct isolated-cell prior when J = 0.  With J > 0 the ferromagnetic coupling amplifies the external field: in the ordered phase (J > J_c ≈ 0.44 for the 2-D Ising model) a small negative h tips almost every cell to state −1, so the true marginal P(ship) at BP convergence will be lower than the naive `p = 17 %`.  This is a calibration artefact, not a bug; hit evidence is still propagated correctly because the large positive H_obs = 10 overwhelms the coupling.  Using J < J_c keeps the prior closer to the target density.

**Empirical performance.**  On a standard 10 × 10 board with seed 42, the greedy BP agent sinks all ships in ≈ 66 shots.  Uniform-random targeting requires ≈ 95 shots in expectation (expected position of the last of 17 targets in a random permutation of 100 cells = 17 × 101 / 18 ≈ 95).  The Ising BP agent therefore reduces the number of shots by roughly **30 %** by concentrating fire near confirmed hits.
