# Tasks

## Phase 1: Data Pipeline & Infrastructure

- [ ] **1.1 Implement UCI Move Mapping**
    - Create `src/chesstrm/data/mapping.py`.
    - Generate/Hardcode the mapping of 1968 UCI moves to indices.
    - Implement `move_to_index(move_str)` and `index_to_move(index)`.
    - *Test:* Verify all legal moves in standard positions are covered.

- [ ] **1.2 Create `ChessDataset`**
    - Create `src/chesstrm/data/dataset.py`.
    - Implement `ChessDataset(torch.utils.data.Dataset)`.
    - Handle H5 file loading (with SWMR support).
    - Implement `__getitem__` to return `x` (19 planes) and `target_index`.
    - *Test:* Load a dummy H5, check shapes and types.

- [ ] **1.3 Create H5 Inspector/Validator**
    - Create `src/chesstrm/utils/inspect_h5.py`.
    - Script to validate H5 file structure (datasets `x`, `y` or similar).
    - *Test:* Run on provided H5 file (if available) or mock one.

## Phase 2: Model Core (ChessTRM)

- [ ] **2.1 Implement Components**
    - Create `src/chesstrm/model/layers.py`.
    - Implement `InputEmbedding` (19 -> d_model).
    - Implement `ActionEmbedding` (1968 -> d_model).
    - Implement `ReadoutHead` (d_model -> 1968).
    - *Test:* Unit tests for shape correctness.

- [ ] **2.2 Implement Transformer Block**
    - Create `src/chesstrm/model/transformer.py`.
    - Implement `RecursiveBlock` with Pre-Norm and Gradient Checkpointing support.
    - *Test:* Forward pass check, gradient flow check.

- [ ] **2.3 Implement `ChessTRM` Class**
    - Create `src/chesstrm/model/trm.py`.
    - Assemble components.
    - Implement `forward(x, n_steps)` with the unrolled loop.
    - Manage state ($x, y, z$).
    - *Test:* Full forward pass with dummy data, check output shape $(Batch, Steps, 1968)$.

## Phase 3: Training Logic

- [ ] **3.1 Implement DIS Loss**
    - Create `src/chesstrm/training/loss.py`.
    - Implement `DISLoss` with linear weight scheduling.
    - *Test:* Calculate loss on dummy logits/targets, verify weighting.

- [ ] **3.2 Implement Training Loop**
    - Create `src/chesstrm/training/trainer.py` (or extend `main.py`).
    - Setup Optimizer (AdamW), Scheduler.
    - Implement loop with `torch.utils.checkpoint`.
    - *Test:* Run 1 epoch on dummy data, check loss decrease.

## Phase 4: Inference & Tools

- [ ] **4.1 FEN to Tensor**
    - Implement helper to convert FEN string to 19-plane tensor.
    - *Test:* Verify against known board states.

- [ ] **4.2 Inference Script**
    - Create `src/chesstrm/inference.py`.
    - Load model, take FEN, run forward, output top moves.
