# ChessTRM: Tiny Recursive Model for Chess

ChessTRM is a minimalist, recursive neural network architecture for chess. It leverages a recursive Transformer block to refine its internal state ($z$) and action intent ($y$) over multiple time steps, mimicking a "thinking" process.

## Features

- **Recursive Architecture**: Uses a shared Transformer block over multiple steps (Deep Equilibrium approach).
- **DIS Loss**: Deep Improvement Supervision loss to encourage improvement at each recursion step.
- **Minimalist Design**: Small parameter count but deep computation graph.
- **Efficient Data Pipeline**: H5-based data loading with SWMR support for concurrent access.

## Installation

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/your-repo/chesstrm.git
cd chesstrm
pip install -e .
```

### Dependencies
- Python 3.8+
- PyTorch
- python-chess
- h5py
- numpy

## Usage

### 1. Data Preparation

ChessTRM expects data in HDF5 (`.h5`) format. The H5 file should contain:
- `input_tensor`: Dataset of shape `(N, 19, 8, 8)` or `(N, 8, 8, 19)` (uint8 or float).
- `target_d1_move` (or similar): Dataset of target moves (indices or UCI strings).

To inspect an H5 file:
```bash
python -m chesstrm.utils.inspect_h5 path/to/data.h5
```

### 2. Training

To train the model, use the `train` command.

```bash
python -m chesstrm.main train --data data/dataset.h5 --epochs 20 --batch-size 64
```

**Common Arguments:**
- `--data`: Path to the H5 dataset (required).
- `--epochs`: Number of training epochs (default: 10).
- `--batch-size`: Batch size (default: 32).
- `--lr`: Learning rate (default: 3e-4).
- `--checkpoint-dir`: Directory to save checkpoints (default: `checkpoints`).
- `--resume`: Path to a checkpoint `.pt` file to resume training.
- `--max-recursion`: Number of recursion steps (default: 16).

### 3. Inference

To run inference on a specific chess position (FEN):

```bash
python -m chesstrm.main inference --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" --model-path checkpoints/final.pt
```

**Arguments:**
- `--fen`: The FEN string of the position to analyze.
- `--model-path`: Path to the trained model checkpoint (optional; if omitted, uses random weights).
- `--top-k`: Number of top moves to display (default: 5).
- `--n-steps`: Number of thinking steps (recursion depth) for inference.

## Architecture Overview

ChessTRM processes a chess position using a recursive loop:

1.  **Input**: The board state is encoded into 19 planes (pieces, color, castling rights, etc.).
2.  **State Initialization**:
    - $x$: Static board embedding.
    - $z_0$: Latent "scratchpad" state (randomly initialized).
    - $y_0$: Action intent (learnable parameter).
3.  **Recursive Step**:
    At each step $t$:
    - The inputs are combined: $Input_t = Embed(x) + Proj(y_{t-1}) + z_{t-1}$.
    - A Transformer block processes $Input_t$.
    - $z_t$ is updated via residual connection.
    - $y_t$ is updated from the new state.
    - A readout head projects $y_t$ to move probabilities (logits).
4.  **Output**: The final logits (or sequence of logits) represent the move distribution.

## Project Structure

```
src/chesstrm/
├── data/           # Data loading, FEN conversion, and move mapping
├── model/          # Neural network architecture (Layers, Transformer, TRM)
├── training/       # Training loop (Trainer) and Loss functions (DISLoss)
├── utils/          # Utility scripts (H5 inspection)
├── inference.py    # Inference logic
├── main.py         # Entry point (CLI)
└── tests/          # Unit tests
```

## Contributing

1.  Install dependencies: `pip install -e .`
2.  Run tests: `pytest src/chesstrm/tests`
