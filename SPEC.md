# ChessTRM Technical Specification

## 1. Introduction

ChessTRM (Tiny Recursive Model) is a neural engine that uses a recursive architecture to refine its understanding of a chess position over multiple iterations. Instead of a deep static network, it uses a small core (Transformer) reused cyclically.

## 2. Data Structures

### 2.1. Input State ($x$)
- **Shape:** $(Batch, 19, 8, 8)$ flattened to $(Batch, 64, 19)$ or projected to $(Batch, 64, d_{model})$.
- **Planes:**
    - 0-5: Active player pieces (P, N, B, R, Q, K)
    - 6-11: Opponent pieces (P, N, B, R, Q, K)
    - 12-13: Repetitions (1x, 2x)
    - 14-17: Castling rights (White O-O, O-O-O, Black ...)
    - 18: Color to move
- **Processing:** Flattened to sequence of 64 tokens + Positional Embeddings.

### 2.2. Action Space ($y$)
- **Size:** 1968 possible moves (UCI format coverage).
- **Representation:**
    - Input: Learnable embedding or projected vector of size $d_{model}$.
    - Output: Logits vector of size 1968.
- **Initialization ($y_0$):** Learnable embedding or noise.

### 2.3. Latent Memory ($z$)
- **Shape:** $(Batch, 64, d_{model})$.
- **Role:** Main "scratchpad" updated at each iteration.

## 3. Model Architecture (`ChessTRM`)

### 3.1. Hyperparameters
- `d_model`: 256 or 512
- `n_heads`: 8
- `n_layers`: 2 (per block)
- `dropout`: 0.1
- `max_recursion` ($T$): 8 to 16 (default for inference)

### 3.2. Components
- **InputProjection:** Projects $x$ (19 channels) to $d_{model}$.
- **PositionalEncoding:** Learnable, size $(64, d_{model})$.
- **ActionEmbedding:** Projects $y$ logits/indices to $d_{model}$ (or broadcast global token).
- **StreamCombiner:** Fuses $x$, $y$, and $z$.
    - Formula: $Input_t = \text{Embed}(x) + \text{Proj}(y_{t-1}) + z_{t-1}$
    - Note: $y$ might be a global token concatened or broadcasted.
- **TransformerBlock:**
    - Pre-Norm architecture.
    - Multi-Head Self Attention.
    - FeedForward (GeLU/SwiGLU).
    - Residual connections.
- **ReadoutHead:** Projects $y_{emb}$ (or $z$) to 1968 logits.

### 3.3. Forward Pass (Unrolled)
1. Initialize $z_0 \sim \mathcal{N}(0, 0.02)$.
2. Initialize $y_0$ (learnable).
3. Loop $t = 1 \dots T$:
    - Combine $x, y_{t-1}, z_{t-1}$.
    - Pass through `TransformerBlock` $\rightarrow z_t$.
    - Update $y_t$ (via readout or auxiliary head).
    - Store logits for DIS loss.
4. Return list of logits $[logits_1, \dots, logits_T]$.

## 4. Training

### 4.1. Deep Improvement Supervision (DIS) Loss
- **Formula:** $\mathcal{L}_{DIS} = \sum_{t=1}^{T} w_t \cdot \mathcal{L}_{CE}(y_t, \text{Target})$
- **Weights ($w_t$):** Linear increasing schedule $w_t = t/T$.
- **Target:** Best move index (constant across steps).

### 4.2. Optimization
- **Gradient Checkpointing:** Used on the `TransformerBlock` to save VRAM.
- **Optimizer:** AdamW.

## 5. Implementation Details

- **Language:** Python 3.9+
- **Framework:** PyTorch
- **Dependencies:** `python-chess` (logic/mappings), `h5py` (data), `numpy`.
- **Project Structure:** `src/chesstrm`
