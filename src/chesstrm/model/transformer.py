import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class RecursiveBlock(nn.Module):
    """
    Standard Transformer block with Pre-Normalization and Residual connections.
    Includes support for gradient checkpointing to save VRAM.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        dim_feedforward: int = 2048,
    ):
        super().__init__()
        self.d_model = d_model

        # Self-Attention
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)

        # FeedForward
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout3 = nn.Dropout(dropout)

        self.use_checkpointing = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional gradient checkpointing.
        x: (Batch, SeqLen, d_model)
        """
        if self.use_checkpointing and self.training:

            # Usually x comes from previous layers so it should be fine.
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-Norm 1 + Self-Attention
        x_norm = self.norm1(x)
        attn_output, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + self.dropout1(attn_output)

        # Pre-Norm 2 + FFN
        x_norm = self.norm2(x)
        ffn_output = self.linear2(self.dropout2(self.activation(self.linear1(x_norm))))
        x = x + self.dropout3(ffn_output)

        return x
