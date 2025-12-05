from typing import List, Optional

import torch
import torch.nn as nn

from src.chesstrm.model.layers import ActionEmbedding, InputEmbedding, ReadoutHead
from src.chesstrm.model.transformer import RecursiveBlock


class ChessTRM(nn.Module):
    """
    ChessTRM: Tiny Recursive Model for Chess.
    Uses a recursive Transformer block to refine state (z) and action (y) over time.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
        max_recursion: int = 16,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_recursion = max_recursion

        # 1. Components
        self.input_projection = InputEmbedding(d_model=d_model, in_channels=19)
        self.pos_encoding = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)

        # Action space: 1968 moves
        # y is the action stream.
        # y_0 is initialized as a learnable parameter.
        self.num_actions = 1968
        self.y_init = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # We need to project y logits back to d_model for the next step?
        # SPEC 3.2: "Projects y logits/indices to d_model (or broadcast global token)."

        # And "y might be a global token concatenated or broadcasted."

        self.action_embedding = ActionEmbedding(
            num_actions=self.num_actions, d_model=d_model
        )

        # Readout Head: d_model -> 1968
        self.readout_head = ReadoutHead(d_model=d_model, num_actions=self.num_actions)

        # 2. Recursive Core
        # The spec says "n_layers (per block): 2".
        # "Le noyau neuronal minimaliste (souvent 2 couches seulement)"

        # SPEC 3.2 "TransformerBlock" seems to be a single block.

        # Standard Transformer Encoder Layer has SelfAttn + FFN.
        # If n_layers=2, we have Block -> Block.
        self.blocks = nn.ModuleList(
            [RecursiveBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )

        # 3. Stream Combination Logic (can be a simple addition if broadcasted)
        # Input_t = Embed(x) + Proj(y_{t-1}) + z_{t-1}

    def forward(
        self, x: torch.Tensor, n_steps: Optional[int] = None
    ) -> List[torch.Tensor]:
        """
        Forward pass implementing the recursive loop.
        x: (Batch, 19, 8, 8)
        n_steps: Number of recursion steps (default: self.max_recursion)
        Returns: List of logits for each step [logits_1, ..., logits_T]
        """
        batch_size = x.size(0)
        steps = n_steps if n_steps is not None else self.max_recursion

        # 1. Initialize State
        # x_emb: (Batch, 64, d_model)
        x_emb = self.input_projection(x) + self.pos_encoding

        # z: "scratchpad" mental. Initialized with noise.
        # Shape: (Batch, 64, d_model)
        z = torch.randn(batch_size, 64, self.d_model, device=x.device) * 0.02

        # y: Intention. Initialized with learnable parameter.
        # Shape: (Batch, 1, d_model) -> Global token
        y_emb = self.y_init.expand(batch_size, -1, -1)

        all_logits = []

        # 2. Recursive Loop
        for t in range(steps):
            # Combine Streams
            # We broadcast y_emb to (Batch, 64, d_model) to add to x and z
            # Or we concat? SPEC 3.2: "Input_t = Embed(x) + Proj(y_{t-1}) + z_{t-1}"
            # This implies addition (same shape).
            combined_input = x_emb + z + y_emb

            # Pass through Recursive Core (Shared Weights)
            curr = combined_input
            for block in self.blocks:
                curr = block(curr)

            # Update z
            # Residual connection: z = z + gating(z_new) or just z_new
            # SPEC 3.3: "z = z + self.gating(z_new) # Gating optionnel"

            # Let's use simple residual for now: z_next = z + output_of_blocks
            z_new = curr
            z = z + z_new

            # Update y
            # We need to extract the "intention" from the updated state z.
            # If y is a global token, how do we get it from z (64 tokens)?
            # Maybe we pool z? Or we should have included y as a token in the sequence?

            # If we used broadcasting (addition), y is mixed in every token.
            # We need to extract a new y_emb.
            # Simple approach: Global Average Pooling of z to get new y representation?
            # Or use the ReadoutHead on z, get logits, then project back?

            # SPEC 3.3: "y_emb_new = self.y_update_layer(z_new)"
            # "logits = self.policy_head(y_emb)"

            # Let's implement a simple pooling to update y_emb
            # z is (Batch, 64, d_model).
            # y_emb should be (Batch, 1, d_model) (or we keep it broadcasted).

            # Let's try Mean Pooling for y update
            y_emb_update = z.mean(dim=1, keepdim=True)  # (Batch, 1, d_model)
            # Or maybe we should have a specific layer for this?
            # SPEC didn't specify `y_update_layer`.


            y_emb = y_emb_update  # Update y_emb for next step (and readout)

            # Readout
            # We project y_emb to logits.
            # y_emb is (Batch, 1, d_model). Output (Batch, 1, 1968).
            logits = self.readout_head(y_emb).squeeze(1)  # (Batch, 1968)
            all_logits.append(logits)

            # For next step: "Proj(y_{t-1})".
            # In our case y_emb is already d_model.
            # Do we re-project logits?



            # Staying in latent space is smoother for gradients.
            # The logits are just "decoded" for supervision.

            # If we follow "y_emb = y_emb_new", we just use the pooled z.
            # However, to be true to "ActionEmbedding", maybe we should use the logits?
            # But "ActionEmbedding" was for indices or logits.




        return all_logits
