import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    """
    Projects the 19 input planes to d_model.
    The input is expected to be (Batch, 19, 8, 8).
    It flattens the spatial dimensions to (Batch, 64, d_model).
    """

    def __init__(self, d_model: int, in_channels: int = 19):
        super().__init__()
        self.d_model = d_model
        # We use a 1x1 convolution to project channels 19 -> d_model
        # Then we will flatten 8x8 -> 64
        self.projection = nn.Conv2d(in_channels, d_model, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, 19, 8, 8)
        x = self.projection(x)  # (Batch, d_model, 8, 8)
        x = x.flatten(2)  # (Batch, d_model, 64)
        x = x.permute(0, 2, 1)  # (Batch, 64, d_model) to match Transformer input
        return x


class ActionEmbedding(nn.Module):
    """
    Projects the action index or logits to d_model.
    """

    def __init__(self, num_actions: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(num_actions, d_model)


        # The spec mentions "Learnable embedding or projected vector".
        # Let's add a linear layer for projecting a full distribution if needed,

        # or a global token.

        # However, SPEC 3.2 says: "Projects y logits/indices to d_model"
        # If we feed back logits (size 1968), we need a Linear layer.
        self.linear = nn.Linear(num_actions, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x can be indices (Batch,) or distributions (Batch, NumActions)
        if x.dtype == torch.long or x.dtype == torch.int:
            return self.embedding(x)
        else:
            return self.linear(x)


class ReadoutHead(nn.Module):
    """
    Projects d_model back to action space (1968 logits).
    Usually takes the global representation or a specific token output.
    """

    def __init__(self, d_model: int, num_actions: int):
        super().__init__()
        self.projection = nn.Linear(d_model, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, d_model) or (Batch, Seq, d_model)
        # If sequence, we typically might pool or take the last token,
        # but the caller should probably handle extraction.
        # Here we just project.
        return self.projection(x)
