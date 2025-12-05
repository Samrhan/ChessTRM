import torch
import torch.nn as nn
from typing import List

class DISLoss(nn.Module):
    """
    Deep Improvement Supervision (DIS) Loss.
    Computes weighted sum of CrossEntropyLoss at each recursion step.
    """
    def __init__(self, schedule: str = "linear"):
        super().__init__()
        self.schedule = schedule
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits_list: List[torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits_list: List of tensors of shape (Batch, NumActions)
            target: Tensor of shape (Batch,) containing correct action indices
        Returns:
            Weighted loss scalar.
        """
        T = len(logits_list)
        total_loss = 0.0

        for t, logits in enumerate(logits_list):
            # t is 0-indexed here, so step is t+1
            step = t + 1

            # Calculate weight
            if self.schedule == "linear":
                w_t = step / T
            elif self.schedule == "uniform":
                w_t = 1.0
            else:
                raise ValueError(f"Unknown schedule: {self.schedule}")

            # Calculate CE loss for this step
            step_loss = self.ce(logits, target)

            total_loss += w_t * step_loss

        return total_loss
