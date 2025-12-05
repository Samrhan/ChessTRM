import torch
import pytest
from src.chesstrm.training.loss import DISLoss

def test_dis_loss_linear():
    """Test DISLoss with linear schedule."""
    criterion = DISLoss(schedule="linear")

    # Dummy data
    batch_size = 2
    num_actions = 4
    T = 3

    # Create dummy logits: List of 3 tensors
    logits_list = [
        torch.randn(batch_size, num_actions),
        torch.randn(batch_size, num_actions),
        torch.randn(batch_size, num_actions)
    ]

    # Targets
    targets = torch.randint(0, num_actions, (batch_size,))

    # Calculate expected loss manually
    loss_vals = []
    ce = torch.nn.CrossEntropyLoss()
    for t in range(T):
        w_t = (t + 1) / T
        l = ce(logits_list[t], targets)
        loss_vals.append(w_t * l)

    expected_loss = sum(loss_vals)

    # Calculate actual loss
    actual_loss = criterion(logits_list, targets)

    assert torch.isclose(actual_loss, expected_loss), f"Expected {expected_loss}, got {actual_loss}"

def test_dis_loss_uniform():
    """Test DISLoss with uniform schedule."""
    criterion = DISLoss(schedule="uniform")

    batch_size = 2
    num_actions = 4
    T = 3

    logits_list = [torch.randn(batch_size, num_actions) for _ in range(T)]
    targets = torch.randint(0, num_actions, (batch_size,))

    loss_vals = []
    ce = torch.nn.CrossEntropyLoss()
    for t in range(T):
        l = ce(logits_list[t], targets)
        loss_vals.append(1.0 * l)

    expected_loss = sum(loss_vals)
    actual_loss = criterion(logits_list, targets)

    assert torch.isclose(actual_loss, expected_loss)
