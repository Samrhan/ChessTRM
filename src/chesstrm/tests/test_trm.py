import torch
import pytest
from src.chesstrm.model.trm import ChessTRM

def test_chesstrm_forward_shape():
    batch_size = 2
    d_model = 64 # Small for test
    n_heads = 4
    n_layers = 2
    max_recursion = 5

    model = ChessTRM(d_model=d_model, n_heads=n_heads, n_layers=n_layers, max_recursion=max_recursion)

    # Input x: (Batch, 19, 8, 8)
    x = torch.randn(batch_size, 19, 8, 8)

    # Run forward
    outputs = model(x)

    # Check output is a list of length max_recursion
    assert isinstance(outputs, list)
    assert len(outputs) == max_recursion

    # Check shape of each output: (Batch, 1968)
    for logits in outputs:
        assert logits.shape == (batch_size, 1968)

def test_chesstrm_variable_steps():
    batch_size = 2
    model = ChessTRM(d_model=32, n_heads=2)
    x = torch.randn(batch_size, 19, 8, 8)

    steps = 3
    outputs = model(x, n_steps=steps)
    assert len(outputs) == steps

def test_chesstrm_gradient_flow():
    # Verify that gradients flow back from the last step to input/weights
    batch_size = 2
    model = ChessTRM(d_model=32, n_heads=2, max_recursion=3)
    x = torch.randn(batch_size, 19, 8, 8, requires_grad=True)

    outputs = model(x)
    loss = sum([o.sum() for o in outputs])

    loss.backward()

    # Check input gradient
    assert x.grad is not None
    # Check weight gradient (e.g. from first block)
    assert model.blocks[0].linear1.weight.grad is not None
