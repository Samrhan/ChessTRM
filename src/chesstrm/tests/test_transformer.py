import torch
import pytest
from src.chesstrm.model.transformer import RecursiveBlock

def test_recursive_block_shape():
    batch_size = 4
    seq_len = 64
    d_model = 256
    n_heads = 8

    model = RecursiveBlock(d_model=d_model, n_heads=n_heads)
    x = torch.randn(batch_size, seq_len, d_model)

    out = model(x)
    assert out.shape == (batch_size, seq_len, d_model)

def test_recursive_block_checkpointing():
    batch_size = 4
    seq_len = 64
    d_model = 256
    n_heads = 8

    model = RecursiveBlock(d_model=d_model, n_heads=n_heads)
    model.use_checkpointing = True
    model.train() # Checkpointing usually active only in training

    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    out = model(x)
    assert out.shape == (batch_size, seq_len, d_model)

    # Check if backward pass works
    loss = out.sum()
    loss.backward()
    assert x.grad is not None

def test_recursive_block_parameter_update():
    # Simple check to ensure parameters are updated
    batch_size = 2
    seq_len = 10
    d_model = 32
    n_heads = 4

    model = RecursiveBlock(d_model=d_model, n_heads=n_heads)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    x = torch.randn(batch_size, seq_len, d_model)

    # Initial weights
    initial_weight = model.linear1.weight.clone()

    out = model(x)
    loss = out.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Weights should change
    assert not torch.equal(model.linear1.weight, initial_weight)
