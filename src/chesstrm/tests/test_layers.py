import torch
import pytest
from src.chesstrm.model.layers import InputEmbedding, ActionEmbedding, ReadoutHead

def test_input_embedding_shape():
    batch_size = 4
    d_model = 256
    model = InputEmbedding(d_model=d_model)
    x = torch.randn(batch_size, 19, 8, 8)
    out = model(x)

    # Expected output: (Batch, 64, d_model)
    assert out.shape == (batch_size, 64, d_model)

def test_action_embedding_indices():
    batch_size = 4
    num_actions = 1968
    d_model = 256
    model = ActionEmbedding(num_actions, d_model)

    # Test with indices
    indices = torch.randint(0, num_actions, (batch_size,))
    out = model(indices)
    assert out.shape == (batch_size, d_model)

def test_action_embedding_logits():
    batch_size = 4
    num_actions = 1968
    d_model = 256
    model = ActionEmbedding(num_actions, d_model)

    # Test with logits/distribution
    logits = torch.randn(batch_size, num_actions)
    out = model(logits)
    assert out.shape == (batch_size, d_model)

def test_readout_head():
    batch_size = 4
    d_model = 256
    num_actions = 1968
    model = ReadoutHead(d_model, num_actions)

    x = torch.randn(batch_size, d_model)
    out = model(x)
    assert out.shape == (batch_size, num_actions)
