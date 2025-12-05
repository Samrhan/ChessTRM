import pytest
import h5py
import numpy as np
import os
import torch
from chesstrm.data.dataset import ChessDataset

@pytest.fixture
def h5_file(tmp_path):
    # Create a temporary H5 file using pytest's tmp_path fixture
    d = tmp_path / "test_data"
    d.mkdir()
    p = d / "test_dataset.h5"

    with h5py.File(p, 'w') as f:
        # Create dummy x data: 10 samples, 19 planes, 8x8
        x = np.random.rand(10, 19, 8, 8).astype(np.float32)
        f.create_dataset('x', data=x)

        # Create dummy y data: mix of strings and indices (if we supported both, but let's test one by one)
        # We use fixed-length strings (S5)
        y_moves = [b"e2e4", b"a7a8q", b"e1g1"] + [b"e2e4"] * 7
        f.create_dataset('y', data=y_moves)

    return str(p)

def test_loading(h5_file):
    ds = ChessDataset(h5_file, use_swmr=False)
    assert len(ds) == 10

    x, y = ds[0]
    assert x.shape == (19, 8, 8)
    assert isinstance(y, torch.Tensor)

    # Test 2nd sample: a7a8q
    x2, y2 = ds[1]
    # Check bounds (0 to 1967)
    assert y2 >= 0 and y2 < 1968

def test_in_memory(h5_file):
    ds = ChessDataset(h5_file, in_memory=True, use_swmr=False)
    x, y = ds[0]
    assert x.shape == (19, 8, 8)
