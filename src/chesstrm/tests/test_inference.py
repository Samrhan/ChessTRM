import torch
import chess
import pytest
from chesstrm.inference import load_model, predict
from chesstrm.model.trm import ChessTRM

def test_inference_pipeline():
    # 1. Setup
    fen = chess.STARTING_FEN
    # Initialize a dummy model (untrained)
    model = load_model(None)

    # 2. Run predict
    top_moves = predict(fen, model, top_k=3, n_steps=2)

    # 3. Verify Output
    assert len(top_moves) == 3
    for move_str, prob in top_moves:
        assert isinstance(move_str, str)
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0
        # Check if move_str is a valid UCI move or valid string at least
        # Since model is random, it might output illegal moves, but they should be valid UCI strings from mapping
        assert len(move_str) >= 4

def test_inference_with_mock_model(monkeypatch):
    # Mock the model output to control test
    class MockModel(torch.nn.Module):
        def __call__(self, x, n_steps=None):
            # Return list of logits. Last one matters.
            # Shape (Batch, 1968)
            batch_size = x.shape[0]
            logits = torch.zeros(batch_size, 1968)
            # Set index 0 (usually a valid move) to high value
            logits[0, 0] = 10.0
            logits[0, 1] = 5.0
            return [logits] # Just one step for mock

    fen = chess.STARTING_FEN
    model = MockModel()

    # We need index_to_move to return something known for index 0 and 1
    # We rely on real mapping.py. Index 0 is typically a move like 'a1a2' (depends on sorting).
    # Let's check what index 0 maps to.
    from chesstrm.data.mapping import index_to_move
    move_0 = index_to_move(0)
    move_1 = index_to_move(1)

    top_moves = predict(fen, model, top_k=2)

    assert top_moves[0][0] == move_0
    assert top_moves[1][0] == move_1
    assert top_moves[0][1] > top_moves[1][1]
