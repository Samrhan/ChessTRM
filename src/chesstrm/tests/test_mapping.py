import pytest
from chesstrm.data.mapping import ALL_MOVES, move_to_index, index_to_move, get_num_actions

def test_total_moves():
    # We expect exactly 1968 moves based on our calculation
    # 1924 (Queen/Knight moves) + 44 (Promotion-specific strings)
    assert len(ALL_MOVES) == 1968
    assert get_num_actions() == 1968

def test_standard_moves():
    # e2e4 is a standard opening move
    assert "e2e4" in ALL_MOVES
    idx = move_to_index("e2e4")
    assert idx is not None
    assert index_to_move(idx) == "e2e4"

def test_promotion_moves():
    # White promotion
    # a7a8 is valid for Rook/King
    assert "a7a8" in ALL_MOVES
    # a7a8q is valid for Pawn
    assert "a7a8q" in ALL_MOVES
    # a7a8n is valid for Pawn
    assert "a7a8n" in ALL_MOVES

    idx_q = move_to_index("a7a8q")
    idx_base = move_to_index("a7a8")
    assert idx_q != idx_base

def test_castling():
    # White Kingside: e1g1
    assert "e1g1" in ALL_MOVES
    # Black Queenside: e8c8
    assert "e8c8" in ALL_MOVES

def test_en_passant_geometry():
    # e5d6 (White capturing on d6 from e5)
    # Geometrically this is a diagonal move, so it should be in the set
    assert "e5d6" in ALL_MOVES
