import torch
import chess
import pytest
from chesstrm.data.fen_to_tensor import fen_to_tensor

def test_fen_to_tensor_start_pos():
    fen = chess.STARTING_FEN
    tensor = fen_to_tensor(fen)

    assert tensor.shape == (19, 8, 8)

    # White to move -> turn=True (White)
    # Active pieces (White)
    # Plane 0: White Pawns. Rank 1 (index 1).
    # tensor[0, 1, :] should be all 1s.
    assert torch.all(tensor[0, 1, :] == 1)

    # Plane 3: White Rooks. A1(0,0), H1(0,7).
    assert tensor[3, 0, 0] == 1
    assert tensor[3, 0, 7] == 1

    # Opponent pieces (Black)
    # Plane 6: Black Pawns. Rank 6 (index 6).
    # tensor[6, 6, :] should be all 1s.
    assert torch.all(tensor[6, 6, :] == 1)

    # Plane 9: Black Rooks. A8(7,0), H8(7,7).
    assert tensor[9, 7, 0] == 1
    assert tensor[9, 7, 7] == 1

    # Castling Rights
    # All true
    assert torch.all(tensor[14, :, :] == 1) # White K
    assert torch.all(tensor[15, :, :] == 1) # White Q
    assert torch.all(tensor[16, :, :] == 1) # Black k
    assert torch.all(tensor[17, :, :] == 1) # Black q

    # Color to move: White -> 0
    assert torch.all(tensor[18, :, :] == 0)

def test_fen_to_tensor_black_move():
    # Force black to move. Simple position.
    # White King on E1, Black King on E8.
    # Black pawn on E7. White pawn on E2.
    # Turn: Black.
    board = chess.Board(None)
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    board.set_piece_at(chess.E2, chess.Piece(chess.PAWN, chess.WHITE))
    board.set_piece_at(chess.E7, chess.Piece(chess.PAWN, chess.BLACK))
    board.turn = chess.BLACK

    fen = board.fen()
    tensor = fen_to_tensor(fen)

    # Black is Active.
    # Board should be rotated 180.

    # Black King at E8 (Rank 7, File 4). Rotated -> Rank 0, File 3 (D1? No).
    # E8 is index 60. 63 - 60 = 3.
    # Index 3 is D1. (Rank 0, File 3).
    # Wait.
    # 0=A1, 1=B1, 2=C1, 3=D1, 4=E1.
    # E1 is 4.
    # E8 is 60.
    # 63 - 60 = 3.
    # 63 - 4 = 59 (D8).
    # If I rotate the board 180 degrees:
    # E8 maps to E1? No.
    # E8 (top, center-right from white) -> E1 (bottom, center-left from white).
    # Let's check coordinates.
    # E file is file 4.
    # E8: rank 7, file 4.
    # Rotated: rank = 7-rank, file = 7-file.
    # 7-7 = 0. 7-4 = 3.
    # Rank 0, File 3 -> D1.

    # Is E8 rotated D1?
    # Center is between rank 3/4 and file 3.5.
    # E(4) mirrors to D(3).
    # Yes.

    # So Active King (Black) should be at (0, 3).
    # Plane 5: Active King.
    assert tensor[5, 0, 3] == 1

    # Active Pawn (Black) at E7 (Rank 7, File 4).
    # Rotated: Rank 0, File 3. No, wait.
    # E7: rank 6, file 4.
    # Rotated: 7-6 = 1. 7-4 = 3.
    # Rank 1, File 3 -> D2.
    # Plane 0: Active Pawn.
    assert tensor[0, 1, 3] == 1

    # Opponent King (White) at E1 (Rank 0, File 4).
    # Rotated: 7-0 = 7. 7-4 = 3.
    # Rank 7, File 3 -> D8.
    # Plane 11: Opponent King.
    assert tensor[11, 7, 3] == 1

    # Opponent Pawn (White) at E2 (Rank 1, File 4).
    # Rotated: 7-1 = 6. 7-4 = 3.
    # Rank 6, File 3 -> D7.
    # Plane 6: Opponent Pawn.
    assert tensor[6, 6, 3] == 1

    # Color to move: Black -> 1
    assert torch.all(tensor[18, :, :] == 1)

def test_castling_rights_black():
    # Fen with Black to move, only Black can castle short.
    # r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b Kkq - 0 1
    # Actually let's construct it.
    board = chess.Board()
    board.turn = chess.BLACK
    # Remove white castling rights
    board.castling_rights = chess.BB_A8 | chess.BB_H8 # Black queenside and kingside

    # Only black castling rights.
    # Since Black is active:
    # Active O-O (14): True
    # Active O-O-O (15): True
    # Opponent O-O (16): False
    # Opponent O-O-O (17): False

    tensor = fen_to_tensor(board.fen())

    assert torch.all(tensor[14, :, :] == 1)
    assert torch.all(tensor[15, :, :] == 1)
    assert torch.all(tensor[16, :, :] == 0)
    assert torch.all(tensor[17, :, :] == 0)
