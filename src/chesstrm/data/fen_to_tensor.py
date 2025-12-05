import chess
import torch

def fen_to_tensor(fen: str) -> torch.Tensor:
    """
    Converts a FEN string to the 19-plane tensor representation used by ChessTRM.

    Planes:
    0-5: Active player pieces (P, N, B, R, Q, K)
    6-11: Opponent pieces (P, N, B, R, Q, K)
    12-13: Repetitions (1x, 2x) - Set to 0 for single FEN
    14-17: Castling rights (Active O-O, Active O-O-O, Opponent O-O, Opponent O-O-O)
    18: Color to move (0 for White, 1 for Black)

    The board is oriented from the perspective of the active player.
    If Black is to move, the board is rotated 180 degrees so that Black pieces start at ranks 0-1.
    """
    board = chess.Board(fen)

    # 19 planes, 8x8
    tensor = torch.zeros((19, 8, 8), dtype=torch.float32)

    turn = board.turn # chess.WHITE (True) or chess.BLACK (False)

    # Define piece types
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

    # Helper to map square index to (row, col)
    # python-chess: A1=0, B1=1 ... H1=7, A2=8 ... H8=63
    # tensor: (8, 8). We usually map rank -> row, file -> col.
    # row 0 = rank 1, row 7 = rank 8.

    def square_to_coord(square):
        rank, file = divmod(square, 8)
        return rank, file

    # 1. Active Player Pieces (0-5)
    for i, pt in enumerate(piece_types):
        # Get squares of pieces of type pt for the active color
        squares = board.pieces(pt, turn)
        for sq in squares:
            if turn == chess.BLACK:
                # Rotate 180: index becomes 63 - index
                sq = 63 - sq

            rank, file = square_to_coord(sq)
            tensor[i, rank, file] = 1.0

    # 2. Opponent Pieces (6-11)
    opponent = not turn
    for i, pt in enumerate(piece_types):
        squares = board.pieces(pt, opponent)
        for sq in squares:
            if turn == chess.BLACK:
                sq = 63 - sq

            rank, file = square_to_coord(sq)
            tensor[6 + i, rank, file] = 1.0

    # 3. Repetitions (12-13)
    # Not available in FEN usually (unless extended). Assuming 0.

    # 4. Castling Rights (14-17)
    # 14: Active O-O
    # 15: Active O-O-O
    # 16: Opponent O-O
    # 17: Opponent O-O-O

    if board.has_kingside_castling_rights(turn):
        tensor[14, :, :] = 1.0
    if board.has_queenside_castling_rights(turn):
        tensor[15, :, :] = 1.0
    if board.has_kingside_castling_rights(opponent):
        tensor[16, :, :] = 1.0
    if board.has_queenside_castling_rights(opponent):
        tensor[17, :, :] = 1.0

    # 5. Color to move (18)
    # White=0, Black=1? Or specific encoding?
    # Assuming White=0, Black=1 as per typical conventions if included.
    # Since we rotate the board, the model "feels" like it is always White (playing 'up').
    # But this plane might indicate "True Color".
    if turn == chess.BLACK:
        tensor[18, :, :] = 1.0

    return tensor
