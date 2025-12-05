import chess

def _generate_uci_moves():
    moves = []

    # Iterate over all possible from_squares and to_squares
    for from_sq in range(64):
        for to_sq in range(64):
            if from_sq == to_sq:
                continue

            # Calculate geometry
            rank_from, file_from = divmod(from_sq, 8)
            rank_to, file_to = divmod(to_sq, 8)

            d_rank = abs(rank_from - rank_to)
            d_file = abs(file_from - file_to)

            is_diagonal = (d_rank == d_file)
            is_straight = (d_rank == 0 or d_file == 0)
            is_knight = (d_rank == 1 and d_file == 2) or (d_rank == 2 and d_file == 1)

            if is_diagonal or is_straight or is_knight:
                # Add the standard "from-to" move
                # This covers all non-promotion moves (e.g., e2e4, a1h8)
                # It also covers the "from-to" part of a promotion if we considered it generic,
                # but in UCI, promotions must have the suffix.
                # However, 'a7a8' IS a valid UCI string for a Rook moving, so we keep it.
                moves.append(chess.Move(from_sq, to_sq).uci())

                # Add promotion moves if applicable (Pawn moving to last rank)
                # White promotion: rank 6 -> 7
                if rank_from == 6 and rank_to == 7:
                    # Pawn captures are diagonal (d_file=1), pushes are straight (d_file=0)
                    if d_file <= 1:
                        for p in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                            moves.append(chess.Move(from_sq, to_sq, promotion=p).uci())

                # Black promotion: rank 1 -> 0
                if rank_from == 1 and rank_to == 0:
                    if d_file <= 1:
                        for p in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                            moves.append(chess.Move(from_sq, to_sq, promotion=p).uci())

    # Sort to ensure deterministic order
    return sorted(list(set(moves)))

# Generate once at module level
ALL_MOVES = _generate_uci_moves()
MOVE_TO_INDEX = {m: i for i, m in enumerate(ALL_MOVES)}
INDEX_TO_MOVE = {i: m for i, m in enumerate(ALL_MOVES)}

def get_num_actions():
    return len(ALL_MOVES)

def move_to_index(move_str: str) -> int:
    """Converts a UCI move string to its index."""
    return MOVE_TO_INDEX.get(move_str)

def index_to_move(index: int) -> str:
    """Converts an index back to a UCI move string."""
    return INDEX_TO_MOVE.get(index)
