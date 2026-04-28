import chess

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


def is_endgame(board: chess.Board):
    # Count non-king pieces
    piece_count = 0
    total_material = 0

    # Count pieces and material on the board
    for piece_type in PIECE_VALUES:
        white_pieces = len(board.pieces(piece_type, chess.WHITE))
        black_pieces = len(board.pieces(piece_type, chess.BLACK))

        piece_count += white_pieces + black_pieces
        total_material += (white_pieces + black_pieces) * PIECE_VALUES[piece_type]

    # Check for queens
    has_queens = (
        len(board.pieces(chess.QUEEN, chess.WHITE)) > 0 or
        len(board.pieces(chess.QUEEN, chess.BLACK)) > 0
    )

    # Heuristics
    few_pieces = piece_count <= 6
    low_material = total_material <= 14
    #no_queens = not has_queens

    #return few_pieces or low_material or no_queens
    #return few_pieces or low_material
    return piece_count, total_material