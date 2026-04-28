import chess

# Piece values
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 300,
    chess.BISHOP: 300,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}

CENTER_SQUARES = [chess.D4, chess.E4, chess.D5, chess.E5]


def evaluate_position(board: chess.Board):
    # --- 1. Terminal conditions ---
    if board.is_checkmate():
        if board.turn:  # White to move and checkmated
            return -float('inf')
        else:  # Black to move and checkmated
            return float('inf')

    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    score = 0

    # --- 2. Material ---
    for piece_type in PIECE_VALUES:
        score += len(board.pieces(piece_type, chess.WHITE)) * PIECE_VALUES[piece_type]
        score -= len(board.pieces(piece_type, chess.BLACK)) * PIECE_VALUES[piece_type]

    # --- 3. Pawn advancement ---
    for square in board.pieces(chess.PAWN, chess.WHITE):
        rank = chess.square_rank(square)
        score += rank * 10  # more advanced = better

    for square in board.pieces(chess.PAWN, chess.BLACK):
        rank = 7 - chess.square_rank(square)
        score -= rank * 10

    # --- 4. Passed pawns ---
    for color in [chess.WHITE, chess.BLACK]:
        pawns = board.pieces(chess.PAWN, color)
        enemy_pawns = board.pieces(chess.PAWN, not color)

        for pawn_sq in pawns:
            file = chess.square_file(pawn_sq)
            rank = chess.square_rank(pawn_sq)

            is_passed = True
            for ep in enemy_pawns:
                ep_file = chess.square_file(ep)
                ep_rank = chess.square_rank(ep)

                if abs(ep_file - file) <= 1:
                    if color == chess.WHITE and ep_rank > rank:
                        is_passed = False
                    if color == chess.BLACK and ep_rank < rank:
                        is_passed = False

            if is_passed:
                bonus = 50
                if color == chess.WHITE:
                    score += bonus
                else:
                    score -= bonus

    # --- 5. King activity (centralization) ---
    white_king_sq = board.king(chess.WHITE)
    black_king_sq = board.king(chess.BLACK)

    def king_centrality(square):
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        return 4 - max(abs(3.5 - file), abs(3.5 - rank))

    score += king_centrality(white_king_sq) * 20
    score -= king_centrality(black_king_sq) * 20

    # --- 6. King proximity to pawns ---
    for pawn_sq in board.pieces(chess.PAWN, chess.WHITE):
        dist = chess.square_distance(black_king_sq, pawn_sq)
        score += max(0, 6 - dist) * 5  # closer enemy king = bad for white

    for pawn_sq in board.pieces(chess.PAWN, chess.BLACK):
        dist = chess.square_distance(white_king_sq, pawn_sq)
        score -= max(0, 6 - dist) * 5

    # --- 7. Mobility (lightweight) ---
    mobility = len(list(board.legal_moves))
    if board.turn == chess.WHITE:
        score += mobility
    else:
        score -= mobility

    return score
