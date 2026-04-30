import chess

# Piece values
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
}

CENTER_SQUARES = [chess.D4, chess.E4, chess.D5, chess.E5]

DIRECTIONS = [
    8, -8, 1, -1,   # rook directions
    9, -9, 7, -7    # bishop directions
]


def is_on_board(square):
    return 0 <= square < 64


def same_line(sq1, sq2, direction):
    # Prevent wrap-around (important for files/ranks)
    if direction in [1, -1]:  # horizontal
        return chess.square_rank(sq1) == chess.square_rank(sq2)
    return True


def scan_ray(board, start_sq, direction):
    """Yield squares in a direction until edge"""
    sq = start_sq
    while True:
        next_sq = sq + direction
        if not is_on_board(next_sq):
            break
        if not same_line(sq, next_sq, direction):
            break
        yield next_sq
        sq = next_sq


def get_piece_value(piece):
    if piece is None:
        return 0
    return PIECE_VALUES[piece.piece_type]

def is_hanging(board: chess.Board, square: int):
    piece = board.piece_at(square)
    if piece is None:
        return False

    attackers = board.attackers(not piece.color, square)
    defenders = board.attackers(piece.color, square)

    # Hanging = attacked and not defended
    return len(attackers) > 0 and len(defenders) == 0


def creates_pin(board_after, move, moved_piece):
    if moved_piece.piece_type not in [chess.BISHOP, chess.ROOK, chess.QUEEN]:
        return False

    for direction in DIRECTIONS:
        ray = list(scan_ray(board_after, move.to_square, direction))

        encountered = []

        for sq in ray:
            piece = board_after.piece_at(sq)
            if piece:
                encountered.append((sq, piece))
                if len(encountered) == 2:
                    break

        if len(encountered) == 2:
            first_sq, first_piece = encountered[0]
            second_sq, second_piece = encountered[1]

            # Check pin pattern: enemy piece in front, king behind
            if (first_piece.color != moved_piece.color and
                second_piece.color != moved_piece.color and
                second_piece.piece_type == chess.KING):
                return True

    return False


def creates_skewer(board_after, move, moved_piece):
    if moved_piece.piece_type not in [chess.BISHOP, chess.ROOK, chess.QUEEN]:
        return False

    for direction in DIRECTIONS:
        ray = list(scan_ray(board_after, move.to_square, direction))

        encountered = []

        for sq in ray:
            piece = board_after.piece_at(sq)
            if piece:
                encountered.append((sq, piece))
                if len(encountered) == 2:
                    break

        if len(encountered) == 2:
            first_sq, first_piece = encountered[0]
            second_sq, second_piece = encountered[1]

            if first_piece.color != moved_piece.color and second_piece.color != moved_piece.color:
                if (get_piece_value(first_piece) >
                        get_piece_value(second_piece)):
                    return True

    return False


def analyze_move(board: chess.Board, move: chess.Move):

    """
    Features:
        is_check
        is_checkmate
        is_capture
        material_gain
        creates_fork
        improves_mobility
        controls_center
        develops_piece
        develops_piece
        is_trade
        is_favorable_trade
        captures_hanging_piece
        creates_hanging_piece
        creates_pin
        creates_skewer
    
    """


    features = {}

    # Copy board
    before = board.copy()
    after = board.copy()
    after.push(move)

    moved_piece = before.piece_at(move.from_square)
    captured_piece = before.piece_at(move.to_square)

    # --- Basic Features ---
    features["is_check"] = after.is_check()
    features["is_checkmate"] = after.is_checkmate()
    features["is_capture"] = captured_piece is not None

    # --- Material Gain ---
    gain = 0
    if captured_piece:
        gain += get_piece_value(captured_piece)

    # Check for immediate recapture (1-ply approximation)
    recapture_loss = 0
    for opp_move in after.legal_moves:
        if opp_move.to_square == move.to_square:
            attacker = after.piece_at(opp_move.from_square)
            if attacker:
                recapture_loss = max(recapture_loss, get_piece_value(moved_piece))
    gain -= recapture_loss

    features["material_gain"] = gain

    # --- Fork Detection ---
    features["creates_fork"] = False
    if moved_piece:
        attacked_squares = after.attacks(move.to_square)
        targets = []

        for sq in attacked_squares:
            piece = after.piece_at(sq)
            if piece and piece.color != moved_piece.color:
                targets.append(piece)

        high_value_targets = [
            p for p in targets if get_piece_value(p) >= 3
        ]

        if len(high_value_targets) >= 2:
            features["creates_fork"] = True

    # --- Mobility ---
    before_mobility = len(list(before.legal_moves))
    after_mobility = len(list(after.legal_moves))
    features["improves_mobility"] = after_mobility > before_mobility

    # --- Center Control ---
    center_control_before = sum(
        1 for sq in CENTER_SQUARES if before.is_attacked_by(before.turn, sq)
    )
    center_control_after = sum(
        1 for sq in CENTER_SQUARES if after.is_attacked_by(after.turn, sq)
    )

    features["controls_center"] = center_control_after > center_control_before

    # --- Development ---
    features["develops_piece"] = False
    if moved_piece:
        if moved_piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
            if move.from_square in chess.SquareSet(chess.BB_BACKRANKS):
                features["develops_piece"] = True

    # --- Trade Detection ---
    features["is_trade"] = False
    features["is_favorable_trade"] = False

    if captured_piece:
        for opp_move in after.legal_moves:
            if opp_move.to_square == move.to_square:
                features["is_trade"] = True
                if gain >= 0:
                    features["is_favorable_trade"] = True
                break

    # --- Hanging Piece Capture ---
    features["captures_hanging_piece"] = False
    if captured_piece:
        if is_hanging(before, move.to_square):
            features["captures_hanging_piece"] = True

    # --- Creates Hanging Piece ---
    features["creates_hanging_piece"] = False

    # Look at all opponent pieces after the move
    for square in chess.SQUARES:
        piece = after.piece_at(square)
        if piece and piece.color != moved_piece.color:
            if is_hanging(after, square):
                features["creates_hanging_piece"] = True
                break

    # --- Pin Detection ---
    features["creates_pin"] = False
    if moved_piece:
        features["creates_pin"] = creates_pin(after, move, moved_piece)

    # --- Skewer Detection ---
    features["creates_skewer"] = False
    if moved_piece:
        features["creates_skewer"] = creates_skewer(after, move, moved_piece)

    return features