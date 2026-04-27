import torch
import chess
from utils.board_encoding import board_to_tensor
from utils.move_masking import get_legal_move_mask

def select_move_1ply(model, board, skill, device):
    model.eval()

    best_move = None
    best_score = -float("inf")

    print(board.legal_moves)

    for move in board.legal_moves:

        # -----------------------------
        # STEP 1: capture info BEFORE push
        # -----------------------------
        print(move)
        captured_piece = board.piece_at(move.to_square)

        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9
        }

        capture_score = 0.0
        #print(board.is_capture(move) and captured_piece)
        if board.is_capture(move) and captured_piece:
            capture_score = piece_values.get(
                captured_piece.piece_type, 0
            ) / 9.0

        # -----------------------------
        # STEP 2: make move
        # -----------------------------
        board.push(move)

        # Immediate win
        if board.is_checkmate():
            board.pop()
            return move

        # Draw
        if board.is_stalemate() or board.is_insufficient_material():
            score = 1

        else:
            # -----------------------------
            # STEP 3: model evaluation
            # -----------------------------
            board_tensor = torch.tensor(
                board_to_tensor(board),
                dtype=torch.float32
            ).unsqueeze(0).to(device)

            skill_tensor = torch.tensor([skill], dtype=torch.long).to(device)

            with torch.no_grad():
                logits, value = model(board_tensor, skill_tensor)

                mask = get_legal_move_mask(board).unsqueeze(0).to(device)
                masked_logits = logits.masked_fill(mask == 0, -1e9)
                probs = torch.softmax(masked_logits, dim=1)

                policy_score = probs.max().item()
                value_score = value.item()

            # -----------------------------
            # STEP 4: final scoring
            # -----------------------------
            print(move)
            print(value_score)
            print(policy_score)
            print(capture_score)
            score = (
                0.2 * value_score +
                0.5 * policy_score +
                0.4 * capture_score
            )

        board.pop()

        if score > best_score:
            best_score = score
            best_move = move

    return best_move

def get_material_value(board):
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9
    }

    value = 0
    for piece_type, v in piece_values.items():
        value += len(board.pieces(piece_type, chess.WHITE)) * v
        value -= len(board.pieces(piece_type, chess.BLACK)) * v

    return value


"""
def select_move_1ply(model, board, skill, device):
    model.eval()

    best_move = None
    best_score = -float("inf")

    for move in board.legal_moves:
        board.push(move)

        # CHECK FOR CHECKMATE
        if board.is_checkmate():
            board.pop()
            return move  # immediate win → always choose

        # CHECK FOR DRAW
        if board.is_stalemate() or board.is_insufficient_material():
            score = 0.0
        else:
            # Normal evaluation
            board_tensor = torch.tensor(
                board_to_tensor(board),
                dtype=torch.float32
            ).unsqueeze(0).to(device)

            skill_tensor = torch.tensor([skill], dtype=torch.long).to(device)

            with torch.no_grad():
                logits, value = model(board_tensor, skill)

                mask = get_legal_move_mask(board).unsqueeze(0).to(device)
                masked_logits = logits.masked_fill(mask == 0, -1e9)

                probs = torch.softmax(masked_logits, dim=1)

                score = probs.max().item()

        board.pop()

        if score > best_score:
            best_score = score
            best_move = move

    return best_move

def evaluate_position(model, board, skill, device, root_turn):
    if board.is_checkmate():
        return -1.0
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0

    board_tensor = torch.tensor(
        board_to_tensor(board),
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    skill_tensor = torch.tensor([skill], dtype=torch.long).to(device)

    with torch.no_grad():
        value = model(board_tensor, skill_tensor)

    value = value.item()

    # If it's opponent's turn, flip perspective
    if board.turn != root_turn:
        value = -value

    return value


def minimax(model, board, depth, alpha, beta, maximizing, skill, device, root_turn):
    # Base case
    if depth == 0 or board.is_game_over():
        return evaluate_position(model, board, skill, device, root_turn)

    if maximizing:
        max_eval = -float("inf")

        for move in board.legal_moves:
            board.push(move)
            eval = minimax(model, board, depth - 1, alpha, beta, False, skill, device, root_turn)
            board.pop()

            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)

            if beta <= alpha:
                break  # alpha-beta pruning

        return max_eval

    else:
        min_eval = float("inf")

        for move in board.legal_moves:
            board.push(move)
            eval = minimax(model, board, depth - 1, alpha, beta, True, skill, device, root_turn)
            board.pop()

            min_eval = min(min_eval, eval)
            beta = min(beta, eval)

            if beta <= alpha:
                break

        return min_eval
    
def select_move_2ply(model, board, skill, device):
    best_move = None
    best_score = -float("inf")

    for move in board.legal_moves:
        board.push(move)

        # Immediate checkmate
        if board.is_checkmate():
            board.pop()
            return move

        score = minimax(
            model,
            board,
            depth=1,              # 2-ply total
            alpha=-float("inf"),
            beta=float("inf"),
            maximizing=False,     # opponent turn
            skill=skill,
            device=device,
            root_turn=board.turn,
        )

        board.pop()

        if score > best_score:
            best_score = score
            best_move = move

    return best_move

"""