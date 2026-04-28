import torch
import chess
from utils.board_encoding import board_to_tensor
from utils.move_masking import get_legal_move_mask

import chess
import math

from evaluate import evaluate_position


def alpha_beta_root(board: chess.Board, depth: int):
    """
    Root function: returns the best move using alpha-beta search.
    """
    best_move = None
    alpha = -math.inf
    beta = math.inf

    maximizing = board.turn == chess.WHITE

    moves = list(board.legal_moves)

    # Move ordering
    # Promotions, then checks, then captures
    moves.sort(key=lambda move: score_move(board, move), reverse=True)

    #print(moves)

    for move in moves:
        board.push(move)
        score = alpha_beta(board, depth - 1, alpha, beta, not maximizing)
        board.pop()

        if maximizing:
            if score > alpha:
                alpha = score
                best_move = move
        else:
            if score < beta:
                beta = score
                best_move = move
        #print(f"move: {move}")
        #print(f"alpha: {alpha}")
        #print(f"beta: {beta}")
        #print(f"score: {score}\n")

    return best_move


def alpha_beta(board: chess.Board, depth: int, alpha: float, beta: float, maximizing: bool):
    """
    Alpha-beta pruning search helper.
    Returns evaluation score.
    """

    # --- 1. Terminal / base case ---
    if board.is_checkmate():
        if board.turn:
            return -100000 - depth
        else:
            return 100000 + depth

    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    if depth == 0:
        return evaluate_position(board)

    moves = list(board.legal_moves)

    # Move ordering
    # Promotions, then checks, then captures
    moves.sort(key=lambda move: score_move(board, move), reverse=True)

    # --- 2. Maximizing player (White) ---
    if maximizing:
        value = -math.inf

        for move in moves:
            board.push(move)
            value = max(value, alpha_beta(board, depth - 1, alpha, beta, False))
            board.pop()

            alpha = max(alpha, value)

            # --- PRUNE ---
            if alpha >= beta:
                break

        return value

    # --- 3. Minimizing player (Black) ---
    else:
        value = math.inf

        for move in moves:
            board.push(move)
            value = min(value, alpha_beta(board, depth - 1, alpha, beta, True))
            board.pop()

            beta = min(beta, value)

            # --- PRUNE ---
            if alpha >= beta:
                break

        return value


def score_move(board: chess.Board, move: chess.Move):

    """
    Helper to score a move for alpha-beta
    Promotions receive highest priority, then checks, then captures
    """

    score = 0

    # --- 1. Promotions (highest priority) ---
    if move.promotion:
        score += 10000

    # --- 2. Checks ---
    if board.gives_check(move):
        score += 5000

    # --- 3. Captures ---
    if board.is_capture(move):
        score += 1000

        # Optional: MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)

        if victim and attacker:
            score += 10 * victim.piece_type - attacker.piece_type

    return score

def tactical_search(board: chess.Board, depth: int = 2):

    """
    Shallow alpha-beta search for opening and middlegame
    """

    best_move = None
    alpha = -float('inf')
    beta = float('inf')

    maximizing = board.turn == chess.WHITE

    moves = list(board.legal_moves)
    moves.sort(key=lambda move: score_move(board, move), reverse=True)

    for move in moves:
        board.push(move)
        score = alpha_beta(board, depth - 1, alpha, beta, not maximizing)
        board.pop()

        if maximizing:
            if score > alpha:
                alpha = score
                best_move = move
        else:
            if score < beta:
                beta = score
                best_move = move

    return best_move, (alpha if maximizing else beta)



def is_blunder(board: chess.Board, move: chess.Move, threshold=200):
    """
    Returns True if the move allows the opponent to gain a large advantage.
    """
    board.push(move)

    # Opponent tries to maximize their gain
    opponent_score = alpha_beta(
        board,
        depth=3,  # opponent gets 2 moves
        alpha=-float('inf'),
        beta=float('inf'),
        maximizing=(board.turn == chess.WHITE)
    )

    board.pop()

    # If it's bad for us, opponent_score will be large (for them)
    # Convert perspective:
    if board.turn == chess.WHITE:
        return opponent_score < -threshold
    else:
        return opponent_score > threshold

"""

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