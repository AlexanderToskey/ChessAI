import torch
import chess
from utils.board_encoding import board_to_tensor
from utils.move_masking import get_legal_move_mask

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
                logits = model(board_tensor, skill_tensor)

                mask = get_legal_move_mask(board).unsqueeze(0).to(device)
                masked_logits = logits.masked_fill(mask == 0, -1e9)

                probs = torch.softmax(masked_logits, dim=1)

                score = probs.max().item()

        board.pop()

        if score > best_score:
            best_score = score
            best_move = move

    return best_move