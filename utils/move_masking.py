
# Imports
import torch
import chess
from utils.move_encoding import uci_to_class

def get_legal_move_mask(board: chess.Board):
    """
    Returns a mask of shape (4096,)
    1 = legal move
    0 = illegal move
    """
    mask = torch.zeros(4096, dtype=torch.float32)

    for move in board.legal_moves:
        move_class = uci_to_class(move.uci())
        mask[move_class] = 1.0

    return mask