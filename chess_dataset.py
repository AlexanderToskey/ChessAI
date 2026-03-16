import torch
from torch.utils.data import Dataset
import json
import chess
from pathlib import Path
from utils.board_encoding import board_to_tensor
from utils.move_encoding import uci_to_class
from utils.elo_processing import elo_to_bucket

class ChessDataset(Dataset):
    """
    PyTorch Dataset for chess positions.
    Expects a JSONL file where each line is:
    {
        "fen": "...",
        "elo": 1650,
        "move": "e2e4"
    }
    """

    def __init__(self, jsonl_file: str):
        self.jsonl_file = Path(jsonl_file)
        self.data = []

        # Load JSONL into memory (can switch to lazy loading if dataset is huge)
        with open(self.jsonl_file, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        fen = sample['fen']
        move_uci = sample['move']
        elo = sample['elo']

        # Convert FEN to (18,8,8) tensor
        board = chess.Board(fen)
        board_tensor = board_to_tensor(board)             # numpy array
        board_tensor = torch.tensor(board_tensor, dtype=torch.float32)

        # Convert move to class index
        move_class = uci_to_class(move_uci)
        move_class = torch.tensor(move_class, dtype=torch.long)

        # Convert ELO to skill bucket
        elo_bucket = elo_to_bucket(elo)
        elo_bucket = torch.tensor(elo_bucket, dtype=torch.long)

        return {
            'board': board_tensor,
            'move_class': move_class,
            'elo_bucket': elo_bucket
        }