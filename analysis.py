from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import json
import os
import numpy as np

def load_data(path):
    moves = []
    fens = []

    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line)
            moves.append(data["move"])
            fens.append(data["fen"])  # using FEN
    
    return moves, fens

def plot_move_distribution(moves):
    move_counts = Counter(moves)
    top_moves = move_counts.most_common(20)
    labels, counts = zip(*top_moves)

    plt.figure()
    plt.bar(labels, counts)
    plt.xticks(rotation=90)
    plt.title("Top 20 Most Frequent Moves")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "figures/move_distribution.png")
    plt.close()

def count_pieces_from_fen(fen):
    board_part = fen.split()[0]
    count = 0
    for char in board_part:
        if char.isalpha():
            count += 1
    return count

def plot_phase_distribution_from_fen(fens):
    piece_counts = [count_pieces_from_fen(fen) for fen in fens]

    plt.figure()
    plt.hist(piece_counts, bins=30)
    plt.title("Game Phase Distribution (by Piece Count)")
    plt.xlabel("Number of Pieces on Board")
    plt.ylabel("Frequency")
    plt.savefig(BASE_DIR / "figures/phase_distribution.png", dpi=300)
    plt.close()

def fen_to_board(fen):
    board_part = fen.split()[0]
    rows = board_part.split('/')
    
    board = []
    for row in rows:
        current_row = []
        for char in row:
            if char.isdigit():
                current_row.extend([0] * int(char))
            else:
                current_row.append(1)
        board.append(current_row)
    
    return np.array(board)

def plot_piece_heatmap(fens):
    heatmap = np.zeros((8, 8))

    for fen in fens:
        board = fen_to_board(fen)
        heatmap += board

    plt.figure()
    plt.imshow(heatmap)
    plt.colorbar()
    plt.title("Piece Presence Heatmap")
    plt.savefig(BASE_DIR / "figures/piece_heatmap.png", dpi=300)
    plt.close()

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Directory which stores the processed positions
DATA_PATH = BASE_DIR / "processed_positions"

# Create a directory to store the plots
os.makedirs(BASE_DIR / "figures", exist_ok=True)

# Get the moves and FENs from the processed positions
moves, fens = load_data(DATA_PATH / "lichess_db_standard_rated_2013-02_filtered_positions.jsonl")

# Generate plots
print("Plotting move distribution")
plot_move_distribution(moves)

print("Plotting game phase distribution")
plot_phase_distribution_from_fen(fens)

print("Generating piece heatmap")
plot_piece_heatmap(fens)