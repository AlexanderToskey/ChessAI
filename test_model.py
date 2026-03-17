
# Imports
import torch
import chess
from pathlib import Path

from model import ChessCNN
from utils.board_encoding import board_to_tensor
from utils.move_encoding import class_to_uci

def main():
    # Base directory
    BASE_DIR = Path(__file__).resolve().parent

    # Directory which contains the trained CNN model
    MODEL_PATH = BASE_DIR / "models" / "chess_cnn_final.pth"

    # Set the training device to CPU
    DEVICE = torch.device("cpu")

    # Load the model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    model = ChessCNN(num_blocks=6).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print(f"Loaded model from: {MODEL_PATH}")

    # Ask the user if they'd like to enter one or multiple FENs
    multipleInputs = input("\nEnter multiple FENs? [Y/n]: ")

    # Ask the user what skill level they want the AI to play at
    elo_bucket = int(input("Enter skill bucket (0–4): "))

    print("\nEnter 'q' to quit")

    # Continue predicting moves
    while True:
        # Ask the user for a FEN input
        fen = input("\nEnter FEN:\n")

        if fen == "q":
            break

        # Convert the FEN to a board representation
        board = chess.Board(fen)

        # Convert the board to a tensor
        board_tensor = torch.tensor(board_to_tensor(board), dtype=torch.float32)
        board_tensor = board_tensor.unsqueeze(0).to(DEVICE)
        skill_tensor = torch.tensor([elo_bucket], dtype=torch.long).to(DEVICE)

        # Predict the next move
        with torch.no_grad():
            output = model(board_tensor, skill_tensor)
            predicted_class = output.argmax(dim=1).item()

        uci_move = class_to_uci(predicted_class)

        print(f"\nPredicted move: {uci_move}")

        # Stop if the user only wants to enter one FEN
        if multipleInputs != "Y":
            break

if __name__ == "__main__":
    main()