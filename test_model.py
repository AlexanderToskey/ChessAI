# Imports
import torch
import chess
from pathlib import Path

from model import ChessCNN
from utils.board_encoding import board_to_tensor
from utils.move_encoding import class_to_uci
from utils.move_masking import get_legal_move_mask

#from utils.search import select_move_1ply
#from utils.search import select_move_2ply

from utils.search import alpha_beta_root
from utils.search import tactical_search
from utils.search import is_blunder
from utils.endgame import is_endgame

# Explanation Imports
from explain.move_analysis import analyze_move
from explain.explain import generate_explanation

def main():
    # Base directory
    BASE_DIR = Path(__file__).resolve().parent

    # Directory which contains the trained CNN model
    MODEL_PATH = BASE_DIR / "models" / "chess_cnn_final.pth"

    # Set the training device to CPU
    DEVICE = torch.device("cpu")

    MATE_SCORE = 100000
    TACTICAL_THRESHOLD = 300  # Free piece

    TACTICAL_DEPTH = 3

    # Load the model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    model = ChessCNN(num_blocks=6).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print(f"Loaded model from: {MODEL_PATH}")

    # Ask the user if they'd like to enter one or multiple FENs
    multipleInputs = input("\nEnter multiple FENs? [y/n]: ")

    # Ask the user what skill level they want the AI to play at
    #elo_bucket = int(input("Enter skill bucket (0–4): "))
    elo_bucket = 4

    print("\nEnter 'q' to quit")

    # Continue predicting moves
    while True:
        # Ask the user for a FEN input
        fen = input("\nEnter FEN:\n")

        if fen == "q":
            break

        # Convert the FEN to a board representation
        board = chess.Board(fen)

        #move = select_move_1ply(model, board, elo_bucket, DEVICE)
        #move = select_move_2ply(model, board, elo_bucket, DEVICE)

        #print(f"\nSelected move: {move.uci()}")

        
        # Convert the board to a tensor
        board_tensor = torch.tensor(board_to_tensor(board), dtype=torch.float32)
        board_tensor = board_tensor.unsqueeze(0).to(DEVICE)
        skill_tensor = torch.tensor([elo_bucket], dtype=torch.long).to(DEVICE)

        move = None

        piece_count, total_material = is_endgame(board)

        # --- 1. Always check for tactics first ---
        tactical_move, tactical_score = tactical_search(board, depth=TACTICAL_DEPTH)

        print(f"Tactical score: {tactical_score}")

        if abs(tactical_score) >= 100000 - 10:
            print("Tactical: Found forced mate")
            move = tactical_move

        elif abs(tactical_score) >= TACTICAL_THRESHOLD:
            print("Tactical: Found winning material")
            move = tactical_move

        # --- 2. Endgame search ---
        elif piece_count <= 4:
            print("Endgame search depth=6")
            move = alpha_beta_root(board, depth=6)

        elif piece_count <= 6:
            print("Endgame search depth=5")
            move = alpha_beta_root(board, depth=5)

        # --- 3. Otherwise use CNN ---
        else:
            print("Using CNN (middlegame/opening)")

            with torch.no_grad():
                logits, values = model(board_tensor, skill_tensor)

                mask = get_legal_move_mask(board)
                mask = mask.unsqueeze(0).to(DEVICE)

                masked_logits = logits.masked_fill(mask == 0, -1e9)
                predicted_class = masked_logits.argmax(dim=1).item()

            candidate_move = chess.Move.from_uci(class_to_uci(predicted_class))
            
            # --- Blunder check ---
            if is_blunder(board, candidate_move):
                print(f"CNN move: {candidate_move} is a blunder, searching alternatives...")

                # Try alternatives using tactical search
                safe_move, safe_score = tactical_search(board, depth=TACTICAL_DEPTH)

                move = safe_move
            else:
                move = candidate_move

        # Print the selected move
        print(f"\nSelected move: {move.uci()}")

        # --- Generate Explanation ---
        try:
            features = analyze_move(board, move)
            explanation = generate_explanation(features)

            print(f"Explanation: {explanation}")

            # Debug
            #print("Features:", features)

        except Exception as e:
            print("Explanation generation failed:", e)

            # Debug
            #print("Features: ", features)
        
        # Stop if the user only wants to enter one FEN
        if multipleInputs != "y":
            break

if __name__ == "__main__":
    main()


