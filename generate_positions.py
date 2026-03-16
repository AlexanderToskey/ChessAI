
# Imports
import os
import bz2
import pickle
import chess
import chess.pgn
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Direcotry which contains the filtered games
FILTERED_DIR = BASE_DIR / "filtered_games"

# Directory to contain the processed positions
OUTPUT_DIR = BASE_DIR / "processed_positions"

# Create the new processed positions directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_game(game, outfile):

    board = game.board()

    white_elo = int(game.headers["WhiteElo"])
    black_elo = int(game.headers["BlackElo"])

    for move in game.mainline_moves():

        fen = board.fen()

        elo = white_elo if board.turn else black_elo

        sample = {
            "fen": fen,
            "elo": elo,
            "move": move.uci()
        }

        outfile.write(json.dumps(sample) + "\n")

        board.push(move)


def process_file(input_path, output_path):

    with open(input_path) as pgn, open(output_path, "w") as outfile:

        game_count = 0
        position_count = 0

        while True:

            game = chess.pgn.read_game(pgn)

            if game is None:
                break

            process_game(game, outfile)

            game_count += 1

            if game_count % 1000 == 0:
                print(f"Processed {game_count} games")

        print(f"Finished {input_path}")


def main():

    for input_path in FILTERED_DIR.glob("*.pgn"):

        output_filename = input_path.stem + "_positions.jsonl"
        output_path = OUTPUT_DIR / output_filename

        if output_path.exists():
            print(f"Skipping {input_path.name} (already processed)")
            continue

        print(f"Generating positions from {input_path.name}")

        process_file(input_path, output_path)


if __name__ == "__main__":
    main()