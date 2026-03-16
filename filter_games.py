
# Imports
import os
import chess.pgn
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Directory which contains raw pgn files
DATA_DIR = BASE_DIR / "data"

# Directory to store the filtered pgn files
OUTPUT_DIR = BASE_DIR / "filtered_games"

# Minimum number of moves required for keeping
MIN_MOVES = 10

# Create the new filtered games directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def is_valid_game(game):
    headers = game.headers

    # Must have ELO ratings
    if not headers["WhiteElo"].isdigit() or not headers["BlackElo"].isdigit():
        return False

    # Must end normally
    if headers.get("Termination") != "Normal":
        return False

    # Must have standard result
    if headers.get("Result") not in {"1-0", "0-1", "1/2-1/2"}:
        return False

    # Count moves
    move_count = sum(1 for _ in game.mainline_moves())

    if move_count < MIN_MOVES:
        return False

    return True


def filter_file(input_path, output_path):

    with open(input_path) as pgn, open(output_path, "w") as outfile:

        game_count = 0
        kept_count = 0

        while True:

            game = chess.pgn.read_game(pgn)

            if game is None:
                break

            game_count += 1

            # If the game is valid, write it to the output file
            if is_valid_game(game):
                kept_count += 1
                print(game, file=outfile, end="\n\n")

            # Print a progress statement every 10,000 games
            if game_count % 10000 == 0:
                print(f"Processed {game_count} games")

        print(f"Finished {input_path}")
        print(f"Kept {kept_count}/{game_count} games")


def main():

    for filename in os.listdir(DATA_DIR):

        if not filename.endswith(".pgn"):
            continue

        input_path = os.path.join(DATA_DIR, filename)
        output_path = os.path.join(
            OUTPUT_DIR,
            filename.replace(".pgn", "_filtered.pgn")
        )

        print(f"Filtering {filename}")
        filter_file(input_path, output_path)


if __name__ == "__main__":
    main()