import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from chess_dataset import ChessDataset 
from model import ChessCNN
from pathlib import Path

def main():
    # Base directory
    BASE_DIR = Path(__file__).resolve().parent

    # Directory which stores the processed positions
    DATA_PATH = BASE_DIR / "processed_positions"

    # Directory which contains the trained CNN model
    MODEL_SAVE_PATH = BASE_DIR / "models" / "chess_cnn_final.pth"

    # Hyperparameters
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    EPOCHS = 2

    # Set the training device to CPU
    DEVICE = torch.device("cpu")

    # Load the dataset
    print(f"Loading dataset from: {DATA_PATH}")

    # If the dataset doesn't exist, raise an error
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    # List every JSONL file in the processed positions directory
    jsonl_files = list(DATA_PATH.glob("*.jsonl"))

    # If there are no JSONL files, raise an error
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in {DATA_PATH}")

    print(f"Found {len(jsonl_files)} dataset files")

    # Extract the training dataset from the process positions
    train_dataset = ChessDataset(jsonl_files)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # Initialize the model
    print("Initializing model...")
    model = ChessCNN(num_blocks=6).to(DEVICE)
    print("Done")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    print("Begin training loop")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            board = batch['board'].to(DEVICE)
            skill = batch['elo_bucket'].to(DEVICE)
            move = batch['move_class'].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(board, skill)
            loss = criterion(outputs, move)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * board.size(0)

        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch}/{EPOCHS}, Loss: {epoch_loss:.4f}")

    # Save the model
    print("Saving model...")
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()