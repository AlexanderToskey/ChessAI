import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)

        return x
    
class ChessCNN(nn.Module):

    def __init__(self, num_blocks=6):

        super().__init__()

        channels = 64

        # Initial board processing
        self.input_conv = nn.Conv2d(18, channels, 3, padding=1)
        self.input_bn = nn.BatchNorm2d(channels)

        # Residual stack
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )

        # Skill embedding
        self.skill_embedding = nn.Embedding(5, 16)

        # Final layers
        # Add 1 to include material
        self.fc1 = nn.Linear(channels * 8 * 8 + 16 + 1, 512)
        
        # Policy head (moves)
        self.policy_head = nn.Linear(512, 4096)

        # Value head (material)
        self.value_head = nn.Linear(512, 1)

    def forward(self, board, skill):

        x = F.relu(self.input_bn(self.input_conv(board)))
        x = self.res_blocks(x)
        x = x.view(x.size(0), -1)

        skill_vec = self.skill_embedding(skill)

        # Material feature
        material = self.compute_material(board)

        x = torch.cat([x, skill_vec, material], dim=1)
        x = F.relu(self.fc1(x))

        policy_logits = self.policy_head(x)
        value = torch.tanh(self.value_head(x))  # constrain to [-1, 1]

        return policy_logits, value
    
    def compute_material(self, board):
        """
        Helper function to compute the material value of pieces still on the board
        board: (B, 18, 8, 8)
        returns: (B, 1)
        """

        # Piece values
        values = torch.tensor([1, 3, 3, 5, 9, 0], device=board.device).view(1, 6, 1, 1)

        # White and black piece planes
        white = board[:, 0:6, :, :]
        black = board[:, 6:12, :, :]

        # Count pieces
        white_count = (white * values).sum(dim=(1,2,3))
        black_count = (black * values).sum(dim=(1,2,3))

        material = white_count - black_count

        # Normalize (important for stability)
        material = material / 39.0  # max theoretical material

        return material.unsqueeze(1)  # (B, 1)