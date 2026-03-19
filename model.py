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
        self.fc1 = nn.Linear(channels * 8 * 8 + 16, 512)
        self.fc2 = nn.Linear(512, 4096)

    def forward(self, board, skill):

        x = F.relu(self.input_bn(self.input_conv(board)))
        x = self.res_blocks(x)
        x = x.view(x.size(0), -1)

        skill_vec = self.skill_embedding(skill)

        x = torch.cat([x, skill_vec], dim=1)
        x = F.relu(self.fc1(x))

        moves = self.fc2(x)

        return moves