"""Checkpoint-compatible architecture recovered from the 2026 study."""
import torch
from torch import nn

class LipNetBackbone(nn.Module):
    """
    LipNet-style:
    - 3D Conv frontend
    - 2-layer BiGRU
    - character logits per timestep
    """
    def __init__(self, num_classes: int, img_size: int = 96, max_frames: int = 32, rnn_units: int = 128, dropout: float = 0.3):
        super().__init__()
        self.max_frames = max_frames

        self.frontend = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3, 5, 5), padding=(1, 2, 2), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            nn.Conv3d(32, 64, kernel_size=(3, 5, 5), padding=(1, 2, 2), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            nn.Conv3d(64, 96, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )

        feat_h = img_size // 8
        feat_w = img_size // 8
        feat_dim = 96 * feat_h * feat_w

        self.gru1 = nn.GRU(
            input_size=feat_dim,
            hidden_size=rnn_units,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.gru2 = nn.GRU(
            input_size=2 * rnn_units,
            hidden_size=rnn_units,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * rnn_units, num_classes)

    def forward(self, x):
        # x: (B,1,T,H,W)
        x = self.frontend(x)  # (B,C,T,H,W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B,T,C,H,W)
        B, T, C, H, W = x.shape
        x = x.view(B, T, C * H * W)
        x, _ = self.gru1(x)
        x = self.drop(x)
        x, _ = self.gru2(x)
        x = self.drop(x)
        logits = self.classifier(x)  # (B,T,C)
        return logits
