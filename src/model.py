import torch
from torch import nn

class Predictor(nn.Module):
    def __init__(self, in_dim, hidden=64, dropout=0.2):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden, 2)

    def forward(self, x):
        h = self.feature(x)
        return h, self.classifier(h)
