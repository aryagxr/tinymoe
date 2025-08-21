
import torch
import torch.nn as nn
from config.config import *

class Expert(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, in_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)



expert1 = Expert(in_dim=10, out_dim=10)
print(expert1)