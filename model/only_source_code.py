import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class SourceCodeProcessModel(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_layers=3, dropout=0.5, pool_type=''):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        return x.squeeze(1)
