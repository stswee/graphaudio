import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class TextGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=4, heads=2):
        super().__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads)
        self.gat2 = GATConv(hidden_dim * heads, num_classes, heads=1)

    def forward(self, x, edge_index, edge_attr=None):
        x = F.elu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        return x

class MelGAT(nn.Module):
    def __init__(self, mel_input_shape, hidden_dim=128, num_classes=4, heads=2):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_pre = nn.Linear(mel_input_shape[0] * mel_input_shape[1], hidden_dim)
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=heads)
        self.gat2 = GATConv(hidden_dim * heads, num_classes, heads=1)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.flatten(x)
        x = F.relu(self.linear_pre(x))
        x = F.elu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        return x

class MultiModalGAT(nn.Module):
    def __init__(self, text_dim, mel_input_shape, hidden_dim=128, num_classes=4, heads=2):
        super().__init__()
        self.flatten = nn.Flatten()
        self.mel_linear = nn.Linear(mel_input_shape[0] * mel_input_shape[1], hidden_dim)
        self.concat_linear = nn.Linear(hidden_dim + text_dim, hidden_dim)
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=heads)
        self.gat2 = GATConv(hidden_dim * heads, num_classes, heads=1)

    def forward(self, text, mel, edge_index, edge_attr=None):
        mel = self.flatten(mel)
        mel = F.relu(self.mel_linear(mel))
        x = torch.cat([text, mel], dim=1)
        x = F.elu(self.concat_linear(x))
        x = F.elu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        return x
