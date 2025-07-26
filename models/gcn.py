import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class TextGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=4):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, x, edge_index, edge_attr=None):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


class MelGCN(nn.Module):
    def __init__(self, mel_input_shape, hidden_dim=128, num_classes=4):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_pre = nn.Linear(mel_input_shape[0] * mel_input_shape[1], hidden_dim)
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.flatten(x)
        x = F.relu(self.linear_pre(x))
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


class MultiModalGCN(nn.Module):
    def __init__(self, text_dim, mel_input_shape, hidden_dim=128, num_classes=4):
        super().__init__()
        self.flatten = nn.Flatten()
        self.mel_linear = nn.Linear(mel_input_shape[0] * mel_input_shape[1], hidden_dim)
        self.concat_linear = nn.Linear(hidden_dim + text_dim, hidden_dim)
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, text, mel, edge_index, edge_attr=None):
        mel = self.flatten(mel)
        mel = F.relu(self.mel_linear(mel))
        x = torch.cat([text, mel], dim=1)
        x = F.relu(self.concat_linear(x))
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
