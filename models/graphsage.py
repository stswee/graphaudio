import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class TextGraphSAGE(nn.Module):
    def __init__(self, input_dim_text=None, input_dim_mel=None, hidden_dim=128, num_classes=4):
        super().__init__()
        assert input_dim_text is not None, "input_dim_text must be provided for TextGraphSAGE"
        self.sage1 = SAGEConv(input_dim_text, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, num_classes)

    def forward(self, x, edge_index, edge_attr=None):
        x = F.relu(self.sage1(x, edge_index))
        x = self.sage2(x, edge_index)
        return x

class MelGraphSAGE(nn.Module):
    def __init__(self, input_dim_text=None, input_dim_mel=None, hidden_dim=128, num_classes=4):
        super().__init__()
        assert input_dim_mel is not None, "input_dim_mel must be provided for MelGraphSAGE"
        self.flatten = nn.Flatten()
        self.linear_pre = nn.Linear(input_dim_mel, hidden_dim)
        self.sage1 = SAGEConv(hidden_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, num_classes)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.flatten(x)
        x = F.relu(self.linear_pre(x))
        x = F.relu(self.sage1(x, edge_index))
        x = self.sage2(x, edge_index)
        return x

class MultiModalGraphSAGE(nn.Module):
    def __init__(self, input_dim_text=None, input_dim_mel=None, hidden_dim=128, num_classes=4):
        super().__init__()
        assert input_dim_text is not None and input_dim_mel is not None, "Both input_dim_text and input_dim_mel must be provided for MultiModalGraphSAGE"
        self.flatten = nn.Flatten()
        self.mel_linear = nn.Linear(input_dim_mel, hidden_dim)
        self.concat_linear = nn.Linear(hidden_dim + input_dim_text, hidden_dim)
        self.sage1 = SAGEConv(hidden_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, num_classes)

    def forward(self, text, mel, edge_index, edge_attr=None):
        mel = self.flatten(mel)
        mel = F.relu(self.mel_linear(mel))
        x = torch.cat([text, mel], dim=1)
        x = F.relu(self.concat_linear(x))
        x = F.relu(self.sage1(x, edge_index))
        x = self.sage2(x, edge_index)
        return x
