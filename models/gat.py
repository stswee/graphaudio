import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class TextGAT(nn.Module):
    def __init__(self, input_dim_text=None, input_dim_mel=None, hidden_dim=128, num_classes=4, heads=2):
        super().__init__()
        assert input_dim_text is not None, "input_dim_text must be provided for TextGAT"
        self.gat1 = GATConv(input_dim_text, hidden_dim, heads=heads)
        self.gat2 = GATConv(hidden_dim * heads, num_classes, heads=1)

    def forward(self, x, edge_index, edge_attr=None):
        x = F.elu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        return x


class ClinicalGAT(nn.Module):
    """GAT for clinical (one-hot + numeric) node features."""
    def __init__(self, input_dim_text=None, input_dim_clinical=None, hidden_dim=128, num_classes=4, heads=2):
        super().__init__()
        in_dim = input_dim_clinical if input_dim_clinical is not None else input_dim_text
        assert in_dim is not None, "Provide input_dim_clinical (preferred) or input_dim_text for ClinicalGAT"
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads)
        self.gat2 = GATConv(hidden_dim * heads, num_classes, heads=1)

    def forward(self, x, edge_index, edge_attr=None):
        x = F.elu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        return x


class MelGAT(nn.Module):
    def __init__(self, input_dim_text=None, input_dim_mel=None, hidden_dim=128, num_classes=4, heads=2):
        super().__init__()
        assert input_dim_mel is not None, "input_dim_mel must be provided for MelGAT"
        self.flatten = nn.Flatten()
        self.linear_pre = nn.Linear(input_dim_mel, hidden_dim)
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=heads)
        self.gat2 = GATConv(hidden_dim * heads, num_classes, heads=1)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.flatten(x)
        x = F.relu(self.linear_pre(x))
        x = F.elu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        return x

class MultiModalGAT(nn.Module):
    def __init__(self, input_dim_text=None, input_dim_mel=None, hidden_dim=128, num_classes=4, heads=2):
        super().__init__()
        assert input_dim_text is not None and input_dim_mel is not None, "Both input_dim_text and input_dim_mel must be provided for MultiModalGAT"
        self.flatten = nn.Flatten()
        self.mel_linear = nn.Linear(input_dim_mel, hidden_dim)
        self.concat_linear = nn.Linear(hidden_dim + input_dim_text, hidden_dim)
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

class MultiModalClinicalGAT(nn.Module):
    """
    Multimodal GAT that fuses clinical and mel features.
    Constructor accepts input_dim_clinical (preferred) or input_dim_text for compatibility.
    Forward expects (clinical, mel, edge_index, ...).
    """
    def __init__(self, input_dim_clinical=None, input_dim_text=None, input_dim_mel=None,
                 hidden_dim=128, num_classes=4, heads=2):
        super().__init__()
        clinical_dim = input_dim_clinical if input_dim_clinical is not None else input_dim_text
        assert clinical_dim is not None and input_dim_mel is not None, \
            "Provide clinical (input_dim_clinical or input_dim_text) and input_dim_mel for MultiModalClinicalGAT"

        self.flatten = nn.Flatten()
        self.mel_linear = nn.Linear(input_dim_mel, hidden_dim)                 # mel -> hidden
        self.concat_linear = nn.Linear(hidden_dim + clinical_dim, hidden_dim)  # [clinical, mel_h] -> hidden
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=heads)
        self.gat2 = GATConv(hidden_dim * heads, num_classes, heads=1)

    def forward(self, clinical, mel, edge_index, edge_attr=None):
        mel = self.flatten(mel)
        mel = F.relu(self.mel_linear(mel))
        x = torch.cat([clinical, mel], dim=1)
        x = F.elu(self.concat_linear(x))
        x = F.elu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        return x
