import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
        num_classes: int = 3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.norm = nn.LayerNorm(out_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, num_classes),
        )

    def forward(self, x, lengths):
        """
        x: (B, T, F)
        lengths: (B,)
        """
        # pack for speed + correct handling of padding
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, c_n) = self.lstm(packed)

        # Use the last layer hidden state for each direction
        # h_n: (num_layers * num_directions, B, hidden_dim)
        if self.lstm.bidirectional:
            # last layer forward = -2, backward = -1
            h_f = h_n[-2]
            h_b = h_n[-1]
            h = torch.cat([h_f, h_b], dim=1)  # (B, 2H)
        else:
            h = h_n[-1]  # (B, H)

        h = self.norm(h)
        logits = self.head(h)
        return logits


class LSTMClassifierNoPack(nn.Module):
    """
    Export-friendly LSTM:
    - No pack/pad ops (ONNX/TRT compatible)
    - Uses last valid timestep (unidirectional recommended)
    """
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        num_classes: int = 3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.norm = nn.LayerNorm(out_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, num_classes),
        )

    def forward(self, x, lengths):
        """
        x: (B, T, F)
        lengths: (B,)
        """
        out, _ = self.lstm(x)  # (B, T, H)
        lengths = lengths.to(out.device)
        # gather last valid timestep for each sample
        idx = (lengths - 1).clamp(min=0).view(-1, 1, 1)
        idx = idx.expand(-1, 1, out.size(-1))
        h = out.gather(1, idx).squeeze(1)  # (B, H)
        h = self.norm(h)
        logits = self.head(h)
        return logits
