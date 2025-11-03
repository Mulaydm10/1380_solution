import math
import torch
from torch import nn

class PositionEmbedding3D(nn.Module):
    def __init__(self, hidden_size, patch_size=(1, 16, 16), max_period=10000):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.max_period = max_period
        self.pos_embed = self._create_3d_pos_embed()

    def _create_3d_pos_embed(self):
        d_model = self.hidden_size
        t_h_w = self.max_period
        pe = torch.zeros(t_h_w, d_model)
        position = torch.arange(0, t_h_w, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(self.max_period) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        seq_len = x.shape[1]
        pos_emb = self.pos_embed[:, :seq_len, :].to(x.device)
        return x + pos_emb
