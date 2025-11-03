import math
import torch
from torch import nn

class PositionEmbedding3D(nn.Module):
    def __init__(self, hidden_size, input_size=(16, 80, 32, 32), patch_size=(1, 2, 2), max_period=10000):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.patch_size = patch_size
        self.max_period = max_period
        self.pos_embed = self._create_3d_pos_embed()

    def _create_3d_pos_embed(self):
        d_model = self.hidden_size
        
        # Calculate max sequence length from input_size and patch_size
        t, h, w = self.input_size[1:]
        pt, ph, pw = self.patch_size
        max_seq_len = (t // pt) * (h // ph) * (w // pw)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(self.max_period) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        seq_len = x.shape[1]
        pos_emb = self.pos_embed[:, :seq_len, :].to(x.device, dtype=x.dtype)
        return x + pos_emb
