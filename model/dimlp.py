import torch
import torch.nn as nn
import math


class TimestepEmbedder(nn.Module):
    def __init__(self, dim, std=1.):
        super().__init__()
        self.weight = nn.Parameter(torch.randn([dim // 2]) * std)

    def forward(self, x: torch.Tensor):
        f = 2 * math.pi * x * self.weight
        return torch.cat([f.cos(), f.sin()], dim=-1)


class DiMLP(nn.Module):
    def __init__(self, num_labels, dim=256, output_size=2):
        super(DiMLP, self).__init__()

        self.t1_embedder = TimestepEmbedder(dim)
        self.t2_embedder = TimestepEmbedder(dim)
        self.e_embedder = nn.Linear(2, dim, bias=False)
        self.c_embedder = nn.Embedding(num_labels + 1, dim)
        self.out = nn.Linear(dim, output_size)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        
    def forward(self, e, t1=None, t2=None, c=None):
        t_embed = self.t1_embedder(t1) - self.t2_embedder(1 - t2) if t1 is not None else 0
        x = self.e_embedder(e) + t_embed + self.c_embedder(c)
        x = self.mlp(x) + x
        return self.out(x)
