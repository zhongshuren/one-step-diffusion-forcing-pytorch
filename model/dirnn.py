import torch
import torch.nn as nn
import math


class TimestepEmbedder(nn.Module):
    def __init__(self, dim, nfreq=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(nfreq, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.nfreq = nfreq

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half_dim, dtype=torch.float32)
            / half_dim
        ).to(device=t.device)
        args = t * freqs
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t = t*1000
        t_freq = self.timestep_embedding(t, self.nfreq)
        t_emb = self.mlp(t_freq)
        return t_emb


class DiRNN(nn.Module):
    def __init__(self, num_labels=0, dim=256, output_size=2):
        super(DiRNN, self).__init__()
        self.num_labels = num_labels
        self.dim = dim

        self.t1_embedder = TimestepEmbedder(dim)
        self.t2_embedder = TimestepEmbedder(dim)
        self.e_embedder = nn.Linear(output_size, dim, bias=False)
        self.c_embedder = nn.Embedding(num_labels + 1, dim)
        self.past_embedder = nn.Linear(output_size, dim, bias=False)
        self.h_embedder = nn.GRU(dim, dim, batch_first=True, num_layers=2)
        # self.h_embedder = nn.LSTM(dim, dim, batch_first=True, num_layers=2)
        self.layer_norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, output_size)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
        )

    def forward(self, e, t1=None, t2=None, c=None, h=None, past_x=None):
        if t1 is None and t2 is None:
            t1 = torch.zeros(e.shape[0], 1, 1, device=e.device)
            t2 = torch.ones(e.shape[0], 1, 1, device=e.device)
        if c is None:
            c = torch.ones(e.shape[0], 1, device=e.device).int() * self.num_labels
        # h_out, h = self.h_embedder(self.past_embedder(past_x), h)
        # gate, h_embed = torch.split(self.expand(h_out), self.dim, dim=-1)
        # x = self.e_embedder(e) * (1 - gate) + self.t1_embedder(t1) + self.t2_embedder(t2) + h_embed * gate
        x = self.e_embedder(e) + self.t1_embedder(t1) + self.t2_embedder(t2) + self.past_embedder(past_x)
        x, h = self.h_embedder(x, h)
        x = self.mlp(x) + x
        x = self.out(x)
        return x, h
