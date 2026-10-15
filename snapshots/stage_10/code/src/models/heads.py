"""Pooling over the sequence and the classifier heads."""
import torch
import torch.nn as nn


class AdditiveAttention(nn.Module):
    """score_l = v^T tanh(W h_l + b); weights = softmax over the sequence (sum to 1); pooled = sum_l w_l h_l."""

    def __init__(self, dim, hidden=128):
        super().__init__()
        self.proj = nn.Linear(dim, hidden)
        self.v = nn.Linear(hidden, 1, bias=False)
        self.out_dim = dim

    def forward(self, h):
        w = torch.softmax(self.v(torch.tanh(self.proj(h))).squeeze(-1).float(), dim=1)  # softmax in fp32
        return torch.bmm(w.unsqueeze(1).to(h.dtype), h).squeeze(1), w


class MeanPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.out_dim = dim

    def forward(self, h):
        return h.mean(dim=1), None


class Classifier(nn.Module):
    """Dropout -> Linear -> 1 logit."""

    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, 1))

    def forward(self, z):
        return self.net(z)


class MLPHead(nn.Module):
    """M05: dim (256) -> 128 -> 1, GELU, dropout 0.3."""

    def __init__(self, dim, hidden=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, z):
        return self.net(z)


def build_pooling(kind, dim, cfg):
    if kind == "attention":
        return AdditiveAttention(dim, cfg.get("attention_hidden", 128))
    if kind == "mean":
        return MeanPool(dim)
    raise ValueError(kind)


def build_head(kind, dim, cfg):
    if kind == "linear":
        return Classifier(dim, cfg.get("dropout", 0.3))
    if kind == "mlp":
        return MLPHead(dim, cfg.get("mlp_hidden", 128), cfg.get("dropout", 0.3))
    raise ValueError(kind)
