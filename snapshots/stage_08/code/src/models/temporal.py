"""Temporal encoders over a sequence (B, L, d_in) -> (B, L, d_out).
RNNs: 1 layer, hidden 128 per direction. Transformer: pre-norm encoder, 2 layers, 4 heads, d=256, dropout 0.1,
learned positional embedding. M10: BiLSTM -> Transformer."""
import torch
import torch.nn as nn


class RNNEncoder(nn.Module):
    def __init__(self, kind, d_in, hidden=128):
        super().__init__()
        bi = kind == "bilstm"
        cls = nn.GRU if kind == "gru" else nn.LSTM
        self.rnn = cls(d_in, hidden, num_layers=1, batch_first=True, bidirectional=bi)
        self.out_dim = hidden * (2 if bi else 1)

    def forward(self, x):
        return self.rnn(x)[0]


class TransformerTemporal(nn.Module):
    def __init__(self, d_in, max_len, d_model=256, n_layers=2, n_heads=4, dropout=0.1, ff_mult=4):
        super().__init__()
        self.inp = nn.Identity() if d_in == d_model else nn.Linear(d_in, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=ff_mult * d_model, dropout=dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)  # final norm required by the pre-norm layout
        self.out_dim = d_model

    def forward(self, x):
        h = self.inp(x) + self.pos[:, : x.shape[1]]
        return self.norm(self.encoder(h))


class BiLSTMTransformer(nn.Module):
    def __init__(self, d_in, max_len, hidden=128, **tf):
        super().__init__()
        self.rnn = RNNEncoder("bilstm", d_in, hidden)
        self.tf = TransformerTemporal(self.rnn.out_dim, max_len, **tf)
        self.out_dim = self.tf.out_dim

    def forward(self, x):
        return self.tf(self.rnn(x))


def build_temporal(cfg, d_in, max_len):
    kind = cfg["type"]
    tf = {k: cfg[k] for k in ["d_model", "n_layers", "n_heads", "dropout", "ff_mult"] if k in cfg}
    if kind in {"lstm", "bilstm", "gru"}:
        return RNNEncoder(kind, d_in, cfg.get("hidden", 128))
    if kind == "transformer":
        return TransformerTemporal(d_in, max_len, **tf)
    if kind == "bilstm_transformer":
        return BiLSTMTransformer(d_in, max_len, cfg.get("hidden", 128), **tf)
    raise ValueError(kind)
