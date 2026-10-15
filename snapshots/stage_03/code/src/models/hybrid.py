"""HybridModel = spatial encoder -> projection -> (+ positional embedding in tokens mode) -> temporal encoder ->
pooling -> classifier. Two input modes (one model code):
  "sequence": x uint8 (B, T, 3, 224, 224); each frame -> pooled backbone vector -> sequence over TIME (L = T)
  "tokens":   x uint8 (B, 3, 224, 224); final feature map -> h*w spatial tokens (+ learned positional embedding)
              -> sequence over SPATIAL REGIONS (L = h*w)
forward(x) -> {"logit": (B,), "attn": (B, L) or None, "features": (B, d)}  (d = pooled dimension)."""
import contextlib

import torch
import torch.nn as nn


class Projection(nn.Module):
    """LayerNorm -> Dropout 0.3 -> Linear -> d (256) -> GELU, applied per sequence element."""

    def __init__(self, d_in, d_out=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_in), nn.Dropout(dropout), nn.Linear(d_in, d_out), nn.GELU())

    def forward(self, x):
        return self.net(x)


class HybridModel(nn.Module):
    def __init__(self, spatial, projection, temporal, pooling, classifier, mode, seq_len):
        super().__init__()
        if mode not in {"sequence", "tokens"}:
            raise ValueError(mode)
        self.mode, self.seq_len = mode, seq_len
        self.spatial, self.projection, self.temporal = spatial, projection, temporal
        self.pooling, self.classifier = pooling, classifier
        d = projection.net[2].out_features
        self.token_pos = None
        if mode == "tokens":
            self.token_pos = nn.Parameter(torch.zeros(1, seq_len, d))
            nn.init.trunc_normal_(self.token_pos, std=0.02)

    # backbone part ---------------------------------------------------------------
    def _backbone_ctx(self):
        return torch.no_grad() if self.spatial.policy == "frozen" else contextlib.nullcontext()

    def backbone_features(self, x):
        """uint8 input -> (B, L, C) backbone features (the part that the D01 feature cache stores)."""
        with self._backbone_ctx():
            if self.mode == "sequence":
                if x.dim() != 5:
                    raise ValueError(f"sequence mode expects (B, T, 3, H, W), got {tuple(x.shape)}")
                B, T = x.shape[:2]
                f = self.spatial.pooled(self.spatial.preprocess(x.flatten(0, 1)))
                return f.view(B, T, -1)
            if x.dim() != 4:
                raise ValueError(f"tokens mode expects (B, 3, H, W), got {tuple(x.shape)}")
            return self.spatial.tokens(self.spatial.preprocess(x))

    # head part ---------------------------------------------------------------------
    def forward_features(self, f):
        """(B, L, C) backbone features -> output dict. Used directly with cached features (frozen D01 backbones)."""
        h = self.projection(f)
        if self.token_pos is not None:
            h = h + self.token_pos
        h = self.temporal(h)
        pooled, attn = self.pooling(h)
        logit = self.classifier(pooled).squeeze(-1)
        return {"logit": logit, "attn": attn, "features": pooled}

    def forward(self, x):
        return self.forward_features(self.backbone_features(x))

    def head_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("spatial.")]
