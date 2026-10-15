"""build_model(model_id, mode, cfg=None) -> HybridModel (on CPU, policy applied); param_groups for the optimizer."""
import os

import torch
import yaml

from src.models.encoders import SpatialEncoder
from src.models.heads import build_head, build_pooling
from src.models.hybrid import HybridModel, Projection
from src.models.temporal import build_temporal
from src.utils.paths import PROJECT_DIR

MODEL_IDS = [f"M{i:02d}" for i in range(1, 11)]
MODES = ["sequence", "tokens"]
SEQ_LEN_T = 8  # D01 clips: 8 x 1-s windows


def load_model_cfg(model_id):
    with open(os.path.join(PROJECT_DIR, "configs", "models", f"{model_id}.yaml")) as fh:
        return yaml.safe_load(fh)


def build_model(model_id, mode, cfg=None, policy=None):
    """model_id 'M01'..'M10'; mode 'sequence' (D01) | 'tokens' (D02); policy overrides cfg['policy'][mode]."""
    if model_id not in MODEL_IDS or mode not in MODES:
        raise ValueError(f"{model_id} / {mode}")
    cfg = cfg or load_model_cfg(model_id)
    torch.manual_seed(cfg.get("seed", 42))  # deterministic head initialisation
    enc = cfg["encoder"]
    input_size = enc.get("input_size", {}).get(mode, 224)
    spatial = SpatialEncoder(enc, input_size, mode)
    spatial.set_policy(policy or cfg["policy"][mode])
    seq_len = SEQ_LEN_T if mode == "sequence" else spatial.n_tokens()
    d = cfg["projection"]["d"]
    projection = Projection(spatial.out_dim, d, cfg["projection"]["dropout"])
    temporal = build_temporal(cfg["temporal"], d, seq_len)
    pooling = build_pooling(cfg["pooling"], temporal.out_dim, cfg["head"])
    head = build_head(cfg["head"]["type"], pooling.out_dim, cfg["head"])
    model = HybridModel(spatial, projection, temporal, pooling, head, mode, seq_len)
    model.model_id, model.cfg = model_id, cfg
    return model


def param_groups(model, cfg=None):
    """[backbone params (backbone_lr), head params (head_lr)]; only parameters with requires_grad."""
    cfg = cfg or model.cfg
    lr = cfg["lr"]
    bb = model.spatial.trainable_backbone_params()
    groups = [{"params": model.head_parameters(), "lr": lr["head_lr"], "name": "head"}]
    if bb:
        groups.insert(0, {"params": bb, "lr": lr["backbone_lr"], "name": "backbone"})
    return groups


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
