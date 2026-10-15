"""Stage 3 model tests: every model M01-M10 x every mode, on cuda with torch.autocast fp16. No real labels are used."""
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
from src.models.factory import MODEL_IDS, MODES, build_model, count_params, load_model_cfg, param_groups  # noqa: E402
from src.models.feature_cache import CLIPS_NPY, build_cache, load_cache  # noqa: E402

DEV = "cuda"
COMBOS = [(m, mode) for m in MODEL_IDS for mode in MODES]
EXPECTED = {  # exact mapping from the Stage 3 brief
    "M01": ("efficientnet_b3", "bilstm", "attention", "linear"),
    "M02": ("custom_cnn", "bilstm", "attention", "linear"),
    "M03": ("resnet50", "lstm", "attention", "linear"),
    "M04": ("convnext_tiny", "bilstm", "attention", "linear"),
    "M05": ("custom_cnn", "transformer", "mean", "mlp"),
    "M06": ("densenet121", "lstm", "attention", "linear"),
    "M07": ("vit_base_patch16_224", "lstm", "attention", "linear"),
    "M08": ("efficientnet_b0", "transformer", "attention", "linear"),
    "M09": ("custom_cnn", "gru", "attention", "linear"),
    "M10": ("resnet50", "bilstm_transformer", "mean", "linear"),
}
FROZEN_D01 = [m for m in MODEL_IDS if load_model_cfg(m)["encoder"]["type"] == "timm"]


def _x(mode, b, g=None, coarse=False):
    """Random uint8 input. coarse=True: random 8x8 colour grids upscaled (distinguishable for pretrained nets)."""
    n = b * 8 if mode == "sequence" else b
    if coarse:
        z = torch.randint(0, 256, (n, 3, 8, 8), generator=g).float()
        x = F.interpolate(z, size=(224, 224), mode="nearest").to(torch.uint8)
    else:
        x = torch.randint(0, 256, (n, 3, 224, 224), dtype=torch.uint8, generator=g)
    return (x.view(b, 8, 3, 224, 224) if mode == "sequence" else x).to(DEV)


def _amp():
    return torch.autocast("cuda", dtype=torch.float16)


def setup_module():
    assert torch.cuda.is_available(), "Stage 3 tests require CUDA"


# ------------------------------------------------------------------ configs / factory
@pytest.mark.parametrize("mid", MODEL_IDS)
def test_config_mapping(mid):
    cfg = load_model_cfg(mid)
    enc, temporal, pooling, head = EXPECTED[mid]
    got_enc = cfg["encoder"].get("timm_name", "custom_cnn").split(".")[0]
    assert (got_enc, cfg["temporal"]["type"], cfg["pooling"], cfg["head"]["type"]) == (enc, temporal, pooling, head)
    assert cfg["model_id"] == mid and cfg["projection"] == {"d": 256, "dropout": 0.3}
    if cfg["temporal"]["type"] in {"transformer", "bilstm_transformer"}:
        assert {k: cfg["temporal"][k] for k in ["d_model", "n_layers", "n_heads", "dropout"]} == \
            {"d_model": 256, "n_layers": 2, "n_heads": 4, "dropout": 0.1}
    custom = cfg["encoder"]["type"] == "custom_cnn"
    assert cfg["policy"] == ({"sequence": "full", "tokens": "full"} if custom
                             else {"sequence": "frozen", "tokens": "last_stage"})
    if not custom:
        assert cfg["lr"]["backbone_lr"] == pytest.approx(cfg["lr"]["head_lr"] / 10)


def test_factory_rejects_unknown():
    with pytest.raises(ValueError):
        build_model("M11", "sequence")
    with pytest.raises(ValueError):
        build_model("M01", "video")


# ------------------------------------------------------------------ forward
@pytest.mark.parametrize("mid,mode", COMBOS)
def test_forward(mid, mode):
    m = build_model(mid, mode).to(DEV).train()
    with _amp():
        out = m(_x(mode, 2))
    assert out["logit"].shape == (2,) and torch.isfinite(out["logit"]).all()
    assert out["features"].shape[0] == 2 and torch.isfinite(out["features"]).all()
    has_attn = EXPECTED[mid][2] == "attention"
    if not has_attn:
        assert out["attn"] is None
    else:
        L = 8 if mode == "sequence" else (196 if EXPECTED[mid][0] in {"vit_base_patch16_224", "custom_cnn"} else 49)
        assert out["attn"].shape == (2, L)
        assert torch.allclose(out["attn"].sum(1), torch.ones(2, device=DEV), atol=1e-4)


# ------------------------------------------------------------------ backward + optimizer step
@pytest.mark.parametrize("mid,mode", COMBOS)
def test_backward_and_step(mid, mode):
    m = build_model(mid, mode).to(DEV).train()
    trainable = {n: p for n, p in m.named_parameters() if p.requires_grad}
    frozen = {n: p for n, p in m.named_parameters() if not p.requires_grad}
    assert trainable
    if m.spatial.policy in {"frozen", "last_stage"}:
        assert frozen and all(n.startswith("spatial.") for n in frozen)
        # modules holding only frozen params are in eval mode (BN statistics not updated)
        for name, mod in m.spatial.backbone.named_modules():
            if not m.spatial._is_trainable(name) and any(True for _ in mod.parameters(recurse=False)):
                assert not mod.training, name
    before_t = {n: p.detach().clone() for n, p in trainable.items()}
    before_f = {n: p.detach().clone() for n, p in frozen.items()}
    bufs = {n: b.detach().clone() for n, b in m.spatial.backbone.named_buffers() if "running" in n}
    opt = torch.optim.AdamW(param_groups(m), weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    y = torch.tensor([0.0, 1.0], device=DEV)
    x = _x(mode, 2)
    for _ in range(5):  # repeat only if the scaler skipped the step (inf/nan grads at this scale)
        opt.zero_grad(set_to_none=True)
        with _amp():
            loss = F.binary_cross_entropy_with_logits(m(x)["logit"].float(), y)
        scaler.scale(loss).backward()
        assert all(p.grad is not None for p in trainable.values()), \
            [n for n, p in trainable.items() if p.grad is None]
        assert all(p.grad is None for p in frozen.values())
        scale = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        if scaler.get_scale() >= scale:
            break
    changed = [n for n, p in trainable.items() if not torch.equal(p.detach(), before_t[n])]
    assert len(changed) == len(trainable), sorted(set(trainable) - set(changed))[:5]
    assert all(torch.equal(p.detach(), before_f[n]) for n, p in frozen.items())
    if m.spatial.policy == "frozen":
        assert all(torch.equal(b, bufs[n]) for n, b in m.spatial.backbone.named_buffers() if n in bufs)
    elif m.spatial.policy == "last_stage":
        frozen_bufs = [n for n in bufs if not m.spatial._is_trainable(n.rsplit(".", 1)[0])]
        assert frozen_bufs or not bufs  # ViT / ConvNeXt have LayerNorm only (no running statistics)
        named = dict(m.spatial.backbone.named_buffers())
        assert all(torch.equal(named[n], bufs[n]) for n in frozen_bufs)


# ------------------------------------------------------------------ determinism
@pytest.mark.parametrize("mid,mode", COMBOS)
def test_determinism_eval(mid, mode):
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    x = _x(mode, 2, torch.Generator().manual_seed(1))
    outs = []
    for _ in range(2):
        torch.manual_seed(42)
        m = build_model(mid, mode).to(DEV).eval()
        with torch.no_grad(), _amp():
            outs.append(m(x)["logit"].float().cpu())
        del m
    assert torch.equal(outs[0], outs[1])


# ------------------------------------------------------------------ tiny overfit on random data (NOT real labels)
@pytest.mark.parametrize("mid,mode", COMBOS)
def test_tiny_overfit_random_data(mid, mode):
    """16 random samples, random 0/1 labels, 60 steps, config learning rates, dropout on: final loss < 30% of the
    first. Frozen backbones: only the head trains, so the head is fed random N(0,1) backbone-sized features
    (random images all map to almost the same frozen feature, mean cosine ~0.998). Otherwise: the full model on
    random coarse uint8 images."""
    g = torch.Generator().manual_seed(0)
    m = build_model(mid, mode).to(DEV).train()
    y = torch.randint(0, 2, (16,), generator=g).float().to(DEV)
    if m.spatial.policy == "frozen":
        inp = torch.randn(16, m.seq_len, m.spatial.out_dim, generator=g).to(DEV)
        fwd = m.forward_features
    else:
        inp = _x(mode, 16, g, coarse=True)
        fwd = m
    opt = torch.optim.AdamW(param_groups(m), weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    losses = []
    for _ in range(60):
        opt.zero_grad(set_to_none=True)
        with _amp():
            loss = F.binary_cross_entropy_with_logits(fwd(inp)["logit"].float(), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        losses.append(loss.item())
    assert losses[-1] < 0.3 * losses[0], (losses[0], losses[-1])


# ------------------------------------------------------------------ feature cache (D01, frozen backbones)
@pytest.mark.parametrize("mid", FROZEN_D01)
def test_feature_cache_equivalence(mid, tmp_path):
    m = build_model(mid, "sequence").to(DEV).eval()
    assert m.spatial.policy == "frozen"
    build_cache(m, root=str(tmp_path), limit=4)
    cached, meta = load_cache(m.spatial.timm_name, root=str(tmp_path))
    assert cached.dtype == np.float16 and cached.shape == (4, 8, m.spatial.out_dim)
    x = torch.from_numpy(np.array(np.load(CLIPS_NPY, mmap_mode="r")[:4])).to(DEV)
    with torch.no_grad(), _amp():
        live = m(x)["logit"].float()
        head = m.forward_features(torch.from_numpy(np.array(cached)).to(DEV).float())["logit"].float()
    assert torch.allclose(live, head, atol=1e-3), (live - head).abs().max()


def test_last_stage_trains_part_of_backbone():
    for mid in FROZEN_D01:
        m = build_model(mid, "tokens")
        bb = list(m.spatial.backbone.parameters())
        n_tr = sum(p.numel() for p in bb if p.requires_grad)
        assert 0 < n_tr < sum(p.numel() for p in bb), mid
        total, trainable = count_params(m)
        assert trainable < total


def test_model_configs_have_batch_sizes():
    for mid in MODEL_IDS:
        with open(os.path.join(PROJECT_DIR, "configs", "models", f"{mid}.yaml")) as fh:
            bs = yaml.safe_load(fh)["batch_size"]
        assert all(bs[mode] in {64, 32, 16, 8, 4} for mode in MODES), (mid, bs)
