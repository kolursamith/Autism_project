"""Per-frame backbone feature cache for D01 with FROZEN backbones (sequence mode).

The frozen backbone is deterministic, so its pooled per-frame features are computed once and stored as fp16:
  cache/features/<timm_name>_D01.npy   shape (N_clips, 8, C), row order = clips_index.csv row_in_npy
  cache/features/<timm_name>_D01.json  provenance (timm name, sha256 of clips_index.csv, shape, dtype)
Training then runs only the head: model.forward_features(torch.from_numpy(cache[idx]).cuda().float()).
Models sharing a backbone share the cache (M03 and M10 -> resnet50)."""
import json
import os

import numpy as np
import torch

from src.data.common import CACHE_DIR, sha256_file

CLIPS_NPY = os.path.join(CACHE_DIR, "D01_eye_tracking", "clips_uint8.npy")
CLIPS_INDEX = os.path.join(CACHE_DIR, "D01_eye_tracking", "clips_index.csv")
FEATURE_DIR = os.path.join(CACHE_DIR, "features")


def cache_path(timm_name, root=FEATURE_DIR):
    return os.path.join(root, f"{timm_name}_D01.npy")


@torch.no_grad()
def compute_features(model, clips, batch_clips=16, device="cuda", amp=True):
    """clips: uint8 array (N, T, 3, H, W) (memmap ok) -> float16 (N, T, C) from model.backbone_features."""
    if model.mode != "sequence" or model.spatial.policy != "frozen":
        raise ValueError("feature cache is only valid for sequence mode with a frozen backbone")
    model.eval()
    out = None
    for s in range(0, len(clips), batch_clips):
        x = torch.from_numpy(np.array(clips[s:s + batch_clips])).to(device, non_blocking=True)  # writable copy
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            f = model.backbone_features(x)
        if out is None:
            out = np.zeros((len(clips), f.shape[1], f.shape[2]), np.float16)
        out[s:s + len(x)] = f.float().cpu().numpy().astype(np.float16)
    return out


def build_cache(model, root=FEATURE_DIR, clips_npy=CLIPS_NPY, clips_index=CLIPS_INDEX, limit=None, batch_clips=16):
    os.makedirs(root, exist_ok=True)
    clips = np.load(clips_npy, mmap_mode="r")
    if limit:
        clips = clips[:limit]
    feats = compute_features(model, clips, batch_clips)
    name = model.spatial.timm_name
    path = cache_path(name, root)
    np.save(path, feats)
    meta = {"timm_name": name, "clips_index_sha256": sha256_file(clips_index), "shape": list(feats.shape),
            "dtype": "float16", "amp": True, "n_clips_used": int(len(clips))}
    with open(path.replace(".npy", ".json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    return path, meta


def load_cache(timm_name, root=FEATURE_DIR, clips_index=CLIPS_INDEX):
    """Loads a cache and refuses it if clips_index.csv changed since it was built."""
    with open(cache_path(timm_name, root).replace(".npy", ".json")) as fh:
        meta = json.load(fh)
    if meta["clips_index_sha256"] != sha256_file(clips_index):
        raise RuntimeError(f"feature cache {timm_name} is stale: clips_index.csv changed")
    return np.load(cache_path(timm_name, root), mmap_mode="r"), meta
