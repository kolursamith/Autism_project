"""Stage 3 - pre-compute frozen-backbone per-frame features for D01 (sequence mode), one cache per backbone:
cache/features/<timm_name>_D01.npy (N, 8, C) float16 + .json provenance. Uses no labels. Log: logs/stage03_feature_cache.log
Usage: .venv/bin/python scripts/03b_build_feature_cache.py"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

from src.data.common import get_logger  # noqa: E402
from src.models.factory import MODEL_IDS, build_model, load_model_cfg  # noqa: E402
from src.models.feature_cache import build_cache  # noqa: E402


def main():
    log = get_logger("stage03_feature_cache")
    torch.backends.cudnn.benchmark = True
    done = {}
    for mid in MODEL_IDS:
        cfg = load_model_cfg(mid)
        if cfg["encoder"]["type"] != "timm" or cfg["policy"]["sequence"] != "frozen":
            continue
        name = cfg["encoder"]["timm_name"]
        if name in done:
            log.info(f"{mid}: shares the {name} cache built for {done[name]}")
            continue
        m = build_model(mid, "sequence").cuda()
        t = time.time()
        path, meta = build_cache(m)
        torch.cuda.synchronize()
        log.info(f"{mid}: {name} -> {path} shape {meta['shape']} {meta['dtype']} in {time.time() - t:.1f} s")
        done[name] = mid
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
