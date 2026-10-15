"""Stage 3 - profile every model x mode on the GPU (F), record pretrained weights (C) and frozen-feature diversity.
Outputs: reports/stage03_model_profile.csv, reports/stage03_weights.csv, reports/stage03_feature_diversity.csv;
writes batch_size.<mode> into configs/models/Mxx.yaml. No training on real labels (random inputs only).
Log: logs/stage03_profile.log.  Usage: .venv/bin/python scripts/03_profile_models.py"""
import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402
from huggingface_hub import try_to_load_from_cache  # noqa: E402
from huggingface_hub.constants import HF_HUB_CACHE  # noqa: E402

from src.data.common import PROJECT_DIR, SPLITS_DIR, get_logger, md_table, sha256_file, write_csv  # noqa: E402
from src.models.factory import MODEL_IDS, MODES, build_model, count_params, load_model_cfg, param_groups  # noqa: E402
from src.models.feature_cache import CLIPS_NPY, compute_features  # noqa: E402

BATCHES = [64, 32, 16, 8, 4]
MEM_LIMIT = 10e9          # bytes ("under ~10 GB" on the 12 GB card)
EPOCHS_FOR_ESTIMATE = 40  # ASSUMPTION for the time estimate only; the real epoch budget is set in Stage 4
WARMUP, TIMED = 5, 20
DATASET_OF_MODE = {"sequence": "D01_eye_tracking", "tokens": "D02_facial_images_A"}


def rand_input(mode, b):
    shape = (b, 8, 3, 224, 224) if mode == "sequence" else (b, 3, 224, 224)
    return torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")


def train_step_fn(m, fwd, inp, y):
    opt = torch.optim.AdamW(param_groups(m), weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = F.binary_cross_entropy_with_logits(fwd(inp)["logit"].float(), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    return step


def timed(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(TIMED):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def mean_train_samples(ds):
    return float(pd.read_csv(os.path.join(SPLITS_DIR, ds, "split_summary.csv"))["n_train_samples"].mean())


def profile(mid, mode, log):
    torch.backends.cudnn.benchmark = True
    m = build_model(mid, mode).cuda().train()
    total, trainable = count_params(m)
    row = {"model_id": mid, "mode": mode, "dataset": DATASET_OF_MODE[mode], "policy": m.spatial.policy,
           "seq_len": m.seq_len, "params_total": total, "params_trainable": trainable}
    chosen = None
    for bs in BATCHES:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            x, y = rand_input(mode, bs), torch.randint(0, 2, (bs,), device="cuda").float()
            train_step_fn(m, m, x, y)()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
        except torch.cuda.OutOfMemoryError:
            peak = float("inf")
        log.info(f"  {mid} {mode} bs={bs}: peak {peak / 1e9:.2f} GB")
        if peak < MEM_LIMIT:
            chosen = bs
            break
        torch.cuda.empty_cache()
    if chosen is None:
        row.update({"batch_size": None, "status": "BLOCKED: no batch size under 10 GB"})
        return row, m
    x, y = rand_input(mode, chosen), torch.randint(0, 2, (chosen,), device="cuda").float()
    torch.cuda.reset_peak_memory_stats()
    s_train = timed(train_step_fn(m, m, x, y))
    peak = torch.cuda.max_memory_allocated()
    m.eval()

    def infer():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            m(x)
    s_inf = timed(infer)
    m.train()
    steps = math.ceil(mean_train_samples(DATASET_OF_MODE[mode]) / chosen)
    row.update({"batch_size": chosen, "peak_mem_gb": round(peak / 1e9, 3), "s_per_train_step": round(s_train, 4),
                "s_per_infer_batch": round(s_inf, 4), "steps_per_epoch": steps,
                "est_fold_hours_live": round(steps * s_train * EPOCHS_FOR_ESTIMATE / 3600, 3),
                "est_fold_min_live": round(steps * s_train * EPOCHS_FOR_ESTIMATE / 60, 2)})
    if m.spatial.policy == "frozen":   # head-only step on cached features (Stage 4 path for D01)
        f = torch.randn(chosen, m.seq_len, m.spatial.out_dim, device="cuda")
        torch.cuda.reset_peak_memory_stats()
        s_head = timed(train_step_fn(m, m.forward_features, f, y))
        row.update({"s_per_train_step_cached": round(s_head, 5),
                    "peak_mem_gb_cached": round(torch.cuda.max_memory_allocated() / 1e9, 3),
                    "est_fold_hours_cached": round(steps * s_head * EPOCHS_FOR_ESTIMATE / 3600, 4),
                    "est_fold_min_cached": round(steps * s_head * EPOCHS_FOR_ESTIMATE / 60, 3)})
    row["status"] = "ok"
    return row, m


def weights_table():
    rows = {}
    for mid in MODEL_IDS:
        enc = load_model_cfg(mid)["encoder"]
        if enc["type"] != "timm":
            rows.setdefault("custom_cnn", {"timm_model": "(none: custom CNN, random init, trained from scratch)",
                                           "pretrained_tag": "", "hf_hub_id": "", "file": "", "size_bytes": "",
                                           "sha256": "", "models": []})["models"].append(mid)
            continue
        name = enc["timm_name"]
        if name not in rows:
            arch, tag = name.split(".", 1)
            hub = f"timm/{name}"
            path = try_to_load_from_cache(hub, "model.safetensors")
            if not isinstance(path, str):
                raise SystemExit(f"weights for {name} not found in the HF cache - STOP (no random fallback)")
            rows[name] = {"timm_model": arch, "pretrained_tag": tag, "hf_hub_id": hub,
                          "size_bytes": os.path.getsize(path), "sha256": sha256_file(path), "models": [],
                          "file": os.path.relpath(path, HF_HUB_CACHE)}  # relative to the HF hub cache
        rows[name]["models"].append(mid)
    df = pd.DataFrame(rows.values())
    df["models"] = df["models"].map(" ".join)
    return df[["models", "timm_model", "pretrained_tag", "hf_hub_id", "size_bytes", "sha256", "file"]]


def feature_diversity(log):
    """Mean cosine similarity of frozen features between 64 real D01 clips (no labels) vs 64 random-noise inputs."""
    clips = np.load(CLIPS_NPY, mmap_mode="r")
    sel = np.sort(np.random.default_rng(42).choice(len(clips), 64, replace=False))
    real = clips[sel]
    noise = np.random.default_rng(42).integers(0, 256, real.shape, dtype=np.uint8)
    out, done = [], set()
    for mid in MODEL_IDS:
        cfg = load_model_cfg(mid)
        if cfg["encoder"]["type"] != "timm" or cfg["encoder"]["timm_name"] in done:
            continue
        done.add(cfg["encoder"]["timm_name"])
        m = build_model(mid, "sequence").cuda()
        r = {"timm_name": cfg["encoder"]["timm_name"]}
        for name, arr in [("real_D01", real), ("random_noise", noise)]:
            f = torch.from_numpy(compute_features(m, arr)).float()
            v = F.normalize(f.mean(1), dim=1)
            fr = F.normalize(f.flatten(0, 1), dim=1)
            r[f"{name}_cos_between_clips"] = round(float(((v @ v.T).sum() - 64) / (64 * 63)), 4)
            r[f"{name}_cos_between_frames"] = round(float(((fr @ fr.T).sum() - 512) / (512 * 511)), 4)
        out.append(r)
        del m
        torch.cuda.empty_cache()
    df = pd.DataFrame(out)
    log.info("frozen feature diversity (mean cosine; lower = more distinguishable):\n" + md_table(df))
    return df


def main():
    log = get_logger("stage03_profile")
    torch.manual_seed(42)
    w = weights_table()
    write_csv(w, os.path.join(PROJECT_DIR, "reports", "stage03_weights.csv"))
    log.info("weights:\n" + md_table(w.drop(columns="file")))
    write_csv(feature_diversity(log), os.path.join(PROJECT_DIR, "reports", "stage03_feature_diversity.csv"))

    rows = []
    for mid in MODEL_IDS:
        for mode in MODES:
            t = time.time()
            row, m = profile(mid, mode, log)
            rows.append(row)
            log.info(f"{mid} {mode}: {row} ({time.time() - t:.0f} s)")
            del m
            torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    write_csv(df, os.path.join(PROJECT_DIR, "reports", "stage03_model_profile.csv"))
    log.info("profile:\n" + md_table(df))

    for mid in MODEL_IDS:
        p = os.path.join(PROJECT_DIR, "configs", "models", f"{mid}.yaml")
        with open(p) as fh:
            header = fh.readline()
            cfg = yaml.safe_load(header + fh.read())
        for mode in MODES:
            cfg["batch_size"][mode] = int(df[(df.model_id == mid) & (df["mode"] == mode)]["batch_size"].iloc[0])
        with open(p, "w") as fh:
            fh.write(header)
            yaml.safe_dump(cfg, fh, sort_keys=False)
    log.info("batch sizes written to configs/models/Mxx.yaml")


if __name__ == "__main__":
    main()
