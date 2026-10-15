"""Training engine: one (dataset, model, fold) run = train_one_fold(). Protocol: configs/training.yaml (fixed in Stage 4).

Fold folder <out_root>/<dataset>/<model>/fold_XX/:
  config.yaml          exact merged config (configs/models/Mxx.yaml + configs/training.yaml + run overrides)
  history.csv          epoch, train_loss, val_loss, val_auc, lr, seconds, peak_gpu_gb (+ val_auc_child for D01)
  last.pt              resume checkpoint, written every epoch (deleted when the run is DONE)
  best_model.pt        model state_dict at the best validation AUC
  predictions.csv      test fold: sample_id, group_id, y_true, y_prob, fold (best checkpoint, predicted once)
  val_predictions.csv  validation part, same columns (used for the Youden threshold)
  attention.npy        test attention weights (N_test, L), rows in predictions.csv order (attention models only)
  metrics.json         test metrics at 0.5 and at the validation-chosen Youden threshold (+ child level for D01)
  run_info.json        status, times, versions, hashes, batch counts, peak memory
Resume: if last.pt exists and the registry status is RUNNING, training continues from it exactly (same config hash
required). Ctrl-C / SIGINT leaves the status RUNNING; any other exception marks the run FAILED."""
import copy
import datetime
import hashlib
import json
import logging
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
import yaml

from src.data.common import LOG_DIR, PROJECT_DIR, SPLITS_DIR, sha256_file, write_csv, write_json
from src.data.splits import load_fold, verify_manifest
from src.models.factory import build_model, load_model_cfg, param_groups
from src.training import registry
from src.training.data import ALIASES, D01, build_datasets, default_num_workers, make_loader, sample_table
from src.training.metrics import binary_metrics, child_level, youden_threshold

TRAINING_YAML = os.path.join(PROJECT_DIR, "configs", "training.yaml")
WEIGHTS_CSV = os.path.join(PROJECT_DIR, "reports", "stage03_weights.csv")
SPLIT_CODES = {"train": 0, "val": 1, "test": 2}


class LeakageError(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------- config
def load_training_cfg(path=TRAINING_YAML):
    with open(path) as fh:
        return yaml.safe_load(fh)


def _set_dotted(d, key, value):
    parts = key.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def merge_config(dataset, model_id, fold, overrides=None, training_yaml=TRAINING_YAML):
    """Model config + training protocol -> the exact config of one run. overrides: {"training.max_epochs": 2, ...}."""
    dataset = ALIASES[dataset]
    t, m = load_training_cfg(training_yaml), load_model_cfg(model_id)
    mode = t["data"][dataset]["mode"]
    custom = m["encoder"]["type"] == "custom_cnn"
    policy = m["policy"][mode]
    head = t["lr"]["head_lr_custom_cnn" if custom else "head_lr_pretrained"]
    factor = {"frozen": 0.0, "last_stage": t["lr"]["backbone_lr_factor_last_stage"],
              "full": t["lr"]["backbone_lr_factor_full"]}[policy]
    cfg = copy.deepcopy(m)
    cfg["seed"] = int(t["seed_base"]) + int(fold)   # also seeds the head initialisation in build_model
    cfg["lr"] = {"head_lr": head, "backbone_lr": head * factor}
    cfg["training"] = {k: copy.deepcopy(v) for k, v in t.items() if k != "data"}
    cfg["data"] = {dataset: copy.deepcopy(t["data"][dataset])}
    cfg["run"] = {"dataset": dataset, "model_id": model_id, "fold": int(fold), "mode": mode, "policy": policy,
                  "batch_size": int(m["batch_size"][mode]),
                  "input": ("cached_features" if dataset == D01 and policy == "frozen"
                            else "clips" if dataset == D01 else "images")}
    for k, v in (overrides or {}).items():
        _set_dotted(cfg, k, v)
    return cfg


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()


# ---------------------------------------------------------------------------------------------- leakage checks
def check_fold(ds, fold, fold_split, table):
    """Raises LeakageError: duplicate ids, id or group shared between train/val/test, unknown ids, a split without
    both classes, or a test set that differs from folds.csv outer_fold == fold."""
    parts = {p: list(fold_split[p]) for p in ["train", "val", "test"]}
    for p, ids in parts.items():
        if len(ids) != len(set(ids)):
            raise LeakageError(f"{ds} fold {fold}: duplicate sample ids in {p}")
        unknown = set(ids) - set(table.index)
        if unknown:
            raise LeakageError(f"{ds} fold {fold}: {len(unknown)} unknown ids in {p}")
    groups = {p: set(table.loc[ids, "group_id"]) for p, ids in parts.items()}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        si = set(parts[a]) & set(parts[b])
        if si:
            raise LeakageError(f"{ds} fold {fold}: {len(si)} sample ids in both {a} and {b}, e.g. {sorted(si)[:3]}")
        gi = groups[a] & groups[b]
        if gi:
            raise LeakageError(f"{ds} fold {fold}: {len(gi)} groups in both {a} and {b}, e.g. {sorted(gi)[:3]}")
    for p, ids in parts.items():
        if table.loc[ids, "label"].nunique() != 2:
            raise LeakageError(f"{ds} fold {fold}: {p} does not contain both classes")
    expected_test = set(table.index[table["outer_fold"] == fold])
    if set(parts["test"]) != expected_test:
        raise LeakageError(f"{ds} fold {fold}: test ids differ from folds.csv outer_fold == {fold}")
    return {p: {"n_samples": len(ids), "n_groups": len(groups[p])} for p, ids in parts.items()}


def assert_no_test_ids(seen, test_ids):
    """seen: {'train': set, 'val': set} of sample ids that were actually put in a batch."""
    for split, ids in seen.items():
        leak = set(ids) & set(test_ids)
        if leak:
            raise LeakageError(f"{len(leak)} test ids were passed to a {split} batch, e.g. {sorted(leak)[:3]}")


# ---------------------------------------------------------------------------------------------- environment info
def code_hash(dirs=("src",), root=PROJECT_DIR):
    """Same definition as scripts/make_snapshot.code_hash(): sha256 of sorted '<sha256>  <size>  <path>' lines."""
    lines = []
    for d in dirs:
        for dp, dns, fns in os.walk(os.path.join(root, d)):
            dns[:] = sorted(x for x in dns if x not in {"__pycache__", ".pytest_cache"})
            for fn in fns:
                if fn.endswith(".pyc"):
                    continue
                rel = os.path.relpath(os.path.join(dp, fn), root)
                lines.append((rel, f"{sha256_file(os.path.join(root, rel))}  {os.path.getsize(os.path.join(root, rel))}  {rel}"))
    return hashlib.sha256(("\n".join(l for _, l in sorted(lines)) + "\n").encode()).hexdigest()


def gpu_info():
    r = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True,
                       text=True)
    return {"gpu_name": torch.cuda.get_device_name(0), "driver": r.stdout.strip().splitlines()[0] if r.stdout else
            "UNKNOWN", "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "torch": torch.__version__,
            "timm": timm.__version__, "python": platform.python_version()}


def weights_info(model):
    if model.spatial.timm_name is None:
        return {"timm_name": None, "note": "custom CNN, random init (seeded), trained from scratch"}
    w = pd.read_csv(WEIGHTS_CSV, dtype=str, keep_default_na=False)
    row = w[w.hf_hub_id == f"timm/{model.spatial.timm_name}"].iloc[0]
    from huggingface_hub import try_to_load_from_cache
    path = try_to_load_from_cache(row.hf_hub_id, "model.safetensors")
    actual = sha256_file(path) if isinstance(path, str) else "NOT_IN_CACHE"
    if actual != row.sha256:
        raise RuntimeError(f"pretrained weights {row.hf_hub_id}: sha256 {actual} != stage03_weights.csv {row.sha256}")
    return {"timm_name": model.spatial.timm_name, "sha256": actual, "verified_against": "reports/stage03_weights.csv"}


# ---------------------------------------------------------------------------------------------- helpers
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all()}


def set_rng_state(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    torch.cuda.set_rng_state_all(s["cuda"])


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def make_scheduler(opt, steps_per_epoch, max_epochs, warmup_epochs):
    warm = max(1, int(warmup_epochs * steps_per_epoch))
    total = max(warm + 1, int(max_epochs * steps_per_epoch))

    def f(step):  # step = number of optimizer steps already taken
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, (step - warm) / (total - warm))))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def default_log_path(out_root, ds, model_id, fold):
    """logs/<ds>/<model>/fold_XX.log for results/; logs/<out_root relative to the project>/<ds>/... otherwise."""
    rel = os.path.relpath(os.path.abspath(out_root), PROJECT_DIR)
    sub = [] if rel == "results" else [os.path.basename(rel)] if rel.startswith("..") else rel.split(os.sep)
    return os.path.join(LOG_DIR, *sub, ds, model_id, f"fold_{fold:02d}.log")


def _logger(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    log = logging.getLogger(f"fold:{path}")
    log.handlers.clear()
    log.setLevel(logging.INFO)
    log.propagate = False
    h = logging.FileHandler(path, mode="a")
    h.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S"))
    log.addHandler(h)
    return log


def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def forward(model, x, cached):
    return model.forward_features(x) if cached else model(x)


@torch.no_grad()
def predict(model, dataset, loader, cached, pos_weight, want_attn=False):
    model.eval()
    probs, losses, idx, attn, n_batches = [], [], [], [], 0
    for x, y, i in loader:
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            out = forward(model, x, cached)
        logit = out["logit"].float()
        losses.append(F.binary_cross_entropy_with_logits(logit, y, pos_weight=pos_weight, reduction="sum").item())
        probs.append(torch.sigmoid(logit).double().cpu().numpy())
        idx.append(i.numpy())
        if want_attn and out["attn"] is not None:
            attn.append(out["attn"].float().cpu().numpy())
        n_batches += 1
    idx = np.concatenate(idx)
    return {"ids": [dataset.ids[j] for j in idx], "idx": idx, "y": dataset.y.numpy()[idx].astype(int),
            "groups": dataset.groups[idx], "prob": np.concatenate(probs), "loss": sum(losses) / len(idx),
            "attn": np.concatenate(attn) if attn else None, "n_batches": n_batches}


def _pred_frame(p, fold):
    return pd.DataFrame({"sample_id": p["ids"], "group_id": p["groups"], "y_true": p["y"], "y_prob": p["prob"],
                         "fold": fold})


# ---------------------------------------------------------------------------------------------- main entry
def train_one_fold(dataset_id, model_id, fold, cfg=None, out_dir=None, out_root=None, force=False, subset=None,
                   log_path=None, stop_after_epoch=None, echo=True):
    """Train / resume / skip one fold. Returns a dict with the final status. subset and stop_after_epoch are for tests
    only (subset = first n ids per class per split; stop_after_epoch raises KeyboardInterrupt after that epoch's
    checkpoint, i.e. a simulated kill)."""
    ds = ALIASES[dataset_id]
    fold = int(fold)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available - STOP (golden rule 10: never fall back to CPU)")
    cfg = cfg or merge_config(ds, model_id, fold)
    out_root = out_root or os.path.join(PROJECT_DIR, "results")
    out_dir = out_dir or os.path.join(out_root, ds, model_id, f"fold_{fold:02d}")
    log = _logger(log_path or default_log_path(out_root, ds, model_id, fold))
    say = (lambda s: (log.info(s), print(s, flush=True))) if echo else log.info

    # 1. splits: hashes + leakage asserts --------------------------------------------------------------------
    verify_manifest()
    fold_split = load_fold(ds, fold)
    table = sample_table(ds)
    split_counts = check_fold(ds, fold, fold_split, table)
    test_ids = set(fold_split["test"])

    # 2. registry: skip / resume / fresh ---------------------------------------------------------------------
    run, reason = registry.should_run(out_root, ds, model_id, fold, force)
    if not run:
        say(f"[{ds} {model_id} fold {fold}] skip: {reason}")
        return {"status": "DONE", "skipped": True}
    status = registry.get_status(out_root, ds, model_id, fold)
    last_pt = os.path.join(out_dir, "last.pt")
    resume = os.path.isfile(last_pt) and status == "RUNNING" and not force
    if not resume and os.path.isdir(out_dir):
        shutil.rmtree(out_dir)  # stale / failed / forced run of this fold only
    os.makedirs(out_dir, exist_ok=True)
    chash = config_hash(cfg)
    session = {"start": _now(), "t0": time.time()}

    T = cfg["training"]
    seed = int(cfg["seed"])
    torch.backends.cudnn.benchmark = bool(T["cudnn_benchmark"]) and not T["deterministic"]
    torch.backends.cudnn.deterministic = bool(T["deterministic"])
    set_seeds(seed)
    run_info = {}
    try:
        model = build_model(model_id, cfg["run"]["mode"], cfg=cfg).cuda()
        cached = cfg["run"]["input"] == "cached_features"
        parts = build_datasets(ds, fold_split, cfg, model, subset=subset)
        y_tr = parts["train"].y
        n_pos, n_neg = int(y_tr.sum()), int(len(y_tr) - y_tr.sum())
        pos_weight = torch.tensor(n_neg / n_pos, device="cuda")
        bs = cfg["run"]["batch_size"]
        nw = default_num_workers() if T["num_workers"] == "auto" else int(T["num_workers"])
        max_epochs = int(T["max_epochs"])
        steps_per_epoch = math.ceil(len(parts["train"]) / bs)
        opt = torch.optim.AdamW(param_groups(model, cfg), weight_decay=T["optimizer"]["weight_decay"])
        sched = make_scheduler(opt, steps_per_epoch, max_epochs, T["schedule"]["warmup_epochs"])
        scaler = torch.amp.GradScaler("cuda", enabled=bool(T["amp"]))
        trainable = [p for p in model.parameters() if p.requires_grad]

        state = {"epoch": 0, "best_auc": -math.inf, "best_epoch": 0, "bad_epochs": 0, "stopped": False,
                 "history": [], "seen": {"train": set(), "val": set()}, "batches": {"train": 0, "val": 0, "test": 0},
                 "sessions": [], "start_time": session["start"]}
        if resume:
            ck = torch.load(last_pt, map_location="cpu", weights_only=False)  # RNG states must stay CPU tensors
            if ck["config_hash"] != chash:
                raise RuntimeError("last.pt was written with a different config - refusing to resume (use --force)")
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["optimizer"])
            sched.load_state_dict(ck["scheduler"])
            scaler.load_state_dict(ck["scaler"])
            state = ck["state"]
            set_rng_state(ck["rng"])
            say(f"[{ds} {model_id} fold {fold}] RESUME from last.pt: epoch {state['epoch']} done, continuing at epoch "
                f"{state['epoch'] + 1} (best val AUC {state['best_auc']:.4f} @ {state['best_epoch']}, "
                f"patience {state['bad_epochs']}/{T['early_stopping']['patience']})")
        else:
            with open(os.path.join(out_dir, "config.yaml"), "w") as fh:
                yaml.safe_dump(cfg, fh, sort_keys=False)
            say(f"[{ds} {model_id} fold {fold}] START ({reason}) input={cfg['run']['input']} policy="
                f"{cfg['run']['policy']} n_train={len(parts['train'])} n_val={len(parts['val'])} "
                f"n_test={len(parts['test'])} pos_weight={n_neg / n_pos:.4f} bs={bs} max_epochs={max_epochs}")
        session["start_epoch"] = state["epoch"] + 1

        run_info = {"dataset": ds, "model_id": model_id, "fold": fold, "status": "RUNNING",
                    "start_time": state["start_time"], "seed": seed, "config_sha256": chash,
                    "code_hash_src": code_hash(("src",)), **gpu_info(), "weights": weights_info(model),
                    "split_files": {f"splits/{ds}/fold_{fold:02d}.json":
                                    sha256_file(os.path.join(SPLITS_DIR, ds, f"fold_{fold:02d}.json")),
                                    f"splits/{ds}/folds.csv": sha256_file(os.path.join(SPLITS_DIR, ds, "folds.csv")),
                                    "splits/SPLITS_MANIFEST.json":
                                    sha256_file(os.path.join(SPLITS_DIR, "SPLITS_MANIFEST.json"))},
                    "split_counts": split_counts, "subset_per_class": subset,
                    "n_used": {p: len(d) for p, d in parts.items()},
                    "test_ids_sha256": hashlib.sha256("\n".join(sorted(test_ids)).encode()).hexdigest(),
                    "pos_weight": n_neg / n_pos, "batch_size": bs, "num_workers": nw, "max_epochs": max_epochs,
                    "cudnn_benchmark": torch.backends.cudnn.benchmark,
                    "cudnn_deterministic": torch.backends.cudnn.deterministic,
                    "nondeterminism_note": ("cudnn.benchmark=True: cuDNN may pick different algorithms between "
                                            "processes, so results can differ in the last digits between runs"
                                            if torch.backends.cudnn.benchmark else "deterministic cuDNN settings"),
                    "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE", "0")}
        if cached:
            run_info["feature_cache"] = {**parts["train"].cache_meta, "sha256": sha256_file(
                os.path.join(PROJECT_DIR, "cache", "features", f"{model.spatial.timm_name}_D01.npy"))}
        write_json(run_info, os.path.join(out_dir, "run_info.json"))
        registry.update(out_root, ds, model_id, fold, status="RUNNING", start=state["start_time"], error="")

        # 3. epochs -----------------------------------------------------------------------------------------------
        es = T["early_stopping"]
        warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step\\(\\)` before")
        for epoch in range(state["epoch"] + 1, max_epochs + 1):
            if state["stopped"]:
                break
            t0 = time.time()
            torch.cuda.reset_peak_memory_stats()
            model.train()
            tl, tn = 0.0, 0
            loader = make_loader(parts["train"], bs, True, seed, epoch, SPLIT_CODES["train"], nw, T["pin_memory"])
            for x, y, i in loader:
                state["seen"]["train"].update(parts["train"].ids[j] for j in i.tolist())
                state["batches"]["train"] += 1
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16, enabled=bool(T["amp"])):
                    logit = forward(model, x, cached)["logit"].float()
                loss = F.binary_cross_entropy_with_logits(logit, y, pos_weight=pos_weight)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable, T["grad_clip_norm"])
                scaler.step(opt)
                scaler.update()
                sched.step()
                tl += loss.item() * len(y)
                tn += len(y)
            vloader = make_loader(parts["val"], bs, False, seed, epoch, SPLIT_CODES["val"], nw, T["pin_memory"])
            v = predict(model, parts["val"], vloader, cached, pos_weight)
            state["seen"]["val"].update(v["ids"])
            state["batches"]["val"] += v["n_batches"]
            vm = binary_metrics(v["y"], v["prob"])
            row = {"epoch": epoch, "train_loss": tl / tn, "val_loss": v["loss"], "val_auc": vm["roc_auc"],
                   "lr": opt.param_groups[-1]["lr"], "seconds": None,
                   "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9}
            if ds == D01:
                _, cy, cp = child_level(v["groups"], v["y"], v["prob"])
                row["val_auc_child"] = binary_metrics(cy, cp)["roc_auc"]
            improved = (not math.isnan(vm["roc_auc"])) and vm["roc_auc"] > state["best_auc"] + es["min_delta"]
            if improved:
                state.update(best_auc=vm["roc_auc"], best_epoch=epoch, bad_epochs=0)
                atomic_save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))
            else:
                state["bad_epochs"] += 1
            state["stopped"] = state["bad_epochs"] >= es["patience"]
            state["epoch"] = epoch
            row["seconds"] = time.time() - t0
            state["history"].append(row)
            write_csv(pd.DataFrame(state["history"]), os.path.join(out_dir, "history.csv"))
            atomic_save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                         "scaler": scaler.state_dict(), "state": state, "rng": rng_state(), "config_hash": chash},
                        last_pt)
            say(f"[{ds} {model_id} fold {fold}] epoch {epoch}/{max_epochs} train_loss {row['train_loss']:.4f} "
                f"val_loss {row['val_loss']:.4f} val_auc {row['val_auc']:.4f}"
                + (f" val_auc_child {row['val_auc_child']:.4f}" if ds == D01 else "")
                + f" lr {row['lr']:.2e} {row['seconds']:.1f}s peak {row['peak_gpu_gb']:.2f}GB"
                + (" *best*" if improved else f" (no improvement {state['bad_epochs']}/{es['patience']})")
                + (" -> early stop" if state["stopped"] else ""))
            if stop_after_epoch is not None and epoch == stop_after_epoch:
                raise KeyboardInterrupt(f"simulated kill after epoch {epoch}")

        # 4. test once with the best checkpoint ------------------------------------------------------------------
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        model.load_state_dict(torch.load(os.path.join(out_dir, "best_model.pt"), map_location="cuda",
                                         weights_only=True))
        mk = lambda s: make_loader(parts[s], bs, False, seed, 0, SPLIT_CODES[s], nw, T["pin_memory"])  # noqa: E731
        v = predict(model, parts["val"], mk("val"), cached, pos_weight)
        state["seen"]["val"].update(v["ids"])
        state["batches"]["val"] += v["n_batches"]
        assert_no_test_ids(state["seen"], test_ids)
        if not state["seen"]["train"] <= set(parts["train"].ids) or not state["seen"]["val"] <= set(parts["val"].ids):
            raise LeakageError("a batch contained an id outside its split")
        te = predict(model, parts["test"], mk("test"), cached, pos_weight, want_attn=True)
        state["batches"]["test"] += te["n_batches"]
        if set(te["ids"]) != set(parts["test"].ids) or len(te["ids"]) != len(parts["test"]):
            raise LeakageError("test predictions do not cover the test ids exactly once")
        thr = youden_threshold(v["y"], v["prob"])  # validation only
        metrics = {"dataset": ds, "model_id": model_id, "fold": fold, "best_epoch": state["best_epoch"],
                   "val_threshold_youden": thr,
                   "test": {"threshold_0.5": binary_metrics(te["y"], te["prob"], 0.5),
                            "threshold_val_youden": binary_metrics(te["y"], te["prob"], thr)},
                   "val_at_best": {"threshold_0.5": binary_metrics(v["y"], v["prob"], 0.5)}}
        if ds == D01:
            _, vcy, vcp = child_level(v["groups"], v["y"], v["prob"])
            _, tcy, tcp = child_level(te["groups"], te["y"], te["prob"])
            cthr = youden_threshold(vcy, vcp)  # validation children only
            metrics["val_threshold_youden_child"] = cthr
            metrics["test_child"] = {"threshold_0.5": binary_metrics(tcy, tcp, 0.5),
                                     "threshold_val_youden": binary_metrics(tcy, tcp, cthr)}
            metrics["val_at_best_child"] = {"threshold_0.5": binary_metrics(vcy, vcp, 0.5)}
        write_csv(_pred_frame(te, fold), os.path.join(out_dir, "predictions.csv"))
        write_csv(_pred_frame(v, fold), os.path.join(out_dir, "val_predictions.csv"))
        if te["attn"] is not None:
            np.save(os.path.join(out_dir, "attention.npy"), te["attn"].astype(np.float32))
        write_json(metrics, os.path.join(out_dir, "metrics.json"))

        # 5. finish ------------------------------------------------------------------------------------------------
        hist = pd.DataFrame(state["history"])
        peak = max(float(hist["peak_gpu_gb"].max()), torch.cuda.max_memory_allocated() / 1e9)
        session.update(end=_now(), end_epoch=state["epoch"], seconds=time.time() - session["t0"])
        state["sessions"].append({k: session[k] for k in ["start", "end", "start_epoch", "end_epoch", "seconds"]})
        seconds = sum(s["seconds"] for s in state["sessions"])
        run_info.update(status="DONE", end_time=session["end"], seconds=seconds, epochs_run=state["epoch"],
                        best_epoch=state["best_epoch"], best_val_auc=state["best_auc"],
                        stopped_early=state["stopped"], peak_gpu_gb=peak, test_seconds=time.time() - t0,
                        sessions=state["sessions"], resumed=len(state["sessions"]) > 1 or resume,
                        batch_counts=state["batches"],
                        n_ids_seen={k: len(s) for k, s in state["seen"].items()},
                        test_ids_in_train_or_val_batches=0)
        write_json(run_info, os.path.join(out_dir, "run_info.json"))
        if T["checkpoint"]["delete_last_when_done"] and os.path.isfile(last_pt):
            os.remove(last_pt)
        test_auc = metrics["test"]["threshold_0.5"]["roc_auc"]
        registry.update(out_root, ds, model_id, fold, status="DONE", end=session["end"], epochs=state["epoch"],
                        best_epoch=state["best_epoch"], val_auc=round(state["best_auc"], 6),
                        test_auc=round(test_auc, 6), peak_gpu_gb=round(peak, 3), seconds=round(seconds, 1),
                        error="")
        say(f"[{ds} {model_id} fold {fold}] DONE epochs {state['epoch']} best_epoch {state['best_epoch']} "
            f"val_auc {state['best_auc']:.4f} test_auc {test_auc:.4f} peak {peak:.2f}GB {seconds:.0f}s "
            f"batches {state['batches']} test ids in train/val batches: 0")
        return {"status": "DONE", "skipped": False, "metrics": metrics, "run_info": run_info}

    except KeyboardInterrupt as e:
        # interrupted: keep status RUNNING so the next start resumes from last.pt
        if run_info:
            if os.path.isfile(last_pt):  # remember this session's span in the checkpoint
                ck = torch.load(last_pt, map_location="cpu", weights_only=False)
                ck["state"]["sessions"].append({"start": session["start"], "end": _now(),
                                                "start_epoch": session.get("start_epoch"),
                                                "end_epoch": ck["state"]["epoch"],
                                                "seconds": time.time() - session["t0"], "interrupted": True})
                atomic_save(ck, last_pt)
            run_info.update(status="RUNNING", interrupted_at=_now())
            write_json(run_info, os.path.join(out_dir, "run_info.json"))
        registry.update(out_root, ds, model_id, fold, status="RUNNING", error=f"interrupted {_now()} {e}"[:200])
        say(f"[{ds} {model_id} fold {fold}] INTERRUPTED ({e}); status stays RUNNING, last.pt kept for resume")
        raise
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        if run_info:
            run_info.update(status="FAILED", error=err, end_time=_now())
            write_json(run_info, os.path.join(out_dir, "run_info.json"))
        registry.update(out_root, ds, model_id, fold, status="FAILED", end=_now(), error=err[:300])
        say(f"[{ds} {model_id} fold {fold}] FAILED: {err}")
        raise
