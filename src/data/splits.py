"""10-fold outer splits + inner validation splits (F1-F4), and the checks every later stage must run before training.

Outer: StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42), stratified by label, grouped by group_id.
Inner (fold k): from the TRAINING groups of fold k, the first split of
StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42 + k) is the validation set; the test fold is never used.
D01 is split at child level (one unit per child, group = child) and the clips inherit their child's assignment."""
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.data.common import PROJECT_DIR, SPLITS_DIR, sha256_file, write_csv, write_json

N_OUTER, N_INNER, SEED = 10, 7, 42
DATASETS = {  # dataset_id -> source manifest (relative to the project dir)
    "D01_eye_tracking": "cache/D01_eye_tracking/clips_index.csv",
    "D02_facial_images_A": "cache/D02_facial_images_A/manifest.csv",
    "D04_autism_prediction": "cache/D04_autism_prediction/clean.csv",
    "D05_toddler_screening": "cache/D05_toddler_screening/clean.csv",
    "D06_early_autism": "cache/D06_early_autism/clean.csv",
}


def load_samples(ds):
    """(sample_id, group_id, label) per sample + the split unit table (unit_id, group_id, label)."""
    src = pd.read_csv(os.path.join(PROJECT_DIR, DATASETS[ds]))
    if ds == "D01_eye_tracking":
        s = pd.DataFrame({"sample_id": src["clip_id"], "group_id": src["participant_id"].map(lambda p: f"p{p:02d}"),
                          "label": src["label"]})
        units = s.groupby("group_id", as_index=False)["label"].first().assign(unit_id=lambda d: d["group_id"])
    elif ds == "D02_facial_images_A":
        s = pd.DataFrame({"sample_id": src["image_id"], "group_id": src["group_id"], "label": src["label"]})
        units = s.assign(unit_id=s["sample_id"])
    else:
        s = src[["sample_id", "group_id", "label"]].copy()
        units = s.assign(unit_id=s["sample_id"])
    if s["sample_id"].duplicated().any():
        raise ValueError(f"{ds}: duplicate sample ids")
    return s.sort_values("sample_id").reset_index(drop=True), units.sort_values("unit_id").reset_index(drop=True)


def assign(units):
    """Outer fold per unit and the validation unit set per fold."""
    u = units.reset_index(drop=True)
    y, g = u["label"].to_numpy(), u["group_id"].to_numpy()
    outer = np.zeros(len(u), int)
    sgkf = StratifiedGroupKFold(n_splits=N_OUTER, shuffle=True, random_state=SEED)
    for k, (_, te) in enumerate(sgkf.split(np.zeros(len(u)), y, g), 1):
        outer[te] = k
    val = {}
    for k in range(1, N_OUTER + 1):
        tr = np.flatnonzero(outer != k)
        inner = StratifiedGroupKFold(n_splits=N_INNER, shuffle=True, random_state=SEED + k)
        _, va = next(inner.split(np.zeros(len(tr)), y[tr], g[tr]))
        val[k] = set(u["unit_id"].to_numpy()[tr[va]])
    return dict(zip(u["unit_id"], outer)), val


def make(ds, out_root=SPLITS_DIR):
    samples, units = load_samples(ds)
    unit_fold, unit_val = assign(units)
    unit_of = samples["group_id"] if ds == "D01_eye_tracking" else samples["sample_id"]
    samples["outer_fold"] = unit_of.map(unit_fold).astype(int)
    out = os.path.join(out_root, ds)
    os.makedirs(out, exist_ok=True)
    write_csv(samples[["sample_id", "group_id", "label", "outer_fold"]], os.path.join(out, "folds.csv"))
    rows = []
    for k in range(1, N_OUTER + 1):
        is_test = samples["outer_fold"] == k
        is_val = ~is_test & unit_of.isin(unit_val[k])
        is_train = ~is_test & ~is_val
        parts = {"train": samples[is_train], "val": samples[is_val], "test": samples[is_test]}
        counts = {p: {"n_samples": len(d), "n_groups": int(d["group_id"].nunique()),
                      "n_label_0": int((d["label"] == 0).sum()), "n_label_1": int((d["label"] == 1).sum())}
                  for p, d in parts.items()}
        write_json({"dataset_id": ds, "fold": k, "counts": counts,
                    **{p: sorted(d["sample_id"].tolist()) for p, d in parts.items()}},
                   os.path.join(out, f"fold_{k:02d}.json"))
        r = {"fold": k}
        for p in ["train", "val", "test"]:
            r.update({f"n_{p}_samples": counts[p]["n_samples"], f"n_{p}_groups": counts[p]["n_groups"],
                      f"{p}_label_0": counts[p]["n_label_0"], f"{p}_label_1": counts[p]["n_label_1"]})
        rows.append(r)
    write_csv(pd.DataFrame(rows), os.path.join(out, "split_summary.csv"))
    return out


def split_files(ds, root=SPLITS_DIR):
    return ["folds.csv", "split_summary.csv"] + [f"fold_{k:02d}.json" for k in range(1, N_OUTER + 1)]


def write_manifest(code_hash, created, root=SPLITS_DIR):
    man = {"created": created, "code_hash_at_split_time": code_hash,
           "code_hash_definition": "sha256 of sorted '<sha256>  <size>  <path>' lines of src/ scripts/ configs/ tests/ "
                                   "(scripts/make_snapshot.py code_hash()); replaces a git commit hash",
           "outer": f"StratifiedGroupKFold(n_splits={N_OUTER}, shuffle=True, random_state={SEED})",
           "inner": f"first split of StratifiedGroupKFold(n_splits={N_INNER}, shuffle=True, random_state={SEED}+k) "
                    "on the training groups of fold k",
           "datasets": {}}
    for ds, src in DATASETS.items():
        man["datasets"][ds] = {
            "source_manifest": {"path": src, "sha256": sha256_file(os.path.join(PROJECT_DIR, src))},
            "files": {f"splits/{ds}/{f}": sha256_file(os.path.join(root, ds, f)) for f in split_files(ds)}}
    write_json(man, os.path.join(root, "SPLITS_MANIFEST.json"))
    return man


def verify_manifest(root=SPLITS_DIR):
    """Every later stage calls this before training: raises if any split file or source manifest hash differs."""
    with open(os.path.join(root, "SPLITS_MANIFEST.json")) as fh:
        man = json.load(fh)
    bad = []
    for ds, e in man["datasets"].items():
        if sha256_file(os.path.join(PROJECT_DIR, e["source_manifest"]["path"])) != e["source_manifest"]["sha256"]:
            bad.append(e["source_manifest"]["path"])
        for rel, h in e["files"].items():
            if sha256_file(os.path.join(root, os.path.relpath(rel, "splits"))) != h:
                bad.append(rel)
    if bad:
        raise RuntimeError(f"split / manifest hash mismatch: {bad}")
    return man


def load_fold(ds, k, root=SPLITS_DIR):
    with open(os.path.join(root, ds, f"fold_{k:02d}.json")) as fh:
        return json.load(fh)
