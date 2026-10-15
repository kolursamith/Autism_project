"""validate_model_results(ds, model): the gate after each model in run_queue.py (the queue STOPS on FAIL).
Checks per fold: registry + run_info status DONE; required files present; metrics.json values valid; predictions.csv
covers exactly the test ids of that fold (no duplicates, labels/groups match folds.csv, probabilities finite in
[0, 1]); history.csv non-empty with finite losses. Across folds: union of test predictions = every sample of the
checked folds exactly once (= the whole dataset when all 10 folds are checked)."""
import json
import os

import numpy as np
import pandas as pd

from src.data.common import SPLITS_DIR
from src.data.splits import load_fold
from src.models.factory import load_model_cfg
from src.training import registry
from src.training.data import ALIASES
from src.training.metrics import RANGE_TOL, is_valid_metric_block

REQUIRED = ["config.yaml", "metrics.json", "predictions.csv", "val_predictions.csv", "history.csv", "best_model.pt",
            "run_info.json"]


def validate_model_results(ds, model, out_root, folds=range(1, 11), return_flags=False):
    """Returns ("PASS" | "FAIL", [reasons]); with return_flags=True also [flags] (explicitly reported, non-failing
    conditions such as an AUC that is undefined because a block has a single class)."""
    ds = ALIASES[ds]
    folds = [int(k) for k in folds]
    reasons, flags = [], []
    reg = registry.read_registry(out_root)
    has_attn = load_model_cfg(model)["pooling"] == "attention"
    folds_csv = pd.read_csv(os.path.join(SPLITS_DIR, ds, "folds.csv")).set_index("sample_id")
    all_pred = []
    for k in folds:
        tag = f"fold {k}"
        d = os.path.join(out_root, ds, model, f"fold_{k:02d}")
        r = reg[(reg.dataset == ds) & (reg.model == model) & (reg.fold.astype(int) == k)]
        if r.empty or r.iloc[0]["status"] != "DONE":
            reasons.append(f"{tag}: registry status {None if r.empty else r.iloc[0]['status']}")
        need = REQUIRED + (["attention.npy"] if has_attn else [])
        missing = [f for f in need if not os.path.isfile(os.path.join(d, f))]
        if missing:
            reasons.append(f"{tag}: missing {missing}")
            continue
        with open(os.path.join(d, "run_info.json")) as fh:
            info = json.load(fh)
        if info.get("status") != "DONE":
            reasons.append(f"{tag}: run_info status {info.get('status')}")
        if info.get("test_ids_in_train_or_val_batches") != 0:
            reasons.append(f"{tag}: test-id leakage counter not 0")
        with open(os.path.join(d, "metrics.json")) as fh:
            m = json.load(fh)
        if not -RANGE_TOL <= m["val_threshold_youden"] <= 1.0 + RANGE_TOL:
            reasons.append(f"{tag}: Youden threshold {m['val_threshold_youden']} outside [0, 1]")
        for b in ["test", "test_child"]:
            for thr, block in m.get(b, {}).items():
                fl = []
                reasons += [f"{tag}: {b}/{thr}: {e}" for e in is_valid_metric_block(block, flags=fl)]
                flags += [f"{tag}: {b}/{thr}: {f}" for f in fl]
        h = pd.read_csv(os.path.join(d, "history.csv"))
        if h.empty or not np.isfinite(h[["train_loss", "val_loss"]].to_numpy()).all():
            reasons.append(f"{tag}: history empty or non-finite losses")
        if len(h) != info.get("epochs_run") or not 1 <= m["best_epoch"] <= len(h):
            reasons.append(f"{tag}: epochs in history ({len(h)}) / run_info / best_epoch inconsistent")
        p = pd.read_csv(os.path.join(d, "predictions.csv"))
        if list(p.columns) != ["sample_id", "group_id", "y_true", "y_prob", "fold"]:
            reasons.append(f"{tag}: predictions.csv columns {list(p.columns)}")
            continue
        expected = set(load_fold(ds, k)["test"])
        if p.sample_id.duplicated().any() or set(p.sample_id) != expected:
            reasons.append(f"{tag}: predictions do not cover exactly the {len(expected)} test ids "
                           f"(got {len(p)}, {p.sample_id.nunique()} unique)")
        ref = folds_csv.reindex(p.sample_id)
        if (ref["label"].to_numpy() != p.y_true.to_numpy()).any() or \
                (ref["group_id"].astype(str).to_numpy() != p.group_id.astype(str).to_numpy()).any():
            reasons.append(f"{tag}: y_true / group_id do not match folds.csv")
        if not (np.isfinite(p.y_prob).all() and p.y_prob.between(-RANGE_TOL, 1.0 + RANGE_TOL).all()) or (p.fold != k).any():
            reasons.append(f"{tag}: y_prob not finite in [0, 1] or wrong fold column")
        if has_attn:
            a = np.load(os.path.join(d, "attention.npy"))
            if a.shape[0] != len(p) or not np.allclose(a.sum(1), 1.0, atol=1e-3):
                reasons.append(f"{tag}: attention.npy shape {a.shape} / rows do not sum to 1")
        all_pred.append(p)
    if all_pred:
        P = pd.concat(all_pred)
        expected_all = set(folds_csv.index[folds_csv.outer_fold.isin(folds)])
        if P.sample_id.duplicated().any() or set(P.sample_id) != expected_all:
            reasons.append(f"union of test predictions ({len(P)}) != the {len(expected_all)} samples of folds "
                           f"{folds} exactly once")
        if sorted(folds) == list(range(1, 11)) and len(expected_all) != len(folds_csv):
            reasons.append("10 folds do not cover the whole dataset")
    verdict = "PASS" if not reasons else "FAIL"
    return (verdict, reasons, flags) if return_flags else (verdict, reasons)
