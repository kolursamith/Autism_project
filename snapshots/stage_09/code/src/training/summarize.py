"""Per model: <out_root>/<ds>/<model>/summary.json = mean, std (ddof=1), min, max over folds of every numeric test metric
in metrics.json, plus pooled out-of-fold metrics from all predictions.csv (each fold's predictions come from its own
model; the pooled Youden-threshold block binarises each fold with that fold's validation threshold).
Per dataset: <out_root>/<ds>/leaderboard.csv (one row per model, sorted by mean test ROC-AUC)."""
import glob
import json
import os

import numpy as np
import pandas as pd

from src.data.common import write_csv, write_json
from src.training.metrics import METRIC_KEYS, binary_metrics

BLOCKS = ["test", "test_child"]


def _flatten(metrics):
    """{'test/threshold_0.5/roc_auc': v, ...} for the numeric scalars of the test blocks + thresholds."""
    out = {k: metrics[k] for k in ["val_threshold_youden", "val_threshold_youden_child", "best_epoch"] if k in metrics}
    for b in BLOCKS:
        for thr, block in metrics.get(b, {}).items():
            for k in METRIC_KEYS + ["sensitivity"]:
                out[f"{b}/{thr}/{k}"] = block[k]
    return out


def fold_dirs(out_root, ds, model):
    return sorted(glob.glob(os.path.join(out_root, ds, model, "fold_[0-9][0-9]")))


def summarize_model(out_root, ds, model, write=True):
    rows, preds = [], []
    for d in fold_dirs(out_root, ds, model):
        mp = os.path.join(d, "metrics.json")
        if not os.path.isfile(mp):
            continue
        with open(mp) as fh:
            m = json.load(fh)
        rows.append({"fold": m["fold"], **_flatten(m)})
        p = pd.read_csv(os.path.join(d, "predictions.csv"))
        p["thr_youden"] = m["val_threshold_youden"]
        p["thr_youden_child"] = m.get("val_threshold_youden_child", np.nan)
        preds.append(p)
    if not rows:
        raise FileNotFoundError(f"no metrics.json under {out_root}/{ds}/{model}")
    df = pd.DataFrame(rows).sort_values("fold")
    per_metric = {}
    for c in df.columns.drop("fold"):
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        per_metric[c] = {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v, ddof=1)) if np.isfinite(v).sum() > 1
                         else float("nan"), "min": float(np.nanmin(v)), "max": float(np.nanmax(v)),
                         "n_folds": int(np.isfinite(v).sum())}
    P = pd.concat(preds, ignore_index=True)
    pooled = {"threshold_0.5": binary_metrics(P.y_true, P.y_prob, 0.5),
              "threshold_val_youden_per_fold": binary_metrics(P.y_true, P.y_prob,
                                                              y_pred=(P.y_prob >= P.thr_youden).astype(int))}
    pooled_child = None
    if "test_child" in metrics_keys(df):
        C = P.groupby(["fold", "group_id"], as_index=False).agg(y=("y_true", "first"), y_n=("y_true", "nunique"),
                                                                 p=("y_prob", "mean"), thr=("thr_youden_child", "first"))
        if (C.y_n != 1).any():
            raise ValueError("label not constant within a child")
        pooled_child = {"threshold_0.5": binary_metrics(C.y, C.p, 0.5),
                        "threshold_val_youden_per_fold": binary_metrics(C.y, C.p, y_pred=(C.p >= C.thr).astype(int))}
    summary = {"dataset": ds, "model_id": model, "folds": df["fold"].tolist(), "n_folds": int(len(df)),
               "std_definition": "sample standard deviation over folds (ddof=1)",
               "per_fold_metrics": per_metric, "pooled_out_of_fold": pooled}
    if pooled_child is not None:
        summary["pooled_out_of_fold_child"] = pooled_child
    if write:
        write_json(summary, os.path.join(out_root, ds, model, "summary.json"))
    return summary


def metrics_keys(df):
    return {c.split("/")[0] for c in df.columns if "/" in c}


def leaderboard(out_root, ds, write=True):
    rows = []
    for mdir in sorted(glob.glob(os.path.join(out_root, ds, "M[0-9][0-9]"))):
        model = os.path.basename(mdir)
        try:
            s = summarize_model(out_root, ds, model, write=write)
        except FileNotFoundError:
            continue
        pm = s["per_fold_metrics"]
        r = {"model_id": model, "n_folds": s["n_folds"]}
        for key, name in [("test/threshold_0.5/roc_auc", "roc_auc"), ("test/threshold_0.5/pr_auc", "pr_auc"),
                          ("test/threshold_0.5/balanced_accuracy", "bal_acc_0.5"),
                          ("test/threshold_val_youden/balanced_accuracy", "bal_acc_youden"),
                          ("test/threshold_0.5/brier", "brier"),
                          ("test_child/threshold_0.5/roc_auc", "child_roc_auc"),
                          ("test_child/threshold_0.5/balanced_accuracy", "child_bal_acc_0.5")]:
            if key in pm:
                r[f"{name}_mean"], r[f"{name}_std"] = pm[key]["mean"], pm[key]["std"]
        r["pooled_oof_roc_auc"] = s["pooled_out_of_fold"]["threshold_0.5"]["roc_auc"]
        if "pooled_out_of_fold_child" in s:
            r["pooled_oof_child_roc_auc"] = s["pooled_out_of_fold_child"]["threshold_0.5"]["roc_auc"]
        rows.append(r)
    lb = pd.DataFrame(rows)
    if len(lb):
        lb = lb.sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)
    if write and len(lb):
        write_csv(lb, os.path.join(out_root, ds, "leaderboard.csv"))
    return lb
