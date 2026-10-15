"""Stage 7 - 10-fold tabular baselines on D04, D05 (set_all = label-definitional upper bound, set_honest), D06.
CPU ONLY and at most 2 threads (runs next to the S5 GPU training). No torch import.
Outputs: results/<ds>/<experiment>/<model>/fold_XX/{config.yaml, metrics.json, predictions.csv, val_predictions.csv,
run_info.json}, results/<ds>/<experiment>/<model>/summary.json, results/<ds>/<experiment>/leaderboard.csv,
results/D05_toddler_screening/set_all/rule_comparison.csv, registry tabular/run_registry_stage7.csv,
checks tabular/stage7_checks.json. DONE runs are skipped unless --force.
Usage: CUDA_VISIBLE_DEVICES="" nice -n 10 .venv/bin/python tabular/run_stage7.py [--force]"""
import os

for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(_v, "2")          # before numpy / sklearn are imported
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse  # noqa: E402
import datetime  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
import yaml  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from src.data.common import SPLITS_DIR, sha256_file, write_csv, write_json  # noqa: E402  (read-only reuse)
from src.data.splits import load_fold, verify_manifest  # noqa: E402
from src.training.metrics import binary_metrics, is_valid_metric_block, youden_threshold  # noqa: E402
from src.training.summarize import summarize_model  # noqa: E402
from tabular.models import (MODEL_IDS, PROJECT_DIR, TAB_DIR, feature_columns, fit_pipeline, load_stage7_cfg,  # noqa: E402
                            load_table)

RESULTS = os.path.join(PROJECT_DIR, "results")
REGISTRY = os.path.join(TAB_DIR, "run_registry_stage7.csv")
REG_COLS = ["dataset", "experiment", "model", "fold", "status", "start", "end", "val_auc", "test_auc", "seconds",
            "error"]
D05_ITEMS = [f"A{i}" for i in range(1, 11)]


class LeakageError(RuntimeError):
    pass


def valid_block(b, tol=1e-9):
    """src.training.metrics.is_valid_metric_block with a floating-point tolerance: sklearn's average precision can
    return 1.0000000000000002 for a perfectly separated fold, which the strict [0, 1] check would flag."""
    clipped = {k: (min(max(v, 0.0), 1.0) if isinstance(v, float) and -tol <= v <= 1 + tol else v) for k, v in b.items()}
    return is_valid_metric_block(clipped)


def check_fold(ds, k, split, table):
    parts = {p: list(split[p]) for p in ["train", "val", "test"]}
    for p, ids in parts.items():
        if len(ids) != len(set(ids)) or set(ids) - set(table.index):
            raise LeakageError(f"{ds} fold {k}: duplicate or unknown ids in {p}")
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        if set(parts[a]) & set(parts[b]):
            raise LeakageError(f"{ds} fold {k}: sample ids shared by {a} and {b}")
        if set(table.loc[parts[a], "group_id"]) & set(table.loc[parts[b], "group_id"]):
            raise LeakageError(f"{ds} fold {k}: groups shared by {a} and {b}")
    folds_csv = pd.read_csv(os.path.join(SPLITS_DIR, ds, "folds.csv")).set_index("sample_id")
    if set(parts["test"]) != set(folds_csv.index[folds_csv.outer_fold == k]):
        raise LeakageError(f"{ds} fold {k}: test ids differ from folds.csv")
    if (folds_csv.loc[table.index, "label"].to_numpy() != table["label"].to_numpy()).any():
        raise LeakageError(f"{ds}: labels in clean.csv differ from folds.csv")
    return parts


def code_hash():
    lines = []
    for dp, dns, fns in os.walk(TAB_DIR):
        dns[:] = sorted(d for d in dns if d not in {"__pycache__"})
        for fn in sorted(fns):
            if fn.endswith((".py", ".yaml")):
                p = os.path.join(dp, fn)
                lines.append(f"{sha256_file(p)}  {os.path.relpath(p, PROJECT_DIR)}")
    return hashlib.sha256("\n".join(sorted(lines)).encode()).hexdigest()


# ------------------------------------------------------------------------------------------------- registry
def read_registry():
    if not os.path.isfile(REGISTRY):
        return pd.DataFrame(columns=REG_COLS)
    return pd.read_csv(REGISTRY, dtype=str, keep_default_na=False)


def update_registry(row):
    df = read_registry()
    key = (df.dataset == row["dataset"]) & (df.experiment == row["experiment"]) & (df.model == row["model"]) & \
          (df.fold == str(row["fold"]))
    df = pd.concat([df[~key], pd.DataFrame([{c: row.get(c, "") for c in REG_COLS}])], ignore_index=True)
    df["fold"] = df.fold.astype(int)
    df = df.sort_values(["dataset", "experiment", "model", "fold"])
    tmp = REGISTRY + ".tmp"
    df.to_csv(tmp, index=False, lineterminator="\n")
    os.replace(tmp, REGISTRY)


def is_done(ds, exp, model, k):
    df = read_registry()
    r = df[(df.dataset == ds) & (df.experiment == exp) & (df.model == model) & (df.fold == str(k))]
    return not r.empty and r.iloc[0]["status"] == "DONE"


# ------------------------------------------------------------------------------------------------- one fold
def run_fold(cfg, e, model_id, k, table, cols, out_root=RESULTS, write=True):
    ds, exp = e["dataset"], e["experiment"]
    seed = int(cfg["seed_base"]) + k
    split = load_fold(ds, k)
    parts = check_fold(ds, k, split, table)
    numeric, categorical = feature_columns(cols, e["feature_set"])
    feats = numeric + categorical
    X = {p: table.loc[ids, feats] for p, ids in parts.items()}
    y = {p: table.loc[ids, "label"].to_numpy().astype(int) for p, ids in parts.items()}
    t0 = time.time()
    with threadpool_limits(limits=int(cfg["n_jobs"])):
        pipe = fit_pipeline(model_id, cfg, seed, numeric, categorical, X["train"], y["train"], X["val"], y["val"])
        p_val = pipe.predict_proba(X["val"])[:, 1]
        p_test = pipe.predict_proba(X["test"])[:, 1]          # test predicted once
    thr = youden_threshold(y["val"], p_val)                       # validation only
    metrics = {"dataset": ds, "experiment": exp, "model_id": model_id, "fold": k, "val_threshold_youden": thr,
               "test": {"threshold_0.5": binary_metrics(y["test"], p_test, 0.5),
                        "threshold_val_youden": binary_metrics(y["test"], p_test, thr)},
               "val": {"threshold_0.5": binary_metrics(y["val"], p_val, 0.5)}}
    clf = pipe.named_steps["clf"]
    if model_id == "mlp":
        metrics["best_epoch"] = int(clf.best_epoch_)
    result = {"pipe": pipe, "metrics": metrics, "parts": parts, "p_val": p_val, "p_test": p_test, "y": y,
              "numeric": numeric, "categorical": categorical, "seconds": time.time() - t0}
    if not write:
        return result
    d = os.path.join(out_root, ds, exp, model_id, f"fold_{k:02d}")
    os.makedirs(d, exist_ok=True)
    run_cfg = {"stage": 7, "dataset": ds, "experiment": exp, "experiment_note": e["note"], "model_id": model_id,
               "fold": k, "seed": seed, "feature_set": e["feature_set"], "numeric_features": numeric,
               "categorical_features": categorical, "model": cfg["models"][model_id],
               "preprocessing": cfg["preprocessing"], "thresholds": cfg["thresholds"], "n_jobs": cfg["n_jobs"],
               "classifier_params": {k2: (list(v) if isinstance(v, tuple) else v)
                                     for k2, v in clf.get_params().items()
                                     if isinstance(v, (int, float, str, bool, tuple, type(None)))}}
    with open(os.path.join(d, "config.yaml"), "w") as fh:
        yaml.safe_dump(run_cfg, fh, sort_keys=False)
    for name, ids, prob, yy in [("predictions.csv", parts["test"], p_test, y["test"]),
                                ("val_predictions.csv", parts["val"], p_val, y["val"])]:
        write_csv(pd.DataFrame({"sample_id": ids, "group_id": table.loc[ids, "group_id"].to_numpy(), "y_true": yy,
                                "y_prob": prob, "fold": k}), os.path.join(d, name))
    write_json(metrics, os.path.join(d, "metrics.json"))
    info = {"status": "DONE", "stage": 7, "dataset": ds, "experiment": exp, "model_id": model_id, "fold": k,
            "seed": seed, "seconds": result["seconds"], "end_time": datetime.datetime.now().isoformat(timespec="seconds"),
            "device": "cpu (CUDA_VISIBLE_DEVICES='')", "n_jobs": cfg["n_jobs"],
            "thread_env": {v: os.environ.get(v) for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]},
            "python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__,
            "pandas": pd.__version__, "code_hash_tabular": code_hash(),
            "config_sha256": hashlib.sha256(json.dumps(run_cfg, sort_keys=True, default=str).encode()).hexdigest(),
            "split_files": {f"splits/{ds}/fold_{k:02d}.json": sha256_file(os.path.join(SPLITS_DIR, ds, f"fold_{k:02d}.json")),
                            f"splits/{ds}/folds.csv": sha256_file(os.path.join(SPLITS_DIR, ds, "folds.csv"))},
            "data_file": {f"cache/{ds}/clean.csv": sha256_file(os.path.join(PROJECT_DIR, "cache", ds, "clean.csv"))},
            "n": {p: len(ids) for p, ids in parts.items()},
            "n_features_after_preprocessing": int(pipe.named_steps["prep"].transform(X["train"].iloc[:1]).shape[1])}
    if model_id == "mlp":
        info.update(mlp_epochs_run=int(clf.epochs_run_), mlp_best_epoch=int(clf.best_epoch_),
                    mlp_best_val_auc=float(clf.best_val_auc_), mlp_n_val_used=int(clf.n_val_seen_))
    write_json(info, os.path.join(d, "run_info.json"))
    return result


# ------------------------------------------------------------------------------------------------- summaries
def leaderboard(ds, exp):
    rows = []
    for m in MODEL_IDS:
        s = summarize_model(os.path.join(RESULTS, ds), exp, m)      # writes <model>/summary.json
        pm = s["per_fold_metrics"]
        r = {"model": m, "n_folds": s["n_folds"]}
        for key in ["roc_auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "accuracy", "brier"]:
            r[f"{key}_mean"], r[f"{key}_std"] = pm[f"test/threshold_0.5/{key}"]["mean"], pm[f"test/threshold_0.5/{key}"]["std"]
        for key in ["balanced_accuracy", "sensitivity", "specificity", "f1"]:
            r[f"{key}_valthr_mean"] = pm[f"test/threshold_val_youden/{key}"]["mean"]
            r[f"{key}_valthr_std"] = pm[f"test/threshold_val_youden/{key}"]["std"]
        r["pooled_oof_roc_auc"] = s["pooled_out_of_fold"]["threshold_0.5"]["roc_auc"]
        rows.append(r)
    lb = pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)
    write_csv(lb, os.path.join(RESULTS, ds, exp, "leaderboard.csv"))
    return lb


def d05_rule_comparison(table):
    """How close each model gets to the exact label rule (label == A1..A10 sum >= 4). NOT autism prediction."""
    rule = (table[D05_ITEMS].sum(axis=1) >= 4).astype(int)
    rows = [{"model": "exact rule: sum(A1..A10) >= 4", "rule_equals_label_all_rows": bool((rule == table.label).all()),
             "n": len(table)}]
    for m in MODEL_IDS:
        P = pd.concat([pd.read_csv(os.path.join(RESULTS, "D05_toddler_screening", "set_all", m, f"fold_{k:02d}",
                                                "predictions.csv")) for k in range(1, 11)])
        r = rule.loc[P.sample_id].to_numpy()
        pred = (P.y_prob >= 0.5).astype(int).to_numpy()
        rows.append({"model": m, "n": len(P), "agreement_with_rule_at_0.5": float((pred == r).mean()),
                     "n_disagreements_at_0.5": int((pred != r).sum()),
                     "pooled_oof_roc_auc_vs_rule": binary_metrics(r, P.y_prob)["roc_auc"]})
    df = pd.DataFrame(rows)
    write_csv(df, os.path.join(RESULTS, "D05_toddler_screening", "set_all", "rule_comparison.csv"))
    return df


def check_results(cfg):
    """All 10 folds x model x experiment present, metrics valid, predictions = test ids exactly once, union = all."""
    out = {}
    for e in cfg["experiments"]:
        ds, exp = e["dataset"], e["experiment"]
        folds_csv = pd.read_csv(os.path.join(SPLITS_DIR, ds, "folds.csv"))
        for m in MODEL_IDS:
            reasons, allp = [], []
            for k in range(1, 11):
                d = os.path.join(RESULTS, ds, exp, m, f"fold_{k:02d}")
                need = ["config.yaml", "metrics.json", "predictions.csv", "val_predictions.csv", "run_info.json"]
                miss = [f for f in need if not os.path.isfile(os.path.join(d, f))]
                if miss:
                    reasons.append(f"fold {k}: missing {miss}")
                    continue
                mt = json.load(open(os.path.join(d, "metrics.json")))
                for thr, b in mt["test"].items():
                    reasons += [f"fold {k} {thr}: {x}" for x in valid_block(b)]
                p = pd.read_csv(os.path.join(d, "predictions.csv"))
                if p.sample_id.duplicated().any() or set(p.sample_id) != set(load_fold(ds, k)["test"]):
                    reasons.append(f"fold {k}: predictions != test ids")
                if not (np.isfinite(p.y_prob).all() and p.y_prob.between(0, 1).all()):
                    reasons.append(f"fold {k}: invalid y_prob")
                allp.append(p)
            if allp:
                P = pd.concat(allp)
                if P.sample_id.duplicated().any() or set(P.sample_id) != set(folds_csv.sample_id):
                    reasons.append("union of test predictions != all samples exactly once")
            out[f"{ds}/{exp}/{m}"] = {"verdict": "PASS" if not reasons else "FAIL", "reasons": reasons}
    write_json(out, os.path.join(TAB_DIR, "stage7_checks.json"))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if "torch" in sys.modules:
        raise RuntimeError("torch must not be imported in Stage 7")
    cfg = load_stage7_cfg()
    verify_manifest()  # raises on any split / source-manifest hash mismatch
    print(f"split hashes OK; CPU only (CUDA_VISIBLE_DEVICES='{os.environ.get('CUDA_VISIBLE_DEVICES')}'), "
          f"n_jobs {cfg['n_jobs']}, OMP_NUM_THREADS {os.environ.get('OMP_NUM_THREADS')}", flush=True)
    t_all = time.time()
    for e in cfg["experiments"]:
        ds, exp = e["dataset"], e["experiment"]
        table, cols = load_table(ds)
        for m in MODEL_IDS:
            for k in range(1, 11):
                if is_done(ds, exp, m, k) and not a.force:
                    continue
                start = datetime.datetime.now().isoformat(timespec="seconds")
                try:
                    r = run_fold(cfg, e, m, k, table, cols)
                except Exception as err:
                    update_registry({"dataset": ds, "experiment": exp, "model": m, "fold": k, "status": "FAILED",
                                     "start": start, "error": f"{type(err).__name__}: {err}"[:300]})
                    raise
                mt = r["metrics"]
                update_registry({"dataset": ds, "experiment": exp, "model": m, "fold": k, "status": "DONE",
                                 "start": start, "end": datetime.datetime.now().isoformat(timespec="seconds"),
                                 "val_auc": round(mt["val"]["threshold_0.5"]["roc_auc"], 6),
                                 "test_auc": round(mt["test"]["threshold_0.5"]["roc_auc"], 6),
                                 "seconds": round(r["seconds"], 2), "error": ""})
            print(f"{ds}/{exp}/{m}: 10 folds done ({time.time() - t_all:.0f} s elapsed)", flush=True)
        leaderboard(ds, exp)
    table, _ = load_table("D05_toddler_screening")
    d05_rule_comparison(table)
    chk = check_results(cfg)
    n_fail = sum(v["verdict"] != "PASS" for v in chk.values())
    print(f"result checks: {len(chk) - n_fail}/{len(chk)} PASS; total {time.time() - t_all:.0f} s", flush=True)
    if "torch" in sys.modules:
        raise RuntimeError("torch was imported during Stage 7")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
