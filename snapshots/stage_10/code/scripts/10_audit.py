"""Stage 10 B - independent re-verification with FRESH code (sklearn / numpy / hashlib only; nothing imported from
src.training, src.analysis or make_snapshot). Read-only. Writes logs/stage10_audit.txt (via tee), logs/stage10_audit.json,
logs/stage10_snapshot_changes.csv, logs/stage10_report_numbers_unmatched.csv.
Usage: CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/10_audit.py | tee logs/stage10_audit.txt"""
import glob
import hashlib
import json
import math
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score, recall_score,
                             roc_auc_score)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOL = 1e-6
MODELS = [f"M{i:02d}" for i in range(1, 11)]
TAB = {"D04_autism_prediction": ["set_all"], "D05_toddler_screening": ["set_all", "set_honest"], "D06_early_autism": ["set_all"]}
TAB_MODELS = ["dummy_majority", "logreg", "random_forest", "hist_gb", "mlp"]
OUT = {}


def P(*a):
    return os.path.join(ROOT, *a)


def jload(p):
    with open(p) as fh:
        return json.load(fh)


def close(a, b, tol=TOL):
    if a is None or b is None:
        return False
    if (isinstance(a, float) and math.isnan(a)) or (isinstance(b, float) and math.isnan(b)):
        return (isinstance(a, float) and math.isnan(a)) and (isinstance(b, float) and math.isnan(b))
    return abs(float(a) - float(b)) <= tol


def block(y, p, thr):
    """Fresh sklearn metrics; prediction rule: positive if p >= thr."""
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    yp = (p >= thr).astype(int)
    two = len(np.unique(y)) == 2
    return {"roc_auc": roc_auc_score(y, p) if two else float("nan"),
            "pr_auc": average_precision_score(y, p) if two else float("nan"),
            "balanced_accuracy": balanced_accuracy_score(y, yp),
            "sensitivity": recall_score(y, yp, pos_label=1, zero_division=0),
            "specificity": recall_score(y, yp, pos_label=0, zero_division=0),
            "f1": f1_score(y, yp, zero_division=0), "brier": float(np.mean((p - y) ** 2))}


def ece10(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    idx = np.clip(np.floor(p * 10).astype(int), 0, 9)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean()) for b in range(10) if (idx == b).any()))


def read_pred(d, deep):
    """Exact parsing: pandas' default C float parser can be 1 ulp off ('0.9999780982691833' -> ...832), so use
    float_precision='round_trip'; deep-model probabilities were float32 and are stored in shortest float32 form, so cast back
    to float32 (exact) before comparing with the stored float32-valued thresholds. Only exact threshold ties are affected."""
    pr = pd.read_csv(os.path.join(d, "predictions.csv"), float_precision="round_trip")
    if deep:
        pr["y_prob"] = pr.y_prob.astype(np.float32).astype(np.float64)
    return pr


def runs():
    """(key, dataset, experiment, model, fold, dir) for all 400 runs."""
    out = []
    for ds in ["D01_eye_tracking", "D02_facial_images_A"]:
        for m in MODELS:
            for k in range(1, 11):
                out.append((ds, None, m, k, P("results", ds, m, f"fold_{k:02d}")))
    for ds, exps in TAB.items():
        for e in exps:
            for m in TAB_MODELS:
                for k in range(1, 11):
                    out.append((ds, e, m, k, P("results", ds, e, m, f"fold_{k:02d}")))
    return out


# ---------------------------------------------------------------------------------------------- B1 + B2
def b1_b2():
    mism, n_checks = [], 0
    fold_vals = defaultdict(lambda: defaultdict(list))   # (table key) -> metric -> [10 values]
    oof = defaultdict(list)
    R = runs()
    for ds, e, m, k, d in R:
        pr = read_pred(d, deep=e is None)
        mt = jload(os.path.join(d, "metrics.json"))
        blocks = [("test", "threshold_0.5", pr.y_true, pr.y_prob, 0.5),
                  ("test", "threshold_val_youden", pr.y_true, pr.y_prob, mt["val_threshold_youden"])]
        if ds == "D01_eye_tracking":
            c = pr.groupby("group_id").agg(y=("y_true", "first"), p=("y_prob", "mean"))
            blocks += [("test_child", "threshold_0.5", c.y, c.p, 0.5),
                       ("test_child", "threshold_val_youden", c.y, c.p, mt["val_threshold_youden_child"])]
        for bname, thr_name, y, p, thr in blocks:
            fresh = block(y, p, thr)
            for met in ["roc_auc", "balanced_accuracy", "sensitivity", "specificity"]:
                n_checks += 1
                if not close(fresh[met], mt[bname][thr_name][met]):
                    mism.append(f"{d}: {bname}/{thr_name}/{met} fresh {fresh[met]} vs stored {mt[bname][thr_name][met]}")
            if thr_name == "threshold_0.5":
                key = {"test": {"D01_eye_tracking": "D01 clip", "D02_facial_images_A": "D02"}.get(ds),
                       "test_child": "D01 child"}[bname] if ds.startswith(("D01", "D02")) else \
                    {"D04_autism_prediction": "D04", "D06_early_autism": "D06"}.get(ds, f"D05 {e}")
                for met, v in fresh.items():
                    fold_vals[(key, m)][met].append(v)
                oof[(key, m)].append((np.asarray(y), np.asarray(p)))
    OUT["b1"] = {"runs": len(R), "checks": n_checks, "mismatches": mism}
    print(f"B1 {len(R)} runs ({sum(r[0].startswith('D0') and r[1] is None for r in R)} deep, "
          f"{sum(r[1] is not None for r in R)} tabular): {n_checks} metric checks (ROC-AUC, balanced accuracy, sensitivity, "
          f"specificity at threshold 0.5 and at the stored validation threshold; D01 also child level) recomputed with "
          f"sklearn from predictions.csv vs metrics.json, tol {TOL}: mismatches {len(mism)}")
    for x in mism[:20]:
        print("   ", x)
    # B2: master table
    M = pd.read_csv(P("results", "final", "master_table.csv"))
    mism2, n2 = [], 0
    for r in M.itertuples():
        fv = fold_vals[(r.dataset, r.model)]
        for met in ["roc_auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "brier"]:
            v = np.asarray(fv[met], float)
            assert len(v) == 10, (r.dataset, r.model, met, len(v))
            for stat, val in [("mean", float(np.mean(v))), ("std", float(np.std(v, ddof=1)))]:
                n2 += 1
                stored = getattr(r, f"{met}_{stat}")
                if not close(val, stored):
                    mism2.append(f"{r.dataset} {r.model} {met}_{stat}: fresh {val} vs table {stored}")
        y = np.concatenate([a for a, _ in oof[(r.dataset, r.model)]])
        p = np.concatenate([b for _, b in oof[(r.dataset, r.model)]])
        for name, val in [("pooled_auc", roc_auc_score(y, p)), ("ece_pooled", ece10(y, p))]:
            n2 += 1
            if not close(val, getattr(r, name)):
                mism2.append(f"{r.dataset} {r.model} {name}: fresh {val} vs table {getattr(r, name)}")
    OUT["b2"] = {"rows": len(M), "checks": n2, "mismatches": mism2}
    print(f"B2 master_table.csv: {len(M)} rows; {n2} values recomputed from predictions.csv (7 metrics x mean/std over 10 "
          f"folds, ddof=1; pooled AUC; ECE 10 bins), tol {TOL}: mismatches {len(mism2)} (bootstrap CIs not recomputed)")
    for x in mism2[:20]:
        print("   ", x)


# ---------------------------------------------------------------------------------------------- B3 leakage
def popcount64(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int64)


def b3():
    res = {}
    bad = []
    datasets = ["D01_eye_tracking", "D02_facial_images_A"] + list(TAB)
    folds = {ds: pd.read_csv(P("splits", ds, "folds.csv")) for ds in datasets}
    fj = {(ds, k): jload(P("splits", ds, f"fold_{k:02d}.json")) for ds in datasets for k in range(1, 11)}
    # per run: predictions == fold test ids; per model: every sample exactly once
    per_model = defaultdict(list)
    for ds, e, m, k, d in runs():
        pr = pd.read_csv(os.path.join(d, "predictions.csv"), usecols=["sample_id"])
        test = set(fj[(ds, k)]["test"])
        if pr.sample_id.duplicated().any() or set(pr.sample_id) != test:
            bad.append(f"{d}: predictions != fold test ids")
        per_model[(ds, e, m)].append(pr.sample_id)
    once = []
    for (ds, e, m), parts in per_model.items():
        s = pd.concat(parts)
        if s.duplicated().any() or set(s) != set(folds[ds].sample_id):
            once.append((ds, e, m))
    res["predictions_equal_fold_test_ids"] = not bad
    res["each_sample_predicted_once_per_model"] = (len(per_model), len(once))
    # train / val / test disjoint by sample and by group
    overlap = []
    for ds in datasets:
        g = folds[ds].set_index("sample_id").group_id
        for k in range(1, 11):
            f = fj[(ds, k)]
            sets = {s: set(f[s]) for s in ["train", "val", "test"]}
            grp = {s: set(g.loc[list(v)]) for s, v in sets.items()}
            for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
                if sets[a] & sets[b] or grp[a] & grp[b]:
                    overlap.append((ds, k, a, b))
            if set(folds[ds].sample_id[folds[ds].outer_fold == k]) != sets["test"]:
                overlap.append((ds, k, "test != folds.csv outer_fold"))
    res["train_val_test_disjoint_sample_and_group"] = not overlap
    # groups in one test fold only
    gfold = {ds: int((folds[ds].groupby("group_id").outer_fold.nunique() > 1).sum()) for ds in datasets}
    res["groups_in_more_than_one_test_fold"] = gfold
    # D02 pHash <= 4 pairs recomputed with fresh popcount
    man = pd.read_csv(P("cache", "D02_facial_images_A", "manifest.csv"))
    fo = man.image_id.map(folds["D02_facial_images_A"].set_index("sample_id").outer_fold).to_numpy()
    h = np.array([int(x, 16) for x in man.phash], dtype=np.uint64)
    cross_pairs = n_pairs = 0
    for i in range(len(h) - 1):
        dist = popcount64(h[i] ^ h[i + 1:])
        close_ = np.flatnonzero(dist <= 4) + i + 1
        n_pairs += len(close_)
        cross_pairs += int((fo[close_] != fo[i]).sum())
    res["D02_phash_le4_pairs_total_and_cross_fold"] = (n_pairs, cross_pairs)
    # D03 never used
    paths = yaml.safe_load(open(P("configs", "paths.yaml")))
    d02f, d03f = paths["datasets"]["D02_facial_images_A"], paths["datasets"]["D03_facial_images_B"]
    d03_hits = []
    for f in glob.glob(P("cache", "**", "*.csv"), recursive=True):
        df = pd.read_csv(f, dtype=str, keep_default_na=False, nrows=None) if os.path.getsize(f) < 50_000_000 else None
        if df is None:
            continue
        for c in df.columns:
            if df[c].str.startswith(d03f + "/").any():
                d03_hits.append(f"{os.path.relpath(f, ROOT)}:{c}")
    res["D03_paths_in_cache"] = d03_hits
    res["D02_manifest_all_under_D02_folder"] = bool(man.rel_path.str.startswith(d02f + "/").all())
    # CARS absent from inputs
    cars = []
    for f in [P("cache", "D01_eye_tracking", "clips_index.csv"), P("cache", "D01_eye_tracking", "quality_per_child.csv"),
              P("cache", "D01_eye_tracking", "participants.csv")]:
        cols = pd.read_csv(f, nrows=1).columns
        cars += [f"{os.path.relpath(f, ROOT)}:{c}" for c in cols if "cars" in c.lower()]
    for ds in TAB:
        cy = yaml.safe_load(open(P("cache", ds, "columns.yaml")))
        feats = [c for c, v in cy["columns"].items() if str(v.get("role", "")).startswith("feature")]
        for s in cy.get("feature_sets", {}).values():
            feats += s
        cars += [f"{ds}:{c}" for c in set(feats) if "cars" in c.lower()]
    code_refs = [f for f in glob.glob(P("src", "training", "*.py")) + glob.glob(P("src", "models", "*.py"))
                 if re.search(r"\bCARS\b|participants\.csv", open(f).read())]
    res["CARS_in_inputs"] = cars
    res["training_code_reading_participants_or_CARS"] = [os.path.relpath(f, ROOT) for f in code_refs]
    # D05 honest
    cy = yaml.safe_load(open(P("cache", "D05_toddler_screening", "columns.yaml")))
    forbidden = {f"A{i}" for i in range(1, 11)} | {c for c in cy["columns"] if "qchat" in c.lower() or "score" in c.lower()}
    honest_sets = cy["feature_sets"]["set_honest"]
    run_feats = set()
    for f in glob.glob(P("results", "D05_toddler_screening", "set_honest", "*", "fold_*", "config.yaml")):
        c = yaml.safe_load(open(f))
        run_feats |= set(c.get("numeric_features", [])) | set(c.get("categorical_features", []))
    res["D05_honest_forbidden_in_set"] = sorted(forbidden & set(honest_sets))
    res["D05_honest_forbidden_in_run_configs"] = sorted(forbidden & run_feats)
    res["D05_honest_run_features"] = sorted(run_feats)
    # D04 test.csv unused
    d4 = pd.read_csv(P("cache", "D04_autism_prediction", "clean.csv"))
    raw = os.path.join(paths["datasets_dir"], paths["datasets"]["D04_autism_prediction"])
    tr = pd.read_csv(os.path.join(raw, "train.csv"))
    te = pd.read_csv(os.path.join(raw, "test.csv"))
    cmp = d4.merge(tr, on="ID", suffixes=("", "_raw"))
    item_cols = [f"A{i}_Score" for i in range(1, 11)] + ["age", "result"]
    rows_match = len(cmp) == len(d4) == len(tr) and all(np.allclose(cmp[c], cmp[c + "_raw"]) for c in item_cols)
    te_match = d4.merge(te, on="ID", suffixes=("", "_te"))
    te_rows_identical = int(sum(np.allclose(r[[c for c in item_cols]].astype(float), r[[c + "_te" for c in item_cols]].astype(float))
                                for _, r in te_match.iterrows())) if len(te_match) else 0
    data_files = {tuple(jload(f)["data_file"]) for f in glob.glob(P("results", "D04_autism_prediction", "set_all", "*", "fold_*", "run_info.json"))}
    res["D04_clean_rows_equal_train_csv"] = bool(rows_match)
    res["D04_rows_identical_to_test_csv_rows"] = te_rows_identical
    res["D04_run_data_files"] = [list(x) for x in data_files]
    OUT["b3"] = {**res, "problems": bad + [str(x) for x in overlap] + [str(x) for x in once]}
    print(f"B3 leakage: predictions == fold test ids for all 400 runs: {res['predictions_equal_fold_test_ids']}; every sample "
          f"predicted exactly once per model: {len(per_model) - len(once)}/{len(per_model)} models; train/val/test disjoint by "
          f"sample AND group in all {len(datasets) * 10} dataset-folds: {not overlap}")
    print(f"B3 groups in more than one test fold: {gfold} (D01 = children, D02 = near-duplicate groups)")
    print(f"B3 D02 pHash <= 4 pairs recomputed (fresh popcount): {n_pairs} pairs, across folds: {cross_pairs}")
    print(f"B3 D03: cache cells starting with the D03 folder '{d03f}/': {len(d03_hits)}; all D02 manifest paths under the D02 "
          f"folder: {res['D02_manifest_all_under_D02_folder']}")
    print(f"B3 CARS: columns named *cars* in D01 caches / tabular feature definitions: {cars or 'none'}; training/model code "
          f"referencing participants.csv or CARS: {res['training_code_reading_participants_or_CARS'] or 'none'}")
    print(f"B3 D05 set_honest: forbidden (A1-A10, Qchat score) in feature set: {res['D05_honest_forbidden_in_set'] or 'none'}; "
          f"in the 50 run configs: {res['D05_honest_forbidden_in_run_configs'] or 'none'}; features used {res['D05_honest_run_features']}")
    print(f"B3 D04: clean.csv rows = raw train.csv rows (ID + A1-A10, age, result identical): {res['D04_clean_rows_equal_train_csv']}; "
          f"clean rows identical to a test.csv row with the same ID: {te_rows_identical}; data file of all D04 runs: {res['D04_run_data_files']}")


# ---------------------------------------------------------------------------------------------- B4 snapshots
_HCACHE = {}


def sha(p):
    if p not in _HCACHE:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 22), b""):
                h.update(b)
        _HCACHE[p] = h.hexdigest()
    return _HCACHE[p]


def scope_files(info):
    out = set()
    for d in info.get("hashed_dirs", []) + info.get("extra_dirs", []):
        for dp, dns, fns in os.walk(P(d)):
            dns[:] = [x for x in dns if x not in ("__pycache__", ".pytest_cache")]
            out |= {os.path.relpath(os.path.join(dp, f), ROOT) for f in fns if not f.endswith(".pyc")}
    for pat in info.get("extra_hash_globs", []):
        out |= {os.path.relpath(f, ROOT) for f in glob.glob(P(pat), recursive=True) if os.path.isfile(f) and not f.endswith(".pt")}
    return out


def b4():
    snaps = sorted(glob.glob(P("snapshots", "stage_*")), key=lambda s: jload(os.path.join(s, "SNAPSHOT_INFO.json"))["datetime"])
    rec = {}
    for s in snaps:
        lines = open(os.path.join(s, "CHECKSUMS.sha256")).read().splitlines()
        rec[os.path.basename(s)] = {l.split("  ", 2)[2]: l.split("  ", 2)[0] for l in lines}
    rows, summary = [], []
    for i, s in enumerate(snaps):
        name = os.path.basename(s)
        info = jload(os.path.join(s, "SNAPSHOT_INFO.json"))
        integrity = sha(os.path.join(s, "CHECKSUMS.sha256")) == info["code_hash"]
        later = [os.path.basename(x) for x in snaps[i + 1:]]
        r = rec[name]
        cur = scope_files(info)
        n_same = 0
        for path, h0 in r.items():
            if not os.path.isfile(P(path)):
                gone = next((L for L in later if path not in rec[L]), "after stage_09")
                rows.append({"snapshot": name, "path": path, "change": "deleted", "explanation": f"absent from {gone}"})
                continue
            h1 = sha(P(path))
            if h1 == h0:
                n_same += 1
                continue
            where = next((L for L in later if rec[L].get(path) == h1), None)
            rows.append({"snapshot": name, "path": path, "change": "changed",
                         "explanation": f"edited later; current version recorded by {where}" if where else "edited in Stage 10 (after the last snapshot)"})
        for path in sorted(cur - set(r)):
            h1 = sha(P(path))
            where = next((L for L in later if rec[L].get(path) == h1), None)
            rows.append({"snapshot": name, "path": path, "change": "added",
                         "explanation": f"added later; current version recorded by {where}" if where else "added/edited in Stage 10 (after the last snapshot)"})
        g = [x for x in rows if x["snapshot"] == name]
        unexplained = [x for x in g if "Stage 10" in x["explanation"] or x["change"] == "deleted"]
        summary.append({"snapshot": name, "datetime": info["datetime"], "checksums_integrity": integrity, "recorded": len(r),
                        "unchanged": n_same, "changed": sum(x["change"] == "changed" for x in g),
                        "added": sum(x["change"] == "added" for x in g), "deleted": sum(x["change"] == "deleted" for x in g),
                        "stage10_or_unexplained": len(unexplained)})
    pd.DataFrame(rows).to_csv(P("logs", "stage10_snapshot_changes.csv"), index=False)
    S = pd.DataFrame(summary)
    OUT["b4"] = {"summary": summary, "stage10_changes": sorted({x["path"] for x in rows if "Stage 10" in x["explanation"]}),
                 "deleted": [x for x in rows if x["change"] == "deleted"]}
    print("B4 snapshots (recomputed sha256 of every recorded file; changes explained by the later snapshot that recorded the "
          "current version) -> logs/stage10_snapshot_changes.csv")
    for r in S.itertuples():
        print(f"   {r.snapshot} ({r.datetime}): integrity {r.checksums_integrity}; recorded {r.recorded}, unchanged {r.unchanged}, "
              f"changed {r.changed}, added {r.added}, deleted {r.deleted}; not recorded by any later snapshot: {r.stage10_or_unexplained}")
    print(f"B4 files whose current version is in no snapshot (Stage 10 edits): {OUT['b4']['stage10_changes']}")
    print(f"B4 deleted files: {len(OUT['b4']['deleted'])}")


# ---------------------------------------------------------------------------------------------- B5 report numbers
def b5():
    text = open(P("reports", "FINAL_RESULTS_REPORT.md")).read()
    vals = set()
    for f in glob.glob(P("results", "final", "*.csv")):
        df = pd.read_csv(f)
        for c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce").dropna()
            for x in v:
                for nd in (1, 2, 3, 4):
                    vals.add(f"{x:.{nd}f}")
                    vals.add(f"{abs(x):.{nd}f}")
                vals.add(f"{x:.2g}")                      # p-values printed with 2 significant digits (e.g. 0.00056, 3.7e-10)
    rep = pd.read_csv(P("results", "final", "reproducibility_table.csv"))
    secs = [float(v) for i, v in zip(rep["item"], rep["value"]) if "seconds" in i and "S7" not in i]
    derived = {f"{s / 60:.1f}" for s in secs} | {f"{sum(secs) / 3600:.2f}"}
    constants = {"0.01": "AdamW weight decay", "0.5": "threshold / chance AUC / min valid ratio", "0.8": "random-crop scale",
                 "1.0": "random-crop scale / gradient clip / D05 AUC", "0.1": "colour jitter", "0.125": "uniform attention 1/8",
                 "0.05": "matching caliper / alpha", "1e-3": "cuDNN loss reproducibility",
                 "97.5": "one-sided bound level (multiplicity note)", "0.975": "one-sided bound level (multiplicity note)"}
    rows, n_scan, n_skip = [], 0, 0
    for ln, line in enumerate(text.splitlines(), 1):
        if line.startswith("|") and ("code_hash" in line or "run_info.json::" in line):
            continue
        if line.startswith("#"):
            n_skip += len(re.findall(r"\d+\.\d+", line))
            continue                                       # section headings
        for mt in re.finditer(r"(?<![\w.])[-+]?\d+\.\d+(?:e[-+]?\d+)?(?![\d.])", line):
            tok = mt.group(0)
            if re.search(r"(sections?|\() *$", line[: mt.start()]) or re.match(r"\d+\.\d+b? = ", line[mt.start():].lstrip("+-")) \
                    or line[mt.end(): mt.end() + 1] == "b":
                n_skip += 1
                continue                                   # cross-references to report sections (3.2, 3.4b, 3.6)
            n_scan += 1
            core = tok.lstrip("+-")
            if core in vals or tok in vals:
                continue
            rows.append({"line": ln, "number": tok, "category": "derived (GPU time from reproducibility_table seconds)" if core in derived
                         else (f"protocol constant: {constants[core]}" if core in constants else "UNMATCHED - manual check"),
                         "context": line.strip()[max(0, mt.start() - 60): mt.end() + 40]})
    U = pd.DataFrame(rows, columns=["line", "number", "category", "context"])
    U.to_csv(P("logs", "stage10_report_numbers_unmatched.csv"), index=False)
    n_all = len(re.findall(r"(?<![\w.])[-+]?\d+\.\d+(?:e[-+]?\d+)?(?![\d.])", text))
    OUT["b5"] = {"decimals_scanned": n_scan, "skipped_headings_and_section_refs": n_skip, "not_in_tables": len(U), "by_category": U.category.value_counts().to_dict(),
                 "manual": U[U.category.str.startswith("UNMATCHED")].to_dict("records")}
    print(f"B5 FINAL_RESULTS_REPORT.md: {n_scan} decimal numbers scanned ({n_skip} section numbers / cross-references skipped); "
          f"{n_scan - len(U)} match a value in results/final/*.csv "
          f"(rounded as printed); not matched: {len(U)} -> {U.category.value_counts().to_dict()}")
    for r in U[U.category.str.startswith("UNMATCHED")].itertuples():
        print(f"   line {r.line}: {r.number}  ...{r.context}...")


def main():
    b1_b2()
    b3()
    b4()
    b5()
    with open(P("logs", "stage10_audit.json"), "w") as fh:
        json.dump(OUT, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
