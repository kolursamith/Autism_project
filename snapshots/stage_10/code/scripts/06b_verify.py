"""Stage 6B - verify the finished S6 (D02) outputs; no run is re-run or modified (reads results/, writes summaries).
Prints the evidence (tee -> logs/stage06_verify.txt) and writes logs/stage06_original_split_auc.csv (descriptive only).
Usage: .venv/bin/python scripts/06b_verify.py | tee logs/stage06_verify.txt"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.data.common import LOG_DIR, PROJECT_DIR, write_csv  # noqa: E402
from src.data.d02_faces import hamming_matrix  # noqa: E402
from src.data.splits import load_fold, verify_manifest  # noqa: E402
from src.training.checks import validate_model_results  # noqa: E402
from src.training.metrics import binary_metrics  # noqa: E402
from src.training.summarize import leaderboard  # noqa: E402

DS = "D02_facial_images_A"
MODELS = [f"M{i:02d}" for i in range(1, 11)]
RES = os.path.join(PROJECT_DIR, "results")


def main():
    # ------------------------------------------------------------------ 1a queue exited
    qlog = open(os.path.join(LOG_DIR, "stage6_queue.log")).read()
    end = re.findall(r"===== (\S+) run_stage6.sh end \(exit (\d+)\)", qlog)
    pidf = os.path.join(LOG_DIR, "stage6.pid.finished")
    pid = int(open(pidf).read().split()[0])
    alive = os.path.exists(f"/proc/{pid}")
    print(f"1a queue: last run_stage6.sh end line {end[-1] if end else None}; 'QUEUE DONE {DS}' in log: "
          f"{f'QUEUE DONE {DS}' in qlog}; PID {pid} alive: {alive}; pid file renamed to logs/stage6.pid.finished: "
          f"{os.path.isfile(pidf) and not os.path.exists(os.path.join(LOG_DIR, 'stage6.pid'))}")
    # ------------------------------------------------------------------ 1b registry
    reg = pd.read_csv(os.path.join(RES, "run_registry.csv"), keep_default_na=False)
    d = reg[reg.dataset == DS]
    print(f"1b registry: {len(d)} D02 rows, status counts {d.status.value_counts().to_dict()}, models "
          f"{sorted(d.model.unique())}, folds per model {d.groupby('model').fold.nunique().unique().tolist()}")
    left = [p for p, _, fs in os.walk(os.path.join(RES, DS)) for f in fs if f == "last.pt"]
    print(f"1b last.pt files left under results/{DS}: {len(left)}")
    # ------------------------------------------------------------------ 1c split hashes, identical test ids
    verify_manifest()
    print("1c split hashes vs SPLITS_MANIFEST.json: OK")
    ok = True
    for k in range(1, 11):
        test = set(load_fold(DS, k)["test"])
        sets = {m: set(pd.read_csv(os.path.join(RES, DS, m, f"fold_{k:02d}", "predictions.csv")).sample_id)
                for m in MODELS}
        same = all(s == test for s in sets.values())
        infos = [json.load(open(os.path.join(RES, DS, m, f"fold_{k:02d}", "run_info.json"))) for m in MODELS]
        hashes = {i["split_files"][f"splits/{DS}/fold_{k:02d}.json"] for i in infos}
        tid = {i["test_ids_sha256"] for i in infos}
        ok &= same and len(hashes) == 1 and len(tid) == 1
        print(f"1c fold {k:2d}: {len(test)} test ids; all 10 models predicted exactly these ids: {same}; fold-file "
              f"sha256 recorded by the 10 runs identical: {len(hashes) == 1}; test_ids_sha256 identical: {len(tid) == 1}")
    print("1c identical test ids per fold across models:", ok)
    # ------------------------------------------------------------------ 1d validate_model_results
    for m in MODELS:
        v, reasons, flags = validate_model_results(DS, m, RES, range(1, 11), return_flags=True)
        print(f"1d validate_model_results {m}: {v}; reasons {len(reasons)}; flags {len(flags)}"
              + "".join(f"\n     reason: {r}" for r in reasons) + "".join(f"\n     flag: {f}" for f in flags))
    # ------------------------------------------------------------------ 1e near-duplicate sanity check
    man = pd.read_csv(os.path.join(PROJECT_DIR, "cache", DS, "manifest.csv"))
    folds = pd.read_csv(os.path.join(PROJECT_DIR, "splits", DS, "folds.csv"))
    man = man.merge(folds[["sample_id", "outer_fold"]], left_on="image_id", right_on="sample_id", how="left")
    assert man.outer_fold.notna().all() and len(man) == len(folds)
    g_folds = man.groupby("group_id").outer_fold.nunique()
    multi = man.group_id.value_counts()
    print(f"1e groups: {man.group_id.nunique()} groups ({int((multi > 1).sum())} with > 1 image); groups spanning "
          f"> 1 outer fold: {int((g_folds > 1).sum())}")
    md5_cross = man.groupby("md5").outer_fold.nunique()
    print(f"1e byte-identical images (same md5): {int((man.md5.value_counts() > 1).sum())} md5 values with > 1 image; "
          f"spanning > 1 outer fold: {int((md5_cross > 1).sum())}")
    max_h = yaml.safe_load(open(os.path.join(PROJECT_DIR, "configs", "data", "D02.yaml")))["near_duplicate"]["max_hamming"]
    H = hamming_matrix(man.phash.tolist())
    iu = np.triu_indices(len(man), 1)
    h = H[iu]
    fo = man.outer_fold.to_numpy()
    cross = fo[iu[0]] != fo[iu[1]]
    print(f"1e pHash pairs recomputed from manifest.csv over all {len(h):,} image pairs (group rule: Hamming <= "
          f"{max_h}):")
    for lo, hi in [(0, max_h), (max_h + 1, 8), (9, 10)]:
        sel = (h >= lo) & (h <= hi)
        print(f"     Hamming {lo}-{hi}: {int(sel.sum())} pairs, of which in different outer folds: {int((sel & cross).sum())}")
    # ------------------------------------------------------------------ 2 summarize
    lb = leaderboard(RES, DS)
    print(f"2 summarize: summary.json for {len(lb)} models + results/{DS}/leaderboard.csv written")
    # ------------------------------------------------------------------ 3 original split subset (descriptive)
    orig = man.set_index("image_id").original_split
    rows = []
    for m in MODELS:
        P = pd.concat([pd.read_csv(os.path.join(RES, DS, m, f"fold_{k:02d}", "predictions.csv")) for k in range(1, 11)])
        P["orig"] = P.sample_id.map(orig)
        assert P.orig.notna().all()
        r = {"model_id": m}
        for name, sel in [("orig_test", P.orig == "test"), ("orig_train_valid", P.orig != "test"),
                          ("all", P.orig.notna())]:
            S = P[sel]
            pooled = binary_metrics(S.y_true, S.y_prob, 0.5)
            per_fold = [binary_metrics(g.y_true, g.y_prob, 0.5)["roc_auc"] for _, g in S.groupby("fold")]
            r[f"{name}_n"] = len(S)
            r[f"{name}_n_pos"] = int(S.y_true.sum())
            r[f"{name}_pooled_roc_auc"] = pooled["roc_auc"]
            r[f"{name}_fold_mean_roc_auc"] = float(np.nanmean(per_fold))
            r[f"{name}_fold_std_roc_auc"] = float(np.nanstd(per_fold, ddof=1))
        rows.append(r)
    out = pd.DataFrame(rows)
    write_csv(out, os.path.join(LOG_DIR, "stage06_original_split_auc.csv"))
    n_t = out.orig_test_n.unique().tolist()
    print(f"3 original-split subsets: logs/stage06_original_split_auc.csv written ({len(out)} models; original 'test' "
          f"images per model {n_t}, rest {out.orig_train_valid_n.unique().tolist()})")


if __name__ == "__main__":
    main()
