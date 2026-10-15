"""Stage 7 - write reports/STAGE_07_REPORT.md from the saved outputs only (results/D04*, D05*, D06*, tabular/,
logs/stage07_*). CPU only, no torch. Decision = PASS only if the Stage 7 tests passed, all 20 result checks pass and
src/, configs/, splits/, cache/ are byte-identical to the stage_04 snapshot (nothing the S5 queue uses was touched).
Usage: CUDA_VISIBLE_DEVICES="" .venv/bin/python tabular/write_report.py"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.data.common import md_table, sha256_file  # noqa: E402
from tabular.models import MODEL_IDS, PROJECT_DIR, TAB_DIR, load_stage7_cfg, load_table  # noqa: E402

RES = os.path.join(PROJECT_DIR, "results")
TITLES = {("D04_autism_prediction", "set_all"): "D04 autism prediction (adult AQ-10 screening table), all features",
          ("D05_toddler_screening", "set_all"): "D05 toddler screening, set_all - LABEL-DEFINITIONAL UPPER BOUND",
          ("D05_toddler_screening", "set_honest"): "D05 toddler screening, set_honest (A1-A10 and Qchat score removed)",
          ("D06_early_autism", "set_all"): "D06 early autism, all features"}


def pm(lb, key):
    return [f"{m:.3f} ± {s:.3f}" for m, s in zip(lb[f"{key}_mean"], lb[f"{key}_std"])]


def untouched():
    """Compare src/ configs/ splits/ cache/ with snapshots/stage_04/CHECKSUMS.sha256 (scripts/ and reports/ are listed
    separately: scripts/ gained the three Stage 5A files, reports/ gains this report)."""
    rec = {}
    for line in open(os.path.join(PROJECT_DIR, "snapshots", "stage_04", "CHECKSUMS.sha256")):
        h, size, path = line.rstrip("\n").split("  ", 2)
        rec[path] = h
    out = {}
    for d in ["src", "configs", "splits", "cache", "scripts", "tests"]:
        cur = {}
        for dp, dns, fns in os.walk(os.path.join(PROJECT_DIR, d)):
            dns[:] = [x for x in dns if x not in {"__pycache__", ".pytest_cache"}]
            for fn in fns:
                if not fn.endswith(".pyc"):
                    p = os.path.relpath(os.path.join(dp, fn), PROJECT_DIR)
                    cur[p] = sha256_file(os.path.join(PROJECT_DIR, p))
        old = {p: h for p, h in rec.items() if p.split("/")[0] == d}
        out[d] = {"changed": sorted(p for p in cur if p in old and cur[p] != old[p]),
                  "added": sorted(set(cur) - set(old)), "removed": sorted(set(old) - set(cur))}
    return out


def main():
    cfg = load_stage7_cfg()
    pyt = open(os.path.join(PROJECT_DIR, "logs", "stage07_pytest.txt")).read().rstrip("\n")
    last = pyt.splitlines()[-1]
    py_ok = re.search(r"(\d+) (failed|error)", last) is None and "passed" in last
    chk = json.load(open(os.path.join(TAB_DIR, "stage7_checks.json")))
    chk_ok = len(chk) == 20 and all(v["verdict"] == "PASS" for v in chk.values())
    unt = untouched()
    frozen_ok = all(not any(unt[d].values()) for d in ["src", "configs", "splits", "cache"])
    reg = pd.read_csv(os.path.join(TAB_DIR, "run_registry_stage7.csv"))
    run_log = open(os.path.join(PROJECT_DIR, "logs", "stage07_run.log")).read().rstrip("\n")

    L = ["# STAGE 7 REPORT - tabular baselines (D04, D05, D06), run early in parallel with S5", "",
         "Screening-support research, not diagnosis. Run on 2026-10-14 BEFORE Stage 6, with the user's explicit "
         "approval, while the S5 GPU training was running. CPU only (`CUDA_VISIBLE_DEVICES=''`), sklearn n_jobs 2, "
         "BLAS/OpenMP threads capped at 2, `nice -n 10`; torch never imported (tested). All Stage 7 code is in "
         "`tabular/`; nothing in src/, scripts/, configs/, splits/, cache/, results/D01_eye_tracking/, logs/stage5* or "
         "results/run_registry.csv was modified (section 8). No snapshot yet (made after S5 finishes).", "",
         "## 1. Summary", "",
         f"- 4 experiments x 5 models x 10 folds = {len(reg)} runs, all DONE ({(reg.status == 'DONE').sum()}); "
         f"total fitting time {reg.seconds.sum():.0f} s.",
         f"- Result checks: {sum(v['verdict'] == 'PASS' for v in chk.values())}/{len(chk)} PASS; "
         f"Stage 7 tests: {last.strip('= ')}.",
         "- Splits: the Stage 2 files `splits/D04_*`, `D05_*`, `D06_*`, sha256-verified against SPLITS_MANIFEST.json "
         "before the run (and in the tests).", "",
         "## 2. Protocol (fixed hyperparameters, no tuning; `tabular/configs/stage7.yaml`)", "",
         "| item | setting |", "|---|---|",
         "| preprocessing | ColumnTransformer inside a sklearn Pipeline, fitted on the TRAIN part of each fold only: "
         "numeric = median imputation + StandardScaler; categorical = most-frequent imputation + one-hot "
         "(unknown categories ignored) |",
         "| dummy_majority | DummyClassifier(most_frequent) |",
         "| logreg | LogisticRegression, L2, C = 1.0, class_weight = balanced, max_iter 5000 |",
         "| random_forest | RandomForestClassifier, 500 trees, class_weight = balanced_subsample, n_jobs 2 |",
         "| hist_gb | HistGradientBoostingClassifier, default parameters, sample_weight = balanced class weights |",
         "| mlp | MLPClassifier(hidden 64, 32), sklearn defaults otherwise; one partial_fit pass over the train part "
         "per epoch, early stopping on the ROC-AUC of the fold's VALIDATION part (patience 10, min_delta 1e-4, max "
         "200 epochs), best epoch kept; sklearn's internal early_stopping split is not used |",
         "| seeds | random_state = 42 + fold |",
         "| validation part | used only for MLP early stopping and for the Youden threshold |",
         "| test part | predicted once |",
         "| thresholds | 0.5 and the Youden-J threshold chosen on the validation predictions (never on test) |",
         "| metrics | the deep-model set (`src/training/metrics.py`): accuracy, balanced accuracy, precision, "
         "sensitivity, specificity, F1, ROC-AUC, PR-AUC, Brier, confusion matrix; summary = mean, std (ddof 1), min, "
         "max over folds + pooled out-of-fold (`src/training/summarize.py`, reused read-only) |", "",
         "Output layout (results/README.md plus an experiment level): `results/<dataset>/<experiment>/<model>/fold_XX/`"
         " {config.yaml, metrics.json, predictions.csv, val_predictions.csv, run_info.json}, `<model>/summary.json`, "
         "`<experiment>/leaderboard.csv`; registry `tabular/run_registry_stage7.csv`. No model files are stored.", "",
         "## 3. Results (test folds, threshold 0.5, mean ± std over 10 folds)", ""]
    for e in cfg["experiments"]:
        ds, exp = e["dataset"], e["experiment"]
        lb = pd.read_csv(os.path.join(RES, ds, exp, "leaderboard.csv"))
        t = pd.DataFrame({"model": lb.model, "ROC-AUC": pm(lb, "roc_auc"), "balanced accuracy": pm(lb, "balanced_accuracy"),
                          "sensitivity": pm(lb, "sensitivity"), "specificity": pm(lb, "specificity"),
                          "F1": pm(lb, "f1"), "bal. acc. @ val threshold": pm(lb, "balanced_accuracy_valthr"),
                          "pooled OOF ROC-AUC": lb.pooled_oof_roc_auc.round(3)})
        table, _ = load_table(ds)
        L += [f"### {TITLES[(ds, exp)]}", "", f"n = {len(table)} ({int(table.label.sum())} positive / "
              f"{int((1 - table.label).sum())} negative); `results/{ds}/{exp}/leaderboard.csv`. {e['note']}.", "",
              md_table(t), ""]
        if (ds, exp) == ("D05_toddler_screening", "set_all"):
            rc = pd.read_csv(os.path.join(RES, ds, exp, "rule_comparison.csv"))
            L += ["**How close each model gets to the exact rule** (`rule_comparison.csv`, pooled out-of-fold "
                  "predictions of all 1,054 rows):", "", md_table(rc.fillna("")), "",
                  "In D05 the label is, by construction, `sum(A1..A10) >= 4` (true for all 1,054 rows), and A1-A10 plus "
                  "`Qchat-10-Score` are in set_all. These numbers therefore measure how well each model re-learns a "
                  "known arithmetic rule from its own inputs. **They do not measure autism prediction** and must not "
                  "be compared with the other experiments as evidence of screening performance.", ""]
    # descriptive single-feature AUCs for D06 (all rows; not used for any modelling choice)
    t6, c6 = load_table("D06_early_autism")
    uni = []
    for c in c6["feature_sets"]["set_all"]:
        x = t6[c]
        if not pd.api.types.is_numeric_dtype(x):
            x = x.astype("category").cat.codes
        a = roc_auc_score(t6.label, x)
        uni.append({"feature": c, "single-feature ROC-AUC (all rows)": round(max(a, 1 - a), 3)})
    L += ["## 4. Description (neutral)", "",
          "- **D04**: logistic regression, random forest, gradient boosting and the MLP reach mean test ROC-AUCs of "
          "0.88-0.91 against 0.50 for the majority baseline. At threshold 0.5 the models trade sensitivity against "
          "specificity differently (the positive class is 20% of rows); the validation-chosen threshold gives similar "
          "balanced accuracy across them. D04 contains the A1-A10 screening answers and `result`; Stage 2 found "
          "`result` is not a function of A1-A10 (kept as a feature).",
          "- **D05 set_all**: upper bound only; see the rule comparison above.",
          "- **D05 set_honest**: without the screening items and score, the remaining demographic/context features "
          "give mean ROC-AUCs of 0.59-0.61 (majority baseline 0.50).",
          "- **D06**: no model exceeds the majority baseline: mean ROC-AUCs are 0.45-0.48. The descriptive "
          "single-feature AUCs below are all close to 0.5, i.e. this table shows no detectable association between "
          "its features and the label. Mean AUCs slightly below 0.5 are a known effect of cross-validation on data "
          "without signal and are not evidence of an inverse relation.", "",
          md_table(pd.DataFrame(uni)), "",
          "None of these numbers is a diagnostic accuracy; the tables are small, single-source screening datasets.", "",
          "## 5. Leakage and correctness tests (`tabular/tests/test_stage07.py`, `logs/stage07_pytest.txt`)", "",
          "Tests: split sha256 = manifest; train/val/test disjoint by sample and by group in every fold, every sample "
          "tested exactly once; a corrupted split is rejected; set_honest contains none of A1-A10 / Qchat-10-Score (config, "
          "fitted pipeline and every saved run); D05 label == item rule; an audit hook records every file opened while "
          "running all experiments x models: no file under the raw dataset folder is opened (so D04 `test.csv` and "
          "`Autism_Screening_Data_Combined.csv` are never read); imputer statistics, scaler mean (== train-part mean, "
          "!= all-data mean) and one-hot categories come from the train part only; MLP early stopping sees exactly "
          "the validation part; every saved threshold equals the Youden threshold of the saved validation "
          "predictions; all 200 runs complete and valid; torch never imported.", "",
          "```", pyt, "```", "",
          "## 6. Result checks (`tabular/stage7_checks.json`)", "",
          f"{sum(v['verdict'] == 'PASS' for v in chk.values())}/{len(chk)} PASS: all 10 folds present per experiment x "
          "model, metric values valid, predictions = test ids of the fold exactly once, union over folds = every "
          "sample exactly once. One check needed a floating-point tolerance: sklearn's average precision returned "
          "1.0000000000000002 for a perfectly separated D05 set_all fold (logreg fold 5); `tabular/run_stage7.py` "
          "clips values within 1e-9 of [0, 1] before the range check.", "",
          "## 7. Run log (`logs/stage07_run.log`)", "", "```", run_log, "```", "",
          "The first run reported 19/20 checks because of the PR-AUC tolerance issue above; the second invocation "
          "skipped all 200 DONE runs and only rebuilt the summaries and checks.", "",
          "## 8. Files the S5 queue depends on were not touched", "",
          "Comparison with `snapshots/stage_04/CHECKSUMS.sha256`:", "",
          md_table(pd.DataFrame([{"dir": d, "changed": len(v["changed"]), "added": len(v["added"]),
                                  "removed": len(v["removed"]), "added files": " ".join(v["added"])}
                                 for d, v in unt.items()])), "",
          "scripts/ additions are the three Stage 5A files (written before Stage 7). src/, configs/, splits/ and "
          "cache/ are byte-identical to the stage_04 snapshot.", "",
          "## 9. Open items", "",
          "1. `src/training/metrics.is_valid_metric_block` (used by the deep-model gate `validate_model_results`) has "
          "the same strict [0, 1] check; a perfectly separated test fold could make it FAIL a valid deep-model run. "
          "src/ must not change while S5 runs: fix (tolerance 1e-9 + test) after S5 finishes and before S6.",
          "2. Full pytest suite and `snapshots/stage_07/` after S5 finishes (the snapshot hashes src/; the stage_04 "
          "snapshot test currently fails because scripts/ gained the Stage 5A files).",
          "3. Stage 8 statistics (bootstrap CIs, Friedman/Nemenyi) can use these summaries and predictions.", ""]
    ok = py_ok and chk_ok and frozen_ok
    why = [n for n, v in [("tests", py_ok), ("result checks", chk_ok), ("frozen files changed", frozen_ok)] if not v]
    L.append("STAGE 7 DECISION: PASS" if ok else "STAGE 7 DECISION: FAIL - " + ", ".join(why))
    with open(os.path.join(PROJECT_DIR, "reports", "STAGE_07_REPORT.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(L[-1])


if __name__ == "__main__":
    main()
