"""Stage 5B - write reports/STAGE_05_REPORT.md from the finished S5 outputs only (results/D01_eye_tracking/, logs/).
No run is re-run or modified. Usage: .venv/bin/python scripts/05b_write_report.py"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.common import LOG_DIR, PROJECT_DIR, md_table  # noqa: E402
from src.training.checks import validate_model_results  # noqa: E402
from src.training.metrics import binary_metrics  # noqa: E402

DS = "D01_eye_tracking"
RES = os.path.join(PROJECT_DIR, "results", DS)
MODELS = [f"M{i:02d}" for i in range(1, 11)]
NAMES = {"M01": "EfficientNetB3 + BiLSTM + Attn", "M02": "CNN + BiLSTM + Attn", "M03": "ResNet50 + LSTM + Attn",
         "M04": "ConvNeXt + BiLSTM + Attn", "M05": "CNN + Transformer + MLP", "M06": "DenseNet121 + LSTM + Attn",
         "M07": "ViT + LSTM + Attn", "M08": "EfficientNet + Transformer + Attn", "M09": "CNN + GRU + Attn",
         "M10": "ResNet50 + BiLSTM + Transformer"}


def R(p):
    with open(p) as fh:
        return fh.read().rstrip("\n")


def ms(pm, key, nd=3):
    e = pm[key]
    return f"{e['mean']:.{nd}f} ± {e['std']:.{nd}f}"


def main():
    # ------------------------------------------------------------------ per-fold records
    rows = []
    for m in MODELS:
        for k in range(1, 11):
            d = os.path.join(RES, m, f"fold_{k:02d}")
            ri = json.load(open(os.path.join(d, "run_info.json")))
            mt = json.load(open(os.path.join(d, "metrics.json")))
            t5, tv = mt["test"]["threshold_0.5"], mt["test"]["threshold_val_youden"]
            single = lambda b: (b["confusion_matrix"][0][1] + b["confusion_matrix"][1][1] == 0) or \
                (b["confusion_matrix"][0][0] + b["confusion_matrix"][1][0] == 0)  # noqa: E731
            rows.append({"model": m, "fold": k, "seconds": ri["seconds"], "epochs": ri["epochs_run"],
                         "best_epoch": ri["best_epoch"], "peak_gb": ri["peak_gpu_gb"], "early": ri["stopped_early"],
                         "status": ri["status"], "auc": t5["roc_auc"], "child_auc": mt["test_child"]["threshold_0.5"]["roc_auc"],
                         "single_class_0.5": single(t5), "single_class_valthr": single(tv),
                         "n_test_children": mt["test_child"]["threshold_0.5"]["n"], "input": ri.get("feature_cache") and
                         "cached features" or "clips (end to end)"})
    F = pd.DataFrame(rows)
    S = {m: json.load(open(os.path.join(RES, m, "summary.json"))) for m in MODELS}
    reg = pd.read_csv(os.path.join(PROJECT_DIR, "results", "run_registry.csv"), keep_default_na=False)
    reg = reg[reg.dataset == DS]
    val = {m: validate_model_results(DS, m, os.path.join(PROJECT_DIR, "results"), range(1, 11), return_flags=True)
           for m in MODELS}
    val_ok = all(v[0] == "PASS" for v in val.values())
    qlog = R(os.path.join(LOG_DIR, "stage5_queue.log"))
    t_start = re.search(r"===== (\S+) run_stage5.sh start", qlog).group(1)
    t_end, rc = re.search(r"===== (\S+) run_stage5.sh end \(exit (\d+)\)", qlog).groups()

    # ------------------------------------------------------------------ tables
    order = sorted(MODELS, key=lambda m: -S[m]["per_fold_metrics"]["test/threshold_0.5/roc_auc"]["mean"])
    main_t, thr_t, child_t = [], [], []
    for m in order:
        pm = S[m]["per_fold_metrics"]
        b = "test/threshold_0.5/"
        main_t.append({"model": m, "architecture": NAMES[m], "ROC-AUC": ms(pm, b + "roc_auc"), "PR-AUC": ms(pm, b + "pr_auc"),
                       "accuracy": ms(pm, b + "accuracy"), "bal. acc.": ms(pm, b + "balanced_accuracy"),
                       "sensitivity": ms(pm, b + "sensitivity"), "specificity": ms(pm, b + "specificity"),
                       "F1": ms(pm, b + "f1"), "Brier": ms(pm, b + "brier")})
        v = "test/threshold_val_youden/"
        thr_t.append({"model": m, "val threshold (mean ± std)": ms(pm, "val_threshold_youden"),
                      "bal. acc.": ms(pm, v + "balanced_accuracy"), "sensitivity": ms(pm, v + "sensitivity"),
                      "specificity": ms(pm, v + "specificity"), "F1": ms(pm, v + "f1")})
        pc = S[m]["pooled_out_of_fold_child"]
        child_t.append({"model": m, "pooled child ROC-AUC (all test children)": round(pc["threshold_0.5"]["roc_auc"], 3),
                        "n children": pc["threshold_0.5"]["n"],
                        "child accuracy @0.5": round(pc["threshold_0.5"]["accuracy"], 3),
                        "child bal. acc. @0.5": round(pc["threshold_0.5"]["balanced_accuracy"], 3),
                        "child accuracy @val thr": round(pc["threshold_val_youden_per_fold"]["accuracy"], 3),
                        "per-fold child AUC (mean ± std)": ms(pm, "test_child/threshold_0.5/roc_auc"),
                        "pooled clip ROC-AUC": round(S[m]["pooled_out_of_fold"]["threshold_0.5"]["roc_auc"], 3)})
    n_children = {r["n children"] for r in child_t}
    mat = F.pivot(index="model", columns="fold", values="auc").loc[order]
    mat.columns = [f"f{c}" for c in mat.columns]
    mat_disp = mat.round(3).copy()
    mat_disp["min (fold)"] = [f"{mat.loc[m].min():.3f} (f{int(mat.loc[m].idxmin()[1:])})" for m in order]
    mat_disp["max (fold)"] = [f"{mat.loc[m].max():.3f} (f{int(mat.loc[m].idxmax()[1:])})" for m in order]
    mat_disp = mat_disp.reset_index()
    stats = F.groupby("model").agg(input=("input", "first"), gpu_s=("seconds", "sum"), mean_s_per_fold=("seconds", "mean"),
                                   epochs_min=("epochs", "min"), epochs_median=("epochs", "median"),
                                   epochs_max=("epochs", "max"), best_min=("best_epoch", "min"),
                                   best_median=("best_epoch", "median"), best_max=("best_epoch", "max"),
                                   peak_gb=("peak_gb", "max")).loc[MODELS].round(2).reset_index()
    best_hist = F.best_epoch.value_counts().sort_index()

    # ------------------------------------------------------------------ flags
    flags = []
    for m in MODELS:
        mean_auc = S[m]["per_fold_metrics"]["test/threshold_0.5/roc_auc"]["mean"]
        if mean_auc < 0.55:
            flags.append(f"{m}: mean clip ROC-AUC {mean_auc:.3f} < 0.55")
    low = F[F.auc < 0.5]
    for m, g in low.groupby("model"):
        flags.append(f"{m}: {len(g)} fold(s) with clip ROC-AUC < 0.5: " + ", ".join(f"f{r.fold} {r.auc:.3f}" for r in g.itertuples()))
    short, long_ = F[F.epochs <= 3], F[F.epochs >= 40]
    flags.append(f"runs stopping at epoch <= 3: {len(short)} (minimum possible with patience 8 is 9 epochs)")
    flags.append(f"runs hitting max_epochs 40: {len(long_)}" + (": " + ", ".join(f"{r.model} f{r.fold}" for r in long_.itertuples()) if len(long_) else ""))
    b1 = F[F.best_epoch <= 3]
    flags.append(f"runs whose best validation epoch was <= 3 (early stop at epoch best+8): {len(b1)} of 100 "
                 f"(per model: {b1.groupby('model').size().reindex(MODELS, fill_value=0).to_dict()})")
    sc5, scv = F[F["single_class_0.5"]], F[F["single_class_valthr"]]
    flags.append(f"folds predicting a single class at threshold 0.5: {len(sc5)}" + (
        " (" + ", ".join(f"{r.model} f{r.fold}" for r in sc5.itertuples()) + ")" if len(sc5) else ""))
    flags.append(f"folds predicting a single class at the validation threshold: {len(scv)}" + (
        " (" + ", ".join(f"{r.model} f{r.fold}" for r in scv.itertuples()) + ")" if len(scv) else ""))

    # ------------------------------------------------------------------ quick shortcut indicator
    q = pd.read_csv(os.path.join(PROJECT_DIR, "cache", DS, "quality_per_child.csv"))
    auc_q = binary_metrics(q.label, q.mean_valid_ratio)["roc_auc"]
    q_dir, q_auc = ("mean valid_ratio (higher -> ASD)", auc_q) if auc_q >= 0.5 else \
        ("1 - mean valid_ratio (lower valid_ratio -> ASD)", 1 - auc_q)
    best_child = max(child_t, key=lambda r: r["pooled child ROC-AUC (all test children)"])

    # ------------------------------------------------------------------ evidence files
    diff = R(os.path.join(LOG_DIR, "stage05_checker_fix.diff"))
    reval = R(os.path.join(LOG_DIR, "stage05_revalidate.txt"))
    ctests = R(os.path.join(LOG_DIR, "stage05_checker_tests.txt"))
    verify = R(os.path.join(LOG_DIR, "stage05_verify.txt"))
    pyt = R(os.path.join(LOG_DIR, "stage05_pytest.txt"))
    last = pyt.splitlines()[-1]
    py_ok = re.search(r"(\d+) (failed|error)", last) is None and "passed" in last
    hashes_ok = "RESULT FILES IDENTICAL" in reval
    ids_ok = "identical test ids per fold across models: True" in verify
    done_ok = len(reg) == 100 and (reg.status == "DONE").all()

    best = order[0]
    means = {m: S[m]["per_fold_metrics"]["test/threshold_0.5/roc_auc"]["mean"] for m in MODELS}
    stds = {m: S[m]["per_fold_metrics"]["test/threshold_0.5/roc_auc"]["std"] for m in MODELS}
    child_pooled = {r["model"]: r["pooled child ROC-AUC (all test children)"] for r in child_t}
    clip_pooled = {r["model"]: r["pooled clip ROC-AUC"] for r in child_t}
    n_child_higher = sum(child_pooled[m] > clip_pooled[m] for m in MODELS)
    n_best_le3 = int((F.best_epoch <= 3).sum())
    pmb = S[best]["per_fold_metrics"]["test/threshold_0.5/roc_auc"]
    L = ["# STAGE 5 REPORT - D01 eye-tracking: 10 hybrid models x 10 folds", "",
         "Screening-support research, not diagnosis. 57 children (27 ASD / 30 TD), 1,241 clips of 8 x 1-s gaze windows; "
         "child-grouped 10-fold CV (5-6 test children and 81-170 test clips per fold); protocol fixed in Stage 4 "
         "(`configs/training.yaml`), no tuning. Test folds were predicted once per run and never used for any choice. "
         "No git: recorded by `snapshots/stage_05/`.", "",
         "## 1. Summary", "",
         f"- Queue: started {t_start}, ended {t_end} (exit {rc}); {int((reg.status == 'DONE').sum())}/100 runs DONE; "
         f"validate_model_results PASS for {sum(v[0] == 'PASS' for v in val.values())}/10 models (re-checked after the "
         "checker fix, no flags).",
         f"- Split hashes re-verified; all 10 models predicted exactly the same test ids in every fold.",
         f"- Highest mean clip-level test ROC-AUC: {best} ({NAMES[best]}) {pmb['mean']:.3f} ± {pmb['std']:.3f}; "
         f"range of model means {min(means.values()):.3f}-{max(means.values()):.3f}.",
         f"- Child level (pooled, each of the {n_children.pop() if len(n_children) == 1 else n_children} children once): "
         f"highest ROC-AUC {best_child['pooled child ROC-AUC (all test children)']:.3f} ({best_child['model']}). "
         f"Quality-only indicator ({q_dir}): child ROC-AUC {q_auc:.3f} (section 7).",
         f"- Checker fix (validation code only): all 630 stored result files byte-identical before/after (section 8).",
         "", "## 2. Run statistics (from run_info.json)", "",
         f"Total GPU-run time (sum of fold run seconds): {F.seconds.sum() / 60:.1f} min ({F.seconds.sum() / 3600:.2f} h); "
         f"queue wall time {t_start[11:19]} -> {t_end[11:19]}. All 100 runs ended by early stopping: "
         f"{bool(F.early.all())}. Peak GPU memory over all runs: {F.peak_gb.max():.2f} GB.", "",
         md_table(stats.rename(columns={"gpu_s": "GPU s (10 folds)", "mean_s_per_fold": "s / fold", "peak_gb": "peak GB"})), "",
         "Best-epoch distribution over the 100 runs (epoch: count): "
         + ", ".join(f"{int(e)}: {int(c)}" for e, c in best_hist.items()) + ".", "",
         "## 3. Main table - clip level, test folds, threshold 0.5 (mean ± std over 10 folds, sorted by ROC-AUC)", "",
         md_table(pd.DataFrame(main_t)), "",
         "### At the validation-chosen threshold (Youden J on each fold's validation clips)", "",
         md_table(pd.DataFrame(thr_t)), "",
         "## 4. Child level", "",
         "Child probability = mean clip probability of that child. Pooled = every child predicted once, by the model of "
         "the fold in which it was a test child (out-of-fold), then one ROC-AUC over all children. Per-fold child AUCs "
         "rest on 5-6 children each and are very coarse.", "",
         md_table(pd.DataFrame(child_t)), "",
         "## 5. Per-fold clip-level test ROC-AUC (models x folds)", "",
         md_table(mat_disp), "",
         "## 6. Flags", "", *[f"- {f}" for f in flags], "",
         "## 7. Quick shortcut indicator (descriptive only; full analysis in Stage 8)", "",
         f"From `cache/{DS}/quality_per_child.csv` (57 children, DD-12 quality features, never a model input): the "
         f"child-level ROC-AUC of mean valid_ratio alone is {auc_q:.3f}; the direction above 0.5 is **{q_dir}**, "
         f"ROC-AUC **{q_auc:.3f}**. This is a single descriptive number over all 57 children (no fitting; the direction "
         "was chosen after looking, so it is optimistic). For comparison, the models' pooled child-level ROC-AUCs "
         f"range {min(r['pooled child ROC-AUC (all test children)'] for r in child_t):.3f}-"
         f"{best_child['pooled child ROC-AUC (all test children)']:.3f} (section 4). Stage 8 must test whether the "
         "models' performance exceeds what tracking quality alone explains.", "",
         "## 8. Checker fix (Stage 5B, validation code only)", "",
         "`src/training/metrics.is_valid_metric_block` and `src/training/checks.validate_model_results`: the range "
         "check is now inclusive with a tolerance (0 - 1e-9 <= value <= 1 + 1e-9) for rates, ROC-AUC, PR-AUC and "
         "Brier (sklearn's average precision can return 1.0000000000000002, seen in Stage 7); NaN is accepted only for "
         "ROC-AUC / PR-AUC when a block genuinely has a single class, and is then reported as an explicit flag "
         "(`validate_model_results(..., return_flags=True)`), never passed silently (before: a NaN AUC passed silently). "
         "No metric computation changed. New tests `tests/test_stage05_checker.py`.", "",
         "<details><summary>diff (logs/stage05_checker_fix.diff)</summary>", "", "```diff", diff, "```", "", "</details>", "",
         "Metrics tests (`logs/stage05_checker_tests.txt`):", "", "```", ctests, "```", "",
         "Re-validation and result-file hash comparison (`logs/stage05_revalidate.txt`; sha256 of all 630 "
         "metrics.json / predictions.csv / summary.json under results/ recorded BEFORE the change):", "",
         "```", reval, "```", "",
         "## 9. Verification (`logs/stage05_verify.txt`)", "", "```", verify, "```", "",
         "## 10. Tests (`.venv/bin/python -m pytest -v`, `logs/stage05_pytest.txt`)", "",
         "Run after the provisional stage_05 snapshot. Before it, the full suite had exactly two failures "
         "(`logs/stage05_pytest_before_snapshot.txt`): the stage_04 snapshot test (expected: scripts/ gained the three "
         "Stage 5A files; src/training/metrics.py and checks.py changed with the checker fix; tests/ gained "
         "test_stage05_checker.py; reports/ gained STAGE_07_REPORT.md) and the CLAUDE.md length test (274 > 260 lines, "
         "caused by the Stage 5A/7 session-log entries; fixed by condensing the session log, not by relaxing the "
         "test).", "",
         f"Result: {last.strip('= ')}", "",
         "<details><summary>full pytest output</summary>", "", "```", pyt, "```", "", "</details>", "",
         "## 11. Neutral description", "",
         f"- On clip level the ten models reach mean test ROC-AUCs between {min(means.values()):.3f} and "
         f"{max(means.values()):.3f}, with fold-to-fold standard deviations of {min(stds.values()):.3f}-"
         f"{max(stds.values()):.3f} (sections 3, 5): with 5-6 test "
         "children per fold, which children land in a fold changes the result strongly. Clips of one child are not "
         "independent, so clip-level numbers overstate the effective sample size.",
         f"- The pooled child-level ROC-AUC is higher than the pooled clip-level ROC-AUC for {n_child_higher} of 10 "
         "models (section 4); a child score averages many clips.",
         f"- {n_best_le3} of 100 runs had their best validation epoch at epoch <= 3 (sections 2, 6); with 7-8 "
         "validation children the validation AUC used for early stopping is noisy.",
         f"- The tracking-quality indicator alone reaches a child-level ROC-AUC of {q_auc:.3f} (section 7). Until Stage 8 "
         "has run the shortcut controls, these results must not be read as evidence that the models learned gaze "
         "behaviour rather than recording-quality differences.",
         "- No result is a diagnosis or a clinical accuracy estimate.", "",
         "## 12. Open items", "",
         "1. Stage 8: quality-only and stimulus-aware baselines with the same folds; bootstrap CIs; Friedman/Nemenyi.",
         "2. Stage 6 (D02) is NOT launched: awaiting the user's review of these results.", ""]
    ok = done_ok and val_ok and hashes_ok and ids_ok and py_ok
    why = [n for n, v in [("100 DONE", done_ok), ("validate_model_results", val_ok), ("result hashes", hashes_ok),
                          ("identical test ids", ids_ok), ("pytest", py_ok)] if not v]
    L.append("STAGE 5 DECISION: PASS" if ok else "STAGE 5 DECISION: FAIL - " + ", ".join(why))
    with open(os.path.join(PROJECT_DIR, "reports", "STAGE_05_REPORT.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(L[-1])


if __name__ == "__main__":
    main()
