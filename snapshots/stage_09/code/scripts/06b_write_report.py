"""Stage 6B - write reports/STAGE_06_REPORT.md from the finished S6 outputs only (results/D02_facial_images_A/, logs/).
No run is re-run or modified. Run scripts/06b_verify.py (-> logs/stage06_verify.txt) and pytest
(-> logs/stage06_pytest.txt) first. Usage: .venv/bin/python scripts/06b_write_report.py"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd  # noqa: E402

from src.data.common import LOG_DIR, PROJECT_DIR, md_table  # noqa: E402
from src.training.checks import validate_model_results  # noqa: E402

DS = "D02_facial_images_A"
RES = os.path.join(PROJECT_DIR, "results", DS)
MODELS = [f"M{i:02d}" for i in range(1, 11)]
CUSTOM_CNN = {"M02", "M05", "M09"}
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


def single(b):
    cm = b["confusion_matrix"]
    return (cm[0][1] + cm[1][1] == 0) or (cm[0][0] + cm[1][0] == 0)


def main():
    # ------------------------------------------------------------------ per-fold records
    rows = []
    for m in MODELS:
        for k in range(1, 11):
            d = os.path.join(RES, m, f"fold_{k:02d}")
            ri = json.load(open(os.path.join(d, "run_info.json")))
            mt = json.load(open(os.path.join(d, "metrics.json")))
            t5, tv = mt["test"]["threshold_0.5"], mt["test"]["threshold_val_youden"]
            rows.append({"model": m, "fold": k, "thr": mt["val_threshold_youden"], "seconds": ri["seconds"], "epochs": ri["epochs_run"],
                         "best_epoch": ri["best_epoch"], "peak_gb": ri["peak_gpu_gb"], "early": ri["stopped_early"],
                         "resumed": ri["resumed"], "status": ri["status"], "auc": t5["roc_auc"],
                         "n_test": t5["n"], "single_class_0.5": single(t5), "single_class_valthr": single(tv),
                         "policy": "full (custom CNN)" if m in CUSTOM_CNN else "last_stage (pretrained)"})
    F = pd.DataFrame(rows)
    S = {m: json.load(open(os.path.join(RES, m, "summary.json"))) for m in MODELS}
    reg = pd.read_csv(os.path.join(PROJECT_DIR, "results", "run_registry.csv"), keep_default_na=False)
    reg = reg[reg.dataset == DS]
    val = {m: validate_model_results(DS, m, os.path.join(PROJECT_DIR, "results"), range(1, 11), return_flags=True)
           for m in MODELS}
    val_ok = all(v[0] == "PASS" for v in val.values())
    n_val_flags = sum(len(v[2]) for v in val.values())
    qlog = R(os.path.join(LOG_DIR, "stage6_queue.log"))
    t_start = re.search(r"===== (\S+) run_stage6.sh start", qlog).group(1)
    t_end, rc = re.findall(r"===== (\S+) run_stage6.sh end \(exit (\d+)\)", qlog)[-1]
    n_launch = len(open(os.path.join(LOG_DIR, "stage6_launch_history.jsonl")).read().splitlines())

    # ------------------------------------------------------------------ tables
    order = sorted(MODELS, key=lambda m: -S[m]["per_fold_metrics"]["test/threshold_0.5/roc_auc"]["mean"])
    main_t, thr_t = [], []
    for m in order:
        pm = S[m]["per_fold_metrics"]
        b = "test/threshold_0.5/"
        main_t.append({"model": m, "architecture": NAMES[m], "ROC-AUC": ms(pm, b + "roc_auc"), "PR-AUC": ms(pm, b + "pr_auc"),
                       "accuracy": ms(pm, b + "accuracy"), "bal. acc.": ms(pm, b + "balanced_accuracy"),
                       "sensitivity": ms(pm, b + "sensitivity"), "specificity": ms(pm, b + "specificity"),
                       "F1": ms(pm, b + "f1"), "Brier": ms(pm, b + "brier")})
        v = "test/threshold_val_youden/"
        thr_t.append({"model": m, "val threshold (mean ± std)": ms(pm, "val_threshold_youden"),
                      "accuracy": ms(pm, v + "accuracy"), "bal. acc.": ms(pm, v + "balanced_accuracy"),
                      "sensitivity": ms(pm, v + "sensitivity"), "specificity": ms(pm, v + "specificity"),
                      "F1": ms(pm, v + "f1"),
                      "pooled OOF ROC-AUC": f"{S[m]['pooled_out_of_fold']['threshold_0.5']['roc_auc']:.3f}"})
    mat = F.pivot(index="model", columns="fold", values="auc").loc[order]
    mat.columns = [f"f{c}" for c in mat.columns]
    mat_disp = mat.round(3).copy()
    mat_disp["min (fold)"] = [f"{mat.loc[m].min():.3f} (f{int(mat.loc[m].idxmin()[1:])})" for m in order]
    mat_disp["max (fold)"] = [f"{mat.loc[m].max():.3f} (f{int(mat.loc[m].idxmax()[1:])})" for m in order]
    mat_disp = mat_disp.reset_index()
    fold_mean = mat.mean(axis=0)
    stats = F.groupby("model").agg(policy=("policy", "first"), gpu_s=("seconds", "sum"), mean_s_per_fold=("seconds", "mean"),
                                   epochs_min=("epochs", "min"), epochs_median=("epochs", "median"),
                                   epochs_max=("epochs", "max"), hit_40=("epochs", lambda e: int((e >= 40).sum())),
                                   best_min=("best_epoch", "min"), best_median=("best_epoch", "median"),
                                   best_max=("best_epoch", "max"), peak_gb=("peak_gb", "max")).loc[MODELS].round(2).reset_index()
    best_hist = F.best_epoch.value_counts().sort_index()

    # ------------------------------------------------------------------ flags (same rules as Stage 5)
    flags = []
    for m in MODELS:
        mean_auc = S[m]["per_fold_metrics"]["test/threshold_0.5/roc_auc"]["mean"]
        if mean_auc < 0.55:
            flags.append(f"{m}: mean ROC-AUC {mean_auc:.3f} < 0.55")
    low = F[F.auc < 0.5]
    for m, g in low.groupby("model"):
        flags.append(f"{m}: {len(g)} fold(s) with ROC-AUC < 0.5: " + ", ".join(f"f{r.fold} {r.auc:.3f}" for r in g.itertuples()))
    flags.append(f"folds with test ROC-AUC < 0.5: {len(low)}; models with mean ROC-AUC < 0.55: "
                 f"{sum(S[m]['per_fold_metrics']['test/threshold_0.5/roc_auc']['mean'] < 0.55 for m in MODELS)}")
    short, long_ = F[F.epochs <= 3], F[F.epochs >= 40]
    flags.append(f"runs stopping at epoch <= 3: {len(short)} (minimum possible with patience 8 is 9 epochs)")
    flags.append(f"runs hitting max_epochs 40: {len(long_)} of 100 (per model: "
                 f"{long_.groupby('model').size().reindex(MODELS, fill_value=0).to_dict()}); "
                 f"{int(long_.model.isin(CUSTOM_CNN).sum())} of them are custom-CNN runs (M02, M05, M09, trained from "
                 f"scratch); {int((~long_.early).sum())} of the {len(long_)} ended without the early-stopping criterion "
                 "being met (validation ROC-AUC still improved within the last 8 epochs), so those runs may not have "
                 "converged under the fixed 40-epoch budget; the others early-stopped exactly at epoch 40")
    b1 = F[F.best_epoch <= 3]
    flags.append(f"runs whose best validation epoch was <= 3: {len(b1)} of 100 "
                 f"(per model: {b1.groupby('model').size().reindex(MODELS, fill_value=0).to_dict()})")
    sc5, scv = F[F["single_class_0.5"]], F[F["single_class_valthr"]]
    flags.append(f"folds predicting a single class at threshold 0.5: {len(sc5)}" + (
        " (" + ", ".join(f"{r.model} f{r.fold}" for r in sc5.itertuples()) + ")" if len(sc5) else ""))
    flags.append(f"folds predicting a single class at the validation threshold: {len(scv)}" + (
        " (" + ", ".join(f"{r.model} f{r.fold}" for r in scv.itertuples()) + ")" if len(scv) else ""))
    ext = F[(F.thr < 0.05) | (F.thr > 0.95)]
    flags.append(f"validation (Youden) thresholds outside [0.05, 0.95]: {len(ext)} of 100 folds (per model: "
                 f"{ext.groupby('model').size().reindex(MODELS, fill_value=0).to_dict()}; overall range "
                 f"{F.thr.min():.3f}-{F.thr.max():.3f}): validation probabilities of these runs are pushed towards 0/1, "
                 "so the chosen threshold sits in a sparse tail and can move a lot between folds")
    flags.append(f"validate_model_results flags (e.g. undefined AUC): {n_val_flags}")
    flags.append(f"resumed runs: {int(F.resumed.sum())}; queue launches: {n_launch}")

    # ------------------------------------------------------------------ evidence files
    verify = R(os.path.join(LOG_DIR, "stage06_verify.txt"))
    pyt = R(os.path.join(LOG_DIR, "stage06_pytest.txt"))
    last = pyt.splitlines()[-1]
    py_ok = re.search(r"(\d+) (failed|error)", last) is None and "passed" in last
    ids_ok = "1c identical test ids per fold across models: True" in verify
    manifest_ok = "1c split hashes vs SPLITS_MANIFEST.json: OK" in verify
    nd = re.search(r"groups spanning > 1 outer fold: (\d+)", verify)
    nd_ok = nd is not None and int(nd.group(1)) == 0
    h04 = re.search(r"Hamming 0-4: (\d+) pairs, of which in different outer folds: (\d+)", verify)
    h58 = re.search(r"Hamming 5-8: (\d+) pairs, of which in different outer folds: (\d+)", verify)
    h910 = re.search(r"Hamming 9-10: (\d+) pairs, of which in different outer folds: (\d+)", verify)
    nd_ok = nd_ok and h04 is not None and int(h04.group(2)) == 0
    files_ok = "unchanged by 06b_verify.py" in verify
    done_ok = len(reg) == 100 and (reg.status == "DONE").all()
    queue_ok = rc == "0" and "QUEUE DONE" in qlog

    O = pd.read_csv(os.path.join(LOG_DIR, "stage06_original_split_auc.csv")).set_index("model_id").loc[order]
    orig_t = pd.DataFrame({
        "model": order,
        "orig 'test' pooled AUC": O.orig_test_pooled_roc_auc.round(3).to_numpy(),
        "orig 'test' per-fold mean ± std": [f"{a:.3f} ± {s:.3f}" for a, s in zip(O.orig_test_fold_mean_roc_auc, O.orig_test_fold_std_roc_auc)],
        "orig train+valid pooled AUC": O.orig_train_valid_pooled_roc_auc.round(3).to_numpy(),
        "orig train+valid per-fold mean ± std": [f"{a:.3f} ± {s:.3f}" for a, s in zip(O.orig_train_valid_fold_mean_roc_auc, O.orig_train_valid_fold_std_roc_auc)],
        "all pooled AUC": O.all_pooled_roc_auc.round(3).to_numpy()})
    diff = (O.orig_test_pooled_roc_auc - O.orig_train_valid_pooled_roc_auc)
    n_orig_test, n_rest = int(O.orig_test_n.iloc[0]), int(O.orig_train_valid_n.iloc[0])

    best = order[0]
    means = {m: S[m]["per_fold_metrics"]["test/threshold_0.5/roc_auc"]["mean"] for m in MODELS}
    stds = {m: S[m]["per_fold_metrics"]["test/threshold_0.5/roc_auc"]["std"] for m in MODELS}
    pmb = S[best]["per_fold_metrics"]["test/threshold_0.5/roc_auc"]
    cnn_means = [means[m] for m in sorted(CUSTOM_CNN)]
    pre_means = [means[m] for m in MODELS if m not in CUSTOM_CNN]
    hardest = fold_mean.idxmin()
    gpu_s = F.seconds.sum()
    L = ["# STAGE 6 REPORT - D02 facial images: 10 hybrid models x 10 folds", "",
         "Screening-support research, not diagnosis. 2,940 facial images (1,470 autistic / 1,470 non-autistic, "
         "web-collected DFDK folder; original train/valid/test sub-folders pooled, DD-05), 2,929 near-duplicate groups "
         "(pHash Hamming <= 4, DD-07); group-wise stratified 10-fold CV, 294 test images per fold; protocol fixed in "
         "Stage 4 (`configs/training.yaml`), no tuning. Each image = one sample; the backbone's final feature map is "
         "the token sequence of the temporal encoder. Test folds were predicted once per run and never used for any "
         "choice. No git: recorded by `snapshots/stage_06/`.", "",
         "## 1. Summary", "",
         f"- Queue: started {t_start}, ended {t_end} (exit {rc}); {n_launch} launch, no resume; "
         f"{int((reg.status == 'DONE').sum())}/100 runs DONE; validate_model_results PASS for "
         f"{sum(v[0] == 'PASS' for v in val.values())}/10 models ({n_val_flags} flags).",
         "- Split hashes re-verified; all 10 models predicted exactly the same 294 test ids in every fold; no "
         "near-duplicate group spans two folds (section 7).",
         f"- Highest mean test ROC-AUC: {best} ({NAMES[best]}) {pmb['mean']:.3f} ± {pmb['std']:.3f}; range of model "
         f"means {min(means.values()):.3f}-{max(means.values()):.3f}.",
         f"- Total GPU-run time {gpu_s / 3600:.2f} h; {len(long_)} of 100 runs reached the 40-epoch limit "
         f"({int(long_.model.isin(CUSTOM_CNN).sum())} of them custom-CNN runs).",
         "", "## 2. Run statistics (from run_info.json)", "",
         f"Total GPU-run time (sum of fold run seconds): {gpu_s / 60:.1f} min ({gpu_s / 3600:.2f} h); queue wall time "
         f"{t_start[11:19]} -> {t_end[11:19]}. Runs ended by early stopping: {int(F.early.sum())}/100 (the other "
         f"{int((~F.early).sum())} reached max_epochs 40). Peak GPU memory over all runs: {F.peak_gb.max():.2f} GB. "
         "Fine-tuning policy: last backbone stage + head for the 7 pretrained models (backbone lr = head lr / 10), the "
         "whole network for the 3 custom CNNs.", "",
         md_table(stats.rename(columns={"gpu_s": "GPU s (10 folds)", "mean_s_per_fold": "s / fold",
                                        "hit_40": "runs at 40", "peak_gb": "peak GB"})), "",
         "Best-epoch distribution over the 100 runs (epoch: count): "
         + ", ".join(f"{int(e)}: {int(c)}" for e, c in best_hist.items()) + ".", "",
         "## 3. Main table - image level, test folds, threshold 0.5 (mean ± std over 10 folds, sorted by ROC-AUC)", "",
         md_table(pd.DataFrame(main_t)), "",
         "### At the validation-chosen threshold (Youden J on each fold's validation images)", "",
         "Pooled OOF ROC-AUC = one ROC-AUC over all 2,940 out-of-fold test predictions (each image predicted once, by "
         "the model of its own fold).", "",
         md_table(pd.DataFrame(thr_t)), "",
         "## 4. Per-fold test ROC-AUC (models x folds)", "",
         md_table(mat_disp), "",
         "Mean over the 10 models per fold: " + ", ".join(f"{c} {v:.3f}" for c, v in fold_mean.items()) + ".", "",
         "## 5. Flags", "", *[f"- {f}" for f in flags], "",
         "## 6. Original dataset split (descriptive only)", "",
         f"The DFDK folder ships its own train/valid/test sub-folders; they were pooled in Stage 2 (DD-05) and kept "
         f"only as `original_split`. Below, the out-of-fold test predictions of each model are split by that column: "
         f"{n_orig_test} images from the original 'test' folder vs the other {n_rest} (original train + valid). "
         "No model was fitted or selected on this; it is a description of where errors fall. Per-fold values on the "
         "original-'test' subset rest on about 30 images per fold and are very coarse "
         "(`logs/stage06_original_split_auc.csv`).", "",
         md_table(orig_t), "",
         f"Pooled AUC on the original 'test' images minus pooled AUC on the rest: {diff.min():+.3f} to "
         f"{diff.max():+.3f} over the 10 models (higher on the original 'test' images for {int((diff > 0).sum())} of 10).",
         "", "## 7. Near-duplicate sanity check (`logs/stage06_verify.txt`, 1e)", "",
         f"- Groups (Stage 2 rule: pHash Hamming <= 4 -> same group): {nd.group(1) if nd else 'UNKNOWN'} groups span "
         "more than one outer fold; no md5 (byte-identical) duplicates exist.",
         f"- Recomputed from manifest.csv over all 4,320,330 image pairs: Hamming 0-4: {h04.group(1)} pairs, "
         f"{h04.group(2)} across folds (rule holds). Below the rule: Hamming 5-8: {h58.group(1)} pairs, {h58.group(2)} "
         f"across folds; Hamming 9-10: {h910.group(1)} pairs, {h910.group(2)} across folds. These pairs were NOT "
         "grouped by the frozen Stage 2 rule (DD-07) and are reported, not acted on; at most "
         f"{2 * int(h58.group(2))} of 2,940 images are in a cross-fold pair at Hamming 5-8. Whether any of them are "
         "the same child (re-crops / edits) is UNKNOWN without subject IDs; Stage 8 may add a sensitivity check that "
         "excludes them.", "",
         "## 8. Limitations", "",
         "These are image-level results. D02 has no subject identifiers, so the same child may appear in several "
         "images and in both training and test folds; grouping by pHash removes only near-identical copies, not "
         "different photos of one child. The images were collected from the web (DFDK dataset), with unknown "
         "diagnosis verification, age range, consent and acquisition conditions, and the two classes may differ in "
         "source, background, lighting or image processing, which a model can exploit instead of facial phenotype. "
         "The results therefore may not generalise to clinical settings or to other populations and must not be read "
         "as diagnostic accuracy. No result here is a diagnosis.", "",
         "## 9. Verification (`logs/stage06_verify.txt`)", "", "```", verify, "```", "",
         "## 10. Tests (`.venv/bin/python -m pytest -v`, `logs/stage06_pytest.txt`)", "",
         "Run after the provisional stage_06 snapshot. `tests/test_stage02_snapshot.py` now picks the latest snapshot "
         "by `SNAPSHOT_INFO.json` datetime instead of by folder name: stage_07 (made early, before S6) sorts after "
         "stage_06 by name, so the name rule would keep checking the outdated stage_07 snapshot. No other test changed.", "",
         f"Result: {last.strip('= ')}", "",
         "<details><summary>full pytest output</summary>", "", "```", pyt, "```", "", "</details>", "",
         "## 11. Neutral description", "",
         f"- The ten models reach mean test ROC-AUCs between {min(means.values()):.3f} and {max(means.values()):.3f} "
         f"with fold-to-fold standard deviations of {min(stds.values()):.3f}-{max(stds.values()):.3f} (sections 3, 4).",
         f"- The three custom CNNs trained from scratch have mean ROC-AUCs {min(cnn_means):.3f}-{max(cnn_means):.3f}; "
         f"the seven pretrained models {min(pre_means):.3f}-{max(pre_means):.3f}. "
         f"{int(long_.model.isin(CUSTOM_CNN).sum())} of the 30 custom-CNN runs reached the 40-epoch limit (section 5).",
         f"- Pooled OOF ROC-AUC (all 2,940 predictions in one ranking) is below the mean per-fold ROC-AUC for "
         f"{sum(S[m]['pooled_out_of_fold']['threshold_0.5']['roc_auc'] < means[m] for m in MODELS)} of 10 models "
         f"(differences {min(S[m]['pooled_out_of_fold']['threshold_0.5']['roc_auc'] - means[m] for m in MODELS):+.3f} to "
         f"{max(S[m]['pooled_out_of_fold']['threshold_0.5']['roc_auc'] - means[m] for m in MODELS):+.3f}); the ten fold "
         "models do not share one probability scale, consistent with the spread of validation thresholds (section 5).",
         f"- Fold {hardest[1:]} has the lowest mean AUC over models ({fold_mean.min():.3f}); fold effects are shared "
         "across models (section 4).",
         "- Differences between models have not been tested statistically; bootstrap CIs and Friedman/Nemenyi follow "
         "in Stage 8.",
         "- No result is a diagnosis or a clinical accuracy estimate.", "",
         "## 12. Open items", "",
         "1. Stage 8: shortcut controls (D01 quality-only baseline; D02 near-duplicate / original-split sensitivity), "
         "bootstrap CIs, Friedman/Nemenyi.",
         "2. Stage 9 figures and the final results report; Stage 10 final audit.", ""]
    ok = queue_ok and done_ok and val_ok and manifest_ok and ids_ok and nd_ok and files_ok and py_ok
    why = [n for n, v in [("queue exit", queue_ok), ("100 DONE", done_ok), ("validate_model_results", val_ok),
                          ("split hashes", manifest_ok), ("identical test ids", ids_ok), ("near-duplicate groups", nd_ok),
                          ("result files unchanged", files_ok), ("pytest", py_ok)] if not v]
    L.append("STAGE 6 DECISION: PASS" if ok else "STAGE 6 DECISION: FAIL - " + ", ".join(why))
    with open(os.path.join(PROJECT_DIR, "reports", "STAGE_06_REPORT.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(L[-1])


if __name__ == "__main__":
    main()
