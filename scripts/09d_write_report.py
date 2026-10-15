"""Stage 9 D - write reports/FINAL_RESULTS_REPORT.md (+ .html via pandoc, self-contained) from results/final/*.csv only
(Stage 8 rules / conclusions copied verbatim from reports/STAGE_08_REPORT.md).
Usage: .venv/bin/python scripts/09d_write_report.py"""
import os
import re
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd  # noqa: E402

from src.analysis.data import CUSTOM_CNN, MODELS, NAMES, TAB_MODELS  # noqa: E402
from src.analysis.final import FINAL_DIR  # noqa: E402
from src.data.common import PROJECT_DIR, md_table  # noqa: E402

REP = os.path.join(PROJECT_DIR, "reports")
OUT_MD = os.path.join(REP, "FINAL_RESULTS_REPORT.md")
OUT_HTML = os.path.join(REP, "FINAL_RESULTS_REPORT.html")


def rd(n):
    return pd.read_csv(os.path.join(FINAL_DIR, n))


MT, SCT, STT = rd("master_table.csv"), rd("shortcut_table.csv"), rd("stats_table.csv")
DST, REPRO, CM = rd("datasets_table.csv"), rd("reproducibility_table.csv"), rd("confusion_matrices_best.csv")
ATT = rd("d01_m07_attention_per_second.csv")


def M(ds, m, col):
    return float(MT[(MT.dataset == ds) & (MT.model == m)][col].iloc[0])


def SC(sec, m, q):
    return float(SCT[(SCT.section == sec) & (SCT.model == m) & (SCT.quantity == q)].value.iloc[0])


def ST(sec, m, q):
    return float(STT[(STT.section == sec) & (STT.model == m) & (STT.quantity == q)].value.iloc[0])


def f3(x):
    return f"{x:.3f}"


def pf(p):
    return f"{p:.2g}" if p < 0.001 else f"{p:.3f}"


def ci(v, lo, hi, signed=False):
    f = "{:+.3f}" if signed else "{:.3f}"
    return f"{f.format(v)} [{f.format(lo)}, {f.format(hi)}]"


def rng(ds, col, models=None):
    g = MT[MT.dataset == ds]
    if models is not None:
        g = g[g.model.isin(models)]
    return g[col].min(), g[col].max()


def best(ds, col="pooled_auc"):
    g = MT[(MT.dataset == ds) & (MT.model != "dummy_majority")]
    return g.sort_values(col, ascending=False).iloc[0]


def ds_table(ds, sort="roc_auc_mean"):
    g = MT[MT.dataset == ds].sort_values(sort, ascending=False)
    pm = lambda c: [f"{a:.3f} ± {b:.3f}" for a, b in zip(g[c + "_mean"], g[c + "_std"])]  # noqa: E731
    return md_table(pd.DataFrame({"model": [f"{m} {NAMES.get(m, '')}".strip() for m in g.model], "ROC-AUC": pm("roc_auc"),
                                  "pooled AUC [95% CI]": [ci(a, b, c) for a, b, c in zip(g.pooled_auc, g.pooled_auc_ci_lo, g.pooled_auc_ci_hi)],
                                  "PR-AUC": pm("pr_auc"), "bal. acc.": pm("balanced_accuracy"), "sens.": pm("sensitivity"),
                                  "spec.": pm("specificity"), "F1": pm("f1"), "Brier": pm("brier"), "ECE": g.ece_pooled.map(f3)}))


def section(text, start, end):
    i = text.index(start)
    j = text.index(end, i)
    return text[i:j].strip()


def repro(item_prefix):
    r = REPRO[REPRO["item"].str.startswith(item_prefix)]
    return r.value.iloc[0]


def main():
    s8 = open(os.path.join(REP, "STAGE_08_REPORT.md")).read()
    rules = section(s8, "### 3.6", "## 4.").replace("### 3.6 ", "#### ")
    claims = section(s8, "## 7. What can and cannot be claimed", "## 8.").replace("## 7. What can and cannot be claimed", "").strip()
    claims = claims.replace("Written from the numbers above and the rule in section 3.6.", "").strip()
    claims = re.sub(r"\*\*(D01 eye-tracking|D02 facial images|Tabular and across datasets)\*\*", r"**\1**\n", claims)

    d01c, d01k, d02 = "D01 clip", "D01 child", "D02"
    q3 = SC("D01 baseline", "quality_LR3", "pooled_auc")
    q3ci = (SC("D01 baseline", "quality_LR3", "pooled_auc_ci_lo"), SC("D01 baseline", "quality_LR3", "pooled_auc_ci_hi"))
    k_lo, k_hi = rng(d01k, "pooled_auc")
    c_lo, c_hi = rng(d01c, "pooled_auc")
    f2_lo, f2_hi = rng(d02, "roc_auc_mean")
    rf = SC("D02 baseline", "image_stats_random_forest", "fold_mean_auc")
    lr = SC("D02 baseline", "image_stats_logreg", "fold_mean_auc")
    nd = [SC("D02", m, "fold_mean_diff") for m in MODELS]
    adds = [m for m in MODELS if SC("D01", m, "adds_information_beyond_quality") == 1]
    worse = [m for m in MODELS if SC("D01", m, "diff_ci_hi_vs_LR3") < 0]
    b_d02_fold = best(d02, "roc_auc_mean")
    b_d02 = best(d02)
    pre = {k: ST(f"pretrained_vs_scratch {k}", "pretrained (M01,M03,M04,M06,M07,M08,M10)", "mean_of_fold_mean_auc") for k in ["D01 clip", "D01 child", "D02 image"]}
    scr = {k: ST(f"pretrained_vs_scratch {k}", "from scratch (M02,M05,M09)", "mean_of_fold_mean_auc") for k in ["D01 clip", "D01 child", "D02 image"]}
    gpu5, gpu6 = float(repro("S5 D01 GPU seconds")), float(repro("S6 D02 GPU seconds"))
    n_d01 = DST.set_index("dataset").loc["D01_eye_tracking"]
    n_d02 = DST.set_index("dataset").loc["D02_facial_images_A"]

    # Stage 10 wording fix: matched-subset statement from results/final/shortcut_table.csv (CI excludes 0.5 only for `adds`)
    excl = [m for m in MODELS if SC("D01", m, "matched_ci_lo") > 0.5 or SC("D01", m, "matched_ci_hi") < 0.5]
    assert len(excl) == 1, excl
    mm = excl[0]
    matched_sentence = (f"only one model ({mm}) had a matched-subset AUC whose 95% bootstrap CI excluded 0.5 "
                        f"({SC('D01', mm, 'matched_auc'):.3f} [{SC('D01', mm, 'matched_ci_lo'):.3f}, {SC('D01', mm, 'matched_ci_hi'):.3f}], "
                        f"{int(SC('D01 baseline', 'matching', 'n_pairs'))} matched pairs); the other models' matched AUCs had CIs "
                        "including 0.5.")
    abstract = (
        f"We trained and compared ten hybrid deep-learning architectures (a pretrained or custom-CNN spatial encoder, a "
        f"recurrent or Transformer temporal encoder and an attention head) for autism risk screening on two public datasets: "
        f"eye-tracking recordings of {int(n_d01.n_groups)} children (D01) and {int(n_d02.n_samples):,} web-collected facial "
        f"images (D02). All models used one fixed training protocol and grouped 10-fold cross-validation; tabular baselines "
        f"were run on three questionnaire datasets. On D01, child-level pooled ROC-AUC ranged from {k_lo:.3f} to {k_hi:.3f}, "
        f"but a logistic regression on three recording-quality features alone reached {q3:.3f}. No model was significantly "
        f"better than this baseline, and {matched_sentence} On D02, models reached mean ROC-AUCs of {f2_lo:.3f}-{f2_hi:.3f}, yet simple non-facial image "
        f"statistics alone reached {rf:.3f}, so dataset artefacts separate the classes. Removing near-duplicate test images "
        f"changed AUCs by at most {max(abs(x) for x in nd):.3f}. Pretrained encoders scored higher on average than custom CNNs on "
        f"both datasets. The results do not show that the models learned autism-related gaze or facial patterns and are not "
        f"clinical accuracy estimates; they motivate fusion that weights each modality by its estimated reliability.")
    n_words = len(abstract.split())

    arch = pd.DataFrame([
        ("M01", "EfficientNet-B3 (efficientnet_b3.ra2_in1k)", "BiLSTM", "temporal attention"),
        ("M02", "custom CNN (from scratch)", "BiLSTM", "temporal attention"),
        ("M03", "ResNet-50 (resnet50.a1_in1k)", "LSTM", "temporal attention"),
        ("M04", "ConvNeXt-Tiny (convnext_tiny.in12k_ft_in1k)", "BiLSTM", "temporal attention"),
        ("M05", "custom CNN (from scratch)", "Transformer encoder", "MLP head"),
        ("M06", "DenseNet-121 (densenet121.ra_in1k)", "LSTM", "temporal attention"),
        ("M07", "ViT-B/16 (vit_base_patch16_224.augreg2_in21k_ft_in1k)", "LSTM", "temporal attention"),
        ("M08", "EfficientNet-B0 (efficientnet_b0.ra_in1k)", "Transformer encoder", "temporal attention"),
        ("M09", "custom CNN (from scratch)", "GRU", "temporal attention"),
        ("M10", "ResNet-50 (resnet50.a1_in1k)", "BiLSTM -> Transformer", "mean pooling")],
        columns=["id", "spatial encoder (timm weights)", "temporal encoder", "head"])
    arch.insert(1, "architecture", arch.id.map(NAMES))

    dtab = DST.copy()
    dtab["n (pos / neg)"] = [f"{int(a):,} ({int(b):,} / {int(c):,})" for a, b, c in zip(dtab.n_samples, dtab.n_positive, dtab.n_negative)]
    plural = {"child": "children", "near-duplicate group": "near-duplicate groups", "duplicate-feature group": "duplicate-feature groups"}
    dtab["groups"] = ["-" if pd.isna(g) else f"{int(g):,} {plural.get(u, u)}" + (f" ({int(gp)} ASD / {int(gn)} TD)" if not pd.isna(gp) else "")
                      for g, u, gp, gn in zip(dtab.n_groups, dtab.group_unit, dtab.n_groups_positive, dtab.n_groups_negative)]
    dtab = dtab[["dataset", "source_folder", "modality", "sample_unit", "n (pos / neg)", "groups", "role", "exclusions"]]

    cm = CM.set_index("dataset")
    fr = {t: (ST(t, "-", "chi2"), ST(t, "-", "p"), int(ST(t, "-", "n_sig_pairs_nemenyi"))) for t in ["D01_clip", "D01_child", "D02_image"]}
    nb_sig = {t: [r.model.split(" vs ")[1] for r in STT[(STT.section == t) & (STT.quantity == "nadeau_bengio_p_holm")].itertuples() if r.value < 0.05]
              for t in ["D01_clip", "D01_child", "D02_image"]}
    nb_best = {t: STT[(STT.section == t) & (STT.quantity == "nadeau_bengio_p")].model.iloc[0].split(" vs ")[0] for t in nb_sig}
    att8 = ATT[ATT.second == 8].set_index("group")
    att1 = ATT[ATT.second == 1].set_index("group")
    fold_md = open(os.path.join(FINAL_DIR, "fold_auc_matrix.md")).read().split("\n", 1)[1].replace("\n## ", "\n### ")
    shortcut_md = open(os.path.join(FINAL_DIR, "shortcut_table.md")).read()
    d01_short = section(shortcut_md, "| model | child AUC", "## D02").strip()
    d02_short = shortcut_md[shortcut_md.index("| model | pooled AUC"):].strip()
    stats_md = open(os.path.join(FINAL_DIR, "stats_table.md")).read()
    snap = REPRO[REPRO["item"].str.contains("code_hash")]
    tab_best = {k: best(k) for k in ["D04", "D05 set_all", "D05 set_honest", "D06"]}
    ece_rng = {k: rng(k, "ece_pooled", MODELS) for k in [d01c, d01k, d02]}

    L = [
        "# Reliability-Weighted Multimodal AI for Early Autism Risk Screening: comparison of ten hybrid deep-learning architectures",
        "", "BITE308L Artificial Intelligence (VIT) - project results report. Team: Samith Kolur (24BIT0117), Iniya K, Chandru Vel Kumar.",
        "", "> Screening-support research only. No result in this report is a diagnosis or an estimate of clinical accuracy.", "",
        "## 1. Abstract", "", abstract, "",
        "## 2. Objective", "",
        "Implement the ten hybrid architectures suggested by our faculty, train and evaluate them with the same protocol and "
        "10-fold cross-validation on a local GPU, and compare them on an eye-tracking dataset (primary) and a facial-image "
        "dataset (secondary), with tabular screening questionnaires as non-deep baselines. A second objective, added after the "
        "first results, was to test whether the models' performance can be explained by shortcuts in the data (recording "
        "quality, image artefacts, near-duplicate images) rather than by autism-related patterns.", "",
        "## 3. Datasets", "",
        md_table(dtab), "",
        "Sources: `results/final/datasets_table.csv` (counts from the frozen split files); full decisions in "
        "`data_registry/DATASET_DECISIONS.md` (DD-01 ... DD-12). No datasets are merged and no predictions are combined across "
        "datasets (DD-11).", "",
        "## 4. Methods", "",
        "### 4.1 Preprocessing", "",
        "- **D01**: gaze rows without a participant ID were dropped and cross-file duplicate rows removed before any windowing; "
        "recordings were split at gaps > 500 ms; each clip = 8 consecutive non-overlapping 1-s windows with at least 50% valid "
        "gaze, each window rendered as a 3-channel 224 x 224 image (scan path, fixation density, time); at most 40 clips per "
        "child. Child score = mean clip probability. Per-child tracking-quality features (valid ratio, blink ratio, tracking "
        "ratio) were stored separately and never used as model inputs.",
        "- **D02**: images resized to 224 and centre-cropped; training-only augmentation (random resized crop 0.8-1.0, "
        "horizontal flip, colour jitter 0.1). Images with pHash Hamming distance <= 4 form one group.",
        "- **Tabular (D04-D06)**: imputers, encoders and scalers fit on each fold's training part only.", "",
        "### 4.2 The ten architectures", "",
        "Shared design: per-step spatial encoder -> linear projection -> temporal encoder -> attention pooling (MLP head for "
        "M05, mean pooling for M10) -> classifier -> P(ASD). D01 sequences run over time (8 one-second images); D02 sequences "
        "run over spatial tokens of the encoder's final feature map (49, or 196 for ViT and the custom CNN). Custom CNN = "
        "4 x [Conv3x3-BN-ReLU-MaxPool], 32-64-128-256 channels, random initialisation.", "",
        md_table(arch), "",
        "### 4.3 Fixed training protocol (no tuning on test data)", "",
        "AdamW (weight decay 0.01); head learning rate 5e-4 for pretrained models and 1e-3 for custom CNNs; D01 pretrained "
        "backbones frozen (cached features), D02 pretrained backbones fine-tune their last stage at lr/10, custom CNNs train "
        "fully; cosine schedule with 2-epoch warm-up; at most 40 epochs; early stopping on validation ROC-AUC (patience 8); "
        "class-weighted BCE; gradient clipping 1.0; mixed precision; batch size 64; seed 42 + fold. Thresholds: 0.5 and Youden J "
        "on the validation part. Hardware: one NVIDIA RTX A2000 12 GB.", "",
        "### 4.4 Cross-validation and leakage controls", "",
        "Outer StratifiedGroupKFold(10, shuffle, seed 42) by group (D01 child, D02 near-duplicate group, tabular identical-"
        "feature-row group); the validation part = one group-wise split of the training groups (about 15%). Splits were made "
        "once, hashed, and verified before every run; every run asserted that no test id appeared in a training or validation "
        "batch; test folds were predicted once from the best validation checkpoint and never used for any choice. All ten "
        "models were checked to predict exactly the same test ids in every fold.", "",
        "### 4.5 Statistics and shortcut controls", "",
        "Pooled out-of-fold AUC with 95% bootstrap CIs (2,000 resamples, seed 42, stratified, resampling children / "
        "near-duplicate groups / feature groups); Friedman test on the 10 x 10 fold-AUC matrix with Nemenyi post-hoc and "
        "critical-difference diagrams; Nadeau-Bengio corrected resampled t-test (best vs each, Holm-adjusted); Brier score and "
        "expected calibration error (10 bins). Shortcut controls (Stage 8, no retraining): D01 quality-only baselines on the "
        "same folds, paired bootstrap and DeLong comparison, Spearman dependence, quality tertiles, 1:1 quality matching "
        "(caliper 0.05) and an exploratory incremental-value analysis; D02 near-duplicate sensitivity (pHash <= 8), an "
        "image-statistics baseline (17 non-facial features, logistic regression and random forest) and an original-split "
        "analysis.", "",
        "## 5. Results per dataset", "",
        "All values: `results/final/master_table.csv` (every number has a `__source` column). ROC-AUC ... Brier = mean ± std "
        "over the 10 test folds at threshold 0.5; pooled AUC = one AUC over all out-of-fold predictions.", "",
        "### 5.1 D01 eye-tracking", "",
        f"Clip level ({int(n_d01.n_samples):,} clips): pooled AUC {c_lo:.3f}-{c_hi:.3f}; child level ({int(n_d01.n_groups)} "
        f"children): {k_lo:.3f}-{k_hi:.3f} (Figure F2). The tracking-quality-only baseline reaches {q3:.3f} "
        f"[{q3ci[0]:.3f}, {q3ci[1]:.3f}] at child level (section 6).", "",
        "Clip level:", "", ds_table(d01c), "", "Child level:", "", ds_table(d01k), "",
        f"Descriptive attention (F9, M07): the mean attention weight on the last second of the clip is "
        f"{att8.loc['ASD', 'mean_weight']:.3f} ± {att8.loc['ASD', 'ci95_half_width']:.3f} for ASD and "
        f"{att8.loc['TD', 'mean_weight']:.3f} ± {att8.loc['TD', 'ci95_half_width']:.3f} for TD children (uniform = 0.125); "
        f"on the first second {att1.loc['ASD', 'mean_weight']:.3f} vs {att1.loc['TD', 'mean_weight']:.3f}. Attention weights "
        "are not explanations and were not tested.", "",
        "### 5.2 D02 facial images", "",
        f"Mean ROC-AUC {f2_lo:.3f}-{f2_hi:.3f}; highest mean fold AUC {b_d02_fold.model} ({b_d02_fold.roc_auc_mean:.3f}), highest "
        f"pooled AUC {b_d02.model} ({ci(b_d02.pooled_auc, b_d02.pooled_auc_ci_lo, b_d02.pooled_auc_ci_hi)}) (F5, F6). The "
        "pooled AUC is lower than the mean fold AUC for most models, consistent with the ten fold models not sharing one "
        "probability scale (Stage 6).", "", ds_table(d02), "",
        "### 5.3 Tabular baselines (D04, D05, D06)", "",
        f"Best pooled AUC (dummy excluded): D04 {tab_best['D04'].model} {ci(tab_best['D04'].pooled_auc, tab_best['D04'].pooled_auc_ci_lo, tab_best['D04'].pooled_auc_ci_hi)}; "
        f"D05 set_honest {tab_best['D05 set_honest'].model} {ci(tab_best['D05 set_honest'].pooled_auc, tab_best['D05 set_honest'].pooled_auc_ci_lo, tab_best['D05 set_honest'].pooled_auc_ci_hi)}; "
        f"D06 {tab_best['D06'].model} {ci(tab_best['D06'].pooled_auc, tab_best['D06'].pooled_auc_ci_lo, tab_best['D06'].pooled_auc_ci_hi)} "
        f"(no model above chance); D05 set_all {tab_best['D05 set_all'].pooled_auc:.3f} is label-definitional (the label is a "
        "rule on the A1-A10 items, which are features in this set) (F10).", "",
        *sum([[f"{k}:", "", ds_table(k), ""] for k in ["D04", "D05 set_all", "D05 set_honest", "D06"]], []),
        "### 5.4 Confusion matrices and calibration of the best model per dataset", "",
        md_table(pd.DataFrame({"dataset": cm.index, "model": cm.model, "TN": cm.tn, "FP": cm.fp, "FN": cm.fn, "TP": cm.tp,
                               "ECE": [f3(M(k, m, "ece_pooled")) for k, m in zip(cm.index, cm.model)]})), "",
        f"Pooled out-of-fold, threshold 0.5 (F8, F12; `results/final/confusion_matrices_best.csv`). ECE ranges over the 10 deep "
        f"models: D01 clip {ece_rng[d01c][0]:.3f}-{ece_rng[d01c][1]:.3f}, D01 child {ece_rng[d01k][0]:.3f}-{ece_rng[d01k][1]:.3f}, "
        f"D02 {ece_rng[d02][0]:.3f}-{ece_rng[d02][1]:.3f}.", "",
        "## 6. Shortcut analysis", "",
        "### 6.1 D01: tracking quality", "",
        f"Quality-only baselines on the same folds: logistic regression on three quality features {q3:.3f} "
        f"[{q3ci[0]:.3f}, {q3ci[1]:.3f}]; 1 - mean valid ratio {SC('D01 baseline', 'one_minus_valid_ratio', 'pooled_auc'):.3f} "
        f"(direction noticed after looking at the data, optimistic); clip level {SC('D01 baseline', 'clip_one_minus_valid_ratio', 'pooled_auc'):.3f}. "
        f"Matching on mean valid ratio left {int(SC('D01 baseline', 'matching', 'n_pairs'))} ASD-TD pairs (standardised mean "
        f"difference {SC('D01 baseline', 'matching', 'smd_all'):.2f} -> {SC('D01 baseline', 'matching', 'smd_matched'):.2f}). "
        "Figures F2-F4.", "", d01_short, "",
        f"Every model's child score is negatively correlated with the mean valid ratio; {len(worse)} models "
        f"({', '.join(worse)}) are significantly worse than the quality-only baseline and none is significantly better.", "",
        "### 6.2 D02: image artefacts and near-duplicates", "",
        f"An image-statistics baseline that never sees a face (size, aspect ratio, file size, JPEG quality, colour and "
        f"brightness statistics, border colour, sharpness) reaches mean AUC {rf:.3f} (random forest) and {lr:.3f} (logistic "
        f"regression): dataset artefacts alone separate the classes. Removing the {int(SC('D02', 'M01', 'n_removed'))} test "
        f"images with a pHash <= 8 neighbour in the fold's training images changes model AUCs by {min(nd):+.3f} to "
        f"{max(nd):+.3f}. All models score higher on the images of the original 'test' folder than on the rest (F5, F6).", "",
        d02_short, "",
        "## 7. Model comparison", "",
        f"Friedman tests: D01 clip chi2 = {fr['D01_clip'][0]:.2f}, p = {pf(fr['D01_clip'][1])} ({fr['D01_clip'][2]} significant "
        f"Nemenyi pairs); D01 child chi2 = {fr['D01_child'][0]:.2f}, p = {pf(fr['D01_child'][1])}; D02 chi2 = "
        f"{fr['D02_image'][0]:.2f}, p = {pf(fr['D02_image'][1])} ({fr['D02_image'][2]} of 45 pairs significant). On D02, "
        f"{nb_best['D02_image']} is significantly better than {len(nb_sig['D02_image'])} of 9 models after Holm correction "
        f"({', '.join(nb_sig['D02_image'])}); on D01 no Nadeau-Bengio comparison survives Holm correction (F7).", "",
        f"Pretrained vs from scratch (mean of model fold-mean AUCs, descriptive; F11): D01 clip {pre['D01 clip']:.3f} vs "
        f"{scr['D01 clip']:.3f}, D01 child {pre['D01 child']:.3f} vs {scr['D01 child']:.3f}, D02 {pre['D02 image']:.3f} vs "
        f"{scr['D02 image']:.3f}.", "",
        stats_md.split("\n", 1)[1].replace("\n## ", "\n### "), "",
        "## 8. Discussion: what can and cannot be claimed", "",
        "The following rule and conclusions are copied unchanged from the Stage 8 report (`reports/STAGE_08_REPORT.md`, "
        "sections 3.6 and 7); section numbers inside them refer to that report (3.2 = model vs quality comparison, 3.4b = "
        "quality-matched subset; here sections 6.1 and F3/F4).", "",
        rules, "", claims, "",
        "**Why this motivates reliability-weighted multimodal fusion.** Both modalities carry a signal whose source is "
        "uncertain: on D01 most of the separability is matched by recording quality, and on D02 by image properties unrelated "
        "to faces. A fused screening system should therefore not trust each modality equally. It should estimate, per child and "
        "per input, how reliable a modality is (for gaze: tracking quality, number of valid clips; for images: acquisition "
        "conditions, distance from the training distribution) and weight or reject that modality's prediction accordingly, "
        "and it should be evaluated against the same shortcut baselines used here. The present datasets do not contain the "
        "same children in both modalities (DD-11), so fusion itself could not be trained or tested in this project.", "",
        "## 9. Limitations", "",
        f"- Sample size: D01 has {int(n_d01.n_groups)} children (5-6 test children per fold); child-level CIs are wide and the "
        f"matched analysis rests on {int(SC('D01 baseline', 'matching', 'n_pairs'))} pairs.",
        "- D02 has no subject identifiers: different photos of the same child can fall into training and test folds; only "
        "near-identical copies were grouped. Results are image-level, not child-level.",
        "- D02 images were collected from the web with unknown diagnosis verification, age range, consent and acquisition "
        "conditions; image artefacts alone separate the classes.",
        "- The custom CNNs were capped at 40 epochs by the fixed protocol and on D02 more than half of their runs reached "
        "the 40-epoch limit without meeting the early-stopping criterion (Stage 6 report), so their comparison with pretrained encoders is conservative for the CNNs.",
        "- cuDNN benchmark mode was on: re-running a fold reproduces results only to about 1e-3 in the losses (Stage 4); "
        "the stored results are the reference.",
        "- D01 clip-level and child-level results answer different questions; clips of one child are not independent.",
        "- The tabular datasets are small public questionnaires; D05 set_all is label-definitional and D06 shows no signal.", "",
        "## 10. Reproducibility", "",
        md_table(REPRO[~REPRO["item"].str.contains("code_hash")]), "",
        f"Total GPU training time: {gpu5 / 60:.1f} min (D01, 100 runs) + {gpu6 / 60:.1f} min (D02, 100 runs) = "
        f"{(gpu5 + gpu6) / 3600:.2f} h on one RTX A2000. Seeds: 42 everywhere (42 + fold for training).", "",
        "Commands per stage (from the project directory, `source .venv/bin/activate`):", "",
        "| stage | command |", "|---|---|",
        "| checks | `python scripts/check_gpu.py`; `python -m pytest -v` |",
        "| S2 data + splits | `python scripts/02a_prepare_d01.py`, `02b_prepare_d02.py`, `02c_prepare_tabular.py`, `02d_make_splits.py` |",
        "| S3 models | `python scripts/03_profile_models.py`; `python scripts/03b_build_feature_cache.py` |",
        "| S5 D01 | `bash scripts/stage5_launch.sh` (runs `scripts/run_stage5.sh` -> `run_queue.py --dataset D01 --models M01-M10`) |",
        "| S6 D02 | `bash scripts/stage6_launch.sh`; verify `python scripts/06b_verify.py` |",
        "| S7 tabular | `CUDA_VISIBLE_DEVICES= python tabular/run_stage7.py` |",
        "| S8 shortcuts + statistics | `python scripts/08a_preflight.py` ... `08f_write_report.py` |",
        "| S9 final tables / figures / report | `python scripts/09a_preflight.py`, `09b_tables.py`, `09c_figures.py`, `09d_write_report.py` |", "",
        "Local snapshots (no git; `snapshots/stage_NN/`, code copy + CHECKSUMS.sha256; code_hash = sha256 of CHECKSUMS):", "",
        md_table(snap.rename(columns={"item": "snapshot", "value": "code_hash"})[["snapshot", "code_hash"]]), "",
        "## 11. Appendix: per-fold test ROC-AUC (models x folds)", "",
        "Source: `results/final/fold_auc_matrix.csv`.", "", fold_md, "",
        "## Figures", "",
        *[f"![{cap}](figures/final/{f}.png)\n\n*{cap}*\n" for f, cap in [
            ("F01_pipeline", "F1. Hybrid model pipeline"),
            ("F02_d01_auc_vs_quality", "F2. D01 model ROC-AUC (pooled, 95% CI) vs the tracking-quality baseline, clip and child level"),
            ("F03_d01_forest_vs_quality", "F3. D01 AUC difference model - quality-only baseline with 95% CIs"),
            ("F04_d01_matched_subset", "F4. D01 AUC on quality-matched children"),
            ("F05_d02_auc_vs_image_stats", "F5. D02 model ROC-AUC vs the image-statistics baseline"),
            ("F06_d02_roc_top3", "F6. D02 pooled ROC curves, top 3 models + image-statistics baseline"),
            ("F07_cd_diagrams", "F7. Critical-difference diagrams, D01 (clip) and D02"),
            ("F08_confusion_matrices", "F8. Confusion matrices of the best model per dataset"),
            ("F09_d01_m07_attention", "F9. D01 M07 attention per second, ASD vs TD (descriptive)"),
            ("F10_tabular_auc", "F10. Tabular baselines: AUC per model"),
            ("F11_pretrained_vs_scratch", "F11. Pretrained vs from-scratch spatial encoders"),
            ("F12_reliability_best", "F12. Reliability diagrams of the best model per dataset")]]]
    with open(OUT_MD, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"abstract: {n_words} words")
    html_note = "pandoc not found: HTML not produced"
    if shutil.which("pandoc"):
        css = os.path.join(REP, "figures", "final", "report.css")
        with open(css, "w") as fh:
            fh.write("body{max-width:1100px;margin:auto;font-family:sans-serif;line-height:1.45;padding:0 16px}"
                     "table{border-collapse:collapse;font-size:12px;margin:8px 0;display:block;overflow-x:auto}"
                     "th,td{border:1px solid #ccc;padding:3px 6px}th{background:#f2f2f2}img{max-width:100%}"
                     "code{font-size:90%}blockquote{color:#444;border-left:3px solid #ccc;margin-left:0;padding-left:12px}\n")
        r = subprocess.run(["pandoc", OUT_MD, "-o", OUT_HTML, "--standalone", "--embed-resources", "--resource-path", REP,
                            "--css", css, "-f", "markdown-tex_math_dollars-yaml_metadata_block", "--metadata",
                            "title=Final results report"], capture_output=True, text=True)
        html_note = f"pandoc exit {r.returncode}" + (f": {r.stderr.strip()[:300]}" if r.stderr.strip() else "")
    print(html_note)


if __name__ == "__main__":
    main()
