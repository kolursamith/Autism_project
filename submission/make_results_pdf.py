"""Build submission/ASD_Hybrid_10fold_Results.pdf: test-fold results, results vs baselines and statistical tests,
verification tests, and the overall final result. Read-only on the project: every number comes from results/final/*.csv,
logs/stage10_* and reports/FINAL_RESULTS_REPORT.md (conclusions copied verbatim). Needs reportlab (not part of the project
requirements): PYTHONPATH=<dir with reportlab> .venv/bin/python submission/make_results_pdf.py"""
import json
import os
import re
from datetime import date

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "submission", "ASD_Hybrid_10fold_Results.pdf")
FIN = os.path.join(ROOT, "results", "final")
FIG = os.path.join(ROOT, "reports", "figures", "final")
FONT_DIR = "/usr/share/fonts/truetype/dejavu"
pdfmetrics.registerFont(TTFont("DV", os.path.join(FONT_DIR, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DVB", os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DVI", os.path.join(FONT_DIR, "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DVB", italic="DVI", boldItalic="DVB")

NAVY, GREY, LIGHT = colors.HexColor("#1f3a5f"), colors.HexColor("#555555"), colors.HexColor("#eef2f7")
ST = {
    "title": ParagraphStyle("title", fontName="DVB", fontSize=20, leading=25, textColor=NAVY, alignment=TA_CENTER),
    "subtitle": ParagraphStyle("sub", fontName="DV", fontSize=11, leading=15, textColor=GREY, alignment=TA_CENTER),
    "h1": ParagraphStyle("h1", fontName="DVB", fontSize=14, leading=18, textColor=NAVY, spaceBefore=10, spaceAfter=6),
    "h2": ParagraphStyle("h2", fontName="DVB", fontSize=11, leading=14, textColor=NAVY, spaceBefore=8, spaceAfter=4),
    "body": ParagraphStyle("body", fontName="DV", fontSize=9, leading=12.5, spaceAfter=5),
    "small": ParagraphStyle("small", fontName="DV", fontSize=7.5, leading=10, textColor=GREY, spaceAfter=4),
    "bullet": ParagraphStyle("bullet", fontName="DV", fontSize=9, leading=12.5, leftIndent=12, bulletIndent=2, spaceAfter=3),
    "cell": ParagraphStyle("cell", fontName="DV", fontSize=7.3, leading=9),
    "cellb": ParagraphStyle("cellb", fontName="DVB", fontSize=7.3, leading=9, textColor=colors.white),
    "box": ParagraphStyle("box", fontName="DV", fontSize=9, leading=12.5),
}


def rd(n):
    return pd.read_csv(os.path.join(FIN, n))


MT, SC, STT = rd("master_table.csv"), rd("shortcut_table.csv"), rd("stats_table.csv")
REPRO, FOLD, CM, DS = rd("reproducibility_table.csv"), rd("fold_auc_matrix.csv"), rd("confusion_matrices_best.csv"), rd("datasets_table.csv")
AUDIT = json.load(open(os.path.join(ROOT, "logs", "stage10_audit.json")))
PYTEST = open(os.path.join(ROOT, "logs", "stage10_pytest.txt")).read().rstrip().splitlines()[-1].strip("= ")
REPORT = open(os.path.join(ROOT, "reports", "FINAL_RESULTS_REPORT.md")).read()
SUBINFO = json.load(open(os.path.join(ROOT, "submission", "SUBMISSION_INFO.json")))
NAMES = dict(zip(MT.model, MT.architecture))


def sc(sec, m, q):
    return float(SC[(SC.section == sec) & (SC.model == m) & (SC.quantity == q)].value.iloc[0])


def st(sec, m, q):
    return float(STT[(STT.section == sec) & (STT.model == m) & (STT.quantity == q)].value.iloc[0])


def P(text, style="body"):
    return Paragraph(text, ST[style])


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def md_inline(t):
    t = esc(t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    return re.sub(r"`(.+?)`", r"<font name='DV' color='#333333'>\1</font>", t)


def table(rows, widths, header_rows=1, zebra=True, align_right_from=None):
    data = [[Paragraph(str(c), ST["cellb"] if i < header_rows else ST["cell"]) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=header_rows)
    style = [("BACKGROUND", (0, 0), (-1, header_rows - 1), NAVY), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("LINEBELOW", (0, -1), (-1, -1), 0.5, NAVY), ("TOPPADDING", (0, 0), (-1, -1), 2.2),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2), ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    if zebra:
        style += [("BACKGROUND", (0, i), (-1, i), LIGHT) for i in range(header_rows + 1, len(rows), 2)]
    t.setStyle(TableStyle(style))
    return t


def figure(name, width_cm, caption):
    path = os.path.join(FIG, f"{name}.png")
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    img = Image(path, width=width_cm * cm, height=width_cm * cm * h / w)
    return KeepTogether([img, P(caption, "small")])


def callout(text):
    t = Table([[Paragraph(text, ST["box"])]], colWidths=[17 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), LIGHT), ("LINEBEFORE", (0, 0), (0, -1), 3, NAVY),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 6),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    return t


def pm(a, b):
    return f"{a:.3f} ± {b:.3f}"


def ci(v, lo, hi, signed=False):
    f = "{:+.3f}" if signed else "{:.3f}"
    return f"{f.format(v)} [{f.format(lo)}, {f.format(hi)}]"


def results_table(key, with_arch=False):
    g = MT[MT.dataset == key].sort_values("roc_auc_mean", ascending=False)
    head = ["Model"] + (["Architecture"] if with_arch else []) + ["ROC-AUC<br/>mean ± SD", "Pooled AUC<br/>[95% CI]",
                                                                    "Bal. acc.", "Sensitivity", "Specificity", "Brier", "ECE"]
    rows = [head]
    for r in g.itertuples():
        rows.append([r.model.replace("_", " ")] + ([r.architecture] if with_arch else []) +
                    [pm(r.roc_auc_mean, r.roc_auc_std), ci(r.pooled_auc, r.pooled_auc_ci_lo, r.pooled_auc_ci_hi),
                     pm(r.balanced_accuracy_mean, r.balanced_accuracy_std), pm(r.sensitivity_mean, r.sensitivity_std),
                     pm(r.specificity_mean, r.specificity_std), f"{r.brier_mean:.3f}", f"{r.ece_pooled:.3f}"])
    widths = ([1.1, 3.6] if with_arch else [1.3]) + [2.4, 3.1, 2.3, 2.3, 2.3, 1.3, 1.3]
    return table(rows, [w * cm for w in widths])


def on_page(c, doc):
    c.saveState()
    c.setFont("DV", 7)
    c.setFillColor(GREY)
    c.drawString(2 * cm, 1.2 * cm, "ASD hybrid 10-fold - test results and final outcome  |  screening-support research, not diagnosis")
    c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"page {doc.page}")
    c.restoreState()


def main():
    abstract = REPORT.split("## 1. Abstract", 1)[1].split("## 2.", 1)[0].strip()
    claims_block = REPORT.split("## 8. Discussion", 1)[1].split("**Why this motivates", 1)[0]
    claims = [l[2:] for l in claims_block.splitlines() if l.startswith("- ")]
    fusion = REPORT.split("**Why this motivates reliability-weighted multimodal fusion.**", 1)[1].split("## 9.", 1)[0].strip()
    limits = [l[2:] for l in REPORT.split("## 9. Limitations", 1)[1].split("## 10.", 1)[0].splitlines() if l.startswith("- ")]
    rep = dict(zip(REPRO["item"], REPRO["value"]))
    gpu5, gpu6 = float(rep["S5 D01 GPU seconds (sum over 100 fold runs)"]), float(rep["S6 D02 GPU seconds (sum over 100 fold runs)"])
    d01 = DS.set_index("dataset").loc["D01_eye_tracking"]
    d02 = DS.set_index("dataset").loc["D02_facial_images_A"]
    q3 = sc("D01 baseline", "quality_LR3", "pooled_auc")
    rf = "image_stats_random_forest"
    b1, b2, b3, b5 = AUDIT["b1"], AUDIT["b2"], AUDIT["b3"], AUDIT["b5"]
    snaps = pd.DataFrame(AUDIT["b4"]["summary"])

    s = []
    # ------------------------------------------------------------------ cover
    s += [Spacer(1, 3 * cm), P("Reliability-Weighted Multimodal AI for Early Autism Risk Screening", "title"), Spacer(1, 6),
          P("Ten hybrid deep-learning architectures, 10-fold cross-validation:<br/>test-fold results, statistical and shortcut "
            "tests, and overall final result", "subtitle"), Spacer(1, 18),
          P("BITE308L Artificial Intelligence (VIT)<br/>Samith Kolur (24BIT0117), Iniya K, Chandru Vel Kumar", "subtitle"),
          Spacer(1, 6), P(date.today().strftime("%d %B %Y"), "subtitle"), Spacer(1, 1.2 * cm),
          callout("<b>Screening-support research only.</b> No result in this document is a diagnosis or an estimate of clinical "
                  "accuracy. All numbers are out-of-fold results on held-out test folds and were re-verified independently "
                  "before this document was produced."), Spacer(1, 0.8 * cm),
          P("Summary", "h1"), P(md_inline(abstract)), PageBreak()]

    # ------------------------------------------------------------------ 1 what was trained
    s += [P("1. What was trained and how it was tested", "h1"),
          P(f"<b>Data.</b> D01 eye-tracking: {int(d01.n_groups)} children ({int(d01.n_groups_positive)} ASD / "
            f"{int(d01.n_groups_negative)} TD), {int(d01.n_samples):,} clips of 8 one-second gaze images. D02 facial images: "
            f"{int(d02.n_samples):,} web-collected photos ({int(d02.n_positive):,} / {int(d02.n_negative):,}), "
            f"{int(d02.n_groups):,} near-duplicate groups. Three questionnaire tables (D04, D05, D06) for non-deep baselines."),
          P("<b>Models.</b> Ten hybrids M01-M10: a spatial encoder (seven ImageNet-pretrained timm backbones or a custom CNN "
            "trained from scratch for M02, M05, M09) -> linear projection -> temporal encoder (LSTM / BiLSTM / GRU / "
            "Transformer) -> attention pooling -> P(ASD). D01 sequences run over time, D02 sequences over spatial tokens."),
          table([["Model", "Architecture", "Model", "Architecture"]] +
                [[f"M{i:02d}", NAMES[f"M{i:02d}"], f"M{i + 5:02d}", NAMES[f"M{i + 5:02d}"]] for i in range(1, 6)],
                [1.2 * cm, 7.3 * cm, 1.2 * cm, 7.3 * cm]), Spacer(1, 6),
          P("<b>Protocol (fixed, never tuned on test data).</b> AdamW, cosine schedule with warm-up, at most 40 epochs, early "
            "stopping on validation ROC-AUC (patience 8), mixed precision, batch 64, seed 42 + fold. Grouped, stratified "
            "10-fold cross-validation (D01 by child, D02 by near-duplicate group); in every fold a validation part of the "
            "training groups chooses the stopping epoch and the threshold, and the test fold is predicted exactly once."),
          P(f"<b>Scale.</b> 200 deep-learning runs (10 models x 2 datasets x 10 folds) on one NVIDIA RTX A2000 12 GB: "
            f"{gpu5 / 60:.1f} min GPU time for D01 and {gpu6 / 60:.1f} min for D02 ({(gpu5 + gpu6) / 3600:.2f} h in total), plus "
            f"200 tabular runs (5 models x 4 settings x 10 folds) on CPU."),
          figure("F01_pipeline", 17, "Figure 1. Hybrid model pipeline and evaluation design."),
          P("<b>Reading the metrics.</b> ROC-AUC mean ± SD is over the 10 test folds; pooled AUC is one AUC over all "
            "out-of-fold test predictions with a 95% bootstrap confidence interval (2,000 resamples of children / "
            "near-duplicate groups). Balanced accuracy, sensitivity and specificity use threshold 0.5. ECE = expected "
            "calibration error (lower is better).", "small"), PageBreak()]

    # ------------------------------------------------------------------ 2 test results
    s += [P("2. Results on the held-out test folds", "h1"),
          KeepTogether([P("2.1 D01 eye-tracking, child level (one score per child = mean clip probability)", "h2"), results_table("D01 child")]),
          Spacer(1, 6), KeepTogether([P("2.2 D01 eye-tracking, clip level", "h2"), results_table("D01 clip")]), Spacer(1, 4),
          figure("F02_d01_auc_vs_quality", 17, "Figure 2. D01 pooled test AUC per model (95% CI) with the tracking-quality-only "
                 "baseline as reference (dashed)."),
          KeepTogether([P("2.3 D02 facial images (image level)", "h2"), results_table("D02")]), Spacer(1, 6),
          figure("F05_d02_auc_vs_image_stats", 14.5, "Figure 3. D02 pooled test AUC per model (95% CI) vs a baseline that uses "
                 "only non-facial image statistics."),
          figure("F06_d02_roc_top3", 9, "Figure 4. D02 pooled out-of-fold ROC curves of the three best models and the "
                 "image-statistics baseline.")]
    tab_rows = [["Dataset / setting", "Best model", "ROC-AUC mean ± SD", "Pooled AUC [95% CI]", "Bal. acc.", "Dummy AUC"]]
    for k, lab in [("D04", "D04 autism prediction"), ("D05 set_honest", "D05 toddler, without A1-A10 / score"),
                   ("D05 set_all", "D05 toddler, all features (label-definitional)"), ("D06", "D06 early autism")]:
        g = MT[(MT.dataset == k) & (MT.model != "dummy_majority")].sort_values("pooled_auc", ascending=False).iloc[0]
        dm = MT[(MT.dataset == k) & (MT.model == "dummy_majority")].iloc[0]
        tab_rows.append([lab, g.model, pm(g.roc_auc_mean, g.roc_auc_std), ci(g.pooled_auc, g.pooled_auc_ci_lo, g.pooled_auc_ci_hi),
                         pm(g.balanced_accuracy_mean, g.balanced_accuracy_std), f"{dm.pooled_auc:.3f}"])
    cm_rows = [["Dataset", "Best model", "TN", "FP", "FN", "TP"]] + [[r.dataset, r.model, r.tn, r.fp, r.fn, r.tp] for r in CM.itertuples()]
    s += [KeepTogether([P("2.4 Tabular baselines (questionnaires)", "h2"), table(tab_rows, [w * cm for w in [5.2, 2.2, 2.6, 3.2, 2.3, 1.5]]),
          P("D05 'all features' contains the A1-A10 items that define its label, so 1.000 is an upper bound by construction, "
            "not a result. D06 shows no signal (best pooled AUC below 0.5).", "small")]), Spacer(1, 4),
          figure("F10_tabular_auc", 17, "Figure 5. Tabular baselines: pooled AUC per model with 95% CI; dashed line = chance."),
          KeepTogether([P("2.5 Confusion matrices of the best model per dataset (pooled out-of-fold, threshold 0.5)", "h2"),
          table(cm_rows, [w * cm for w in [3.4, 2.6, 1.6, 1.6, 1.6, 1.6]])]), PageBreak()]

    # ------------------------------------------------------------------ 3 results vs baselines / shortcut tests
    d1 = [["Model", "Child AUC [95% CI]", "Model - quality baseline [95% CI]", "DeLong p", "Matched-subset AUC [95% CI]", "Adds info (rule)"]]
    for m in [f"M{i:02d}" for i in range(1, 11)]:
        d1.append([m, ci(sc("D01", m, "child_auc"), sc("D01", m, "child_auc_ci_lo"), sc("D01", m, "child_auc_ci_hi")),
                   ci(sc("D01", m, "diff_vs_LR3"), sc("D01", m, "diff_ci_lo_vs_LR3"), sc("D01", m, "diff_ci_hi_vs_LR3"), True),
                   f"{sc('D01', m, 'delong_p_vs_LR3'):.3f}",
                   ci(sc("D01", m, "matched_auc"), sc("D01", m, "matched_ci_lo"), sc("D01", m, "matched_ci_hi")),
                   "yes" if sc("D01", m, "adds_information_beyond_quality") else "no"])
    d2 = [["Model", "Pooled AUC [95% CI]", "Fold-mean AUC", "Near-duplicates removed", "Original 'test' folder - rest [95% CI]"]]
    for m in [f"M{i:02d}" for i in range(1, 11)]:
        d2.append([m, ci(sc("D02", m, "pooled_auc"), sc("D02", m, "pooled_ci_lo"), sc("D02", m, "pooled_ci_hi")),
                   f"{sc('D02', m, 'fold_mean_auc_all'):.3f}", f"{sc('D02', m, 'fold_mean_auc_removed'):.3f} ({sc('D02', m, 'fold_mean_diff'):+.3f})",
                   ci(sc("D02", m, "original_split_diff"), sc("D02", m, "original_split_diff_ci_lo"), sc("D02", m, "original_split_diff_ci_hi"), True)])
    rf = "image_stats_random_forest"
    s += [P("3. Results vs baselines: shortcut tests", "h1"),
          P("A high test AUC is only meaningful if it cannot be explained by something simpler than autism-related patterns. "
            "Both datasets were tested against such shortcuts, without retraining any model."),
          P("3.1 D01: deep models vs tracking quality alone", "h2"),
          P(f"A logistic regression on three per-child recording-quality features (valid-gaze ratio, blink ratio, tracking "
            f"ratio), fitted per fold on training children only, reaches child-level pooled AUC <b>{q3:.3f}</b> "
            f"[{sc('D01 baseline', 'quality_LR3', 'pooled_auc_ci_lo'):.3f}, {sc('D01 baseline', 'quality_LR3', 'pooled_auc_ci_hi'):.3f}]. "
            f"After 1:1 matching of ASD and TD children on tracking quality ({int(sc('D01 baseline', 'matching', 'n_pairs'))} "
            f"pairs), the quality score itself drops to {sc('D01 baseline', 'one_minus_valid_ratio', 'matched_auc'):.3f}."),
          table(d1, [w * cm for w in [1.2, 3.1, 3.9, 1.6, 3.6, 1.8]]), Spacer(1, 4),
          figure("F03_d01_forest_vs_quality", 12, "Figure 6. D01: child-level AUC difference model - quality baseline (95% paired "
                 "bootstrap CI); right of 0 = model better."), PageBreak(),
          figure("F04_d01_matched_subset", 12, "Figure 7. D01: AUC on quality-matched children (95% CI); grey band = quality-only "
                 "score on the same children."),
          P("3.2 D02: deep models vs image artefacts and near-duplicates", "h2"),
          P(f"A random forest on 17 non-facial image statistics (size, file size, JPEG quality, colour and brightness, border "
            f"colour, sharpness) reaches fold-mean AUC <b>{sc('D02 baseline', rf, 'fold_mean_auc'):.3f}</b> ± "
            f"{sc('D02 baseline', rf, 'fold_std_auc'):.3f}, so dataset artefacts alone separate the classes. Removing the "
            f"{int(sc('D02', 'M01', 'n_removed'))} test images that have a near-duplicate (pHash <= 8) in the training images "
            f"changes model AUCs by at most {max(abs(sc('D02', m, 'fold_mean_diff')) for m in [f'M{i:02d}' for i in range(1, 11)]):.3f}."),
          table(d2, [w * cm for w in [1.2, 3.4, 2.4, 3.8, 4.4]]), PageBreak()]

    # ------------------------------------------------------------------ 4 statistical tests
    fr_rows = [["Comparison", "Friedman chi²", "p", "Nemenyi CD", "Significant pairs (of 45)", "Best average rank"]]
    for tag, lab in [("D01_clip", "D01 clip-level fold AUC"), ("D01_child", "D01 child-level fold AUC"), ("D02_image", "D02 fold AUC")]:
        ranks = STT[(STT.section == tag) & (STT.quantity == "avg_rank")].sort_values("value").iloc[0]
        p = st(tag, "-", "p")
        fr_rows.append([lab, f"{st(tag, '-', 'chi2'):.2f}", f"{p:.2g}" if p < 0.001 else f"{p:.3f}", f"{st(tag, '-', 'nemenyi_cd'):.2f}",
                        int(st(tag, "-", "n_sig_pairs_nemenyi")), f"{ranks.model} ({ranks.value:.2f})"])
    nb = STT[(STT.section == "D02_image") & (STT.quantity == "nadeau_bengio_p_holm")]
    nb_sig = [r.model.split(" vs ")[1] for r in nb.itertuples() if r.value < 0.05]
    ps_rows = [["Analysis", "Pretrained (7 models)", "From scratch (3 custom CNNs)"]]
    for k in ["D01 clip", "D01 child", "D02 image"]:
        ps_rows.append([k, f"{st(f'pretrained_vs_scratch {k}', 'pretrained (M01,M03,M04,M06,M07,M08,M10)', 'mean_of_fold_mean_auc'):.3f}",
                        f"{st(f'pretrained_vs_scratch {k}', 'from scratch (M02,M05,M09)', 'mean_of_fold_mean_auc'):.3f}"])
    s += [P("4. Statistical tests: are the models different?", "h1"),
          P("Friedman test on the 10 folds x 10 models matrix of test AUCs, Nemenyi post-hoc test (alpha 0.05) and the "
            "Nadeau-Bengio corrected resampled t-test (best model vs each other model, Holm-adjusted)."),
          table(fr_rows, [w * cm for w in [4.2, 2.2, 1.8, 2.0, 3.3, 3.5]]), Spacer(1, 4),
          P(f"On D02, M04 is significantly better than {len(nb_sig)} of 9 models after Holm correction ({', '.join(nb_sig)}); on D01 "
            "no pairwise comparison survives correction."),
          figure("F07_cd_diagrams", 17, "Figure 8. Critical-difference diagrams (average rank, 1 = best; bars join models that are "
                 "not significantly different)."),
          P("Pretrained vs from-scratch encoders (mean of model fold-mean AUCs, descriptive)", "h2"),
          table(ps_rows, [w * cm for w in [4.5, 5, 5]]), PageBreak(),
          figure("F12_reliability_best", 17, "Figure 9. Calibration (reliability diagrams) of the best model per dataset."),
          figure("F09_d01_m07_attention", 11, "Figure 10. D01 M07 temporal attention per second, ASD vs TD (descriptive)."), PageBreak()]

    # ------------------------------------------------------------------ 5 verification tests
    leak = [["Check", "Result"],
            ["Predictions of every run = that fold's test ids (400 runs)", "passed" if b3["predictions_equal_fold_test_ids"] else "FAILED"],
            ["Every sample predicted exactly once per model", f"{b3['each_sample_predicted_once_per_model'][0] - b3['each_sample_predicted_once_per_model'][1]}/{b3['each_sample_predicted_once_per_model'][0]} models"],
            ["Train / validation / test disjoint by sample and by group (50 dataset-folds)", "passed" if b3["train_val_test_disjoint_sample_and_group"] else "FAILED"],
            ["No child (D01) or near-duplicate group (D02) in two test folds", "passed" if not any(b3["groups_in_more_than_one_test_fold"].values()) else "FAILED"],
            ["D02 near-identical image pairs (pHash <= 4) across folds", f"{b3['D02_phash_le4_pairs_total_and_cross_fold'][1]} of {b3['D02_phash_le4_pairs_total_and_cross_fold'][0]}"],
            ["Excluded duplicate dataset D03 never used", "passed" if not b3["D03_paths_in_cache"] else "FAILED"],
            ["CARS score (label proxy) absent from every input", "passed" if not b3["CARS_in_inputs"] else "FAILED"],
            ["D05 'honest' setting contains no A1-A10 items / score", "passed" if not b3["D05_honest_forbidden_in_run_configs"] else "FAILED"],
            ["D04 unlabelled test.csv never used", "passed" if b3["D04_clean_rows_equal_train_csv"] else "FAILED"]]
    snap_ok = bool(snaps.checksums_integrity.all()) and not AUDIT["b4"]["deleted"]
    s += [P("5. How the results themselves were tested", "h1"),
          P("Before this document was produced, every stored result was re-derived by independent code (scikit-learn directly "
            "from the saved predictions, without the project's own metric code), checked for data leakage, and covered by the "
            "project's automated test suite."),
          table([["Verification", "Result"],
                 [f"Automated test suite (pytest, run with results frozen read-only)", PYTEST],
                 [f"Per-run metrics recomputed from predictions ({b1['runs']} runs, {b1['checks']:,} checks, tolerance 1e-6)", f"{len(b1['mismatches'])} mismatches"],
                 [f"Final master table recomputed ({b2['checks']} values: means, SDs, pooled AUC, ECE)", f"{len(b2['mismatches'])} mismatches"],
                 [f"Numbers quoted in the final report found in the result tables ({b5['decimals_scanned']:,} numbers)",
                  f"{b5['decimals_scanned'] - b5['not_in_tables']:,} matched; {b5['not_in_tables']} explained (derived or constants); {len(b5['manual'])} unexplained"],
                 [f"Local snapshots of all 10 stages (code + checksums)", "all intact, no file lost" if snap_ok else "PROBLEM"],
                 ["Frozen result manifest and submission package", f"{json.load(open(os.path.join(ROOT, 'results', 'FINAL_MANIFEST.json')))['n_files']:,} files hashed; zip {SUBINFO['n_files']} files, {SUBINFO['size_mb']} MB"]],
                [11.5 * cm, 5.5 * cm]), Spacer(1, 8),
          P("Leakage checks", "h2"), table(leak, [12.5 * cm, 4.5 * cm]), Spacer(1, 6),
          P("Each stage (S1-S10) ended with a written report, passing tests and a local snapshot before the next stage started.", "small"),
          PageBreak()]

    # ------------------------------------------------------------------ 6 overall final result
    s += [P("6. Overall final result", "h1"),
          callout(f"<b>In short:</b> the hybrid models reach moderate test AUCs on eye-tracking (child level "
                  f"{MT[MT.dataset == 'D01 child'].pooled_auc.min():.3f}-{MT[MT.dataset == 'D01 child'].pooled_auc.max():.3f}) "
                  f"and high test AUCs on facial images ({MT[MT.dataset == 'D02'].roc_auc_mean.min():.3f}-"
                  f"{MT[MT.dataset == 'D02'].roc_auc_mean.max():.3f}). But tracking quality alone reaches {q3:.3f} on D01, "
                  f"and non-facial image statistics alone reach {sc('D02 baseline', rf, 'fold_mean_auc'):.3f} on D02. The "
                  "results therefore support <b>no claim</b> that the models learned autism-related gaze or facial "
                  "patterns, and they are not clinical accuracy estimates."), Spacer(1, 6),
          P("What can and cannot be claimed (copied unchanged from the final report)", "h2")]
    s += [Paragraph(md_inline(c), ST["bullet"], bulletText="•") for c in claims]
    s += [P("Why this motivates reliability-weighted multimodal fusion", "h2"), P(md_inline(fusion)),
          P("Limitations", "h2")] + [Paragraph(md_inline(c), ST["bullet"], bulletText="•") for c in limits]
    s += [Spacer(1, 8), P("Source files: reports/FINAL_RESULTS_REPORT.md (full report), results/final/*.csv (all numbers with "
                          "source references), reports/STAGE_01..10_REPORT.md (stage records), logs/stage10_audit.txt "
                          "(independent re-verification).", "small"), PageBreak()]

    # ------------------------------------------------------------------ appendix
    s += [P("Appendix A. Per-fold test ROC-AUC", "h1")]
    for key in ["D01 clip", "D01 child", "D02"]:
        g = FOLD[FOLD.dataset == key].pivot(index="model", columns="fold", values="roc_auc")
        rows = [["Model"] + [f"F{c}" for c in g.columns] + ["Mean"]]
        rows += [[m] + [f"{v:.3f}" for v in g.loc[m]] + [f"{g.loc[m].mean():.3f}"] for m in g.index]
        s += [P(key, "h2"), table(rows, [1.3 * cm] + [1.35 * cm] * 10 + [1.4 * cm]), Spacer(1, 4)]
    env = [["Item", "Value"]] + [[k, str(rep[k])] for k in ["python", "torch", "cuda", "gpu_name", "driver", "timm", "tabular sklearn"]]
    s += [P("Appendix B. Environment", "h1"), table(env, [5 * cm, 8 * cm])]

    doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.7 * cm, bottomMargin=1.8 * cm,
                            title="ASD hybrid 10-fold: test results and final outcome",
                            author="Samith Kolur, Iniya K, Chandru Vel Kumar", subject="BITE308L project results")
    doc.build(s, onFirstPage=on_page, onLaterPages=on_page)
    print(OUT, os.path.getsize(OUT))


if __name__ == "__main__":
    main()
