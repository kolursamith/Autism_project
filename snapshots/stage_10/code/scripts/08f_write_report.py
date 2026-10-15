"""Stage 8 F - write reports/STAGE_08_REPORT.md from results/stage8/* and logs/stage08_*.txt only; re-checks that
results/D0*/ is byte-identical to the pre-flight manifest (-> logs/stage08_results_unchanged.txt).
Usage: .venv/bin/python scripts/08f_write_report.py"""
import importlib.util
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.analysis.data import CUSTOM_CNN, D01, D02, MODELS, NAMES, OUT_DIR  # noqa: E402
from src.data.common import LOG_DIR, PROJECT_DIR, md_table  # noqa: E402

FIGS = ["stage08_cd_D01.png", "stage08_cd_D02.png", "stage08_reliability_D01.png", "stage08_reliability_D02.png",
        "stage08_d01_model_vs_quality_forest.png", "stage08_d02_near_dup_sensitivity.png"]


def R(p):
    with open(p) as fh:
        return fh.read().rstrip("\n")


def rd(name):
    return pd.read_csv(os.path.join(OUT_DIR, name))


def f3(x):
    return "NaN" if pd.isna(x) else f"{x:.3f}"


def ci(a, lo, hi, signed=False):
    fmt = "{:+.3f}" if signed else "{:.3f}"
    return f"{fmt.format(a)} [{fmt.format(lo)}, {fmt.format(hi)}]"


def pfmt(p):
    return f"{p:.2g}" if p < 0.001 else f"{p:.3f}"


def results_unchanged():
    spec = importlib.util.spec_from_file_location("pf", os.path.join(PROJECT_DIR, "scripts", "08a_preflight.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rec = R(os.path.join(OUT_DIR, "preflight_results_manifest.sha256")).splitlines()
    cur = mod.results_manifest_lines()
    same = rec == cur
    n_pm = sum(("/predictions.csv" in l) or ("/metrics.json" in l) for l in cur)
    txt = (f"results/D0*/ (no *.pt): {len(rec)} files recorded at pre-flight, {len(cur)} now; identical sha256 + size + "
           f"path for all: {same} (incl. {n_pm} predictions.csv / metrics.json)")
    if not same:
        txt += "\nchanged: " + ", ".join(sorted({l.split('  ', 2)[2] for l in set(rec) ^ set(cur)})[:20])
    with open(os.path.join(LOG_DIR, "stage08_results_unchanged.txt"), "w") as fh:
        fh.write(txt + "\n")
    return same, txt


def main():
    same, unchanged_txt = results_unchanged()
    pre = json.load(open(os.path.join(OUT_DIR, "preflight.json")))
    B1, B2, B3, B4a = rd("d01_quality_baselines.csv"), rd("d01_model_vs_quality.csv"), rd("d01_spearman_valid_ratio.csv"), rd("d01_quality_tertiles.csv")
    B4b, B5, B6 = rd("d01_matched_auc.csv"), rd("d01_incremental_value.csv"), rd("d01_interpretation.csv")
    bal = json.load(open(os.path.join(OUT_DIR, "d01_matching_balance.json")))
    cuts = json.load(open(os.path.join(OUT_DIR, "d01_quality_tertile_cuts.json")))
    C1, C1c, C2, C2u, C3 = (rd("d02_near_dup_sensitivity.csv"), rd("d02_near_dup_counts.csv"), rd("d02_image_stats_baseline.csv"),
                            rd("d02_image_stats_univariate.csv"), rd("d02_original_split.csv"))
    nd_img = rd("d02_near_dup_test_images.csv")
    CI, CAL, NB, PS = rd("stats_bootstrap_ci.csv"), rd("stats_calibration.csv"), rd("stats_nadeau_bengio.csv"), rd("stats_pretrained_vs_scratch.csv")
    FR = json.load(open(os.path.join(OUT_DIR, "stats_friedman_nemenyi.json")))

    # ------------------------------------------------------------------ B tables
    b1 = B1.assign(pooled_auc=B1.pooled_auc.map(f3), per_fold=[f"{m:.3f} ± {s:.3f}" for m, s in zip(B1.fold_mean_auc, B1.fold_std_auc)])
    b1 = b1[["baseline", "level", "description", "n", "pooled_auc", "per_fold"]].rename(
        columns={"pooled_auc": "pooled AUC", "per_fold": "per-fold AUC mean ± std"})
    lr3 = B2[B2.baseline == "quality_LR3"].set_index("model").loc[MODELS]
    mvr = B2[B2.baseline == "one_minus_valid_ratio"].set_index("model").loc[MODELS]
    b2 = pd.DataFrame({"model": MODELS, "child AUC [95% CI]": [ci(r.model_auc, r.model_auc_ci_lo, r.model_auc_ci_hi) for r in lr3.itertuples()],
                       "diff vs quality LR3 [95% CI]": [ci(r.diff, r.diff_ci_lo, r.diff_ci_hi, True) for r in lr3.itertuples()],
                       "DeLong p (LR3)": [pfmt(p) for p in lr3.delong_p],
                       "diff vs 1 - valid_ratio [95% CI]": [ci(r.diff, r.diff_ci_lo, r.diff_ci_hi, True) for r in mvr.itertuples()],
                       "DeLong p (1-vr)": [pfmt(p) for p in mvr.delong_p]})
    sp = B3.pivot(index="model", columns="children", values="spearman_rho").loc[MODELS]
    spp = B3.pivot(index="model", columns="children", values="p").loc[MODELS]
    b3 = pd.DataFrame({"model": MODELS, **{f"rho {g} (n={int(B3[B3.children == g].n.iloc[0])})": [f"{sp.loc[m, g]:+.2f} (p {pfmt(spp.loc[m, g])})" for m in MODELS]
                                            for g in ["all", "ASD", "TD"]}})
    tcounts = B4a.drop_duplicates(["level", "tertile"])[["level", "tertile", "n", "n_pos", "n_neg"]]
    t_order = ["low", "mid", "high"]
    b4a = {}
    for lev in ["clip", "child"]:
        sub = B4a[B4a.level == lev]
        piv = sub.pivot(index="model", columns="tertile", values="auc")
        idx = MODELS + [m for m in piv.index if m not in MODELS]
        piv = piv.loc[idx, t_order].map(f3).reset_index()
        b4a[lev] = piv
    b4b = B4b.assign(**{"matched AUC [95% CI]": [ci(r.auc_matched, r.ci_lo, r.ci_hi) for r in B4b.itertuples()]})[
        ["model", "n_pairs", "matched AUC [95% CI]", "ci_excludes_0.5_above"]]
    b5 = B5.assign(**{"pooled AUC quality": B5.pooled_auc_quality.map(f3), "pooled AUC quality + score": B5.pooled_auc_quality_plus_score.map(f3),
                      "pooled diff": B5.pooled_diff.map(lambda x: f"{x:+.3f}"),
                      "per-fold diff mean ± std": [f"{m:+.3f} ± {s:.3f}" for m, s in zip(B5.fold_mean_diff, B5.fold_std_diff)],
                      "folds better / worse": [f"{a} / {b}" for a, b in zip(B5.folds_improved, B5.folds_worse)]})[
        ["model", "pooled AUC quality", "pooled AUC quality + score", "pooled diff", "per-fold diff mean ± std", "folds better / worse"]]
    b6 = B6.copy()
    adds = list(B6.model[B6.adds_information_beyond_tracking_quality])
    not_adds = [m for m in MODELS if m not in adds]
    worse = list(lr3.index[lr3.diff_ci_hi < 0])
    better = list(lr3.index[lr3.diff_ci_lo > 0])
    r2_models = list(B6.model[B6["rule2_matched_ci_excludes_0.5"]])
    q_lr3 = B1.set_index("baseline").loc["quality_LR3"]
    q_mvr = B1.set_index("baseline").loc["one_minus_valid_ratio"]
    q_clip = B1.set_index("baseline").loc["clip_one_minus_valid_ratio"]
    qm = B4b.set_index("model")

    # ------------------------------------------------------------------ C tables
    c1 = C1.assign(**{"fold-mean AUC all": C1.fold_mean_auc_all.map(f3), "fold-mean AUC without near-dups": C1.fold_mean_auc_removed.map(f3),
                      "diff (fold mean)": C1.fold_mean_diff.map(lambda x: f"{x:+.3f}"), "pooled AUC all": C1.pooled_auc_all.map(f3),
                      "pooled AUC without": C1.pooled_auc_removed.map(f3), "AUC on the removed images only": C1.auc_on_near_dup_images_only.map(f3)})[
        ["model", "n_removed", "fold-mean AUC all", "fold-mean AUC without near-dups", "diff (fold mean)", "pooled AUC all",
         "pooled AUC without", "AUC on the removed images only"]]
    n_nd = int(C1.n_removed.iloc[0])
    nd_same = nd_img[nd_img.near_dup]
    c2 = C2.assign(**{"AUC mean ± std (10 folds)": [f"{m:.3f} ± {s:.3f}" for m, s in zip(C2.fold_mean_auc, C2.fold_std_auc)],
                      "pooled AUC": C2.pooled_auc.map(f3)})[["baseline", "AUC mean ± std (10 folds)", "pooled AUC", "fold_aucs", "params"]]
    c2u = C2u.assign(**{"AUC (signed)": C2u.auc_signed.map(f3), "direction-free AUC": C2u.auc_direction_free.map(f3),
                        "mean autistic": C2u.mean_autistic.map(lambda x: f"{x:.2f}"), "mean non-autistic": C2u.mean_non_autistic.map(lambda x: f"{x:.2f}")})[
        ["feature", "AUC (signed)", "direction-free AUC", "mean autistic", "mean non-autistic"]]
    artefact = bool(C2.artefacts_alone_separate_classes.any())
    best_c2 = C2.sort_values("fold_mean_auc", ascending=False).iloc[0]
    c3 = C3.assign(**{"AUC orig 'test' [95% CI]": [ci(r.auc_orig_test, r.ci_lo_orig_test, r.ci_hi_orig_test) for r in C3.itertuples()],
                      "AUC orig train+valid [95% CI]": [ci(r.auc_rest, r.ci_lo_rest, r.ci_hi_rest) for r in C3.itertuples()],
                      "difference [95% CI]": [ci(r.diff, r.diff_ci_lo, r.diff_ci_hi, True) for r in C3.itertuples()]})[
        ["model", "n_orig_test", "n_rest", "AUC orig 'test' [95% CI]", "AUC orig train+valid [95% CI]", "difference [95% CI]"]]
    c3_sig = list(C3.model[C3.diff_ci_lo > 0])

    # ------------------------------------------------------------------ D tables
    def ci_tab(ds, level):
        g = CI[(CI.dataset == ds) & (CI.level == level)].copy()
        g["pooled AUC [95% CI]"] = [ci(r.pooled_auc, r.ci_lo, r.ci_hi) for r in g.itertuples()]
        g = g.sort_values("pooled_auc", ascending=False)
        return g[["model", "pooled AUC [95% CI]", "n", "n_groups", "resampling_unit", "stratified"]]
    d1 = {k: ci_tab(*k) for k in [(D01, "clip"), (D01, "child"), (D02, "image")]}
    tab = CI[~CI.dataset.isin([D01, D02])].copy()
    tab["pooled AUC [95% CI]"] = [ci(r.pooled_auc, r.ci_lo, r.ci_hi) for r in tab.itertuples()]
    tab_t = tab.pivot(index="model", columns="dataset", values="pooled AUC [95% CI]").reset_index()
    top = []
    for (ds, lev), g in CI[~CI.model.str.startswith("baseline") & (CI.model != "dummy_majority")].groupby(["dataset", "level"], sort=False):
        b = g.sort_values("pooled_auc", ascending=False).iloc[0]
        top.append({"dataset": ds, "level": lev, "top model (pooled OOF AUC)": b.model,
                    "pooled AUC [95% CI]": ci(b.pooled_auc, b.ci_lo, b.ci_hi), "resampling unit": b.resampling_unit})
    top = pd.DataFrame(top)
    fr_rows = []
    for tag, f in FR.items():
        fr_rows.append({"analysis": tag, "metric": f["metric"], "Friedman chi2": f"{f['chi2']:.2f}", "p": pfmt(f["p"]),
                        "Iman-Davenport F (p)": f"{f['iman_davenport_F']:.2f} ({pfmt(f['iman_davenport_p'])})",
                        "Nemenyi CD": f"{f['nemenyi_cd']:.2f}", "significant Nemenyi pairs (of 45)": f["n_sig_pairs_nemenyi"],
                        "best avg rank": min(f["avg_ranks"], key=f["avg_ranks"].get)})
    fr_t = pd.DataFrame(fr_rows)
    rank_t = pd.DataFrame({tag: {m: f"{v:.2f}" for m, v in f["avg_ranks"].items()} for tag, f in FR.items()}).loc[MODELS].reset_index().rename(columns={"index": "model"})
    nem_pairs = {}
    for tag, f in FR.items():
        P = f["nemenyi_p"]
        nem_pairs[tag] = [f"{a}-{b} (p {pfmt(P[a][b])})" for i, a in enumerate(MODELS) for b in MODELS[i + 1:] if P[a][b] < 0.05]
    nb = NB.assign(**{"mean diff": NB.mean_diff.map(lambda x: f"{x:+.3f}"), "t": NB.t.map(lambda x: f"{x:.2f}"), "p": NB.p.map(pfmt),
                      "Holm p": NB.p_holm.map(pfmt), "mean AUC best": NB.mean_auc_best.map(f3), "mean AUC other": NB.mean_auc_other.map(f3)})[
        ["analysis", "best", "other", "mean AUC best", "mean AUC other", "mean diff", "t", "p", "Holm p"]]
    nb_sig = {tag: list(NB[(NB.analysis == tag) & (NB.p_holm < 0.05)].other) for tag in FR}
    cal_t = CAL[CAL.dataset.isin([D01, D02])].copy()
    cal_t["key"] = cal_t.dataset.str[:3] + " " + cal_t.level
    cal_p = cal_t.pivot(index="model", columns="key", values="ece_10").loc[MODELS].map(f3)
    cal_b = cal_t.pivot(index="model", columns="key", values="brier").loc[MODELS].map(f3)
    cal_tab = cal_p.add_prefix("ECE ").join(cal_b.add_prefix("Brier ")).reset_index()
    cal_tabular = CAL[~CAL.dataset.isin([D01, D02])].assign(brier=lambda d: d.brier.map(f3), ece_10=lambda d: d.ece_10.map(f3))[
        ["dataset", "model", "n", "brier", "ece_10"]]
    ps = PS.assign(**{"mean of model fold-mean AUC": PS.mean_of_fold_mean_auc.map(f3),
                      "range": [f"{a:.3f}-{b:.3f}" for a, b in zip(PS.min_fold_mean_auc, PS.max_fold_mean_auc)],
                      "mean pooled OOF AUC": PS.mean_pooled_auc.map(f3)})[["analysis", "group", "n_models", "mean of model fold-mean AUC", "range", "mean pooled OOF AUC"]]

    # ------------------------------------------------------------------ evidence
    pyt = R(os.path.join(LOG_DIR, "stage08_pytest.txt"))
    last = pyt.splitlines()[-1]
    py_ok = re.search(r"(\d+) (failed|error)", last) is None and "passed" in last
    figs_ok = all(os.path.isfile(os.path.join(PROJECT_DIR, "reports", "figures", f)) for f in FIGS)
    pre_ok = bool(pre["preflight_ok"])
    d01ci = CI[(CI.dataset == D01) & (CI.level == "child")].set_index("model")
    d02ci = CI[(CI.dataset == D02)].set_index("model")
    m07 = qm.loc[adds[0]] if adds else None

    L = ["# STAGE 8 REPORT - Shortcut controls and statistics", "",
         "Screening-support research, not diagnosis. Stage 8 trains NO deep model: every analysis reads the saved "
         "out-of-fold predictions / metrics of Stages 5-7 (results/D0*/ read-only, verified byte-identical before and "
         "after, section 9) plus the Stage 2 caches. New code: `src/analysis/` (stats.py, quality.py, image_stats.py, "
         "data.py), `scripts/08a_preflight.py`, `08b_d01_quality.py`, `08c_d02_shortcuts.py`, `08d_statistics.py`, "
         "`08e_figures.py`, `08f_write_report.py`; outputs `results/stage8/`, `reports/figures/stage08_*.png`; logs "
         "`logs/stage08_*.txt`. CPU only (CUDA_VISIBLE_DEVICES=\"\"). Bootstrap: 2,000 resamples, seed 42, stratified "
         "by class at the level of the resampling unit (child / near-duplicate group / duplicate-feature group), the same "
         "resamples for every model of a dataset (paired). Pooled AUC = one AUC over all out-of-fold predictions.", "",
         "## 1. Summary", "",
         f"- **D01**: the tracking-quality-only baseline (logistic regression on 3 per-child quality features, fit per fold "
         f"on training children only) reaches child-level pooled AUC {q_lr3.pooled_auc:.3f} (1 - mean valid_ratio alone: "
         f"{q_mvr.pooled_auc:.3f}); the deep models reach {lr3.model_auc.min():.3f}-{lr3.model_auc.max():.3f}. No model "
         f"is significantly better than the quality baseline (all lower CI bounds of the difference < 0); "
         f"{len(worse)} are significantly worse ({', '.join(worse)}). After 1:1 matching on valid_ratio ({bal['n_pairs']} "
         f"pairs) only {', '.join(r2_models) if r2_models else 'no model'} keeps a bootstrap CI above 0.5. By the "
         f"pre-specified rule (section 3.6): adds information beyond tracking quality: {', '.join(adds) if adds else 'none'}; "
         f"not distinguishable from the shortcut: {', '.join(not_adds)}.",
         f"- **D02**: an image-statistics baseline without any face model reaches AUC "
         f"{best_c2.fold_mean_auc:.3f} ± {best_c2.fold_std_auc:.3f} ({best_c2.baseline}); "
         + ("**dataset artefacts alone separate the classes** (>= 0.75). " if artefact else "below the 0.75 artefact threshold. ")
         + f"Removing the {n_nd} test images with a pHash <= 8 training neighbour changes model AUCs by "
         f"{C1.fold_mean_diff.min():+.3f} to {C1.fold_mean_diff.max():+.3f}.",
         f"- **Friedman** (10 deep models x 10 folds): D01 clip p = {pfmt(FR['D01_clip']['p'])}, D01 child p = "
         f"{pfmt(FR['D01_child']['p'])}, D02 p = {pfmt(FR['D02_image']['p'])}.",
         "- Top model per dataset with 95% CI: section 5.1.", "",
         "## 2. Pre-flight (A, `logs/stage08_preflight.txt`)", "", "```", R(os.path.join(LOG_DIR, "stage08_preflight.txt")), "```", "",
         "The snapshot comparison separates data/results files (splits, cache, hashed result files: all identical) from "
         "code/report files added after each snapshot (expected: later stages add scripts and reports).", "",
         "## 3. B - D01 eye-tracking: tracking-quality shortcut (DD-12)", "",
         "Quality features (`cache/D01_eye_tracking/quality_per_child.csv`, never model inputs): mean_valid_ratio, "
         "mean_blink_ratio, tracking_ratio_before_filtering (= overall tracking ratio). 57 children (27 ASD / 30 TD), "
         "1,241 clips. Child score of a deep model = mean of its out-of-fold clip probabilities.", "",
         "### 3.1 Quality-only baselines on the same 10 outer folds", "",
         md_table(b1), "",
         "Fitted baselines are fit on the fold's training (train + val) children only and predict its test children "
         "(`results/stage8/d01_quality_baseline_folds.csv` lists both sets per fold; checked by tests). "
         "1 - mean valid_ratio needs no fitting, but its direction was noticed in Stage 5 after looking at the data, so it is "
         "optimistic; the fitted LR on valid_ratio learns the direction per fold. Per-fold child AUCs rest on 5-6 test "
         "children; the three child baselines rank the test children identically in every fold, hence identical per-fold "
         "mean ± std. The primary comparator below is quality_LR3 (fitted, no look-ahead).", "",
         "### 3.2 Model vs quality-only, child level (paired bootstrap over children + DeLong)", "",
         md_table(b2), "",
         "### 3.3 Dependence of model child scores on mean valid_ratio (Spearman)", "",
         md_table(b3), "",
         f"All {int((sp['all'] < 0).sum())} models have negative overall correlations (lower tracking quality -> higher "
         f"P(ASD)); within ASD children rho is {sp['ASD'].min():+.2f} to {sp['ASD'].max():+.2f} (p < 0.05 for "
         f"{int((spp['ASD'] < 0.05).sum())} of 10), within TD children {sp['TD'].min():+.2f} to {sp['TD'].max():+.2f} "
         f"(p < 0.05 for {int((spp['TD'] < 0.05).sum())} of 10).", "",
         "### 3.4 Quality-controlled performance", "",
         f"(a) AUC within quality tertiles. Clip tertile cut points of clip valid_ratio: {cuts['clip_valid_ratio_cuts'][0]:.3f}, "
         f"{cuts['clip_valid_ratio_cuts'][1]:.3f}; child cut points of mean valid_ratio: {cuts['child_mean_valid_ratio_cuts'][0]:.3f}, "
         f"{cuts['child_mean_valid_ratio_cuts'][1]:.3f}. Class counts:", "",
         md_table(tcounts), "",
         "Clip level (pooled out-of-fold clips within the tertile):", "", md_table(b4a["clip"]), "",
         "Child level (tertiles are strongly class-imbalanced, e.g. the low tertile has "
         f"{int(tcounts[(tcounts.level == 'child') & (tcounts.tertile == 'low')].n_neg.iloc[0])} TD children; AUCs on "
         "19 children are very coarse):", "", md_table(b4a["child"]), "",
         f"(b) Quality-matched subset: greedy 1:1 nearest-neighbour matching of ASD to TD children on mean valid_ratio, "
         f"caliper {bal['caliper']}, without replacement: **{bal['n_pairs']} pairs** (of {bal['n_asd']} ASD / {bal['n_td']} TD; "
         f"max within-pair difference {bal['max_abs_diff_in_pair']:.3f}). Mean valid_ratio ASD/TD {bal['mean_valid_ratio_asd_all']:.3f}/"
         f"{bal['mean_valid_ratio_td_all']:.3f} (SMD {bal['smd_all']:.2f}) before, {bal['mean_valid_ratio_asd_matched']:.3f}/"
         f"{bal['mean_valid_ratio_td_matched']:.3f} (SMD {bal['smd_matched']:.2f}) after. CI: 2,000 bootstrap resamples of "
         "matched pairs, seed 42. The quality-only scores on the matched children are a balance check (residual imbalance).", "",
         md_table(b4b), "",
         "### 3.5 Incremental value (EXPLORATORY)", "",
         "Per outer fold: logistic regression on the training children with [3 quality features] vs [3 quality features + "
         "the model's child score], evaluated on the fold's test children. Scores are out-of-fold, but the score of a "
         "TRAINING child of fold k comes from the model of that child's own fold, which was trained on data that included "
         "fold k's test children: a mild, indirect information path, acceptable only for an exploratory analysis.", "",
         md_table(b5), "",
         "### 3.6 Interpretation rules (written before applying them) and their application", "",
         "A model **adds information beyond tracking quality** only if (rule 1) the 95% CI of its child-level AUC difference "
         "vs the quality-only baseline (quality_LR3, section 3.2) lies entirely above 0, OR (rule 2) its matched-subset AUC "
         "(section 3.4b) has a bootstrap 95% CI entirely above 0.5. Otherwise its D01 results are **not distinguishable "
         "from the tracking-quality shortcut**. No adjustment for testing 10 models is part of the rule; this is noted in "
         "section 7.", "",
         md_table(b6.rename(columns={"rule1_diff_ci_excludes_0_for_model": "rule 1", "rule2_matched_ci_excludes_0.5": "rule 2",
                                     "adds_information_beyond_tracking_quality": "adds information",
                                     "secondary_vs_1mvr_ci_lo": "lower CI vs 1-valid_ratio"}).round(3)), "",
         f"Result: rule 1 holds for {len(better)} models; rule 2 holds for {len(r2_models)} ({', '.join(r2_models) or '-'}).", "",
         "## 4. C - D02 facial images: dataset shortcuts", "",
         "### 4.1 Near-duplicate sensitivity (no retraining)", "",
         f"For each fold, test images whose nearest pHash neighbour among that fold's training images (train + val = all "
         f"images the fold's model could use) is at Hamming distance <= 8: {n_nd} of 2,940 (per fold "
         f"{C1c.n_near_dup_le8.tolist()}; at <= 4: {int(C1c.n_le4.sum())}, as required by the Stage 2 groups). The nearest "
         f"neighbour has the same label for {int((nd_same.label == nd_same.nearest_train_label).sum())} of {len(nd_same)}.", "",
         md_table(c1), "",
         "### 4.2 Image-statistics baseline (no face model)", "",
         "17 non-facial features per image from the raw files (read-only): width, height, aspect ratio, file size, JPEG "
         "quality estimate (IJG scale from the luminance quantisation table; all 2,940 files are JPEG), mean/std of R, G, B, "
         "brightness mean/std, mean R/G/B of the outer 10% border, sharpness = log(1 + variance of the Laplacian). Logistic "
         "regression (standardised) and random forest with fixed parameters, trained on each fold's train + val images, "
         "tested on its test images (`results/stage8/d02_image_stats.csv`, `d02_image_stats_oof.csv`).", "",
         md_table(c2), "",
         ("**The image-statistics baseline reaches >= 0.75: dataset artefacts alone (colour, brightness, image size/"
          "compression, sharpness) separate the two classes to a large degree, without looking at facial features.**"
          if artefact else "The image-statistics baseline stays below 0.75."), "",
         "Single features (no fitting; signed AUC > 0.5 = higher in autistic images):", "", md_table(c2u), "",
         "### 4.3 Original dataset split", "",
         "AUC of each model's out-of-fold predictions on the images from the original 'test' folder vs the rest (original "
         "train + valid); CIs from 2,000 stratified resamples of near-duplicate groups (seed 42), the difference computed "
         "within each resample.", "",
         md_table(c3), "",
         f"The CI of the difference lies above 0 for {len(c3_sig)} of 10 models ({', '.join(c3_sig)}); descriptive only.", "",
         "## 5. D - Statistics for all datasets", "",
         "### 5.1 Pooled out-of-fold AUC with 95% bootstrap CIs", "",
         md_table(top), "",
         "D01 clip level (resampling children):", "", md_table(d1[(D01, "clip")]), "",
         "D01 child level (resampling children; quality baselines included for reference):", "", md_table(d1[(D01, "child")]), "",
         "D02 image level (resampling near-duplicate groups; image-statistics baselines included):", "", md_table(d1[(D02, "image")]), "",
         "Tabular (Stage 7 models; resampling duplicate-feature groups, stratified where groups are label-pure):", "",
         md_table(tab_t), "",
         "D05 set_all is the label-definitional upper bound (label = A1-A10 sum rule; Stage 7 experiment a), not a result.", "",
         "### 5.2 Model comparison across the 10 deep models", "",
         "Friedman test on the 10 folds x 10 models matrix of per-fold test ROC-AUC (from metrics.json; rank 1 = best per "
         "fold), Nemenyi post-hoc (alpha 0.05, studentized range, infinite df), Nadeau-Bengio corrected resampled t-test of "
         "the model with the highest mean fold AUC vs each other model (df 9; correction 1/10 + n_test/n_train with the mean "
         "fold sizes) with Holm adjustment over the 9 comparisons.", "",
         md_table(fr_t), "", "Average ranks:", "", md_table(rank_t), "",
         *[f"- Nemenyi pairs with p < 0.05, {tag}: " + (", ".join(v) if v else "none") for tag, v in nem_pairs.items()], "",
         md_table(nb), "",
         *[f"- Nadeau-Bengio Holm p < 0.05, {tag} (best {NB[NB.analysis == tag].best.iloc[0]}): " + (", ".join(v) if v else "none")
           for tag, v in nb_sig.items()], "",
         "### 5.3 Calibration (pooled out-of-fold, ECE with 10 equal-width bins)", "",
         md_table(cal_tab), "", "Tabular:", "", md_table(cal_tabular), "",
         "### 5.4 Pretrained vs from-scratch (descriptive)", "",
         md_table(ps), "",
         "## 6. Figures (`reports/figures/`), each opened and checked", "",
         f"- `stage08_cd_D01.png`: two CD diagrams; at clip level the Friedman p is {pfmt(FR['D01_clip']['p'])} but one "
         f"Nemenyi bar (CD {FR['D01_clip']['nemenyi_cd']:.2f}) spans all ten models; at child level (p "
         f"{pfmt(FR['D01_child']['p'])}) all ten models are joined as well.",
         f"- `stage08_cd_D02.png`: M04 and M07 hold the best average ranks (1.40, 2.10), the three custom CNNs the worst "
         f"three; overlapping bars, {FR['D02_image']['n_sig_pairs_nemenyi']} of 45 pairs significantly different.",
         "- `stage08_reliability_D01.png`: clip-level curves are flatter than the diagonal for all models (over-confident "
         "at both ends); M09 puts many predictions in the top bin with only ~40% ASD there and M02 has almost no low "
         "predictions; ECE 0.10-0.41.",
         "- `stage08_reliability_D02.png`: U-shaped prediction histograms (most predictions in the 0-0.1 and 0.9-1 bins); "
         "curves lie above the diagonal in the lower and middle bins (more ASD than predicted) and slightly below it in the top bin; ECE 0.06-0.13.",
         "- `stage08_d01_model_vs_quality_forest.png`: left, every model's AUC difference vs the quality baseline is "
         "negative or straddles 0 (M07 the only point estimate above 0, CI crossing 0); right, matched-subset AUCs with wide "
         "CIs, only M07's CI clear of 0.5, the quality-only band 0.45-0.68 overlapping most models.",
         "- `stage08_d02_near_dup_sensitivity.png`: the removed-near-duplicate points lie just below the all-image points for "
         "every model (differences ~0.002); all models sit above the two image-statistics baseline lines (0.78, 0.80).", "",
         "## 7. What can and cannot be claimed", "",
         "Written from the numbers above and the rule in section 3.6.", "",
         "**D01 eye-tracking**",
         f"- Can be claimed: per-child tracking quality alone predicts the label about as well as or better than every deep "
         f"model (quality_LR3 pooled child AUC {q_lr3.pooled_auc:.3f}; models {lr3.model_auc.min():.3f}-{lr3.model_auc.max():.3f}); "
         f"no model is significantly better than this baseline, and {len(worse)} models ({', '.join(worse)}) are significantly "
         f"worse (CI of the difference below 0). Model child scores fall as tracking quality rises, also within the ASD group.",
         (f"- By the pre-specified rule, {', '.join(adds)} adds information beyond tracking quality, via rule 2 only: matched-subset "
          f"AUC {ci(m07.auc_matched, m07.ci_lo, m07.ci_hi)} on {bal['n_pairs']} pairs. This is weak evidence: rule 1 fails "
          f"(difference vs quality_LR3 {ci(lr3.loc[adds[0]]["diff"], lr3.loc[adds[0]].diff_ci_lo, lr3.loc[adds[0]].diff_ci_hi, True)}); "
          f"10 models were tested without multiplicity adjustment (with no signal beyond quality, the chance that at least one "
          f"of 10 independent one-sided 97.5% bounds lies above 0.5 is 1 - 0.975^10 = {1 - 0.975 ** 10:.0%}; the models are "
          f"correlated, so this is only a guide); the matched set is small and keeps some residual imbalance (quality-only AUC on it "
          f"{qm.loc['quality: 1 - valid_ratio'].auc_matched:.3f}, SMD {bal['smd_matched']:.2f}); the exploratory incremental "
          f"analysis gives {B5.set_index('model').loc[adds[0]].pooled_diff:+.3f} pooled AUC."
          if adds else "- No model meets either rule."),
         f"- For {', '.join(not_adds)}: the D01 results are **not distinguishable from the tracking-quality shortcut**.",
         "- Cannot be claimed: that the D01 models learned gaze behaviour characteristic of autism; that any D01 model "
         "outperforms a recording-quality measure; any clinical accuracy.", "",
         "**D02 facial images**",
         (f"- Can be claimed: simple non-facial image statistics alone reach AUC {C2.fold_mean_auc.min():.3f}-{C2.fold_mean_auc.max():.3f}, "
          "so the two classes differ systematically in image properties (colour, brightness, size/compression, sharpness); "
          f"the deep models ({C1.fold_mean_auc_all.min():.3f}-{C1.fold_mean_auc_all.max():.3f}) score higher than this baseline, "
          "but how much of their performance comes from facial phenotype rather than such artefacts is not identified by "
          "these analyses." if artefact else "- The image-statistics baseline stays below 0.75."),
         f"- Near-duplicate leakage at pHash <= 8 does not explain the results (AUC change {C1.fold_mean_diff.min():+.3f} to "
         f"{C1.fold_mean_diff.max():+.3f}); different photos of the same child cannot be detected without subject IDs, so "
         "subject-level leakage is neither shown nor excluded.",
         f"- Models differ (Friedman p {pfmt(FR['D02_image']['p'])}); {NB[NB.analysis == 'D02_image'].best.iloc[0]} has the highest "
         f"mean fold AUC and is significantly better than {len(nb_sig['D02_image'])} of 9 models after Holm "
         f"({', '.join(nb_sig['D02_image'])}); pretrained backbones average higher than the custom CNNs.",
         f"- AUC is higher on the original 'test' images than on the rest for all 10 models (CI above 0 for {len(c3_sig)}): the "
         "original sub-folders are not exchangeable, a further sign of source heterogeneity.",
         "- Cannot be claimed: that facial phenotype was learned; clinical validity; generalisation to clinical photos or "
         "other populations (web-collected images, no subject IDs, unknown diagnosis verification).", "",
         "**Tabular and across datasets**",
         f"- D04: tabular models reach pooled AUC up to {tab[tab.dataset.str.startswith('D04') & (tab.model != 'dummy_majority')].pooled_auc.max():.3f}; "
         f"D05 set_honest up to {tab[tab.dataset.str.endswith('set_honest') & (tab.model != 'dummy_majority')].pooled_auc.max():.3f}; "
         f"D06 no model above 0.5 (max {tab[tab.dataset.str.startswith('D06') & (tab.model != 'dummy_majority')].pooled_auc.max():.3f}); "
         "D05 set_all = 1.000 is label-definitional.",
         "- No result in this project is a diagnosis or a clinical accuracy estimate.", "",
         "## 8. Tests (`.venv/bin/python -m pytest -v`, `logs/stage08_pytest.txt`)", "",
         "New `tests/test_stage08.py` (10 tests): bootstrap reproducible with seed and cluster-preserving; DeLong vs a "
         "brute-force placement implementation on a known example (AUC 8/9) and with ties; Friedman chi2 = 8 on a toy "
         "matrix and Nemenyi q_alpha vs Demšar's table; Nadeau-Bengio / Holm / ECE / matching; the quality baseline uses "
         "only training children (test labels flipped -> identical predictions; overlap refused; fold log = split files); "
         "results/D0*/ unchanged since pre-flight.", "",
         f"Result: {last.strip('= ')}", "",
         "<details><summary>full pytest output</summary>", "", "```", pyt, "```", "", "</details>", "",
         "## 9. Read-only check of earlier results", "", "```", unchanged_txt, "```", "",
         "## 10. Notes and deviations", "",
         "- Bootstrap resampling is stratified by class at the resampling-unit level (guarantees both classes in every "
         f"resample); stratified for {int(CI.stratified.sum())} of {len(CI)} rows (column `stratified`; unstratified "
         f"cluster resampling where some groups contain both labels: {', '.join(sorted(set(CI.dataset[~CI.stratified]))) or 'none'}).",
         "- 'Training images' in C1 = the fold's train + val images (all non-test images available to that fold's model).",
         "- Friedman / Nemenyi / Nadeau-Bengio use per-fold clip-level AUC (D01 primary) and per-fold child-level AUC "
         "(D01 secondary, 5-6 children per fold) as stored in metrics.json; the Nadeau-Bengio 'best' model is the one with "
         "the highest mean fold AUC, which can differ from the best average rank.",
         "- The D01 quality-matched subset and the tertile analyses are small-sample; their CIs are wide.", ""]
    ok = pre_ok and same and py_ok and figs_ok
    why = [n for n, v in [("pre-flight", pre_ok), ("results unchanged", same), ("pytest", py_ok), ("figures", figs_ok)] if not v]
    L.append("STAGE 8 DECISION: PASS" if ok else "STAGE 8 DECISION: FAIL - " + ", ".join(why))
    with open(os.path.join(PROJECT_DIR, "reports", "STAGE_08_REPORT.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(unchanged_txt)
    print(L[-1])


if __name__ == "__main__":
    main()
