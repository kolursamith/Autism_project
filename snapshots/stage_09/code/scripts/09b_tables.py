"""Stage 9 B - collect (never recompute) the final tables into results/final/*.csv + *.md. Every number is read through
src.analysis.final.resolve from the source reference stored next to it, so tests can re-check each value.
Usage: .venv/bin/python scripts/09b_tables.py"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd  # noqa: E402

from src.analysis.data import D01, D02, MODELS, NAMES, TAB_MODELS  # noqa: E402
from src.analysis.final import FINAL_DIR, resolve  # noqa: E402
from src.data.common import PROJECT_DIR, md_table, write_csv  # noqa: E402

S8 = "results/stage8"
CI_CSV, CAL_CSV = f"{S8}/stats_bootstrap_ci.csv", f"{S8}/stats_calibration.csv"
# key, dataset id, experiment, metrics block, stage8 CI dataset, stage8 level, models
DSETS = [("D01 clip", D01, None, "test", D01, "clip", MODELS),
         ("D01 child", D01, None, "test_child", D01, "child", MODELS),
         ("D02", D02, None, "test", D02, "image", MODELS),
         ("D04", "D04_autism_prediction", "set_all", "test", "D04_autism_prediction/set_all", "row", TAB_MODELS),
         ("D05 set_all", "D05_toddler_screening", "set_all", "test", "D05_toddler_screening/set_all", "row", TAB_MODELS),
         ("D05 set_honest", "D05_toddler_screening", "set_honest", "test", "D05_toddler_screening/set_honest", "row", TAB_MODELS),
         ("D06", "D06_early_autism", "set_all", "test", "D06_early_autism/set_all", "row", TAB_MODELS)]
FOLD_METRICS = [("roc_auc", "roc_auc"), ("pr_auc", "pr_auc"), ("balanced_accuracy", "balanced_accuracy"),
                ("sensitivity", "sensitivity"), ("specificity", "specificity"), ("f1", "f1"), ("brier", "brier")]


def mdir(ds, exp, model):
    return "/".join(["results", ds] + ([exp] if exp else []) + [model])


def put(row, col, ref):
    row[col] = resolve(ref)
    row[col + "__source"] = ref


def pm(x, s, nd=3):
    return f"{x:.{nd}f} ± {s:.{nd}f}"


def cis(v, lo, hi, signed=False):
    f = "{:+.3f}" if signed else "{:.3f}"
    return f"{f.format(v)} [{f.format(lo)}, {f.format(hi)}]"


def master():
    rows = []
    for key, ds, exp, block, cids, lev, models in DSETS:
        for m in models:
            r = {"dataset": key, "level": lev, "model": m, "architecture": NAMES.get(m, m)}
            sj = f"{mdir(ds, exp, m)}/summary.json"
            put(r, "n_folds", f"{sj}::n_folds")
            for col, met in FOLD_METRICS:
                for stat in ["mean", "std"]:
                    put(r, f"{col}_{stat}", f"{sj}::per_fold_metrics::{block}/threshold_0.5/{met}::{stat}")
            flt = f"dataset={cids}&level={lev}&model={m}"
            put(r, "pooled_auc", f"{CI_CSV}::{flt}::pooled_auc")
            put(r, "pooled_auc_ci_lo", f"{CI_CSV}::{flt}::ci_lo")
            put(r, "pooled_auc_ci_hi", f"{CI_CSV}::{flt}::ci_hi")
            put(r, "ece_pooled", f"{CAL_CSV}::{flt}::ece_10")
            rows.append(r)
    M = pd.DataFrame(rows)
    write_csv(M, os.path.join(FINAL_DIR, "master_table.csv"))
    L = ["# Master table", "", "Test folds, threshold 0.5. ROC-AUC ... Brier: mean ± std over 10 folds (sample std). Pooled "
         "AUC: one AUC over all out-of-fold predictions with a 95% bootstrap CI (2,000 resamples, seed 42; children / "
         "near-duplicate groups / duplicate-feature groups). ECE: pooled out-of-fold, 10 bins. Sources: the `__source` "
         "columns of master_table.csv.", ""]
    for key, *_ in DSETS:
        g = M[M.dataset == key].sort_values("roc_auc_mean", ascending=False)
        t = pd.DataFrame({"model": g.model, "ROC-AUC": [pm(a, b) for a, b in zip(g.roc_auc_mean, g.roc_auc_std)],
                          "pooled AUC [95% CI]": [cis(a, b, c) for a, b, c in zip(g.pooled_auc, g.pooled_auc_ci_lo, g.pooled_auc_ci_hi)],
                          "PR-AUC": [pm(a, b) for a, b in zip(g.pr_auc_mean, g.pr_auc_std)],
                          "bal. acc.": [pm(a, b) for a, b in zip(g.balanced_accuracy_mean, g.balanced_accuracy_std)],
                          "sensitivity": [pm(a, b) for a, b in zip(g.sensitivity_mean, g.sensitivity_std)],
                          "specificity": [pm(a, b) for a, b in zip(g.specificity_mean, g.specificity_std)],
                          "F1": [pm(a, b) for a, b in zip(g.f1_mean, g.f1_std)],
                          "Brier": [pm(a, b) for a, b in zip(g.brier_mean, g.brier_std)], "ECE": g.ece_pooled.map("{:.3f}".format)})
        L += [f"## {key} ({g.level.iloc[0]} level)", "", md_table(t), ""]
    open(os.path.join(FINAL_DIR, "master_table.md"), "w").write("\n".join(L))
    return M


def long_table(rows, name, title, note):
    T = pd.DataFrame(rows)
    T["value"] = [resolve(s) for s in T.source]
    T = T[["section", "model", "quantity", "value", "source"]]
    write_csv(T, os.path.join(FINAL_DIR, f"{name}.csv"))
    return T


def shortcut():
    rows = []
    add = lambda sec, m, q, ref: rows.append({"section": sec, "model": m, "quantity": q, "source": ref})  # noqa: E731
    mvq = f"{S8}/d01_model_vs_quality.csv"
    for m in MODELS:
        for base, tag in [("quality_LR3", "LR3"), ("one_minus_valid_ratio", "1mvr")]:
            f = f"model={m}&baseline={base}"
            if tag == "LR3":
                for q in ["model_auc", "model_auc_ci_lo", "model_auc_ci_hi"]:
                    add("D01", m, f"child_{q.replace('model_', '')}", f"{mvq}::{f}::{q}")
            for q in ["diff", "diff_ci_lo", "diff_ci_hi", "delong_p"]:
                add("D01", m, f"{q}_vs_{tag}", f"{mvq}::{f}::{q}")
        for q in ["auc_matched", "ci_lo", "ci_hi"]:
            add("D01", m, f"matched_{q.replace('auc_matched', 'auc')}", f"{S8}/d01_matched_auc.csv::model={m}::{q}")
        for g in ["all", "ASD", "TD"]:
            add("D01", m, f"spearman_valid_ratio_{g}", f"{S8}/d01_spearman_valid_ratio.csv::model={m}&children={g}::spearman_rho")
        add("D01", m, "adds_information_beyond_quality", f"{S8}/d01_interpretation.csv::model={m}::adds_information_beyond_tracking_quality")
    for b in ["quality_LR3", "quality_LR_valid_ratio", "one_minus_valid_ratio", "clip_one_minus_valid_ratio", "clip_LR_valid_blink"]:
        for q in ["pooled_auc", "fold_mean_auc", "fold_std_auc"]:
            add("D01 baseline", b, q, f"{S8}/d01_quality_baselines.csv::baseline={b}::{q}")
    for b, lab in [("baseline: quality_LR3", "quality_LR3"), ("baseline: 1 - valid_ratio", "one_minus_valid_ratio")]:
        for q in ["ci_lo", "ci_hi"]:
            add("D01 baseline", lab, f"pooled_auc_{q}", f"{CI_CSV}::dataset={D01}&level=child&model={b}::{q}")
    for b, lab in [("quality: 1 - valid_ratio", "one_minus_valid_ratio"), ("quality_LR3", "quality_LR3")]:
        for q in ["auc_matched", "ci_lo", "ci_hi", "n_pairs"]:
            add("D01 baseline", lab, f"matched_{q.replace('auc_matched', 'auc')}", f"{S8}/d01_matched_auc.csv::model={b}::{q}")
    for q in ["n_pairs", "smd_all", "smd_matched"]:
        add("D01 baseline", "matching", q, f"{S8}/d01_matching_balance.json::{q}")
    nds = f"{S8}/d02_near_dup_sensitivity.csv"
    for m in MODELS:
        for q in ["pooled_auc", "ci_lo", "ci_hi"]:
            add("D02", m, f"pooled_{q.replace('pooled_auc', 'auc')}", f"{CI_CSV}::dataset={D02}&level=image&model={m}::{q}")
        for q in ["fold_mean_auc_all", "fold_mean_auc_removed", "fold_mean_diff", "pooled_auc_removed", "n_removed"]:
            add("D02", m, q, f"{nds}::model={m}::{q}")
        for q in ["auc_orig_test", "auc_rest", "diff", "diff_ci_lo", "diff_ci_hi"]:
            add("D02", m, f"original_split_{q}", f"{S8}/d02_original_split.csv::model={m}::{q}")
    for b in ["image_stats_logreg", "image_stats_random_forest"]:
        for q in ["fold_mean_auc", "fold_std_auc", "pooled_auc"]:
            add("D02 baseline", b, q, f"{S8}/d02_image_stats_baseline.csv::baseline={b}::{q}")
        for q in ["ci_lo", "ci_hi"]:
            add("D02 baseline", b, f"pooled_auc_{q}", f"{CI_CSV}::dataset={D02}&level=image&model=baseline: {b}::{q}")
    T = long_table(rows, "shortcut_table", "", "")
    v = lambda sec, m, q: T[(T.section == sec) & (T.model == m) & (T.quantity == q)].value.iloc[0]  # noqa: E731
    d1 = pd.DataFrame({"model": MODELS,
                       "child AUC [95% CI]": [cis(v("D01", m, "child_auc"), v("D01", m, "child_auc_ci_lo"), v("D01", m, "child_auc_ci_hi")) for m in MODELS],
                       "diff vs quality LR3 [95% CI]": [cis(v("D01", m, "diff_vs_LR3"), v("D01", m, "diff_ci_lo_vs_LR3"), v("D01", m, "diff_ci_hi_vs_LR3"), True) for m in MODELS],
                       "DeLong p": [f"{v('D01', m, 'delong_p_vs_LR3'):.3f}" for m in MODELS],
                       "matched AUC [95% CI]": [cis(v("D01", m, "matched_auc"), v("D01", m, "matched_ci_lo"), v("D01", m, "matched_ci_hi")) for m in MODELS],
                       "Spearman rho (all / ASD / TD)": [" / ".join(f"{v('D01', m, f'spearman_valid_ratio_{g}'):+.2f}" for g in ["all", "ASD", "TD"]) for m in MODELS],
                       "adds information (rule)": ["yes" if v("D01", m, "adds_information_beyond_quality") else "no" for m in MODELS]})
    b = lambda m, q: v("D01 baseline", m, q)  # noqa: E731
    d2 = pd.DataFrame({"model": MODELS,
                       "pooled AUC [95% CI]": [cis(v("D02", m, "pooled_auc"), v("D02", m, "pooled_ci_lo"), v("D02", m, "pooled_ci_hi")) for m in MODELS],
                       "fold-mean AUC": [f"{v('D02', m, 'fold_mean_auc_all'):.3f}" for m in MODELS],
                       "fold-mean AUC, near-dups removed": [f"{v('D02', m, 'fold_mean_auc_removed'):.3f}" for m in MODELS],
                       "change": [f"{v('D02', m, 'fold_mean_diff'):+.3f}" for m in MODELS],
                       "orig 'test' - rest [95% CI]": [cis(v("D02", m, "original_split_diff"), v("D02", m, "original_split_diff_ci_lo"), v("D02", m, "original_split_diff_ci_hi"), True) for m in MODELS]})
    rf, lr = "image_stats_random_forest", "image_stats_logreg"
    L = ["# Shortcut table", "", "Source of every value: shortcut_table.csv (long format, column `source`).", "",
         "## D01 eye-tracking: deep models vs tracking-quality-only baseline (child level)", "",
         f"Quality-only baseline (LR on 3 per-child quality features, fit per fold on training children): pooled AUC "
         f"{cis(b('quality_LR3', 'pooled_auc'), b('quality_LR3', 'pooled_auc_ci_lo'), b('quality_LR3', 'pooled_auc_ci_hi'))}; "
         f"1 - mean valid_ratio: {cis(b('one_minus_valid_ratio', 'pooled_auc'), b('one_minus_valid_ratio', 'pooled_auc_ci_lo'), b('one_minus_valid_ratio', 'pooled_auc_ci_hi'))}; "
         f"clip level 1 - valid_ratio: {b('clip_one_minus_valid_ratio', 'pooled_auc'):.3f}. Matched subset: "
         f"{int(b('matching', 'n_pairs'))} ASD-TD pairs (SMD {b('matching', 'smd_all'):.2f} -> {b('matching', 'smd_matched'):.2f}); "
         f"quality-only matched AUC {cis(b('one_minus_valid_ratio', 'matched_auc'), b('one_minus_valid_ratio', 'matched_ci_lo'), b('one_minus_valid_ratio', 'matched_ci_hi'))}.", "",
         md_table(d1), "",
         "## D02 facial images: deep models vs image-statistics baseline and near-duplicate removal", "",
         f"Image-statistics baseline (no face model): random forest fold-mean AUC {v('D02 baseline', rf, 'fold_mean_auc'):.3f} ± "
         f"{v('D02 baseline', rf, 'fold_std_auc'):.3f}, pooled {cis(v('D02 baseline', rf, 'pooled_auc'), v('D02 baseline', rf, 'pooled_auc_ci_lo'), v('D02 baseline', rf, 'pooled_auc_ci_hi'))}; "
         f"logistic regression {v('D02 baseline', lr, 'fold_mean_auc'):.3f} ± {v('D02 baseline', lr, 'fold_std_auc'):.3f}, pooled "
         f"{cis(v('D02 baseline', lr, 'pooled_auc'), v('D02 baseline', lr, 'pooled_auc_ci_lo'), v('D02 baseline', lr, 'pooled_auc_ci_hi'))}. "
         f"Near-duplicates removed: {int(v('D02', 'M01', 'n_removed'))} test images with a pHash <= 8 training neighbour.", "",
         md_table(d2), ""]
    open(os.path.join(FINAL_DIR, "shortcut_table.md"), "w").write("\n".join(L))
    return T


def stats():
    rows = []
    add = lambda sec, m, q, ref: rows.append({"section": sec, "model": m, "quantity": q, "source": ref})  # noqa: E731
    fj = f"{S8}/stats_friedman_nemenyi.json"
    fr = json.load(open(os.path.join(PROJECT_DIR, fj)))
    for tag, f in fr.items():
        for q in ["chi2", "p", "iman_davenport_F", "iman_davenport_p", "nemenyi_cd", "n_sig_pairs_nemenyi", "n_folds_used"]:
            add(tag, "-", q, f"{fj}::{tag}::{q}")
        for m in MODELS:
            add(tag, m, "avg_rank", f"{fj}::{tag}::avg_ranks::{m}")
        for i, a in enumerate(MODELS):
            for c in MODELS[i + 1:]:
                if f["nemenyi_p"][a][c] < 0.05:
                    add(tag, f"{a}-{c}", "nemenyi_p", f"{fj}::{tag}::nemenyi_p::{a}::{c}")
    nb = pd.read_csv(os.path.join(PROJECT_DIR, S8, "stats_nadeau_bengio.csv"))
    for r in nb.itertuples():
        for q in ["mean_auc_best", "mean_auc_other", "mean_diff", "t", "p", "p_holm"]:
            add(r.analysis, f"{r.best} vs {r.other}", f"nadeau_bengio_{q}",
                f"{S8}/stats_nadeau_bengio.csv::analysis={r.analysis}&other={r.other}::{q}")
    ps = pd.read_csv(os.path.join(PROJECT_DIR, S8, "stats_pretrained_vs_scratch.csv"))
    for r in ps.itertuples():
        for q in ["mean_of_fold_mean_auc", "min_fold_mean_auc", "max_fold_mean_auc", "mean_pooled_auc"]:
            add(f"pretrained_vs_scratch {r.analysis}", r.group, q,
                f"{S8}/stats_pretrained_vs_scratch.csv::analysis={r.analysis}&group={r.group}::{q}")
    T = long_table(rows, "stats_table", "", "")
    L = ["# Statistics table", "", "Source of every value: stats_table.csv (column `source`). Friedman on the 10 folds x 10 "
         "models matrix of per-fold test ROC-AUC; Nemenyi alpha 0.05; Nadeau-Bengio corrected resampled t-test of the model "
         "with the highest mean fold AUC vs each other model, Holm-adjusted over 9 comparisons.", ""]
    for tag in fr:
        g = T[T.section == tag]
        q = lambda name: g[(g.quantity == name)].value.iloc[0]  # noqa: E731
        pf = lambda x: f"{x:.2g}" if x < 0.001 else f"{x:.3f}"  # noqa: E731
        ranks = g[g.quantity == "avg_rank"].sort_values("value")
        nbt = g[g.quantity.str.startswith("nadeau_bengio")].pivot(index="model", columns="quantity", values="value")
        nbt = nbt.reindex(sorted(nbt.index, key=lambda s: s.split(" vs ")[1]))
        L += [f"## {tag}", "", f"Friedman chi2 = {q('chi2'):.2f}, p = {pf(q('p'))}; Iman-Davenport F = {q('iman_davenport_F'):.2f}, "
              f"p = {pf(q('iman_davenport_p'))}; Nemenyi CD = {q('nemenyi_cd'):.2f}; significant Nemenyi pairs: "
              f"{int(q('n_sig_pairs_nemenyi'))} of 45" + (" (" + ", ".join(f"{r.model} p {pf(r.value)}" for r in g[g.quantity == 'nemenyi_p'].itertuples()) + ")"
                                                        if (g.quantity == "nemenyi_p").any() else "") + ".", "",
              "Average ranks (1 = best): " + ", ".join(f"{r.model} {r.value:.2f}" for r in ranks.itertuples()) + ".", "",
              md_table(pd.DataFrame({"comparison": nbt.index, "mean AUC best": nbt.nadeau_bengio_mean_auc_best.map("{:.3f}".format),
                                     "mean AUC other": nbt.nadeau_bengio_mean_auc_other.map("{:.3f}".format),
                                     "mean diff": nbt.nadeau_bengio_mean_diff.map("{:+.3f}".format), "t": nbt.nadeau_bengio_t.map("{:.2f}".format),
                                     "p": nbt.nadeau_bengio_p.map(pf), "Holm p": nbt.nadeau_bengio_p_holm.map(pf)})), ""]
    g = T[T.section.str.startswith("pretrained_vs_scratch")]
    pv = g.pivot_table(index=["section", "model"], columns="quantity", values="value").reset_index()
    L += ["## Pretrained vs from scratch (descriptive)", "",
          md_table(pd.DataFrame({"analysis": pv.section.str.replace("pretrained_vs_scratch ", ""), "group": pv.model,
                                 "mean of model fold-mean AUC": pv.mean_of_fold_mean_auc.map("{:.3f}".format),
                                 "range": [f"{a:.3f}-{b:.3f}" for a, b in zip(pv.min_fold_mean_auc, pv.max_fold_mean_auc)],
                                 "mean pooled AUC": pv.mean_pooled_auc.map("{:.3f}".format)})), ""]
    open(os.path.join(FINAL_DIR, "stats_table.md"), "w").write("\n".join(L))
    return T


def fold_matrix():
    rows = []
    for key, ds, exp, block, cids, lev, models in DSETS:
        for m in models:
            for k in range(1, 11):
                ref = f"{mdir(ds, exp, m)}/fold_{k:02d}/metrics.json::{block}::threshold_0.5::roc_auc"
                rows.append({"dataset": key, "model": m, "fold": k, "roc_auc": resolve(ref), "source": ref})
    F = pd.DataFrame(rows)
    write_csv(F, os.path.join(FINAL_DIR, "fold_auc_matrix.csv"))
    L = ["# Per-fold test ROC-AUC (threshold-free), models x folds", "", "Source: fold_auc_matrix.csv (column `source`).", ""]
    for key, *_ in DSETS:
        p = F[F.dataset == key].pivot(index="model", columns="fold", values="roc_auc")
        p.columns = [f"f{c}" for c in p.columns]
        p = p.map(lambda x: "NaN" if pd.isna(x) else f"{x:.3f}").reset_index()
        L += [f"## {key}", "", md_table(p), ""]
    open(os.path.join(FINAL_DIR, "fold_auc_matrix.md"), "w").write("\n".join(L))


def datasets():
    """Dataset facts: counts from the frozen split files (count of rows / positives / groups) + Stage 1-2 decisions."""
    meta = {D01: ("Eye-Tracking Dataset", "eye-tracking gaze time series -> clips of 8 x 1-s gaze images", "clip", "child",
                  "PRIMARY: 10 hybrid models",
                  "IDs 12, 16 have no gaze rows; rows without participant ID ('Unidentified', ~40%) dropped; cross-file "
                  "duplicate rows removed (DD-02); CARS score never used (label proxy, DD-03)"),
            D02: ("DFDK facial images (AutismDataset sub-folder)", "facial RGB images", "image", "near-duplicate group",
                  "SECONDARY: 10 hybrid models", "original train/valid/test folders pooled (DD-05); pHash <= 4 pairs grouped (DD-07)"),
            "D04_autism_prediction": ("Autism_prediction dataset (train.csv)", "screening questionnaire table", "row",
                                      "duplicate-feature group", "tabular baselines", "test.csv has no labels -> unused (DD-08)"),
            "D05_toddler_screening": ("Autism_dataset (Toddler Autism dataset July 2018.csv)", "toddler screening questionnaire",
                                      "row", "duplicate-feature group", "tabular baselines (set_all = label-definitional upper bound; set_honest without A1-A10 / Qchat score)",
                                      "label = (A1..A10 sum >= 4) (DD-09); Autism_Screening_Data_Combined.csv EXCLUDED: it embeds the toddler file with 728 labels changed"),
            "D06_early_autism": ("early_autism (asd.csv)", "screening table", "row", "duplicate-feature group", "tabular baselines",
                                 "empty column 'Unnamed: 10' dropped (DD-10)")}
    rows = []
    for ds, (src, mod, unit, grp, role, excl) in meta.items():
        f = pd.read_csv(os.path.join(PROJECT_DIR, "splits", ds, "folds.csv"))
        lab = f.groupby("group_id").label.first() if ds == D01 else None
        rows.append({"dataset": ds, "source_folder": src, "modality": mod, "sample_unit": unit, "n_samples": len(f),
                     "n_positive": int(f.label.sum()), "n_negative": int((1 - f.label).sum()), "group_unit": grp,
                     "n_groups": int(f.group_id.nunique()),
                     "n_groups_positive": int(lab.sum()) if lab is not None else "",
                     "n_groups_negative": int((1 - lab).sum()) if lab is not None else "",
                     "role": role, "exclusions": excl, "source": f"splits/{ds}/folds.csv (row / label / group_id counts); "
                     "data_registry/DATASET_DECISIONS.md"})
    rows.append({"dataset": "D03_facial_images_B", "source_folder": "AutismDataset", "modality": "facial RGB images",
                 "sample_unit": "image", "n_samples": 2940, "n_positive": 1470, "n_negative": 1470, "group_unit": "", "n_groups": "",
                 "n_groups_positive": "", "n_groups_negative": "", "role": "EXCLUDED",
                 "exclusions": "copy of D02: 2,541 byte-identical + 399 re-saved images (DD-06)",
                 "source": "data_registry/DATASET_REGISTRY.md; data_registry/DATASET_DECISIONS.md DD-06"})
    D = pd.DataFrame(rows)
    write_csv(D, os.path.join(FINAL_DIR, "datasets_table.csv"))
    open(os.path.join(FINAL_DIR, "datasets_table.md"), "w").write(
        "# Datasets\n\n" + md_table(D.drop(columns=["source"])) + "\n\nSources: datasets_table.csv (column `source`).\n")


def reproducibility():
    rows = []
    add = lambda item, ref: rows.append({"item": item, "value": None, "source": ref})  # noqa: E731
    ri = f"results/{D02}/M01/fold_01/run_info.json"
    for k in ["python", "torch", "cuda", "cudnn", "gpu_name", "driver", "timm", "cudnn_benchmark", "num_workers"]:
        add(k, f"{ri}::{k}")
    rt = "results/D04_autism_prediction/set_all/logreg/fold_01/run_info.json"
    for k in ["sklearn", "numpy", "pandas"]:
        add(f"tabular {k}", f"{rt}::{k}")
    R = []
    for r in rows:
        path, *keys = r["source"].split("::")
        v = json.load(open(os.path.join(PROJECT_DIR, path)))
        for k in keys:
            v = v[k]
        R.append({**r, "value": v})
    for ds, tag in [(D01, "S5 D01 GPU seconds (sum over 100 fold runs)"), (D02, "S6 D02 GPU seconds (sum over 100 fold runs)")]:
        files = sorted(glob.glob(os.path.join(PROJECT_DIR, "results", ds, "M*", "fold_*", "run_info.json")))
        R.append({"item": tag, "value": round(sum(json.load(open(f))["seconds"] for f in files), 1),
                  "source": f"sum of 'seconds' in results/{ds}/M*/fold_*/run_info.json ({len(files)} files)"})
    tf = sorted(glob.glob(os.path.join(PROJECT_DIR, "results", "D0[456]*", "*", "*", "fold_*", "run_info.json")))
    R.append({"item": "S7 tabular CPU seconds (sum over fold runs)", "value": round(sum(json.load(open(f))["seconds"] for f in tf), 1),
              "source": f"sum of 'seconds' in results/D04-D06/*/*/fold_*/run_info.json ({len(tf)} files)"})
    for s in sorted(glob.glob(os.path.join(PROJECT_DIR, "snapshots", "stage_*"))):
        info = json.load(open(os.path.join(s, "SNAPSHOT_INFO.json")))
        rel = os.path.relpath(os.path.join(s, "SNAPSHOT_INFO.json"), PROJECT_DIR)
        R.append({"item": f"{os.path.basename(s)} code_hash ({info['datetime']}, {info['decision']})", "value": info["code_hash"],
                  "source": f"{rel}::code_hash"})
    T = pd.DataFrame(R)
    write_csv(T, os.path.join(FINAL_DIR, "reproducibility_table.csv"))
    open(os.path.join(FINAL_DIR, "reproducibility_table.md"), "w").write(
        "# Reproducibility\n\n" + md_table(T) + "\n")


def main():
    os.makedirs(FINAL_DIR, exist_ok=True)
    M = master()
    S = shortcut()
    T = stats()
    fold_matrix()
    datasets()
    reproducibility()
    print(f"master_table.csv: {len(M)} rows x {sum(not c.endswith('__source') for c in M.columns)} value columns; "
          f"shortcut_table.csv: {len(S)} values; stats_table.csv: {len(T)} values; + fold_auc_matrix, datasets_table, "
          "reproducibility_table (.csv + .md)")


if __name__ == "__main__":
    main()
