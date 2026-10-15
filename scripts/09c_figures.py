"""Stage 9 C - final figures (reports/figures/final/F*.png at 300 dpi + .svg) from results/final/*.csv, results/stage8/*
and saved out-of-fold predictions (read-only). One colour per model in every figure (src.analysis.final.MODEL_COLORS).
Usage: .venv/bin/python scripts/09c_figures.py"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from src.analysis import stats as S  # noqa: E402
from src.analysis.data import D01, D02, MODELS, NAMES, PRETRAINED, CUSTOM_CNN, TAB_MODELS, child_scores, load_oof  # noqa: E402
from src.analysis.final import BASELINE_COLOR, FIG_FINAL, FINAL_DIR, MODEL_COLORS, TAB_COLORS, apply_style  # noqa: E402
from src.data.common import PROJECT_DIR, write_csv  # noqa: E402

S8 = os.path.join(PROJECT_DIR, "results", "stage8")
TAB = {"D04": ("D04_autism_prediction", "set_all"), "D05 set_all": ("D05_toddler_screening", "set_all"),
       "D05 set_honest": ("D05_toddler_screening", "set_honest"), "D06": ("D06_early_autism", "set_all")}
apply_style(plt)


def save(fig, name):
    fig.savefig(os.path.join(FIG_FINAL, f"{name}.png"), dpi=300)
    fig.savefig(os.path.join(FIG_FINAL, f"{name}.svg"))
    plt.close(fig)


def master():
    return pd.read_csv(os.path.join(FINAL_DIR, "master_table.csv"))


def sc(sec, model, q):
    T = pd.read_csv(os.path.join(FINAL_DIR, "shortcut_table.csv"))
    return float(T[(T.section == sec) & (T.model == model) & (T.quantity == q)].value.iloc[0])


def dot_ci(ax, y, x, lo, hi, colors):
    for yi, xi, a, b, c in zip(y, x, lo, hi, colors):
        ax.errorbar(xi, yi, xerr=[[xi - a], [b - xi]], fmt="o", ms=5, color=c, capsize=2.5, lw=1.4)


# ---------------------------------------------------------------------------------------------- F1
def f1_pipeline():
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    def box(x, y, w, h, text, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08", fc=fc, ec="#444", lw=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8, linespacing=1.3)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color="#444", lw=1))
    c_in, c_pre, c_mod, c_out = "#e8f1fa", "#eef7ec", "#fdf3e3", "#f6e8f2"
    box(0.1, 3.0, 1.9, 1.15, "D01 eye-tracking\ngaze time series\n(children, ~60 Hz)", c_in)
    box(0.1, 1.2, 1.9, 1.15, "D02 facial images\nweb-collected\nRGB photos", c_in)
    box(2.35, 3.0, 2.1, 1.15, "8 x 1-s windows per clip,\neach rendered 224x224:\nscan path, fixation density,\ntime channel", c_pre)
    box(2.35, 1.2, 2.1, 1.15, "resize 224 + centre crop;\ntrain only: random crop\n0.8-1.0, h-flip, colour\njitter 0.1", c_pre)
    box(4.85, 1.85, 1.75, 1.65, "Spatial encoder\n\npretrained timm\nbackbone (7 models)\nor custom CNN\n(M02, M05, M09)", c_mod)
    box(6.95, 2.25, 1.15, 0.85, "Linear\nprojection", c_mod)
    box(8.45, 1.85, 1.6, 1.65, "Temporal encoder\n\nLSTM / BiLSTM /\nGRU / Transformer\n\nD01: 8 time steps\nD02: 49 / 196 tokens", c_mod)
    box(10.35, 2.6, 1.55, 0.9, "Temporal attention\n(M05: MLP head;\nM10: mean pooling)", c_mod)
    box(10.35, 1.4, 1.55, 0.8, "Classifier\n-> P(ASD)", c_out)
    arrow(2.0, 3.57, 2.35, 3.57)
    arrow(2.0, 1.77, 2.35, 1.77)
    arrow(4.45, 3.57, 4.85, 3.0)
    arrow(4.45, 1.77, 4.85, 2.35)
    arrow(6.6, 2.67, 6.95, 2.67)
    arrow(8.1, 2.67, 8.45, 2.67)
    arrow(10.05, 2.9, 10.35, 3.0)
    arrow(11.12, 2.6, 11.12, 2.2)
    box(0.1, 0.1, 11.8, 0.75, "Evaluation: grouped, stratified 10-fold CV (D01 by child, D02 by near-duplicate group); validation part = "
        "15% of training groups -> early stopping and threshold;\nout-of-fold test predictions only -> metrics, bootstrap CIs, "
        "Friedman / Nemenyi, shortcut controls (tracking quality, image statistics, near-duplicates)", "#f2f2f2")
    ax.text(5.95, 3.62, "D01 frozen backbone (cached features)\nD02 last stage fine-tuned", ha="center", va="bottom",
            fontsize=7, color="#555")
    ax.set_title("F1. Hybrid model pipeline (one model id = one spatial encoder + one temporal encoder + one head)", loc="left")
    save(fig, "F01_pipeline")


# ---------------------------------------------------------------------------------------------- F2 / F5
def f2_d01_auc(M):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    y = np.arange(len(MODELS))[::-1]
    for ax, key, title in [(axes[0], "D01 clip", "clip level (1,241 clips)"), (axes[1], "D01 child", "child level (57 children)")]:
        g = M[M.dataset == key].set_index("model").loc[MODELS]
        dot_ci(ax, y, g.pooled_auc, g.pooled_auc_ci_lo, g.pooled_auc_ci_hi, [MODEL_COLORS[m] for m in MODELS])
        ax.axvline(0.5, color="#bbb", lw=0.8)
        ax.set_title(f"D01 {title}")
        ax.set_xlabel("pooled out-of-fold ROC-AUC (95% bootstrap CI)")
        ax.set_xlim(0.3, 1.0)
    q_clip = sc("D01 baseline", "clip_one_minus_valid_ratio", "pooled_auc")
    axes[0].axvline(q_clip, color=BASELINE_COLOR, ls="--", lw=1.1, label=f"quality only: clip 1 - valid_ratio ({q_clip:.3f})")
    q = sc("D01 baseline", "quality_LR3", "pooled_auc")
    axes[1].axvspan(sc("D01 baseline", "quality_LR3", "pooled_auc_ci_lo"), sc("D01 baseline", "quality_LR3", "pooled_auc_ci_hi"),
                    color="#999", alpha=0.18, lw=0)
    axes[1].axvline(q, color=BASELINE_COLOR, ls="--", lw=1.1, label=f"quality only: LR on 3 quality features ({q:.3f}, CI shaded)")
    for ax in axes:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"{m}  {NAMES[m]}" for m in MODELS])
    fig.suptitle("F2. D01 eye-tracking: model ROC-AUC vs the tracking-quality-only baseline (CIs: children resampled)", x=0.01, ha="left")
    fig.subplots_adjust(wspace=0.12)
    save(fig, "F02_d01_auc_vs_quality")


def f5_d02_auc(M):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    y = np.arange(len(MODELS))[::-1]
    g = M[M.dataset == "D02"].set_index("model").loc[MODELS]
    dot_ci(ax, y, g.pooled_auc, g.pooled_auc_ci_lo, g.pooled_auc_ci_hi, [MODEL_COLORS[m] for m in MODELS])
    for b, ls, lab in [("image_stats_random_forest", "--", "random forest"), ("image_stats_logreg", ":", "logistic regression")]:
        v = sc("D02 baseline", b, "pooled_auc")
        ax.axvspan(sc("D02 baseline", b, "pooled_auc_ci_lo"), sc("D02 baseline", b, "pooled_auc_ci_hi"), color="#999", alpha=0.15, lw=0)
        ax.axvline(v, color=BASELINE_COLOR, ls=ls, lw=1.1, label=f"image statistics only, {lab} ({v:.3f}, CI shaded)")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{m}  {NAMES[m]}" for m in MODELS])
    ax.set_xlim(0.74, 0.93)
    ax.set_xlabel("pooled out-of-fold ROC-AUC (95% bootstrap CI, near-duplicate groups resampled)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.16), frameon=False)
    ax.set_title("F5. D02 facial images: model ROC-AUC vs a no-face image-statistics baseline", loc="left")
    save(fig, "F05_d02_auc_vs_image_stats")


# ---------------------------------------------------------------------------------------------- F3 / F4
def f3_forest():
    fig, ax = plt.subplots(figsize=(7, 4.2))
    y = np.arange(len(MODELS))[::-1]
    d = [sc("D01", m, "diff_vs_LR3") for m in MODELS]
    dot_ci(ax, y, d, [sc("D01", m, "diff_ci_lo_vs_LR3") for m in MODELS], [sc("D01", m, "diff_ci_hi_vs_LR3") for m in MODELS],
           [MODEL_COLORS[m] for m in MODELS])
    ax.axvline(0, color=BASELINE_COLOR, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(MODELS)
    ax.set_xlabel("child-level pooled AUC: model - quality-only baseline (95% paired bootstrap CI)")
    ax.text(0.99, 0.02, "right of 0 = model better", transform=ax.transAxes, ha="right", fontsize=7, color="#555")
    ax.set_title("F3. D01: AUC difference model - tracking-quality baseline (LR on 3 features)", loc="left")
    save(fig, "F03_d01_forest_vs_quality")


def f4_matched():
    fig, ax = plt.subplots(figsize=(7, 4.2))
    y = np.arange(len(MODELS))[::-1]
    a = [sc("D01", m, "matched_auc") for m in MODELS]
    dot_ci(ax, y, a, [sc("D01", m, "matched_ci_lo") for m in MODELS], [sc("D01", m, "matched_ci_hi") for m in MODELS],
           [MODEL_COLORS[m] for m in MODELS])
    q = sc("D01 baseline", "one_minus_valid_ratio", "matched_auc")
    ax.axvspan(sc("D01 baseline", "one_minus_valid_ratio", "matched_ci_lo"), sc("D01 baseline", "one_minus_valid_ratio", "matched_ci_hi"),
               color="#999", alpha=0.18, lw=0)
    ax.axvline(q, color=BASELINE_COLOR, ls="--", lw=1.1, label=f"quality only (1 - valid_ratio) on the matched set: {q:.3f} (CI shaded)")
    ax.axvline(0.5, color="#bbb", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(MODELS)
    n = int(sc("D01 baseline", "matching", "n_pairs"))
    ax.set_xlabel(f"child-level AUC on {n} quality-matched ASD-TD pairs (95% bootstrap CI over pairs)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)
    ax.set_title("F4. D01: AUC after 1:1 matching on mean valid_ratio (caliper 0.05)", loc="left")
    save(fig, "F04_d01_matched_subset")


# ---------------------------------------------------------------------------------------------- F6
def f6_roc(M):
    g = M[M.dataset == "D02"].sort_values("pooled_auc", ascending=False).head(3)
    fig, ax = plt.subplots(figsize=(5.2, 5))
    for r in g.itertuples():
        P = load_oof(D02, r.model)
        fpr, tpr, _ = roc_curve(P.y_true, P.y_prob)
        ax.plot(fpr, tpr, color=MODEL_COLORS[r.model], lw=1.6, label=f"{r.model} {NAMES[r.model]} (AUC {r.pooled_auc:.3f})")
    X = pd.read_csv(os.path.join(S8, "d02_image_stats_oof.csv"))
    fpr, tpr, _ = roc_curve(X.label, X.p_rf)
    ax.plot(fpr, tpr, color=BASELINE_COLOR, ls="--", lw=1.3,
            label=f"image statistics only, random forest (AUC {sc('D02 baseline', 'image_stats_random_forest', 'pooled_auc'):.3f})")
    ax.plot([0, 1], [0, 1], color="#bbb", lw=0.8)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=7)
    ax.set_title("F6. D02: pooled out-of-fold ROC curves, top 3 models", loc="left")
    save(fig, "F06_d02_roc_top3")


# ---------------------------------------------------------------------------------------------- F7
def cd_diagram(ax, avg_ranks, cd, title):
    names = np.array(list(avg_ranks))
    R = np.array([avg_ranks[n] for n in names])
    o = np.argsort(R)
    names, R = names[o], R[o]
    k = len(R)
    ax.set_xlim(0.2, k + 0.8)
    ax.set_ylim(0, 1)
    ax.axis("off")
    y0 = 0.78
    ax.hlines(y0, 1, k, color="k", lw=1)
    for t in range(1, k + 1):
        ax.vlines(t, y0, y0 + 0.03, color="k", lw=1)
        ax.text(t, y0 + 0.05, str(t), ha="center", va="bottom", fontsize=8)
    ax.hlines(y0 + 0.16, 1, 1 + cd, color="#c0392b", lw=2)
    ax.text(1 + cd / 2, y0 + 0.175, f"Nemenyi CD = {cd:.2f}", ha="center", va="bottom", fontsize=8, color="#c0392b")
    half = (k + 1) // 2
    for i, (n, r) in enumerate(zip(names, R)):
        left = i < half
        yy = y0 - 0.1 - 0.075 * (i if left else k - 1 - i)
        xx = 0.6 if left else k + 0.4
        ax.plot([r, r, xx], [y0, yy, yy], color=MODEL_COLORS[n], lw=1)
        ax.text(xx + (-0.05 if left else 0.05), yy, f"{n} ({r:.2f})", ha="right" if left else "left", va="center",
                fontsize=8, color=MODEL_COLORS[n])
    cl = []
    for i in range(k):
        j = max(jj for jj in range(i, k) if R[jj] - R[i] < cd)
        if j > i and not any(a <= i and j <= b for a, b in cl):
            cl.append((i, j))
    for c, (i, j) in enumerate(cl):
        ax.hlines(y0 - 0.035 - 0.025 * c, R[i] - 0.03, R[j] + 0.03, color="#333", lw=3)
    ax.set_title(title, fontsize=9, loc="left")


def f7_cd():
    fr = json.load(open(os.path.join(S8, "stats_friedman_nemenyi.json")))
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.6))
    for ax, tag, lab in [(axes[0], "D01_clip", "D01 eye-tracking (clip-level fold AUC)"), (axes[1], "D02_image", "D02 facial images (fold AUC)")]:
        f = fr[tag]
        p = f"{f['p']:.2g}" if f["p"] < 0.001 else f"{f['p']:.3f}"
        cd_diagram(ax, f["avg_ranks"], f["nemenyi_cd"], f"{lab}: Friedman chi2 = {f['chi2']:.2f}, p = {p}")
    fig.text(0.01, 0.005, "Average rank over 10 folds (1 = best). Black bars join models that are not significantly different "
             "(Nemenyi, alpha 0.05).", fontsize=7)
    fig.suptitle("F7. Critical-difference diagrams", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "F07_cd_diagrams")


# ---------------------------------------------------------------------------------------------- F8
def best_models(M):
    out = {}
    for key in ["D01 clip", "D01 child", "D02", "D04", "D05 set_all", "D05 set_honest", "D06"]:
        g = M[(M.dataset == key) & (M.model != "dummy_majority")]
        out[key] = g.sort_values("pooled_auc", ascending=False).iloc[0].model
    return out


def summary_path(key, model):
    if key.startswith("D01"):
        return os.path.join(PROJECT_DIR, "results", D01, model, "summary.json")
    if key == "D02":
        return os.path.join(PROJECT_DIR, "results", D02, model, "summary.json")
    ds, exp = TAB[key]
    return os.path.join(PROJECT_DIR, "results", ds, exp, model, "summary.json")


def f8_confusion(M):
    best = best_models(M)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.2))
    rows = []
    for ax, (key, m) in zip(axes.ravel(), best.items()):
        s = json.load(open(summary_path(key, m)))
        blk = s["pooled_out_of_fold_child" if key == "D01 child" else "pooled_out_of_fold"]["threshold_0.5"]
        cm = np.array(blk["confusion_matrix"])
        rows.append({"dataset": key, "model": m, "tn": cm[0, 0], "fp": cm[0, 1], "fn": cm[1, 0], "tp": cm[1, 1],
                     "source": os.path.relpath(summary_path(key, m), PROJECT_DIR) + ("::pooled_out_of_fold_child" if key == "D01 child"
                                                                                       else "::pooled_out_of_fold") + "::threshold_0.5::confusion_matrix"})
        frac = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n({frac[i, j]:.0%})", ha="center", va="center", fontsize=8,
                        color="white" if frac[i, j] > 0.6 else "black")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["pred. neg.", "pred. pos."])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["true neg.", "true pos."])
        ax.set_title(f"{key}: {m}", fontsize=9, color=MODEL_COLORS.get(m, TAB_COLORS.get(m, "k")))
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes.ravel()[-1].axis("off")
    axes.ravel()[-1].text(0.0, 0.5, "Best model per dataset = highest pooled\nout-of-fold AUC (dummy excluded).\n"
                          "Pooled out-of-fold predictions,\nthreshold 0.5; row-normalised colour;\npositive = ASD / autistic.\n"
                          "D06: no model exceeds chance\n(best pooled AUC < 0.5).", fontsize=8, va="center")
    write_csv(pd.DataFrame(rows), os.path.join(FINAL_DIR, "confusion_matrices_best.csv"))
    fig.suptitle("F8. Confusion matrices of the best model per dataset", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "F08_confusion_matrices")


# ---------------------------------------------------------------------------------------------- F9
def f9_attention():
    m = "M07"
    parts = []
    for k in range(1, 11):
        d = os.path.join(PROJECT_DIR, "results", D01, m, f"fold_{k:02d}")
        p = pd.read_csv(os.path.join(d, "predictions.csv"))
        a = np.load(os.path.join(d, "attention.npy"))
        assert a.shape == (len(p), 8)
        parts.append(pd.concat([p[["group_id", "y_true"]], pd.DataFrame(a, columns=[f"s{i + 1}" for i in range(8)])], axis=1))
    A = pd.concat(parts, ignore_index=True)
    C = A.groupby("group_id").agg({"y_true": "first", **{f"s{i + 1}": "mean" for i in range(8)}})
    rows = []
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    sec = np.arange(1, 9)
    for lab, name, col, off in [(1, "ASD", "#c0392b", -0.06), (0, "TD", "#2471a3", 0.06)]:
        g = C[C.y_true == lab][[f"s{i}" for i in sec]]
        mu, se = g.mean().to_numpy(), g.std(ddof=1).to_numpy() / np.sqrt(len(g))
        ax.errorbar(sec + off, mu, yerr=1.96 * se, fmt="o-", ms=4, capsize=2, color=col, label=f"{name} ({len(g)} children)")
        rows += [{"group": name, "second": int(s), "n_children": len(g), "mean_weight": float(v), "ci95_half_width": float(1.96 * e)}
                 for s, v, e in zip(sec, mu, se)]
    ax.axhline(1 / 8, color="#999", ls=":", lw=1, label="uniform (1/8)")
    ax.set_xticks(sec)
    ax.set_xlabel("second within the 8-s clip")
    ax.set_ylabel("mean attention weight")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("F9. D01 M07 (ViT + LSTM + Attn): temporal attention per second\n(out-of-fold test clips, averaged per child; "
                 "mean ± 95% CI over children; descriptive)", loc="left", fontsize=9)
    write_csv(pd.DataFrame(rows), os.path.join(FINAL_DIR, "d01_m07_attention_per_second.csv"))
    save(fig, "F09_d01_m07_attention")


# ---------------------------------------------------------------------------------------------- F10
def f10_tabular(M):
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.6), sharey=True)
    y = np.arange(len(TAB_MODELS))[::-1]
    for ax, key in zip(axes, TAB):
        g = M[M.dataset == key].set_index("model").loc[TAB_MODELS]
        dot_ci(ax, y, g.pooled_auc, g.pooled_auc_ci_lo, g.pooled_auc_ci_hi, [TAB_COLORS[m] for m in TAB_MODELS])
        ax.axvline(0.5, color=BASELINE_COLOR, ls="--", lw=1)
        ax.set_xlim(0.3, 1.05)
        ax.set_title(key + (" (label-definitional)" if key == "D05 set_all" else ""), fontsize=9)
        ax.set_xlabel("pooled OOF ROC-AUC (95% CI)")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([m.replace("dummy_majority", "dummy (majority)") for m in TAB_MODELS])
    fig.suptitle("F10. Tabular baselines: ROC-AUC per model (dashed line = chance / dummy baseline 0.5)", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "F10_tabular_auc")


# ---------------------------------------------------------------------------------------------- F11
def f11_pretrained(M):
    T = pd.read_csv(os.path.join(FINAL_DIR, "stats_table.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))
    for ax, key, sec in [(axes[0], "D01 clip", "D01 clip"), (axes[1], "D01 child", "D01 child"), (axes[2], "D02", "D02 image")]:
        g = M[M.dataset == key].set_index("model")
        for x, ms in [(0, PRETRAINED), (1, CUSTOM_CNN)]:
            jit = np.linspace(-0.15, 0.15, len(ms))
            for j, m in zip(jit, ms):
                ax.plot(x + j, g.loc[m, "roc_auc_mean"], "o", ms=6, color=MODEL_COLORS[m])
            grp = "pretrained (M01,M03,M04,M06,M07,M08,M10)" if x == 0 else "from scratch (M02,M05,M09)"
            mv = T[(T.section == f"pretrained_vs_scratch {sec}") & (T.model == grp) & (T.quantity == "mean_of_fold_mean_auc")].value.iloc[0]
            ax.hlines(mv, x - 0.28, x + 0.28, color="k", lw=2)
            ax.text(x + 0.31, mv, f"{mv:.3f}", va="center", fontsize=8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["pretrained\n(7 models)", "from scratch\n(3 custom CNNs)"])
        ax.set_xlim(-0.5, 1.6)
        ax.set_title(key)
    axes[0].set_ylabel("mean test ROC-AUC over 10 folds")
    handles = [plt.Line2D([], [], marker="o", ls="", color=MODEL_COLORS[m], label=m) for m in MODELS]
    fig.legend(handles=handles, loc="lower center", ncol=10, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("F11. Pretrained vs from-scratch spatial encoders (black bar = group mean; descriptive)", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "F11_pretrained_vs_scratch")


# ---------------------------------------------------------------------------------------------- F12
def f12_reliability(M):
    best = best_models(M)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.2), sharex=True, sharey=True)
    rows = []
    for ax, (key, m) in zip(axes.ravel(), best.items()):
        if key.startswith("D01"):
            P = load_oof(D01, m)
            if key == "D01 child":
                C = child_scores(P)
                y, p = C.y.to_numpy(), C.p.to_numpy()
            else:
                y, p = P.y_true.to_numpy(), P.y_prob.to_numpy()
        elif key == "D02":
            P = load_oof(D02, m)
            y, p = P.y_true.to_numpy(), P.y_prob.to_numpy()
        else:
            P = load_oof(*TAB[key][:1], m, TAB[key][1])
            y, p = P.y_true.to_numpy(), P.y_prob.to_numpy()
        b = pd.DataFrame(S.reliability_bins(y, p))
        rows += [{"dataset": key, "model": m, **r} for r in b.to_dict("records")]
        b = b[b.n > 0]
        col = MODEL_COLORS.get(m, TAB_COLORS.get(m, "k"))
        ax.plot([0, 1], [0, 1], color="#bbb", lw=0.8)
        ax.plot(b.mean_p, b.frac_pos, "o-", ms=3.5, color=col)
        ax2 = ax.twinx()
        ax2.bar((b.bin + 0.5) / 10, b.n / b.n.sum(), width=0.09, color="#999", alpha=0.25)
        ax2.set_ylim(0, 1)
        ax2.set_yticks([])
        ax2.spines["right"].set_visible(False)
        ece = M[(M.dataset == key) & (M.model == m)].ece_pooled.iloc[0]
        ax.set_title(f"{key}: {m} (ECE {ece:.3f})", fontsize=9, color=col)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0, 3].tick_params(labelbottom=True)
    axes[0, 3].set_xlabel("mean predicted P(positive)")
    for ax in axes[1]:
        ax.set_xlabel("mean predicted P(positive)")
    for ax in axes[:, 0]:
        ax.set_ylabel("observed fraction positive")
    axes.ravel()[-1].axis("off")
    axes.ravel()[-1].text(0.05, 0.5, "Pooled out-of-fold predictions,\n10 equal-width bins; grey bars =\nshare of predictions per bin;\n"
                          "diagonal = perfect calibration.\nD01 child: mean clip probability.", fontsize=8, va="center",
                          transform=axes.ravel()[-1].transAxes)
    write_csv(pd.DataFrame(rows), os.path.join(FINAL_DIR, "reliability_bins_best.csv"))
    fig.suptitle("F12. Reliability diagrams of the best model per dataset", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "F12_reliability_best")


def main():
    os.makedirs(FIG_FINAL, exist_ok=True)
    M = master()
    f1_pipeline()
    f2_d01_auc(M)
    f3_forest()
    f4_matched()
    f5_d02_auc(M)
    f6_roc(M)
    f7_cd()
    f8_confusion(M)
    f9_attention()
    f10_tabular(M)
    f11_pretrained(M)
    f12_reliability(M)
    print(sorted(os.listdir(FIG_FINAL)))


if __name__ == "__main__":
    main()
