"""Stage 8 E - figures from results/stage8/*.csv|json only -> reports/figures/stage08_*.png.
Usage: .venv/bin/python scripts/08e_figures.py"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.analysis.data import CUSTOM_CNN, D01, D02, FIG_DIR, MODELS, OUT_DIR  # noqa: E402

DPI = 150


def cd_diagram(ax, avg_ranks, cd, title):
    """Demšar-style critical-difference diagram: rank axis (1 = best, left), models attached left/right, thick bars join
    models whose average ranks differ by less than the Nemenyi CD (not significantly different)."""
    names = np.array(list(avg_ranks.keys()))
    R = np.array(list(avg_ranks.values()))
    o = np.argsort(R)
    names, R = names[o], R[o]
    k = len(R)
    lo, hi = 1, k
    ax.set_xlim(lo - 0.5, hi + 0.5)
    ax.set_ylim(0, 1)
    ax.axis("off")
    y0 = 0.80
    ax.hlines(y0, lo, hi, color="k", lw=1)
    for t in range(lo, hi + 1):
        ax.vlines(t, y0, y0 + 0.03, color="k", lw=1)
        ax.text(t, y0 + 0.05, str(t), ha="center", va="bottom", fontsize=8)
    ax.hlines(y0 + 0.14, lo, lo + cd, color="tab:red", lw=2)
    ax.text(lo + cd / 2, y0 + 0.155, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=8, color="tab:red")
    half = (k + 1) // 2
    for i, (n, r) in enumerate(zip(names, R)):
        left = i < half
        yy = y0 - 0.08 - 0.07 * (i if left else k - 1 - i)
        xx = lo - 0.4 if left else hi + 0.4
        ax.plot([r, r, xx], [y0, yy, yy], color="k", lw=0.8)
        ax.text(xx + (-0.05 if left else 0.05), yy, f"{n} ({r:.2f})", ha="right" if left else "left", va="center",
                fontsize=8, color="tab:orange" if n in CUSTOM_CNN else "k")
    cliques = []
    for i in range(k):
        j = max(jj for jj in range(i, k) if R[jj] - R[i] < cd)
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))
    for c, (i, j) in enumerate(cliques):
        yy = y0 - 0.03 - 0.025 * c
        ax.hlines(yy, R[i] - 0.03, R[j] + 0.03, color="tab:blue", lw=3)
    ax.set_title(title, fontsize=9)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    fr = json.load(open(os.path.join(OUT_DIR, "stats_friedman_nemenyi.json")))
    # ---------------------------------------------------------------- CD diagrams
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.0))
    for ax, tag, lab in [(axes[0], "D01_clip", "clip-level"), (axes[1], "D01_child", "child-level")]:
        f = fr[tag]
        cd_diagram(ax, f["avg_ranks"], f["nemenyi_cd"],
                   f"D01 eye-tracking, {lab} fold AUC: Friedman chi2 = {f['chi2']:.2f}, p = {f['p']:.3g} (10 folds)")
    fig.text(0.01, 0.01, "Average rank over 10 folds (1 = best). Blue bars: not significantly different (Nemenyi, alpha 0.05). "
             "Orange = custom CNN.", fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(FIG_DIR, "stage08_cd_D01.png"), dpi=DPI)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 2.7))
    f = fr["D02_image"]
    cd_diagram(ax, f["avg_ranks"], f["nemenyi_cd"],
               f"D02 facial images, fold AUC: Friedman chi2 = {f['chi2']:.2f}, p = {f['p']:.2g} (10 folds)")
    fig.text(0.01, 0.01, "Average rank over 10 folds (1 = best). Blue bars: not significantly different (Nemenyi, alpha 0.05). "
             "Orange = custom CNN.", fontsize=7)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(FIG_DIR, "stage08_cd_D02.png"), dpi=DPI)
    plt.close(fig)

    # ---------------------------------------------------------------- reliability diagrams
    rel = pd.read_csv(os.path.join(OUT_DIR, "stats_reliability_bins.csv"))
    cal = pd.read_csv(os.path.join(OUT_DIR, "stats_calibration.csv"))
    for ds, short, lev in [(D01, "D01", "clip"), (D02, "D02", "image")]:
        fig, axes = plt.subplots(2, 5, figsize=(13, 5.6), sharex=True, sharey=True)
        for ax, m in zip(axes.ravel(), MODELS):
            b = rel[(rel.dataset == ds) & (rel.model == m) & (rel.n > 0)]
            c = cal[(cal.dataset == ds) & (cal.model == m) & (cal.level == lev)].iloc[0]
            ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=0.8)
            ax.plot(b.mean_p, b.frac_pos, "o-", ms=3, color="tab:blue")
            ax2 = ax.twinx()
            ax2.bar((b.bin + 0.5) / 10, b.n / b.n.sum(), width=0.09, color="tab:grey", alpha=0.25)
            ax2.set_ylim(0, 1)
            ax2.set_yticks([])
            ax.set_title(f"{m}: ECE {c.ece_10:.3f}, Brier {c.brier:.3f}", fontsize=8)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        for ax in axes[1]:
            ax.set_xlabel("mean predicted P(ASD)", fontsize=8)
        for ax in axes[:, 0]:
            ax.set_ylabel("observed fraction ASD", fontsize=8)
        fig.suptitle(f"{short} reliability diagrams, pooled out-of-fold {lev}-level predictions (10 equal-width bins; "
                     "grey bars = share of predictions per bin)", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f"stage08_reliability_{short}.png"), dpi=DPI)
        plt.close(fig)

    # ---------------------------------------------------------------- D01 forest plot
    B2 = pd.read_csv(os.path.join(OUT_DIR, "d01_model_vs_quality.csv"))
    B4 = pd.read_csv(os.path.join(OUT_DIR, "d01_matched_auc.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0), sharey=True)
    y = np.arange(len(MODELS))[::-1]
    for off, base, col, lab in [(0.12, "quality_LR3", "tab:blue", "vs quality LR (3 features, fit per fold)"),
                                (-0.12, "one_minus_valid_ratio", "tab:grey", "vs 1 - mean valid_ratio")]:
        d = B2[B2.baseline == base].set_index("model").loc[MODELS]
        axes[0].errorbar(d["diff"], y + off, xerr=[d["diff"] - d.diff_ci_lo, d.diff_ci_hi - d["diff"]], fmt="o", ms=4,
                         color=col, capsize=2, label=lab)
    axes[0].axvline(0, color="k", lw=0.8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(MODELS)
    axes[0].set_xlabel("child-level pooled AUC: model - quality-only (95% paired bootstrap CI)", fontsize=8)
    axes[0].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    axes[0].set_title("D01: deep model vs tracking-quality-only baseline", fontsize=9)
    d = B4.set_index("model")
    dm = d.loc[MODELS]
    axes[1].errorbar(dm.auc_matched, y, xerr=[dm.auc_matched - dm.ci_lo, dm.ci_hi - dm.auc_matched], fmt="o", ms=4,
                     color="tab:green", capsize=2)
    q = d.loc["quality: 1 - valid_ratio"]
    axes[1].axvspan(q.ci_lo, q.ci_hi, color="tab:grey", alpha=0.15, label=f"quality 1 - valid_ratio: {q.auc_matched:.2f} (CI)")
    axes[1].axvline(0.5, color="k", lw=0.8)
    axes[1].set_xlabel(f"AUC on quality-matched children ({int(d.n_pairs.iloc[0])} ASD-TD pairs, 95% bootstrap CI)", fontsize=8)
    axes[1].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    axes[1].set_title("D01: AUC after 1:1 matching on mean valid_ratio (caliper 0.05)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "stage08_d01_model_vs_quality_forest.png"), dpi=DPI)
    plt.close(fig)

    # ---------------------------------------------------------------- D02 near-dup sensitivity
    C1 = pd.read_csv(os.path.join(OUT_DIR, "d02_near_dup_sensitivity.csv")).set_index("model").loc[MODELS]
    C2 = pd.read_csv(os.path.join(OUT_DIR, "d02_image_stats_baseline.csv"))
    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.arange(len(MODELS))
    ax.plot(x - 0.1, C1.fold_mean_auc_all, "o", color="tab:blue", label="all test images")
    ax.plot(x + 0.1, C1.fold_mean_auc_removed, "s", color="tab:orange",
            label=f"without {int(C1.n_removed.iloc[0])} test images with a pHash <= 8 training neighbour")
    for i in x:
        ax.plot([i - 0.1, i + 0.1], [C1.fold_mean_auc_all.iloc[i], C1.fold_mean_auc_removed.iloc[i]], color="grey", lw=0.8)
    for r, ls in zip(C2.itertuples(), ["--", ":"]):
        ax.axhline(r.fold_mean_auc, color="tab:red", ls=ls, lw=1,
                   label=f"{r.baseline.replace('_', ' ')} (no face model): {r.fold_mean_auc:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylabel("mean test ROC-AUC over 10 folds")
    ax.set_ylim(0.7, 1.0)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("D02: near-duplicate sensitivity (no retraining) and image-statistics baseline", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "stage08_d02_near_dup_sensitivity.png"), dpi=DPI)
    plt.close(fig)
    print("figures:", sorted(f for f in os.listdir(FIG_DIR) if f.startswith("stage08_")))


if __name__ == "__main__":
    main()
