"""Stage 2 figures (C6, D4, F5) -> reports/figures/. Log: logs/stage02_figures.log
Usage: .venv/bin/python scripts/02e_stage02_figures.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from src.data import splits as sp  # noqa: E402
from src.data.common import CACHE_DIR, DATASETS_DIR, FIG_DIR, SPLITS_DIR, get_logger, set_seed  # noqa: E402


def d01_examples(log):
    d = os.path.join(CACHE_DIR, "D01_eye_tracking")
    idx = pd.read_csv(os.path.join(d, "clips_index.csv"))
    arr = np.load(os.path.join(d, "clips_uint8.npy"), mmap_mode="r")
    rng = np.random.default_rng(42)
    picks = []
    for lab in [1, 0]:
        kids = rng.choice(sorted(idx[idx.label == lab].participant_id.unique()), 3, replace=False)
        for p in sorted(kids):
            c = idx[idx.participant_id == p]
            picks.append(c.iloc[int(rng.integers(len(c)))])
    fig, axes = plt.subplots(6, 8, figsize=(16, 13))
    for i, r in enumerate(picks):
        clip = arr[int(r.row_in_npy)]                      # (8, 3, 224, 224)
        for w in range(8):
            ax = axes[i, w]
            ax.imshow(np.transpose(clip[w], (1, 2, 0)))    # R = scan path, G = fixation density, B = time
            ax.set_xticks([]), ax.set_yticks([])
            if w == 0:
                ax.set_ylabel(f"{'ASD' if r.label == 1 else 'TD'} p{int(r.participant_id):02d}\n"
                              f"valid {r.valid_ratio:.2f}", fontsize=10)
            if i == 0:
                ax.set_title(f"second {w + 1}", fontsize=10)
    fig.suptitle("D01: 3 ASD + 3 TD clips (8 x 1-s windows). R = scan path, G = fixation density, B = time", fontsize=13)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "stage02_d01_examples.png")
    fig.savefig(path, dpi=80)
    plt.close(fig)
    log.info(f"{path}: clips {[r.clip_id for r in picks]}")


def d02_neardups(log):
    d = os.path.join(CACHE_DIR, "D02_facial_images_A")
    man = pd.read_csv(os.path.join(d, "manifest.csv"))
    pairs = pd.read_csv(os.path.join(d, "near_duplicate_pairs.csv"))
    sizes = man.groupby("group_id").size()
    groups = sorted(sizes[sizes > 1].index)[:8]
    width = max(sizes[groups])
    fig, axes = plt.subplots(width, len(groups), figsize=(2.4 * len(groups), 2.8 * width), squeeze=False)
    for j, g in enumerate(groups):
        members = man[man.group_id == g].sort_values("image_id")
        ham = pairs[pairs.image_a.isin(members.image_id) & pairs.image_b.isin(members.image_id)].hamming.tolist()
        for i in range(width):
            ax = axes[i, j]
            ax.set_xticks([]), ax.set_yticks([])
            if i < len(members):
                m = members.iloc[i]
                with Image.open(os.path.join(DATASETS_DIR, m.rel_path)) as im:
                    ax.imshow(im.convert("RGB"))
                ax.set_xlabel(f"{m.image_id}\n({m.original_split}, label {m.label})", fontsize=7)
            if i == 0:
                ax.set_title(f"{g}  Hamming {ham}", fontsize=8)
    fig.suptitle("D02: near-duplicate groups (pHash Hamming <= 4), one column per group", fontsize=11)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "stage02_d02_neardups.png")
    fig.savefig(path, dpi=90)
    plt.close(fig)
    log.info(f"{path}: groups {groups}")


def fold_balance(log):
    fig, axes = plt.subplots(1, len(sp.DATASETS), figsize=(4 * len(sp.DATASETS), 3.6))
    for ax, ds in zip(axes, sp.DATASETS):
        f = pd.read_csv(os.path.join(SPLITS_DIR, ds, "folds.csv"))
        if ds == "D01_eye_tracking":
            f = f.drop_duplicates("group_id")
            unit = "children"
        else:
            unit = "samples"
        c = f.groupby(["outer_fold", "label"]).size().unstack(fill_value=0).reindex(columns=[0, 1], fill_value=0)
        x = c.index.to_numpy()
        ax.bar(x, c[0], color="#4C78A8", label="label 0 (TD / non-ASD)")
        ax.bar(x, c[1], bottom=c[0], color="#F58518", label="label 1 (ASD)")
        ax.set_title(ds, fontsize=9)
        ax.set_xlabel("outer test fold")
        ax.set_ylabel(f"test {unit}")
        ax.set_xticks(x)
        log.info(f"{ds} test-fold {unit} per class:\n{c.T.to_string()}")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Test-fold class counts across the 10 outer folds", fontsize=11)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "stage02_fold_balance.png")
    fig.savefig(path, dpi=90)
    plt.close(fig)
    log.info(path)


def main():
    set_seed(42)
    os.makedirs(FIG_DIR, exist_ok=True)
    log = get_logger("stage02_figures")
    d01_examples(log)
    d02_neardups(log)
    fold_balance(log)


if __name__ == "__main__":
    main()
