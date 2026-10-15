"""Stage 2 D02 facial-image integrity tests (DD-05..DD-07)."""
import os
import sys

import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
from src.data import d02_faces as f  # noqa: E402
from src.data.common import CACHE_DIR, DATASETS, DATASETS_DIR, PROJECT_DIR as P, SPLITS_DIR  # noqa: E402

C = os.path.join(CACHE_DIR, "D02_facial_images_A")
D03_PREFIX = DATASETS["D03_facial_images_B"] + os.sep


def _man():
    return pd.read_csv(os.path.join(C, "manifest.csv"))


def test_manifest_pooled_and_complete():
    m = _man()
    assert len(m) == 2940 and m.image_id.is_unique
    assert set(m.original_split) == {"train", "valid", "test"}
    assert m.label.value_counts().to_dict() == {1: 1470, 0: 1470}
    assert m.rel_path.str.startswith(DATASETS["D02_facial_images_A"] + os.sep).all()
    assert all(os.path.isfile(os.path.join(DATASETS_DIR, p)) for p in m.rel_path)


def test_no_d03_path_anywhere():
    hits = []
    for root in [os.path.join(CACHE_DIR, "D02_facial_images_A"), os.path.join(SPLITS_DIR, "D02_facial_images_A")]:
        for dp, _, fns in os.walk(root):
            for fn in fns:
                with open(os.path.join(dp, fn), errors="ignore") as fh:
                    for line in fh:
                        if any(tok.startswith(D03_PREFIX) for tok in line.replace('"', ",").split(",")):
                            hits.append(fn)
                            break
    assert not hits, hits
    # the code never builds a path to the D03 folder
    for fn in ["src/data/d02_faces.py", "scripts/02b_prepare_d02.py"]:
        with open(os.path.join(P, fn)) as fh:
            assert "D03_facial_images_B" not in fh.read()


def test_near_duplicate_pairs_recomputed_lie_in_same_outer_fold():
    m = _man()
    d = f.hamming_matrix(m.phash.tolist())
    folds = pd.read_csv(os.path.join(SPLITS_DIR, "D02_facial_images_A", "folds.csv")).set_index("sample_id")
    fold = folds.loc[m.image_id, "outer_fold"].to_numpy()
    import numpy as np
    i, j = np.nonzero(np.triu(d <= 4, k=1))
    assert len(i) == 11
    assert (fold[i] == fold[j]).all()
    assert (m.group_id.to_numpy()[i] == m.group_id.to_numpy()[j]).all()


def test_no_group_has_mixed_labels():
    m = _man()
    assert m.groupby("group_id").label.nunique().max() == 1
