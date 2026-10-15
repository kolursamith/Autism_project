"""Stage 2 split tests (all datasets, all folds): disjointness, coverage, class presence, determinism, manifest."""
import json
import os
import subprocess
import sys

import pandas as pd
import pytest

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
from src.data import splits as sp  # noqa: E402
from src.data.common import SPLITS_DIR, sha256_file  # noqa: E402

DATASETS = list(sp.DATASETS)
FOLDS = range(1, 11)


def _folds(ds):
    return pd.read_csv(os.path.join(SPLITS_DIR, ds, "folds.csv"))


@pytest.mark.parametrize("ds", DATASETS)
@pytest.mark.parametrize("k", FOLDS)
def test_parts_disjoint_in_samples_and_groups(ds, k):
    f = sp.load_fold(ds, k)
    grp = _folds(ds).set_index("sample_id")["group_id"]
    parts = {p: set(f[p]) for p in ["train", "val", "test"]}
    groups = {p: set(grp[list(s)]) for p, s in parts.items()}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not parts[a] & parts[b], f"{ds} fold {k}: {a}/{b} share samples"
        assert not groups[a] & groups[b], f"{ds} fold {k}: {a}/{b} share groups"
    assert parts["train"] | parts["val"] | parts["test"] == set(grp.index)
    for p in parts:
        assert f["counts"][p]["n_samples"] == len(parts[p])


@pytest.mark.parametrize("ds", DATASETS)
def test_every_sample_in_exactly_one_test_fold(ds):
    seen = []
    for k in FOLDS:
        seen += sp.load_fold(ds, k)["test"]
    allids = set(_folds(ds)["sample_id"])
    assert len(seen) == len(set(seen)), "a sample is in two test folds"
    assert set(seen) == allids
    f = _folds(ds)
    for k in FOLDS:
        assert set(f.loc[f.outer_fold == k, "sample_id"]) == set(sp.load_fold(ds, k)["test"])


@pytest.mark.parametrize("ds", DATASETS)
@pytest.mark.parametrize("k", FOLDS)
def test_test_and_val_contain_both_classes(ds, k):
    lab = _folds(ds).set_index("sample_id")["label"]
    f = sp.load_fold(ds, k)
    for p in ["val", "test", "train"]:
        assert set(lab[f[p]]) == {0, 1}, f"{ds} fold {k} {p} lacks a class"


def test_d01_test_folds_have_asd_and_td_children():
    f = _folds("D01_eye_tracking").drop_duplicates("group_id")
    for k in FOLDS:
        kids = f[f.outer_fold == k]
        assert (kids.label == 1).sum() >= 1 and (kids.label == 0).sum() >= 1
    # every child is in exactly one outer fold
    assert _folds("D01_eye_tracking").groupby("group_id").outer_fold.nunique().max() == 1


def test_splits_manifest_hashes_match():
    man = sp.verify_manifest()
    assert set(man["datasets"]) == set(DATASETS)
    for ds in DATASETS:
        assert len(man["datasets"][ds]["files"]) == 12


def test_rerun_make_splits_is_byte_identical(tmp_path):
    r = subprocess.run([sys.executable, os.path.join(PROJECT_DIR, "scripts", "02d_make_splits.py"), "--out",
                        str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    with open(os.path.join(SPLITS_DIR, "SPLITS_MANIFEST.json")) as fh:
        man = json.load(fh)
    for ds in DATASETS:
        for f in sp.split_files(ds):
            assert sha256_file(os.path.join(tmp_path, ds, f)) == man["datasets"][ds]["files"][f"splits/{ds}/{f}"], f
