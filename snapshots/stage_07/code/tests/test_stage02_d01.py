"""Stage 2 D01 eye-tracking integrity tests."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
from src.data import d01_eye as d  # noqa: E402
from src.data.common import CACHE_DIR, SPLITS_DIR, load_cfg  # noqa: E402

C = os.path.join(CACHE_DIR, "D01_eye_tracking")
CFG = load_cfg("D01")


@pytest.fixture(scope="module")
def gaze():
    return pd.read_pickle(os.path.join(C, "gaze_clean.pkl"))


@pytest.fixture(scope="module")
def idx():
    return pd.read_csv(os.path.join(C, "clips_index.csv"))


@pytest.fixture(scope="module")
def rebuilt():
    """The full pipeline re-run from the raw CSVs (independent of the cached gaze table)."""
    df, *_ = d.build_clean_gaze(CFG)
    return d.recordings(df)


def test_no_unidentified_rows(gaze):
    assert pd.api.types.is_integer_dtype(gaze["participant_id"])
    meta = d.read_metadata()
    assert set(gaze["participant_id"]) <= set(meta["participant_id"])
    assert not {12, 16} & set(gaze["participant_id"])


def test_no_duplicate_participant_time_trial(gaze):
    assert not gaze.duplicated(["participant_id", "t_ms", "trial"]).any()


def test_no_cars_anywhere_in_cache_or_splits():
    hits = []
    for root in [C, SPLITS_DIR]:
        for dp, _, fns in os.walk(root):
            for fn in fns:
                p = os.path.join(dp, fn)
                if fn.endswith(".npy"):
                    continue
                if fn.endswith(".pkl"):
                    cols = pd.read_pickle(p).columns
                    hits += [f"{p}:{c}" for c in cols if "cars" in c.lower()]
                    continue
                with open(p, errors="ignore") as fh:
                    if "cars" in fh.read().lower():
                        hits.append(p)
    assert not hits, hits


def test_npy_shape_matches_index(idx):
    arr = np.load(os.path.join(C, "clips_uint8.npy"), mmap_mode="r")
    assert arr.dtype == np.uint8
    assert arr.shape == (len(idx), 8, 3, 224, 224)
    assert (idx["row_in_npy"].to_numpy() == np.arange(len(idx))).all()
    assert idx["clip_id"].is_unique


def test_clip_rules(idx):
    assert (idx["valid_ratio"] >= CFG["clips"]["min_valid_ratio"]).all()
    assert idx.groupby("participant_id").size().max() <= CFG["clips"]["max_clips_per_child"]
    assert np.allclose(idx["end_ms"] - idx["start_ms"], 8000)
    unc = pd.read_csv(os.path.join(C, "clips_index_uncapped.csv"))
    assert (unc.loc[unc.status.isin(["kept", "dropped_by_cap"]), "max_gap_ms"] <= CFG["clips"]["gap_max_ms"]).all()
    assert set(unc.loc[unc.status == "kept", "clip_id"]) == set(idx["clip_id"])


def test_no_clip_spans_two_trials(idx, gaze):
    """No sample of another trial (same file + child) lies inside a clip's time span, and the clip's own samples
    all belong to its trial."""
    g = gaze.groupby(["source_file", "participant_id"])
    for r in idx.itertuples():
        rows = g.get_group((r.source_file, r.participant_id))
        inside = rows[(rows.t_ms >= r.start_ms) & (rows.t_ms < r.end_ms)]
        assert set(inside["trial"]) == {r.trial}, r.clip_id
        assert len(inside) == r.n_samples, r.clip_id


def test_20_random_clips_rerendered_from_raw_are_identical(idx, rebuilt):
    arr = np.load(os.path.join(C, "clips_uint8.npy"), mmap_mode="r")
    groups = rebuilt.groupby(["source_file", "participant_id", "trial"], sort=False).indices
    pick = np.random.default_rng(2024).choice(len(idx), size=20, replace=False)
    for i in pick:
        r = idx.iloc[i]
        rec = rebuilt.iloc[groups[(r.source_file, r.participant_id, r.trial)]]
        img = d.render_clip(rec, r.start_ms, CFG)
        assert np.array_equal(img, arr[int(r.row_in_npy)]), r.clip_id
        assert img.any(), f"{r.clip_id} is completely black"


def test_rebuilt_gaze_equals_cached(gaze, rebuilt):
    cols = ["source_file", "row_in_file", "participant_id", "trial", "t_ms", "x_px", "y_px", "valid", "event"]
    pd.testing.assert_frame_equal(rebuilt[cols].reset_index(drop=True), gaze[cols].reset_index(drop=True))


def test_quality_table_not_in_model_inputs(idx):
    q = pd.read_csv(os.path.join(C, "quality_per_child.csv"))
    assert set(q["participant_id"]) == set(idx["participant_id"])
    assert "tracking_ratio" not in idx.columns  # quality features are only in quality_per_child.csv
