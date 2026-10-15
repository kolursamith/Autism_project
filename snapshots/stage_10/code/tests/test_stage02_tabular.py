"""Stage 2 tabular tests (DD-08..DD-10) + raw-data snapshot."""
import glob
import importlib
import os
import sys

import pandas as pd
import yaml

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "scripts"))
from src.data import tabular as tb  # noqa: E402
from src.data.common import CACHE_DIR, dataset_dir  # noqa: E402


def _clean(ds):
    return pd.read_csv(os.path.join(CACHE_DIR, ds, "clean.csv"))


def _cols(ds):
    with open(os.path.join(CACHE_DIR, ds, "columns.yaml")) as fh:
        return yaml.safe_load(fh)


def test_d04_only_labelled_train_rows():
    c = _clean("D04_autism_prediction")
    tr = pd.read_csv(os.path.join(dataset_dir("D04_autism_prediction"), "train.csv"), dtype=str)
    te = pd.read_csv(os.path.join(dataset_dir("D04_autism_prediction"), "test.csv"), dtype=str)
    assert len(c) == 800 and sorted(c["ID"].astype(str)) == sorted(tr["ID"])
    # no test.csv row (all columns except ID) equals a used row
    cols = [x for x in te.columns if x != "ID"]
    used = set(map(tuple, tr[cols].to_numpy()))
    assert not used & set(map(tuple, te[cols].to_numpy()))
    assert _cols("D04_autism_prediction")["source_file"] == "train.csv"


def test_d04_result_decision_recorded():
    import json
    with open(os.path.join(CACHE_DIR, "D04_autism_prediction", "result_column_test.json")) as fh:
        r = json.load(fh)
    in_features = "result" in _cols("D04_autism_prediction")["feature_sets"]["set_all"]
    assert in_features == (not r["is_function_of_items"])


def test_d05_honest_set_excludes_items_and_score():
    fs = _cols("D05_toddler_screening")["feature_sets"]
    banned = {f"A{i}" for i in range(1, 11)} | {"Qchat-10-Score"}
    assert not banned & set(fs["set_honest"])
    assert set(fs["set_honest"]) == set(fs["set_all"]) - banned
    assert "Case_No" not in fs["set_all"] and "label" not in fs["set_all"]


def test_d05_label_rule_on_clean_data():
    acc, _ = tb.item_sum_rule(_clean("D05_toddler_screening"), [f"A{i}" for i in range(1, 11)], 4)
    assert acc == 1.0


def test_d05_other_csv_never_read_by_stage2_code():
    files = glob.glob(os.path.join(PROJECT_DIR, "src", "data", "*.py")) + \
        glob.glob(os.path.join(PROJECT_DIR, "scripts", "02*.py")) + \
        glob.glob(os.path.join(PROJECT_DIR, "configs", "data", "*.yaml"))
    for p in files:
        with open(p) as fh:
            txt = fh.read()
        assert "Combined" not in txt and "Screening_Data" not in txt, p
    assert _cols("D05_toddler_screening")["source_file"] == "Toddler Autism dataset July 2018.csv"


def test_d06_empty_column_absent():
    c = _clean("D06_early_autism")
    assert "Unnamed: 10" not in c.columns
    assert _cols("D06_early_autism")["columns"]["Unnamed: 10"]["role"] == "excluded"


def test_identical_feature_rows_share_group():
    for ds in ["D04_autism_prediction", "D05_toddler_screening", "D06_early_autism"]:
        c = _clean(ds)
        spec = _cols(ds)
        feats = [k for k, v in spec["columns"].items() if v["role"].startswith("feature")]
        key = c[feats].astype(object).where(c[feats].notna(), "<NA>").astype(str).agg("|".join, axis=1)
        assert c.groupby(key).group_id.nunique().max() == 1, ds
        assert key.groupby(c.group_id).nunique().max() == 1, ds


def test_every_column_has_a_role():
    for ds in ["D04_autism_prediction", "D05_toddler_screening", "D06_early_autism"]:
        spec = _cols(ds)["columns"]
        roles = {"id", "feature_numeric", "feature_categorical", "label", "excluded"}
        assert all(v["role"] in roles for v in spec.values())
        assert set(_clean(ds).columns) <= set(spec)


def _read_tsv(path):
    with open(path) as fh:
        next(fh)
        return [tuple(line.rstrip("\n").split("\t")) for line in fh]


def test_raw_data_snapshot_unchanged():
    snap = importlib.import_module("00_snapshot_raw")
    start = _read_tsv(os.path.join(PROJECT_DIR, "data_registry", "raw_snapshot_start.tsv"))
    assert start == [tuple(map(str, r)) for r in snap.snapshot()]
