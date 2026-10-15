"""Stage 1 checks: skeleton, audit outputs, raw data untouched, CUDA available."""
import json
import os
import sys

import pytest
import torch

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_DIR, "scripts"))
import importlib  # noqa: E402

snap = importlib.import_module("00_snapshot_raw")

REG = os.path.join(PROJECT_DIR, "data_registry")
IDS = ["D01_eye_tracking", "D02_facial_images_A", "D03_facial_images_B",
       "D04_autism_prediction", "D05_toddler_screening", "D06_early_autism"]
DIRS = ["configs", "src/data", "src/models", "src/training", "src/evaluation", "src/utils", "scripts",
        "tests", "data_registry", "splits", "cache", "results", "reports", "logs"]


@pytest.mark.parametrize("d", DIRS)
def test_skeleton_dir_exists(d):
    assert os.path.isdir(os.path.join(PROJECT_DIR, d))


def test_results_readme_and_gitignore():
    assert os.path.isfile(os.path.join(PROJECT_DIR, "results", "README.md"))
    gi = open(os.path.join(PROJECT_DIR, ".gitignore")).read()
    for pat in ["cache/", "results/**/best_model.pt", "*.pt", "__pycache__", ".venv"]:
        assert pat in gi


@pytest.mark.parametrize("did", IDS)
def test_audit_json_loads_and_nonzero(did):
    with open(os.path.join(REG, f"{did}_audit.json")) as fh:
        a = json.load(fh)
    assert a["dataset_id"] == did
    assert a["n_samples"] > 0
    assert os.path.isfile(os.path.join(REG, f"{did}_audit.md"))
    if "images" in a:
        assert sum(a["images"]["class_totals"].values()) > 0
        assert os.path.isfile(os.path.join(PROJECT_DIR, a["images"]["contact_sheet"]))
    if "tabular" in a:
        assert all(f["shape"][0] > 0 for f in a["tabular"]["files"])
    if "eye_tracking" in a:
        assert a["eye_tracking"]["total_gaze_rows"] > 0
        assert a["eye_tracking"]["participants_with_metadata_and_gaze"] > 0


def test_registry_exists():
    txt = open(os.path.join(REG, "DATASET_REGISTRY.md")).read()
    for did in IDS:
        assert did in txt


def _read(path):
    with open(path) as fh:
        next(fh)
        return [tuple(line.rstrip("\n").split("\t")) for line in fh]


def test_raw_data_unmodified():
    start = _read(os.path.join(REG, "raw_snapshot_start.tsv"))
    end = _read(os.path.join(REG, "raw_snapshot_end.tsv"))
    live = [tuple(map(str, r)) for r in snap.snapshot()]
    assert len(start) > 0
    assert start == end, "START and END snapshots differ"
    assert start == live, "raw data changed since the START snapshot"


def test_cuda_available():
    assert torch.cuda.is_available()
