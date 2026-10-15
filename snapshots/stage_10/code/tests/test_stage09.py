"""Stage 9: every number in results/final/ tables matches its source file; every final figure exists (PNG 300 dpi +
SVG, non-empty); results/ (except results/final/) unchanged since the Stage 9 pre-flight; final report present."""
import math
import os
import re
import sys

import pandas as pd
import pytest
from PIL import Image

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
from src.analysis.final import FIG_FINAL, FINAL_DIR, resolve, results_manifest_lines  # noqa: E402

FIGS = ["F01_pipeline", "F02_d01_auc_vs_quality", "F03_d01_forest_vs_quality", "F04_d01_matched_subset",
        "F05_d02_auc_vs_image_stats", "F06_d02_roc_top3", "F07_cd_diagrams", "F08_confusion_matrices",
        "F09_d01_m07_attention", "F10_tabular_auc", "F11_pretrained_vs_scratch", "F12_reliability_best"]


def _same(a, b):
    return (math.isnan(a) and math.isnan(b)) or a == b or math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15)


def test_master_table_every_number_matches_its_source():
    M = pd.read_csv(os.path.join(FINAL_DIR, "master_table.csv"))
    src_cols = [c for c in M.columns if c.endswith("__source")]
    assert len(M) == 50 and len(src_cols) == 19
    n = 0
    for c in src_cols:
        col = c[: -len("__source")]
        for v, ref in zip(M[col], M[c]):
            assert _same(float(v), resolve(ref)), (col, ref, v)
            n += 1
    assert n == 50 * 19


@pytest.mark.parametrize("name", ["shortcut_table.csv", "stats_table.csv", "fold_auc_matrix.csv"])
def test_long_tables_match_sources(name):
    T = pd.read_csv(os.path.join(FINAL_DIR, name))
    vcol = "roc_auc" if name == "fold_auc_matrix.csv" else "value"
    for v, ref in zip(T[vcol], T.source):
        assert _same(float(v), resolve(ref)), (ref, v)


def test_resolver_rejects_ambiguous_filters():
    with pytest.raises(KeyError):
        resolve("results/stage8/stats_bootstrap_ci.csv::dataset=D01_eye_tracking::pooled_auc")


@pytest.mark.parametrize("fig", FIGS)
def test_figure_files_exist_nonempty_300dpi(fig):
    png, svg = os.path.join(FIG_FINAL, f"{fig}.png"), os.path.join(FIG_FINAL, f"{fig}.svg")
    assert os.path.getsize(png) > 10_000 and os.path.getsize(svg) > 5_000
    with Image.open(png) as im:
        assert round(im.info["dpi"][0]) == 300


def test_results_unchanged_since_stage9_preflight():
    rec = open(os.path.join(FINAL_DIR, "preflight_results_manifest.sha256")).read().splitlines()
    cur = results_manifest_lines()
    assert len(rec) > 2500
    assert rec == cur, sorted(set(rec) ^ set(cur))[:10]


def test_final_report_present_with_abstract_length():
    md = open(os.path.join(PROJECT_DIR, "reports", "FINAL_RESULTS_REPORT.md")).read()
    abstract = md.split("## 1. Abstract", 1)[1].split("## 2.", 1)[0].strip()
    assert 150 <= len(abstract.split()) <= 200
    for k in range(1, 12):
        assert re.search(rf"^## {k}\. ", md, re.M), k
    html = os.path.join(PROJECT_DIR, "reports", "FINAL_RESULTS_REPORT.html")
    assert os.path.getsize(html) > 100_000 and open(html).read().count("data:image/png;base64") == 12
