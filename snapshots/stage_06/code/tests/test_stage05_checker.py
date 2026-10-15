"""Stage 5B checker fix: inclusive [0, 1] range check with 1e-9 tolerance; NaN only where genuinely undefined, and then
always flagged explicitly (src/training/metrics.is_valid_metric_block). Validation only: no metric is recomputed."""
import os
import sys

import pytest

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
from src.training.metrics import binary_metrics, is_valid_metric_block  # noqa: E402

BASE = binary_metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], 0.5)


def _block(**kw):
    b = dict(BASE)
    b.update(kw)
    return b


def test_reference_block_is_valid():
    assert is_valid_metric_block(BASE) == []


def test_perfect_pr_auc_passes():
    perfect = binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.5)
    assert perfect["pr_auc"] == pytest.approx(1.0) and perfect["roc_auc"] == 1.0
    assert is_valid_metric_block(perfect) == []
    assert is_valid_metric_block(_block(pr_auc=1.0)) == []
    assert is_valid_metric_block(_block(pr_auc=1.0000000000000002)) == []  # the float seen in Stage 7 (within 1e-9)


def test_auc_zero_passes():
    worst = binary_metrics([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1], 0.5)
    assert worst["roc_auc"] == 0.0
    assert is_valid_metric_block(worst) == []
    assert is_valid_metric_block(_block(roc_auc=0.0, brier=0.0)) == []


def test_value_above_one_fails():
    errs = is_valid_metric_block(_block(pr_auc=1.0000001))
    assert len(errs) == 1 and "pr_auc" in errs[0] and "outside [0, 1]" in errs[0]


def test_negative_value_fails():
    errs = is_valid_metric_block(_block(specificity=-0.01))
    assert len(errs) == 1 and "specificity" in errs[0]
    assert is_valid_metric_block(_block(brier=-1e-10)) == []  # inside the tolerance


def test_nan_is_flagged_never_silent():
    single = binary_metrics([1, 1, 1], [0.2, 0.6, 0.9], 0.5)   # single class: ROC-AUC / PR-AUC undefined
    flags = []
    assert is_valid_metric_block(single, flags=flags) == []
    assert len(flags) == 2 and all("undefined (NaN): single class" in f for f in flags)
    errs = is_valid_metric_block(single)                          # no flag list -> reported as errors, not passed
    assert len(errs) == 2 and all("not accepted without flags=" in e for e in errs)


def test_nan_where_defined_fails():
    flags = []
    errs = is_valid_metric_block(_block(roc_auc=float("nan")), flags=flags)   # both classes present -> defined
    assert len(errs) == 1 and "NaN although it is defined" in errs[0] and flags == []
    errs = is_valid_metric_block(_block(accuracy=float("nan")), flags=flags)  # never undefined
    assert len(errs) == 1 and "accuracy" in errs[0]


def test_missing_or_non_numeric_fails():
    b = dict(BASE)
    del b["f1"]
    assert any("f1 missing" in e for e in is_valid_metric_block(b))
    assert any("not a number" in e for e in is_valid_metric_block(_block(recall="0.5")))
    assert any("confusion matrix" in e for e in is_valid_metric_block(_block(n=5)))
