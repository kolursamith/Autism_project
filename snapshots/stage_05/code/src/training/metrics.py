"""Binary classification metrics (positive class = ASD = 1) and the validation-only Youden threshold."""
import math

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

METRIC_KEYS = ["accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc",
               "brier"]


def _safe(fn, y, p):
    if len(np.unique(y)) < 2:  # one class only -> undefined
        return float("nan")
    return float(fn(y, p))


def binary_metrics(y_true, y_prob, threshold=0.5, y_pred=None):
    """All metrics for one set of predictions. y_pred overrides the threshold (pooled out-of-fold use)."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    yp = (p >= threshold).astype(int) if y_pred is None else np.asarray(y_pred).astype(int)
    tp = int(((yp == 1) & (y == 1)).sum())
    tn = int(((yp == 0) & (y == 0)).sum())
    fp = int(((yp == 1) & (y == 0)).sum())
    fn = int(((yp == 0) & (y == 1)).sum())
    div = lambda a, b: a / b if b else 0.0  # noqa: E731  (sklearn zero_division=0 convention)
    rec, spec, prec = div(tp, tp + fn), div(tn, tn + fp), div(tp, tp + fp)
    return {
        "n": int(len(y)), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
        "threshold": None if y_pred is not None else float(threshold),
        "accuracy": div(tp + tn, len(y)),
        "balanced_accuracy": (rec + spec) / 2,
        "precision": prec, "recall": rec, "sensitivity": rec, "specificity": spec,
        "f1": div(2 * prec * rec, prec + rec),
        "roc_auc": _safe(roc_auc_score, y, p),
        "pr_auc": _safe(average_precision_score, y, p),
        "brier": float(np.mean((p - y) ** 2)),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def youden_threshold(y_val, p_val):
    """Threshold maximising Youden's J = TPR - FPR on VALIDATION data only (ties -> the highest threshold, i.e. the
    first one sklearn returns). Returns 0.5 if the validation set has a single class."""
    y = np.asarray(y_val).astype(int)
    if y.min() == y.max():
        return 0.5
    fpr, tpr, thr = roc_curve(y, np.asarray(p_val, dtype=float))
    ok = np.isfinite(thr)
    j = np.where(ok, tpr - fpr, -np.inf)
    t = float(thr[int(np.argmax(j))])
    return min(max(t, 0.0), 1.0)


def child_level(sample_groups, y_true, y_prob):
    """Mean probability per group (child); label must be constant within a child. Returns groups, y, p arrays."""
    import pandas as pd
    d = pd.DataFrame({"g": sample_groups, "y": np.asarray(y_true).astype(int), "p": np.asarray(y_prob, float)})
    agg = d.groupby("g", sort=True).agg(y=("y", "first"), y_n=("y", "nunique"), p=("p", "mean"))
    if (agg["y_n"] != 1).any():
        raise ValueError("label not constant within a child")
    return agg.index.to_numpy(), agg["y"].to_numpy(), agg["p"].to_numpy()


RANGE_TOL = 1e-9  # absorbs floating-point results such as average precision = 1.0000000000000002
UNDEFINED_IF_SINGLE_CLASS = {"roc_auc", "pr_auc"}


def is_valid_metric_block(m, flags=None, tol=RANGE_TOL):
    """Validation only (no metric is recomputed). Returns a list of errors for one metrics block:
    - every rate / AUC / PR-AUC / Brier must be a number with 0 - tol <= value <= 1 + tol (inclusive);
    - NaN is accepted ONLY for ROC-AUC / PR-AUC when the block genuinely has a single class (n_pos == 0 or n_neg == 0).
      Such a case is never passed silently: it is appended to `flags` (a list supplied by the caller) or, when no
      list is supplied, returned as an error;
    - a missing value, NaN anywhere else, or a non-numeric value is an error;
    - the confusion matrix must sum to n."""
    errs = []
    single = m.get("n_pos") == 0 or m.get("n_neg") == 0
    for k in METRIC_KEYS:
        v = m.get(k)
        if v is None:
            errs.append(f"{k} missing")
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            errs.append(f"{k}={v!r} is not a number")
            continue
        if math.isnan(v):
            if k in UNDEFINED_IF_SINGLE_CLASS and single:
                msg = f"{k} undefined (NaN): single class in this block (n_pos={m.get('n_pos')}, n_neg={m.get('n_neg')})"
                (flags.append(msg) if flags is not None else errs.append(msg + " [not accepted without flags=]"))
            else:
                errs.append(f"{k} is NaN although it is defined (n_pos={m.get('n_pos')}, n_neg={m.get('n_neg')})")
            continue
        if not (-tol <= v <= 1.0 + tol):
            errs.append(f"{k}={v!r} outside [0, 1]")
    cm = np.asarray(m.get("confusion_matrix", [[0, 0], [0, 0]]))
    if cm.sum() != m.get("n"):
        errs.append(f"confusion matrix sums to {cm.sum()} != n {m.get('n')}")
    return errs
