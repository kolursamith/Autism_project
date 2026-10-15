# STAGE 5 REPORT - D01 eye-tracking: 10 hybrid models x 10 folds

Screening-support research, not diagnosis. 57 children (27 ASD / 30 TD), 1,241 clips of 8 x 1-s gaze windows; child-grouped 10-fold CV (5-6 test children and 81-170 test clips per fold); protocol fixed in Stage 4 (`configs/training.yaml`), no tuning. Test folds were predicted once per run and never used for any choice. No git: recorded by `snapshots/stage_05/`.

## 1. Summary

- Queue: started 2026-10-14T10:01:09+05:30, ended 2026-10-14T10:45:38+05:30 (exit 0); 100/100 runs DONE; validate_model_results PASS for 10/10 models (re-checked after the checker fix, no flags).
- Split hashes re-verified; all 10 models predicted exactly the same test ids in every fold.
- Highest mean clip-level test ROC-AUC: M08 (EfficientNet + Transformer + Attn) 0.686 ± 0.128; range of model means 0.595-0.686.
- Child level (pooled, each of the 57 children once): highest ROC-AUC 0.832 (M07). Quality-only indicator (1 - mean valid_ratio (lower valid_ratio -> ASD)): child ROC-AUC 0.842 (section 7).
- Checker fix (validation code only): all 630 stored result files byte-identical before/after (section 8).

## 2. Run statistics (from run_info.json)

Total GPU-run time (sum of fold run seconds): 43.8 min (0.73 h); queue wall time 10:01:09 -> 10:45:38. All 100 runs ended by early stopping: True. Peak GPU memory over all runs: 3.18 GB.

| model | input | GPU s (10 folds) | s / fold | epochs_min | epochs_median | epochs_max | best_min | best_median | best_max | peak GB |
|---|---|---|---|---|---|---|---|---|---|---|
| M01 | cached features | 73.23 | 7.32 | 9 | 14.0 | 23 | 1 | 6.0 | 15 | 0.12 |
| M02 | clips (end to end) | 552.44 | 55.24 | 9 | 14.0 | 22 | 1 | 6.0 | 14 | 3.17 |
| M03 | cached features | 106.61 | 10.66 | 9 | 17.0 | 23 | 1 | 9.0 | 15 | 0.23 |
| M04 | cached features | 104.88 | 10.49 | 9 | 15.5 | 21 | 1 | 7.5 | 13 | 0.26 |
| M05 | clips (end to end) | 646.05 | 64.61 | 11 | 13.0 | 35 | 3 | 5.0 | 27 | 3.18 |
| M06 | cached features | 113.01 | 11.3 | 11 | 15.0 | 27 | 3 | 7.0 | 19 | 0.09 |
| M07 | cached features | 148.11 | 14.81 | 9 | 14.0 | 21 | 1 | 6.0 | 13 | 0.71 |
| M08 | cached features | 82.59 | 8.26 | 9 | 10.5 | 21 | 1 | 2.5 | 13 | 0.09 |
| M09 | clips (end to end) | 695.34 | 69.53 | 12 | 16.5 | 32 | 4 | 8.5 | 24 | 3.17 |
| M10 | cached features | 104.52 | 10.45 | 9 | 13.0 | 18 | 1 | 5.0 | 10 | 0.26 |

Best-epoch distribution over the 100 runs (epoch: count): 1: 16, 2: 2, 3: 11, 4: 6, 5: 12, 6: 6, 7: 8, 8: 2, 9: 9, 10: 6, 11: 7, 13: 4, 14: 3, 15: 4, 19: 2, 24: 1, 27: 1.

## 3. Main table - clip level, test folds, threshold 0.5 (mean ± std over 10 folds, sorted by ROC-AUC)

| model | architecture | ROC-AUC | PR-AUC | accuracy | bal. acc. | sensitivity | specificity | F1 | Brier |
|---|---|---|---|---|---|---|---|---|---|
| M08 | EfficientNet + Transformer + Attn | 0.686 ± 0.128 | 0.572 ± 0.154 | 0.661 ± 0.113 | 0.605 ± 0.092 | 0.503 ± 0.249 | 0.707 ± 0.177 | 0.485 ± 0.164 | 0.218 ± 0.056 |
| M04 | ConvNeXt + BiLSTM + Attn | 0.683 ± 0.081 | 0.589 ± 0.129 | 0.639 ± 0.100 | 0.614 ± 0.057 | 0.552 ± 0.193 | 0.676 ± 0.219 | 0.501 ± 0.101 | 0.227 ± 0.042 |
| M07 | ViT + LSTM + Attn | 0.678 ± 0.099 | 0.567 ± 0.158 | 0.657 ± 0.070 | 0.630 ± 0.092 | 0.522 ± 0.183 | 0.738 ± 0.096 | 0.493 ± 0.150 | 0.221 ± 0.031 |
| M05 | CNN + Transformer + MLP | 0.657 ± 0.115 | 0.523 ± 0.152 | 0.595 ± 0.130 | 0.543 ± 0.059 | 0.481 ± 0.385 | 0.605 ± 0.366 | 0.380 ± 0.221 | 0.276 ± 0.079 |
| M01 | EfficientNetB3 + BiLSTM + Attn | 0.645 ± 0.108 | 0.538 ± 0.171 | 0.625 ± 0.129 | 0.588 ± 0.083 | 0.484 ± 0.130 | 0.692 ± 0.206 | 0.470 ± 0.118 | 0.247 ± 0.063 |
| M06 | DenseNet121 + LSTM + Attn | 0.639 ± 0.108 | 0.522 ± 0.175 | 0.632 ± 0.094 | 0.596 ± 0.083 | 0.481 ± 0.157 | 0.710 ± 0.143 | 0.465 ± 0.122 | 0.234 ± 0.050 |
| M10 | ResNet50 + BiLSTM + Transformer | 0.638 ± 0.134 | 0.521 ± 0.162 | 0.598 ± 0.139 | 0.573 ± 0.096 | 0.557 ± 0.319 | 0.589 ± 0.343 | 0.455 ± 0.144 | 0.262 ± 0.060 |
| M03 | ResNet50 + LSTM + Attn | 0.635 ± 0.107 | 0.508 ± 0.128 | 0.618 ± 0.135 | 0.579 ± 0.090 | 0.416 ± 0.131 | 0.741 ± 0.203 | 0.424 ± 0.128 | 0.245 ± 0.061 |
| M09 | CNN + GRU + Attn | 0.605 ± 0.087 | 0.501 ± 0.127 | 0.472 ± 0.174 | 0.518 ± 0.026 | 0.726 ± 0.385 | 0.310 ± 0.423 | 0.438 ± 0.170 | 0.411 ± 0.173 |
| M02 | CNN + BiLSTM + Attn | 0.595 ± 0.134 | 0.488 ± 0.166 | 0.416 ± 0.154 | 0.513 ± 0.059 | 0.773 ± 0.288 | 0.253 ± 0.338 | 0.449 ± 0.141 | 0.333 ± 0.101 |

### At the validation-chosen threshold (Youden J on each fold's validation clips)

| model | val threshold (mean ± std) | bal. acc. | sensitivity | specificity | F1 |
|---|---|---|---|---|---|
| M08 | 0.394 ± 0.164 | 0.613 ± 0.059 | 0.621 ± 0.201 | 0.605 ± 0.197 | 0.507 ± 0.104 |
| M04 | 0.444 ± 0.068 | 0.621 ± 0.069 | 0.646 ± 0.123 | 0.596 ± 0.212 | 0.535 ± 0.090 |
| M07 | 0.432 ± 0.143 | 0.612 ± 0.054 | 0.632 ± 0.179 | 0.592 ± 0.175 | 0.504 ± 0.111 |
| M05 | 0.427 ± 0.286 | 0.613 ± 0.083 | 0.596 ± 0.182 | 0.631 ± 0.218 | 0.511 ± 0.108 |
| M01 | 0.456 ± 0.171 | 0.572 ± 0.089 | 0.590 ± 0.215 | 0.553 ± 0.220 | 0.462 ± 0.112 |
| M06 | 0.482 ± 0.155 | 0.583 ± 0.095 | 0.520 ± 0.158 | 0.646 ± 0.193 | 0.464 ± 0.132 |
| M10 | 0.447 ± 0.233 | 0.578 ± 0.095 | 0.587 ± 0.219 | 0.568 ± 0.224 | 0.474 ± 0.108 |
| M03 | 0.354 ± 0.176 | 0.599 ± 0.086 | 0.653 ± 0.181 | 0.546 ± 0.179 | 0.499 ± 0.112 |
| M09 | 0.674 ± 0.336 | 0.559 ± 0.057 | 0.604 ± 0.163 | 0.514 ± 0.165 | 0.472 ± 0.114 |
| M02 | 0.623 ± 0.181 | 0.557 ± 0.099 | 0.534 ± 0.197 | 0.579 ± 0.203 | 0.456 ± 0.141 |

## 4. Child level

Child probability = mean clip probability of that child. Pooled = every child predicted once, by the model of the fold in which it was a test child (out-of-fold), then one ROC-AUC over all children. Per-fold child AUCs rest on 5-6 children each and are very coarse.

| model | pooled child ROC-AUC (all test children) | n children | child accuracy @0.5 | child bal. acc. @0.5 | child accuracy @val thr | per-fold child AUC (mean ± std) | pooled clip ROC-AUC |
|---|---|---|---|---|---|---|---|
| M08 | 0.786 | 57 | 0.737 | 0.731 | 0.737 | 0.828 ± 0.210 | 0.689 |
| M04 | 0.762 | 57 | 0.684 | 0.681 | 0.667 | 0.856 ± 0.137 | 0.657 |
| M07 | 0.832 | 57 | 0.772 | 0.765 | 0.772 | 0.800 ± 0.219 | 0.675 |
| M05 | 0.605 | 57 | 0.596 | 0.596 | 0.667 | 0.767 ± 0.187 | 0.61 |
| M01 | 0.714 | 57 | 0.719 | 0.713 | 0.632 | 0.761 ± 0.219 | 0.634 |
| M06 | 0.784 | 57 | 0.684 | 0.681 | 0.667 | 0.783 ± 0.207 | 0.652 |
| M10 | 0.673 | 57 | 0.632 | 0.63 | 0.632 | 0.761 ± 0.193 | 0.628 |
| M03 | 0.735 | 57 | 0.684 | 0.676 | 0.614 | 0.744 ± 0.195 | 0.625 |
| M09 | 0.574 | 57 | 0.491 | 0.506 | 0.702 | 0.706 ± 0.203 | 0.541 |
| M02 | 0.578 | 57 | 0.456 | 0.474 | 0.632 | 0.678 ± 0.283 | 0.57 |

## 5. Per-fold clip-level test ROC-AUC (models x folds)

| model | f1 | f2 | f3 | f4 | f5 | f6 | f7 | f8 | f9 | f10 | min (fold) | max (fold) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M08 | 0.732 | 0.604 | 0.784 | 0.907 | 0.794 | 0.651 | 0.68 | 0.605 | 0.669 | 0.438 | 0.438 (f10) | 0.907 (f4) |
| M04 | 0.598 | 0.664 | 0.683 | 0.795 | 0.813 | 0.631 | 0.711 | 0.679 | 0.704 | 0.548 | 0.548 (f10) | 0.813 (f5) |
| M07 | 0.607 | 0.67 | 0.751 | 0.811 | 0.814 | 0.539 | 0.699 | 0.634 | 0.712 | 0.541 | 0.539 (f6) | 0.814 (f5) |
| M05 | 0.622 | 0.678 | 0.757 | 0.703 | 0.814 | 0.502 | 0.73 | 0.532 | 0.743 | 0.488 | 0.488 (f10) | 0.814 (f5) |
| M01 | 0.668 | 0.657 | 0.687 | 0.802 | 0.764 | 0.468 | 0.68 | 0.587 | 0.659 | 0.478 | 0.468 (f6) | 0.802 (f4) |
| M06 | 0.557 | 0.621 | 0.791 | 0.785 | 0.766 | 0.508 | 0.661 | 0.573 | 0.609 | 0.522 | 0.508 (f6) | 0.791 (f3) |
| M10 | 0.603 | 0.65 | 0.649 | 0.879 | 0.837 | 0.447 | 0.663 | 0.582 | 0.556 | 0.516 | 0.447 (f6) | 0.879 (f4) |
| M03 | 0.658 | 0.647 | 0.674 | 0.769 | 0.799 | 0.44 | 0.666 | 0.575 | 0.605 | 0.519 | 0.440 (f6) | 0.799 (f5) |
| M09 | 0.634 | 0.723 | 0.682 | 0.639 | 0.673 | 0.458 | 0.643 | 0.524 | 0.575 | 0.499 | 0.458 (f6) | 0.723 (f2) |
| M02 | 0.666 | 0.737 | 0.767 | 0.599 | 0.689 | 0.415 | 0.637 | 0.507 | 0.574 | 0.358 | 0.358 (f10) | 0.767 (f3) |

## 6. Flags

- M01: 2 fold(s) with clip ROC-AUC < 0.5: f6 0.468, f10 0.478
- M02: 2 fold(s) with clip ROC-AUC < 0.5: f6 0.415, f10 0.358
- M03: 1 fold(s) with clip ROC-AUC < 0.5: f6 0.440
- M05: 1 fold(s) with clip ROC-AUC < 0.5: f10 0.488
- M08: 1 fold(s) with clip ROC-AUC < 0.5: f10 0.438
- M09: 2 fold(s) with clip ROC-AUC < 0.5: f6 0.458, f10 0.499
- M10: 1 fold(s) with clip ROC-AUC < 0.5: f6 0.447
- runs stopping at epoch <= 3: 0 (minimum possible with patience 8 is 9 epochs)
- runs hitting max_epochs 40: 0
- runs whose best validation epoch was <= 3 (early stop at epoch best+8): 29 of 100 (per model: {'M01': 4, 'M02': 2, 'M03': 3, 'M04': 3, 'M05': 1, 'M06': 3, 'M07': 5, 'M08': 5, 'M09': 0, 'M10': 3})
- folds predicting a single class at threshold 0.5: 9 (M02 f4, M02 f5, M02 f7, M05 f4, M05 f7, M05 f9, M09 f2, M09 f4, M09 f7)
- folds predicting a single class at the validation threshold: 0

## 7. Quick shortcut indicator (descriptive only; full analysis in Stage 8)

From `cache/D01_eye_tracking/quality_per_child.csv` (57 children, DD-12 quality features, never a model input): the child-level ROC-AUC of mean valid_ratio alone is 0.158; the direction above 0.5 is **1 - mean valid_ratio (lower valid_ratio -> ASD)**, ROC-AUC **0.842**. This is a single descriptive number over all 57 children (no fitting; the direction was chosen after looking, so it is optimistic). For comparison, the models' pooled child-level ROC-AUCs range 0.574-0.832 (section 4). Stage 8 must test whether the models' performance exceeds what tracking quality alone explains.

## 8. Checker fix (Stage 5B, validation code only)

`src/training/metrics.is_valid_metric_block` and `src/training/checks.validate_model_results`: the range check is now inclusive with a tolerance (0 - 1e-9 <= value <= 1 + 1e-9) for rates, ROC-AUC, PR-AUC and Brier (sklearn's average precision can return 1.0000000000000002, seen in Stage 7); NaN is accepted only for ROC-AUC / PR-AUC when a block genuinely has a single class, and is then reported as an explicit flag (`validate_model_results(..., return_flags=True)`), never passed silently (before: a NaN AUC passed silently). No metric computation changed. New tests `tests/test_stage05_checker.py`.

<details><summary>diff (logs/stage05_checker_fix.diff)</summary>

```diff
--- /tmp/claude-1001/-home-score-Documents-AI-project/4f92d53a-edd7-465a-9ad4-b89e6fdbac58/scratchpad/metrics.py.orig	2026-10-14 10:47:34.162156484 +0530
+++ src/training/metrics.py	2026-10-14 10:47:55.407254965 +0530
@@ -62,17 +62,37 @@
     return agg.index.to_numpy(), agg["y"].to_numpy(), agg["p"].to_numpy()
 
 
-def is_valid_metric_block(m):
-    """Every rate in [0, 1] (NaN allowed only for AUCs), confusion matrix sums to n."""
+RANGE_TOL = 1e-9  # absorbs floating-point results such as average precision = 1.0000000000000002
+UNDEFINED_IF_SINGLE_CLASS = {"roc_auc", "pr_auc"}
+
+
+def is_valid_metric_block(m, flags=None, tol=RANGE_TOL):
+    """Validation only (no metric is recomputed). Returns a list of errors for one metrics block:
+    - every rate / AUC / PR-AUC / Brier must be a number with 0 - tol <= value <= 1 + tol (inclusive);
+    - NaN is accepted ONLY for ROC-AUC / PR-AUC when the block genuinely has a single class (n_pos == 0 or n_neg == 0).
+      Such a case is never passed silently: it is appended to `flags` (a list supplied by the caller) or, when no
+      list is supplied, returned as an error;
+    - a missing value, NaN anywhere else, or a non-numeric value is an error;
+    - the confusion matrix must sum to n."""
     errs = []
+    single = m.get("n_pos") == 0 or m.get("n_neg") == 0
     for k in METRIC_KEYS:
         v = m.get(k)
-        if v is None or (isinstance(v, float) and math.isnan(v)):
-            if k not in {"roc_auc", "pr_auc"}:
-                errs.append(f"{k} missing/NaN")
+        if v is None:
+            errs.append(f"{k} missing")
+            continue
+        if isinstance(v, bool) or not isinstance(v, (int, float)):
+            errs.append(f"{k}={v!r} is not a number")
+            continue
+        if math.isnan(v):
+            if k in UNDEFINED_IF_SINGLE_CLASS and single:
+                msg = f"{k} undefined (NaN): single class in this block (n_pos={m.get('n_pos')}, n_neg={m.get('n_neg')})"
+                (flags.append(msg) if flags is not None else errs.append(msg + " [not accepted without flags=]"))
+            else:
+                errs.append(f"{k} is NaN although it is defined (n_pos={m.get('n_pos')}, n_neg={m.get('n_neg')})")
             continue
-        if not 0.0 <= v <= 1.0:
-            errs.append(f"{k}={v} outside [0, 1]")
+        if not (-tol <= v <= 1.0 + tol):
+            errs.append(f"{k}={v!r} outside [0, 1]")
     cm = np.asarray(m.get("confusion_matrix", [[0, 0], [0, 0]]))
     if cm.sum() != m.get("n"):
         errs.append(f"confusion matrix sums to {cm.sum()} != n {m.get('n')}")
--- /tmp/claude-1001/-home-score-Documents-AI-project/4f92d53a-edd7-465a-9ad4-b89e6fdbac58/scratchpad/checks.py.orig	2026-10-14 10:47:34.164046681 +0530
+++ src/training/checks.py	2026-10-14 10:47:56.457668872 +0530
@@ -14,17 +14,18 @@
 from src.models.factory import load_model_cfg
 from src.training import registry
 from src.training.data import ALIASES
-from src.training.metrics import is_valid_metric_block
+from src.training.metrics import RANGE_TOL, is_valid_metric_block
 
 REQUIRED = ["config.yaml", "metrics.json", "predictions.csv", "val_predictions.csv", "history.csv", "best_model.pt",
             "run_info.json"]
 
 
-def validate_model_results(ds, model, out_root, folds=range(1, 11)):
-    """Returns ("PASS" | "FAIL", [reasons])."""
+def validate_model_results(ds, model, out_root, folds=range(1, 11), return_flags=False):
+    """Returns ("PASS" | "FAIL", [reasons]); with return_flags=True also [flags] (explicitly reported, non-failing
+    conditions such as an AUC that is undefined because a block has a single class)."""
     ds = ALIASES[ds]
     folds = [int(k) for k in folds]
-    reasons = []
+    reasons, flags = [], []
     reg = registry.read_registry(out_root)
     has_attn = load_model_cfg(model)["pooling"] == "attention"
     folds_csv = pd.read_csv(os.path.join(SPLITS_DIR, ds, "folds.csv")).set_index("sample_id")
@@ -48,11 +49,13 @@
             reasons.append(f"{tag}: test-id leakage counter not 0")
         with open(os.path.join(d, "metrics.json")) as fh:
             m = json.load(fh)
-        if not 0.0 <= m["val_threshold_youden"] <= 1.0:
+        if not -RANGE_TOL <= m["val_threshold_youden"] <= 1.0 + RANGE_TOL:
             reasons.append(f"{tag}: Youden threshold {m['val_threshold_youden']} outside [0, 1]")
         for b in ["test", "test_child"]:
             for thr, block in m.get(b, {}).items():
-                reasons += [f"{tag}: {b}/{thr}: {e}" for e in is_valid_metric_block(block)]
+                fl = []
+                reasons += [f"{tag}: {b}/{thr}: {e}" for e in is_valid_metric_block(block, flags=fl)]
+                flags += [f"{tag}: {b}/{thr}: {f}" for f in fl]
         h = pd.read_csv(os.path.join(d, "history.csv"))
         if h.empty or not np.isfinite(h[["train_loss", "val_loss"]].to_numpy()).all():
             reasons.append(f"{tag}: history empty or non-finite losses")
@@ -70,7 +73,7 @@
         if (ref["label"].to_numpy() != p.y_true.to_numpy()).any() or \
                 (ref["group_id"].astype(str).to_numpy() != p.group_id.astype(str).to_numpy()).any():
             reasons.append(f"{tag}: y_true / group_id do not match folds.csv")
-        if not (np.isfinite(p.y_prob).all() and p.y_prob.between(0, 1).all()) or (p.fold != k).any():
+        if not (np.isfinite(p.y_prob).all() and p.y_prob.between(-RANGE_TOL, 1.0 + RANGE_TOL).all()) or (p.fold != k).any():
             reasons.append(f"{tag}: y_prob not finite in [0, 1] or wrong fold column")
         if has_attn:
             a = np.load(os.path.join(d, "attention.npy"))
@@ -85,4 +88,5 @@
                            f"{folds} exactly once")
         if sorted(folds) == list(range(1, 11)) and len(expected_all) != len(folds_csv):
             reasons.append("10 folds do not cover the whole dataset")
-    return ("PASS" if not reasons else "FAIL"), reasons
+    verdict = "PASS" if not reasons else "FAIL"
+    return (verdict, reasons, flags) if return_flags else (verdict, reasons)
```

</details>

Metrics tests (`logs/stage05_checker_tests.txt`):

```
tests/test_stage05_checker.py::test_reference_block_is_valid PASSED      [  4%]
tests/test_stage05_checker.py::test_perfect_pr_auc_passes PASSED         [  8%]
tests/test_stage05_checker.py::test_auc_zero_passes PASSED               [ 12%]
tests/test_stage05_checker.py::test_value_above_one_fails PASSED         [ 16%]
tests/test_stage05_checker.py::test_negative_value_fails PASSED          [ 20%]
tests/test_stage05_checker.py::test_nan_is_flagged_never_silent PASSED   [ 25%]
tests/test_stage05_checker.py::test_nan_where_defined_fails PASSED       [ 29%]
tests/test_stage05_checker.py::test_missing_or_non_numeric_fails PASSED  [ 33%]
tests/test_stage04_units.py::test_binary_metrics_match_sklearn[0.5-0] PASSED [ 37%]
tests/test_stage04_units.py::test_binary_metrics_match_sklearn[0.5-1] PASSED [ 41%]
tests/test_stage04_units.py::test_binary_metrics_match_sklearn[0.5-2] PASSED [ 45%]
tests/test_stage04_units.py::test_binary_metrics_match_sklearn[0.3-0] PASSED [ 50%]
tests/test_stage04_units.py::test_binary_metrics_match_sklearn[0.3-1] PASSED [ 54%]
tests/test_stage04_units.py::test_binary_metrics_match_sklearn[0.3-2] PASSED [ 58%]
tests/test_stage04_units.py::test_binary_metrics_match_sklearn[0.72-0] PASSED [ 62%]
tests/test_stage04_units.py::test_binary_metrics_match_sklearn[0.72-1] PASSED [ 66%]
tests/test_stage04_units.py::test_binary_metrics_match_sklearn[0.72-2] PASSED [ 70%]
tests/test_stage04_units.py::test_binary_metrics_known_values PASSED     [ 75%]
tests/test_stage04_units.py::test_youden_threshold_brute_force_on_validation_only[0] PASSED [ 79%]
tests/test_stage04_units.py::test_youden_threshold_brute_force_on_validation_only[1] PASSED [ 83%]
tests/test_stage04_units.py::test_youden_threshold_brute_force_on_validation_only[2] PASSED [ 87%]
tests/test_stage04_units.py::test_youden_threshold_brute_force_on_validation_only[3] PASSED [ 91%]
tests/test_stage04_units.py::test_child_level_mean_probability PASSED    [ 95%]
tests/test_stage04_units.py::test_summary_aggregation_on_fake_folds PASSED [100%]
====================== 24 passed, 24 deselected in 2.68s =======================
```

Re-validation and result-file hash comparison (`logs/stage05_revalidate.txt`; sha256 of all 630 metrics.json / predictions.csv / summary.json under results/ recorded BEFORE the change):

```
D01 M01: PASS  reasons=[]  flags=[]
D01 M02: PASS  reasons=[]  flags=[]
D01 M03: PASS  reasons=[]  flags=[]
D01 M04: PASS  reasons=[]  flags=[]
D01 M05: PASS  reasons=[]  flags=[]
D01 M06: PASS  reasons=[]  flags=[]
D01 M07: PASS  reasons=[]  flags=[]
D01 M08: PASS  reasons=[]  flags=[]
D01 M09: PASS  reasons=[]  flags=[]
D01 M10: PASS  reasons=[]  flags=[]
ALL PASS
16 passed in 8.08s
RESULT FILES IDENTICAL: all 630 sha256 match
```

## 9. Verification (`logs/stage05_verify.txt`)

```
4a registry: 100 D01 rows, status counts {'DONE': 100}, models ['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'], folds per model [10]
4a split hashes vs SPLITS_MANIFEST.json: OK
4a fold  1: 128 test ids; all 10 models predicted exactly these ids: True; fold-file sha256 recorded by the 10 runs identical: True
4a fold  2: 124 test ids; all 10 models predicted exactly these ids: True; fold-file sha256 recorded by the 10 runs identical: True
4a fold  3: 119 test ids; all 10 models predicted exactly these ids: True; fold-file sha256 recorded by the 10 runs identical: True
4a fold  4: 149 test ids; all 10 models predicted exactly these ids: True; fold-file sha256 recorded by the 10 runs identical: True
4a fold  5: 119 test ids; all 10 models predicted exactly these ids: True; fold-file sha256 recorded by the 10 runs identical: True
4a fold  6: 81 test ids; all 10 models predicted exactly these ids: True; fold-file sha256 recorded by the 10 runs identical: True
4a fold  7: 170 test ids; all 10 models predicted exactly these ids: True; fold-file sha256 recorded by the 10 runs identical: True
4a fold  8: 141 test ids; all 10 models predicted exactly these ids: True; fold-file sha256 recorded by the 10 runs identical: True
4a fold  9: 92 test ids; all 10 models predicted exactly these ids: True; fold-file sha256 recorded by the 10 runs identical: True
4a fold 10: 118 test ids; all 10 models predicted exactly these ids: True; fold-file sha256 recorded by the 10 runs identical: True
4a identical test ids per fold across models: True
4b summarize: summary.json for 10 models + results/D01_eye_tracking/leaderboard.csv written
D01 summary.json after re-summarize: identical to the pre-change manifest (10 files)
```

## 10. Tests (`.venv/bin/python -m pytest -v`, `logs/stage05_pytest.txt`)

Run after the provisional stage_05 snapshot. Before it, the full suite had exactly two failures (`logs/stage05_pytest_before_snapshot.txt`): the stage_04 snapshot test (expected: scripts/ gained the three Stage 5A files; src/training/metrics.py and checks.py changed with the checker fix; tests/ gained test_stage05_checker.py; reports/ gained STAGE_07_REPORT.md) and the CLAUDE.md length test (274 > 260 lines, caused by the Stage 5A/7 session-log entries; fixed by condensing the session log, not by relaxing the test).

Result: 331 passed in 127.46s (0:02:07)

<details><summary>full pytest output</summary>

```
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0 -- /home/score/Documents/AI_project/asd_hybrid_10fold/.venv/bin/python
rootdir: /home/score/Documents/AI_project/asd_hybrid_10fold
configfile: pytest.ini
testpaths: tests
plugins: anyio-4.15.1
collecting ... collected 331 items

tests/test_setup.py::test_project_dir_and_symlink PASSED                 [  0%]
tests/test_setup.py::test_paths_yaml_every_path_exists PASSED            [  0%]
tests/test_setup.py::test_datasets_dir_outside_project PASSED            [  0%]
tests/test_setup.py::test_no_absolute_paths_outside_paths_yaml PASSED    [  1%]
tests/test_setup.py::test_venv_imports PASSED                            [  1%]
tests/test_setup.py::test_cuda_available PASSED                          [  1%]
tests/test_setup.py::test_check_gpu_script_exits_zero PASSED             [  2%]
tests/test_setup.py::test_claude_md_sections PASSED                      [  2%]
tests/test_setup.py::test_raw_data_snapshot_matches PASSED               [  2%]
tests/test_setup.py::test_gitignore[.venv/] PASSED                       [  3%]
tests/test_setup.py::test_gitignore[cache/] PASSED                       [  3%]
tests/test_setup.py::test_gitignore[*.pt] PASSED                         [  3%]
tests/test_setup.py::test_gitignore[logs/*.log] PASSED                   [  3%]
tests/test_stage01.py::test_skeleton_dir_exists[configs] PASSED          [  4%]
tests/test_stage01.py::test_skeleton_dir_exists[src/data] PASSED         [  4%]
tests/test_stage01.py::test_skeleton_dir_exists[src/models] PASSED       [  4%]
tests/test_stage01.py::test_skeleton_dir_exists[src/training] PASSED     [  5%]
tests/test_stage01.py::test_skeleton_dir_exists[src/evaluation] PASSED   [  5%]
tests/test_stage01.py::test_skeleton_dir_exists[src/utils] PASSED        [  5%]
tests/test_stage01.py::test_skeleton_dir_exists[scripts] PASSED          [  6%]
tests/test_stage01.py::test_skeleton_dir_exists[tests] PASSED            [  6%]
tests/test_stage01.py::test_skeleton_dir_exists[data_registry] PASSED    [  6%]
tests/test_stage01.py::test_skeleton_dir_exists[splits] PASSED           [  6%]
tests/test_stage01.py::test_skeleton_dir_exists[cache] PASSED            [  7%]
tests/test_stage01.py::test_skeleton_dir_exists[results] PASSED          [  7%]
tests/test_stage01.py::test_skeleton_dir_exists[reports] PASSED          [  7%]
tests/test_stage01.py::test_skeleton_dir_exists[logs] PASSED             [  8%]
tests/test_stage01.py::test_results_readme_and_gitignore PASSED          [  8%]
tests/test_stage01.py::test_audit_json_loads_and_nonzero[D01_eye_tracking] PASSED [  8%]
tests/test_stage01.py::test_audit_json_loads_and_nonzero[D02_facial_images_A] PASSED [  9%]
tests/test_stage01.py::test_audit_json_loads_and_nonzero[D03_facial_images_B] PASSED [  9%]
tests/test_stage01.py::test_audit_json_loads_and_nonzero[D04_autism_prediction] PASSED [  9%]
tests/test_stage01.py::test_audit_json_loads_and_nonzero[D05_toddler_screening] PASSED [  9%]
tests/test_stage01.py::test_audit_json_loads_and_nonzero[D06_early_autism] PASSED [ 10%]
tests/test_stage01.py::test_registry_exists PASSED                       [ 10%]
tests/test_stage01.py::test_raw_data_unmodified PASSED                   [ 10%]
tests/test_stage01.py::test_cuda_available PASSED                        [ 11%]
tests/test_stage02_d01.py::test_no_unidentified_rows PASSED              [ 11%]
tests/test_stage02_d01.py::test_no_duplicate_participant_time_trial PASSED [ 11%]
tests/test_stage02_d01.py::test_no_cars_anywhere_in_cache_or_splits PASSED [ 12%]
tests/test_stage02_d01.py::test_npy_shape_matches_index PASSED           [ 12%]
tests/test_stage02_d01.py::test_clip_rules PASSED                        [ 12%]
tests/test_stage02_d01.py::test_no_clip_spans_two_trials PASSED          [ 12%]
tests/test_stage02_d01.py::test_20_random_clips_rerendered_from_raw_are_identical PASSED [ 13%]
tests/test_stage02_d01.py::test_rebuilt_gaze_equals_cached PASSED        [ 13%]
tests/test_stage02_d01.py::test_quality_table_not_in_model_inputs PASSED [ 13%]
tests/test_stage02_d02.py::test_manifest_pooled_and_complete PASSED      [ 14%]
tests/test_stage02_d02.py::test_no_d03_path_anywhere PASSED              [ 14%]
tests/test_stage02_d02.py::test_near_duplicate_pairs_recomputed_lie_in_same_outer_fold PASSED [ 14%]
tests/test_stage02_d02.py::test_no_group_has_mixed_labels PASSED         [ 15%]
tests/test_stage02_snapshot.py::test_latest_snapshot_checksums_match_current_files PASSED [ 15%]
tests/test_stage02_snapshot.py::test_latest_snapshot_info_and_code_copy PASSED [ 15%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[1-D01_eye_tracking] PASSED [ 16%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[1-D02_facial_images_A] PASSED [ 16%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[1-D04_autism_prediction] PASSED [ 16%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[1-D05_toddler_screening] PASSED [ 16%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[1-D06_early_autism] PASSED [ 17%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[2-D01_eye_tracking] PASSED [ 17%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[2-D02_facial_images_A] PASSED [ 17%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[2-D04_autism_prediction] PASSED [ 18%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[2-D05_toddler_screening] PASSED [ 18%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[2-D06_early_autism] PASSED [ 18%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[3-D01_eye_tracking] PASSED [ 19%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[3-D02_facial_images_A] PASSED [ 19%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[3-D04_autism_prediction] PASSED [ 19%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[3-D05_toddler_screening] PASSED [ 19%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[3-D06_early_autism] PASSED [ 20%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[4-D01_eye_tracking] PASSED [ 20%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[4-D02_facial_images_A] PASSED [ 20%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[4-D04_autism_prediction] PASSED [ 21%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[4-D05_toddler_screening] PASSED [ 21%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[4-D06_early_autism] PASSED [ 21%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[5-D01_eye_tracking] PASSED [ 22%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[5-D02_facial_images_A] PASSED [ 22%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[5-D04_autism_prediction] PASSED [ 22%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[5-D05_toddler_screening] PASSED [ 22%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[5-D06_early_autism] PASSED [ 23%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[6-D01_eye_tracking] PASSED [ 23%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[6-D02_facial_images_A] PASSED [ 23%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[6-D04_autism_prediction] PASSED [ 24%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[6-D05_toddler_screening] PASSED [ 24%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[6-D06_early_autism] PASSED [ 24%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[7-D01_eye_tracking] PASSED [ 25%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[7-D02_facial_images_A] PASSED [ 25%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[7-D04_autism_prediction] PASSED [ 25%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[7-D05_toddler_screening] PASSED [ 25%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[7-D06_early_autism] PASSED [ 26%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[8-D01_eye_tracking] PASSED [ 26%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[8-D02_facial_images_A] PASSED [ 26%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[8-D04_autism_prediction] PASSED [ 27%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[8-D05_toddler_screening] PASSED [ 27%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[8-D06_early_autism] PASSED [ 27%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[9-D01_eye_tracking] PASSED [ 28%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[9-D02_facial_images_A] PASSED [ 28%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[9-D04_autism_prediction] PASSED [ 28%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[9-D05_toddler_screening] PASSED [ 29%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[9-D06_early_autism] PASSED [ 29%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[10-D01_eye_tracking] PASSED [ 29%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[10-D02_facial_images_A] PASSED [ 29%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[10-D04_autism_prediction] PASSED [ 30%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[10-D05_toddler_screening] PASSED [ 30%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[10-D06_early_autism] PASSED [ 30%]
tests/test_stage02_splits.py::test_every_sample_in_exactly_one_test_fold[D01_eye_tracking] PASSED [ 31%]
tests/test_stage02_splits.py::test_every_sample_in_exactly_one_test_fold[D02_facial_images_A] PASSED [ 31%]
tests/test_stage02_splits.py::test_every_sample_in_exactly_one_test_fold[D04_autism_prediction] PASSED [ 31%]
tests/test_stage02_splits.py::test_every_sample_in_exactly_one_test_fold[D05_toddler_screening] PASSED [ 32%]
tests/test_stage02_splits.py::test_every_sample_in_exactly_one_test_fold[D06_early_autism] PASSED [ 32%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[1-D01_eye_tracking] PASSED [ 32%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[1-D02_facial_images_A] PASSED [ 32%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[1-D04_autism_prediction] PASSED [ 33%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[1-D05_toddler_screening] PASSED [ 33%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[1-D06_early_autism] PASSED [ 33%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[2-D01_eye_tracking] PASSED [ 34%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[2-D02_facial_images_A] PASSED [ 34%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[2-D04_autism_prediction] PASSED [ 34%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[2-D05_toddler_screening] PASSED [ 35%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[2-D06_early_autism] PASSED [ 35%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[3-D01_eye_tracking] PASSED [ 35%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[3-D02_facial_images_A] PASSED [ 35%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[3-D04_autism_prediction] PASSED [ 36%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[3-D05_toddler_screening] PASSED [ 36%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[3-D06_early_autism] PASSED [ 36%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[4-D01_eye_tracking] PASSED [ 37%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[4-D02_facial_images_A] PASSED [ 37%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[4-D04_autism_prediction] PASSED [ 37%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[4-D05_toddler_screening] PASSED [ 38%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[4-D06_early_autism] PASSED [ 38%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[5-D01_eye_tracking] PASSED [ 38%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[5-D02_facial_images_A] PASSED [ 38%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[5-D04_autism_prediction] PASSED [ 39%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[5-D05_toddler_screening] PASSED [ 39%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[5-D06_early_autism] PASSED [ 39%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[6-D01_eye_tracking] PASSED [ 40%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[6-D02_facial_images_A] PASSED [ 40%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[6-D04_autism_prediction] PASSED [ 40%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[6-D05_toddler_screening] PASSED [ 41%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[6-D06_early_autism] PASSED [ 41%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[7-D01_eye_tracking] PASSED [ 41%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[7-D02_facial_images_A] PASSED [ 41%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[7-D04_autism_prediction] PASSED [ 42%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[7-D05_toddler_screening] PASSED [ 42%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[7-D06_early_autism] PASSED [ 42%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[8-D01_eye_tracking] PASSED [ 43%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[8-D02_facial_images_A] PASSED [ 43%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[8-D04_autism_prediction] PASSED [ 43%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[8-D05_toddler_screening] PASSED [ 44%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[8-D06_early_autism] PASSED [ 44%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[9-D01_eye_tracking] PASSED [ 44%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[9-D02_facial_images_A] PASSED [ 45%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[9-D04_autism_prediction] PASSED [ 45%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[9-D05_toddler_screening] PASSED [ 45%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[9-D06_early_autism] PASSED [ 45%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[10-D01_eye_tracking] PASSED [ 46%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[10-D02_facial_images_A] PASSED [ 46%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[10-D04_autism_prediction] PASSED [ 46%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[10-D05_toddler_screening] PASSED [ 47%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[10-D06_early_autism] PASSED [ 47%]
tests/test_stage02_splits.py::test_d01_test_folds_have_asd_and_td_children PASSED [ 47%]
tests/test_stage02_splits.py::test_splits_manifest_hashes_match PASSED   [ 48%]
tests/test_stage02_splits.py::test_rerun_make_splits_is_byte_identical PASSED [ 48%]
tests/test_stage02_tabular.py::test_d04_only_labelled_train_rows PASSED  [ 48%]
tests/test_stage02_tabular.py::test_d04_result_decision_recorded PASSED  [ 48%]
tests/test_stage02_tabular.py::test_d05_honest_set_excludes_items_and_score PASSED [ 49%]
tests/test_stage02_tabular.py::test_d05_label_rule_on_clean_data PASSED  [ 49%]
tests/test_stage02_tabular.py::test_d05_other_csv_never_read_by_stage2_code PASSED [ 49%]
tests/test_stage02_tabular.py::test_d06_empty_column_absent PASSED       [ 50%]
tests/test_stage02_tabular.py::test_identical_feature_rows_share_group PASSED [ 50%]
tests/test_stage02_tabular.py::test_every_column_has_a_role PASSED       [ 50%]
tests/test_stage02_tabular.py::test_raw_data_snapshot_unchanged PASSED   [ 51%]
tests/test_stage03_models.py::test_config_mapping[M01] PASSED            [ 51%]
tests/test_stage03_models.py::test_config_mapping[M02] PASSED            [ 51%]
tests/test_stage03_models.py::test_config_mapping[M03] PASSED            [ 51%]
tests/test_stage03_models.py::test_config_mapping[M04] PASSED            [ 52%]
tests/test_stage03_models.py::test_config_mapping[M05] PASSED            [ 52%]
tests/test_stage03_models.py::test_config_mapping[M06] PASSED            [ 52%]
tests/test_stage03_models.py::test_config_mapping[M07] PASSED            [ 53%]
tests/test_stage03_models.py::test_config_mapping[M08] PASSED            [ 53%]
tests/test_stage03_models.py::test_config_mapping[M09] PASSED            [ 53%]
tests/test_stage03_models.py::test_config_mapping[M10] PASSED            [ 54%]
tests/test_stage03_models.py::test_factory_rejects_unknown PASSED        [ 54%]
tests/test_stage03_models.py::test_forward[M01-sequence] PASSED          [ 54%]
tests/test_stage03_models.py::test_forward[M01-tokens] PASSED            [ 54%]
tests/test_stage03_models.py::test_forward[M02-sequence] PASSED          [ 55%]
tests/test_stage03_models.py::test_forward[M02-tokens] PASSED            [ 55%]
tests/test_stage03_models.py::test_forward[M03-sequence] PASSED          [ 55%]
tests/test_stage03_models.py::test_forward[M03-tokens] PASSED            [ 56%]
tests/test_stage03_models.py::test_forward[M04-sequence] PASSED          [ 56%]
tests/test_stage03_models.py::test_forward[M04-tokens] PASSED            [ 56%]
tests/test_stage03_models.py::test_forward[M05-sequence] PASSED          [ 57%]
tests/test_stage03_models.py::test_forward[M05-tokens] PASSED            [ 57%]
tests/test_stage03_models.py::test_forward[M06-sequence] PASSED          [ 57%]
tests/test_stage03_models.py::test_forward[M06-tokens] PASSED            [ 58%]
tests/test_stage03_models.py::test_forward[M07-sequence] PASSED          [ 58%]
tests/test_stage03_models.py::test_forward[M07-tokens] PASSED            [ 58%]
tests/test_stage03_models.py::test_forward[M08-sequence] PASSED          [ 58%]
tests/test_stage03_models.py::test_forward[M08-tokens] PASSED            [ 59%]
tests/test_stage03_models.py::test_forward[M09-sequence] PASSED          [ 59%]
tests/test_stage03_models.py::test_forward[M09-tokens] PASSED            [ 59%]
tests/test_stage03_models.py::test_forward[M10-sequence] PASSED          [ 60%]
tests/test_stage03_models.py::test_forward[M10-tokens] PASSED            [ 60%]
tests/test_stage03_models.py::test_backward_and_step[M01-sequence] PASSED [ 60%]
tests/test_stage03_models.py::test_backward_and_step[M01-tokens] PASSED  [ 61%]
tests/test_stage03_models.py::test_backward_and_step[M02-sequence] PASSED [ 61%]
tests/test_stage03_models.py::test_backward_and_step[M02-tokens] PASSED  [ 61%]
tests/test_stage03_models.py::test_backward_and_step[M03-sequence] PASSED [ 61%]
tests/test_stage03_models.py::test_backward_and_step[M03-tokens] PASSED  [ 62%]
tests/test_stage03_models.py::test_backward_and_step[M04-sequence] PASSED [ 62%]
tests/test_stage03_models.py::test_backward_and_step[M04-tokens] PASSED  [ 62%]
tests/test_stage03_models.py::test_backward_and_step[M05-sequence] PASSED [ 63%]
tests/test_stage03_models.py::test_backward_and_step[M05-tokens] PASSED  [ 63%]
tests/test_stage03_models.py::test_backward_and_step[M06-sequence] PASSED [ 63%]
tests/test_stage03_models.py::test_backward_and_step[M06-tokens] PASSED  [ 64%]
tests/test_stage03_models.py::test_backward_and_step[M07-sequence] PASSED [ 64%]
tests/test_stage03_models.py::test_backward_and_step[M07-tokens] PASSED  [ 64%]
tests/test_stage03_models.py::test_backward_and_step[M08-sequence] PASSED [ 64%]
tests/test_stage03_models.py::test_backward_and_step[M08-tokens] PASSED  [ 65%]
tests/test_stage03_models.py::test_backward_and_step[M09-sequence] PASSED [ 65%]
tests/test_stage03_models.py::test_backward_and_step[M09-tokens] PASSED  [ 65%]
tests/test_stage03_models.py::test_backward_and_step[M10-sequence] PASSED [ 66%]
tests/test_stage03_models.py::test_backward_and_step[M10-tokens] PASSED  [ 66%]
tests/test_stage03_models.py::test_determinism_eval[M01-sequence] PASSED [ 66%]
tests/test_stage03_models.py::test_determinism_eval[M01-tokens] PASSED   [ 67%]
tests/test_stage03_models.py::test_determinism_eval[M02-sequence] PASSED [ 67%]
tests/test_stage03_models.py::test_determinism_eval[M02-tokens] PASSED   [ 67%]
tests/test_stage03_models.py::test_determinism_eval[M03-sequence] PASSED [ 67%]
tests/test_stage03_models.py::test_determinism_eval[M03-tokens] PASSED   [ 68%]
tests/test_stage03_models.py::test_determinism_eval[M04-sequence] PASSED [ 68%]
tests/test_stage03_models.py::test_determinism_eval[M04-tokens] PASSED   [ 68%]
tests/test_stage03_models.py::test_determinism_eval[M05-sequence] PASSED [ 69%]
tests/test_stage03_models.py::test_determinism_eval[M05-tokens] PASSED   [ 69%]
tests/test_stage03_models.py::test_determinism_eval[M06-sequence] PASSED [ 69%]
tests/test_stage03_models.py::test_determinism_eval[M06-tokens] PASSED   [ 70%]
tests/test_stage03_models.py::test_determinism_eval[M07-sequence] PASSED [ 70%]
tests/test_stage03_models.py::test_determinism_eval[M07-tokens] PASSED   [ 70%]
tests/test_stage03_models.py::test_determinism_eval[M08-sequence] PASSED [ 70%]
tests/test_stage03_models.py::test_determinism_eval[M08-tokens] PASSED   [ 71%]
tests/test_stage03_models.py::test_determinism_eval[M09-sequence] PASSED [ 71%]
tests/test_stage03_models.py::test_determinism_eval[M09-tokens] PASSED   [ 71%]
tests/test_stage03_models.py::test_determinism_eval[M10-sequence] PASSED [ 72%]
tests/test_stage03_models.py::test_determinism_eval[M10-tokens] PASSED   [ 72%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M01-sequence] PASSED [ 72%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M01-tokens] PASSED [ 73%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M02-sequence] PASSED [ 73%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M02-tokens] PASSED [ 73%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M03-sequence] PASSED [ 74%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M03-tokens] PASSED [ 74%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M04-sequence] PASSED [ 74%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M04-tokens] PASSED [ 74%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M05-sequence] PASSED [ 75%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M05-tokens] PASSED [ 75%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M06-sequence] PASSED [ 75%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M06-tokens] PASSED [ 76%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M07-sequence] PASSED [ 76%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M07-tokens] PASSED [ 76%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M08-sequence] PASSED [ 77%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M08-tokens] PASSED [ 77%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M09-sequence] PASSED [ 77%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M09-tokens] PASSED [ 77%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M10-sequence] PASSED [ 78%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M10-tokens] PASSED [ 78%]
tests/test_stage03_models.py::test_feature_cache_equivalence[M01] PASSED [ 78%]
tests/test_stage03_models.py::test_feature_cache_equivalence[M03] PASSED [ 79%]
tests/test_stage03_models.py::test_feature_cache_equivalence[M04] PASSED [ 79%]
tests/test_stage03_models.py::test_feature_cache_equivalence[M06] PASSED [ 79%]
tests/test_stage03_models.py::test_feature_cache_equivalence[M07] PASSED [ 80%]
tests/test_stage03_models.py::test_feature_cache_equivalence[M08] PASSED [ 80%]
tests/test_stage03_models.py::test_feature_cache_equivalence[M10] PASSED [ 80%]
tests/test_stage03_models.py::test_last_stage_trains_part_of_backbone PASSED [ 80%]
tests/test_stage03_models.py::test_model_configs_have_batch_sizes PASSED [ 81%]
tests/test_stage04_engine.py::test_interrupt_leaves_running_and_checkpoint[M01] PASSED [ 81%]
tests/test_stage04_engine.py::test_resume_identical_history_and_predictions[M01] PASSED [ 81%]
tests/test_stage04_engine.py::test_resumed_run_info_and_cleanup[M01] PASSED [ 82%]
tests/test_stage04_engine.py::test_youden_threshold_comes_from_validation_predictions[M01] PASSED [ 82%]
tests/test_stage04_engine.py::test_done_run_is_skipped_unless_forced[M01] PASSED [ 82%]
tests/test_stage04_engine.py::test_checks_detect_incomplete_results[M01] PASSED [ 83%]
tests/test_stage04_engine.py::test_interrupt_leaves_running_and_checkpoint[M09] PASSED [ 83%]
tests/test_stage04_engine.py::test_resume_identical_history_and_predictions[M09] PASSED [ 83%]
tests/test_stage04_engine.py::test_resumed_run_info_and_cleanup[M09] PASSED [ 83%]
tests/test_stage04_engine.py::test_youden_threshold_comes_from_validation_predictions[M09] PASSED [ 84%]
tests/test_stage04_engine.py::test_done_run_is_skipped_unless_forced[M09] PASSED [ 84%]
tests/test_stage04_engine.py::test_checks_detect_incomplete_results[M09] PASSED [ 84%]
tests/test_stage04_engine.py::test_engine_refuses_corrupted_split PASSED [ 85%]
tests/test_stage04_engine.py::test_resume_refuses_changed_config PASSED  [ 85%]
tests/test_stage04_units.py::test_training_protocol_values_fixed PASSED  [ 85%]
tests/test_stage04_units.py::test_merge_config[M01-D01_eye_tracking-1-0.0005-0.0-cached_features] PASSED [ 86%]
tests/test_stage04_units.py::test_merge_config[M01-D02_facial_images_A-3-0.0005-5e-05-images] PASSED [ 86%]
tests/test_stage04_units.py::test_merge_config[M07-D02_facial_images_A-10-0.0005-5e-05-images] PASSED [ 86%]
tests/test_stage04_units.py::test_merge_config[M09-D01_eye_tracking-2-0.001-0.001-clips] PASSED [ 87%]
tests/test_stage04_units.py::test_merge_config[M05-D02_facial_images_A-4-0.001-0.001-images] PASSED [ 87%]
tests/test_stage04_units.py::test_model_config_lr_mirrors_training_yaml PASSED [ 87%]
tests/test_stage04_units.py::test_scheduler_warmup_and_cosine PASSED     [ 87%]
tests/test_stage04_units.py::test_d02_transforms_uint8 PASSED            [ 88%]
tests/test_stage04_units.py::test_real_splits_pass_check_fold[D01_eye_tracking] PASSED [ 88%]
tests/test_stage04_units.py::test_real_splits_pass_check_fold[D02_facial_images_A] PASSED [ 88%]
tests/test_stage04_units.py::test_leakage_asserts_trigger_on_corrupted_split[id_in_train_and_test-D01_eye_tracking] PASSED [ 89%]
tests/test_stage04_units.py::test_leakage_asserts_trigger_on_corrupted_split[id_in_train_and_test-D02_facial_images_A] PASSED [ 89%]
tests/test_stage04_units.py::test_leakage_asserts_trigger_on_corrupted_split[duplicate-D01_eye_tracking] PASSED [ 89%]
tests/test_stage04_units.py::test_leakage_asserts_trigger_on_corrupted_split[duplicate-D02_facial_images_A] PASSED [ 90%]
tests/test_stage04_units.py::test_leakage_asserts_trigger_on_corrupted_split[group_overlap-D01_eye_tracking] PASSED [ 90%]
tests/test_stage04_units.py::test_leakage_asserts_trigger_on_corrupted_split[group_overlap-D02_facial_images_A] PASSED [ 90%]
tests/test_stage04_units.py::test_leakage_asserts_trigger_on_corrupted_split[unknown_id-D01_eye_tracking] PASSED [ 90%]
tests/test_stage04_units.py::test_leakage_asserts_trigger_on_corrupted_split[unknown_id-D02_facial_images_A] PASSED [ 91%]
tests/test_stage04_units.py::test_test_set_must_match_folds_csv PASSED   [ 91%]
tests/test_stage04_units.py::test_verify_manifest_detects_modified_temp_copy PASSED [ 91%]
tests/test_stage04_units.py::test_assert_no_test_ids PASSED              [ 92%]
tests/test_stage04_units.py::test_binary_metrics_match_sklearn[0.5-0] PASSED [ 92%]
tests/test_stage04_units.py::test_binary_metrics_match_sklearn[0.5-1] PASSED [ 92%]
tests/test_stage04_units.py::test_binary_metrics_match_sklearn[0.5-2] PASSED [ 93%]
tests/test_stage04_units.py::test_binary_metrics_match_sklearn[0.3-0] PASSED [ 93%]
tests/test_stage04_units.py::test_binary_metrics_match_sklearn[0.3-1] PASSED [ 93%]
tests/test_stage04_units.py::test_binary_metrics_match_sklearn[0.3-2] PASSED [ 93%]
tests/test_stage04_units.py::test_binary_metrics_match_sklearn[0.72-0] PASSED [ 94%]
tests/test_stage04_units.py::test_binary_metrics_match_sklearn[0.72-1] PASSED [ 94%]
tests/test_stage04_units.py::test_binary_metrics_match_sklearn[0.72-2] PASSED [ 94%]
tests/test_stage04_units.py::test_binary_metrics_known_values PASSED     [ 95%]
tests/test_stage04_units.py::test_youden_threshold_brute_force_on_validation_only[0] PASSED [ 95%]
tests/test_stage04_units.py::test_youden_threshold_brute_force_on_validation_only[1] PASSED [ 95%]
tests/test_stage04_units.py::test_youden_threshold_brute_force_on_validation_only[2] PASSED [ 96%]
tests/test_stage04_units.py::test_youden_threshold_brute_force_on_validation_only[3] PASSED [ 96%]
tests/test_stage04_units.py::test_child_level_mean_probability PASSED    [ 96%]
tests/test_stage04_units.py::test_registry_skip_force_resume PASSED      [ 96%]
tests/test_stage04_units.py::test_registry_file_lock_no_lost_rows PASSED [ 97%]
tests/test_stage04_units.py::test_summary_aggregation_on_fake_folds PASSED [ 97%]
tests/test_stage05_checker.py::test_reference_block_is_valid PASSED      [ 97%]
tests/test_stage05_checker.py::test_perfect_pr_auc_passes PASSED         [ 98%]
tests/test_stage05_checker.py::test_auc_zero_passes PASSED               [ 98%]
tests/test_stage05_checker.py::test_value_above_one_fails PASSED         [ 98%]
tests/test_stage05_checker.py::test_negative_value_fails PASSED          [ 99%]
tests/test_stage05_checker.py::test_nan_is_flagged_never_silent PASSED   [ 99%]
tests/test_stage05_checker.py::test_nan_where_defined_fails PASSED       [ 99%]
tests/test_stage05_checker.py::test_missing_or_non_numeric_fails PASSED  [100%]

======================= 331 passed in 127.46s (0:02:07) ========================
```

</details>

## 11. Neutral description

- On clip level the ten models reach mean test ROC-AUCs between 0.595 and 0.686, with fold-to-fold standard deviations of 0.081-0.134 (sections 3, 5): with 5-6 test children per fold, which children land in a fold changes the result strongly. Clips of one child are not independent, so clip-level numbers overstate the effective sample size.
- The pooled child-level ROC-AUC is higher than the pooled clip-level ROC-AUC for 9 of 10 models (section 4); a child score averages many clips.
- 29 of 100 runs had their best validation epoch at epoch <= 3 (sections 2, 6); with 7-8 validation children the validation AUC used for early stopping is noisy.
- The tracking-quality indicator alone reaches a child-level ROC-AUC of 0.842 (section 7). Until Stage 8 has run the shortcut controls, these results must not be read as evidence that the models learned gaze behaviour rather than recording-quality differences.
- No result is a diagnosis or a clinical accuracy estimate.

## 12. Open items

1. Stage 8: quality-only and stimulus-aware baselines with the same folds; bootstrap CIs; Friedman/Nemenyi.
2. Stage 6 (D02) is NOT launched: awaiting the user's review of these results.

STAGE 5 DECISION: PASS
