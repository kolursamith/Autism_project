# STAGE 7 REPORT - tabular baselines (D04, D05, D06), run early in parallel with S5

Screening-support research, not diagnosis. Run on 2026-10-14 BEFORE Stage 6, with the user's explicit approval, while the S5 GPU training was running. CPU only (`CUDA_VISIBLE_DEVICES=''`), sklearn n_jobs 2, BLAS/OpenMP threads capped at 2, `nice -n 10`; torch never imported (tested). All Stage 7 code is in `tabular/`; nothing in src/, scripts/, configs/, splits/, cache/, results/D01_eye_tracking/, logs/stage5* or results/run_registry.csv was modified (section 8). No snapshot yet (made after S5 finishes).

## 1. Summary

- 4 experiments x 5 models x 10 folds = 200 runs, all DONE (200); total fitting time 50 s.
- Result checks: 20/20 PASS; Stage 7 tests: 16 passed in 7.69s.
- Splits: the Stage 2 files `splits/D04_*`, `D05_*`, `D06_*`, sha256-verified against SPLITS_MANIFEST.json before the run (and in the tests).

## 2. Protocol (fixed hyperparameters, no tuning; `tabular/configs/stage7.yaml`)

| item | setting |
|---|---|
| preprocessing | ColumnTransformer inside a sklearn Pipeline, fitted on the TRAIN part of each fold only: numeric = median imputation + StandardScaler; categorical = most-frequent imputation + one-hot (unknown categories ignored) |
| dummy_majority | DummyClassifier(most_frequent) |
| logreg | LogisticRegression, L2, C = 1.0, class_weight = balanced, max_iter 5000 |
| random_forest | RandomForestClassifier, 500 trees, class_weight = balanced_subsample, n_jobs 2 |
| hist_gb | HistGradientBoostingClassifier, default parameters, sample_weight = balanced class weights |
| mlp | MLPClassifier(hidden 64, 32), sklearn defaults otherwise; one partial_fit pass over the train part per epoch, early stopping on the ROC-AUC of the fold's VALIDATION part (patience 10, min_delta 1e-4, max 200 epochs), best epoch kept; sklearn's internal early_stopping split is not used |
| seeds | random_state = 42 + fold |
| validation part | used only for MLP early stopping and for the Youden threshold |
| test part | predicted once |
| thresholds | 0.5 and the Youden-J threshold chosen on the validation predictions (never on test) |
| metrics | the deep-model set (`src/training/metrics.py`): accuracy, balanced accuracy, precision, sensitivity, specificity, F1, ROC-AUC, PR-AUC, Brier, confusion matrix; summary = mean, std (ddof 1), min, max over folds + pooled out-of-fold (`src/training/summarize.py`, reused read-only) |

Output layout (results/README.md plus an experiment level): `results/<dataset>/<experiment>/<model>/fold_XX/` {config.yaml, metrics.json, predictions.csv, val_predictions.csv, run_info.json}, `<model>/summary.json`, `<experiment>/leaderboard.csv`; registry `tabular/run_registry_stage7.csv`. No model files are stored.

## 3. Results (test folds, threshold 0.5, mean ± std over 10 folds)

### D04 autism prediction (adult AQ-10 screening table), all features

n = 800 (161 positive / 639 negative); `results/D04_autism_prediction/set_all/leaderboard.csv`. features per columns.yaml (A1-A10 scores, age, result, demographics); test.csv (unlabelled) never read.

| model | ROC-AUC | balanced accuracy | sensitivity | specificity | F1 | bal. acc. @ val threshold | pooled OOF ROC-AUC |
|---|---|---|---|---|---|---|---|
| mlp | 0.912 ± 0.031 | 0.773 ± 0.067 | 0.628 ± 0.164 | 0.917 ± 0.061 | 0.634 ± 0.106 | 0.833 ± 0.038 | 0.905 |
| logreg | 0.905 ± 0.040 | 0.832 ± 0.037 | 0.839 ± 0.052 | 0.826 ± 0.069 | 0.669 ± 0.073 | 0.833 ± 0.053 | 0.899 |
| random_forest | 0.900 ± 0.041 | 0.762 ± 0.036 | 0.596 ± 0.055 | 0.928 ± 0.044 | 0.637 ± 0.074 | 0.833 ± 0.047 | 0.901 |
| hist_gb | 0.881 ± 0.036 | 0.783 ± 0.036 | 0.683 ± 0.082 | 0.884 ± 0.052 | 0.640 ± 0.058 | 0.813 ± 0.034 | 0.88 |
| dummy_majority | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.500 ± 0.000 | 0.5 |

### D05 toddler screening, set_all - LABEL-DEFINITIONAL UPPER BOUND

n = 1054 (728 positive / 326 negative); `results/D05_toddler_screening/set_all/leaderboard.csv`. LABEL-DEFINITIONAL UPPER BOUND: label == (A1..A10 sum >= 4); does NOT measure autism prediction.

| model | ROC-AUC | balanced accuracy | sensitivity | specificity | F1 | bal. acc. @ val threshold | pooled OOF ROC-AUC |
|---|---|---|---|---|---|---|---|
| logreg | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.994 ± 0.007 | 1.0 |
| random_forest | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.992 ± 0.013 | 1.0 |
| hist_gb | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.0 |
| mlp | 0.999 ± 0.003 | 0.978 ± 0.028 | 0.995 ± 0.010 | 0.961 ± 0.054 | 0.989 ± 0.013 | 0.980 ± 0.022 | 0.999 |
| dummy_majority | 0.500 ± 0.000 | 0.500 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.817 ± 0.003 | 0.500 ± 0.000 | 0.5 |

**How close each model gets to the exact rule** (`rule_comparison.csv`, pooled out-of-fold predictions of all 1,054 rows):

| model | rule_equals_label_all_rows | n | agreement_with_rule_at_0.5 | n_disagreements_at_0.5 | pooled_oof_roc_auc_vs_rule |
|---|---|---|---|---|---|
| exact rule: sum(A1..A10) >= 4 | True | 1054 |  |  |  |
| dummy_majority |  | 1054 | 0.6907020872865275 | 326.0 | 0.5 |
| logreg |  | 1054 | 1.0 | 0.0 | 1.0 |
| random_forest |  | 1054 | 1.0 | 0.0 | 1.0 |
| hist_gb |  | 1054 | 1.0 | 0.0 | 1.0 |
| mlp |  | 1054 | 0.9838709677419356 | 17.0 | 0.9988454796737004 |

In D05 the label is, by construction, `sum(A1..A10) >= 4` (true for all 1,054 rows), and A1-A10 plus `Qchat-10-Score` are in set_all. These numbers therefore measure how well each model re-learns a known arithmetic rule from its own inputs. **They do not measure autism prediction** and must not be compared with the other experiments as evidence of screening performance.

### D05 toddler screening, set_honest (A1-A10 and Qchat score removed)

n = 1054 (728 positive / 326 negative); `results/D05_toddler_screening/set_honest/leaderboard.csv`. A1-A10 and Qchat-10-Score removed (age, sex, ethnicity, jaundice, family history, respondent).

| model | ROC-AUC | balanced accuracy | sensitivity | specificity | F1 | bal. acc. @ val threshold | pooled OOF ROC-AUC |
|---|---|---|---|---|---|---|---|
| random_forest | 0.612 ± 0.070 | 0.563 ± 0.062 | 0.696 ± 0.094 | 0.429 ± 0.102 | 0.711 ± 0.062 | 0.572 ± 0.068 | 0.607 |
| hist_gb | 0.612 ± 0.074 | 0.573 ± 0.073 | 0.618 ± 0.094 | 0.527 ± 0.097 | 0.673 ± 0.071 | 0.580 ± 0.061 | 0.61 |
| mlp | 0.597 ± 0.069 | 0.538 ± 0.043 | 0.948 ± 0.056 | 0.129 ± 0.108 | 0.810 ± 0.025 | 0.568 ± 0.050 | 0.587 |
| logreg | 0.594 ± 0.068 | 0.577 ± 0.062 | 0.661 ± 0.053 | 0.494 ± 0.114 | 0.700 ± 0.041 | 0.559 ± 0.049 | 0.591 |
| dummy_majority | 0.500 ± 0.000 | 0.500 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.817 ± 0.003 | 0.500 ± 0.000 | 0.5 |

### D06 early autism, all features

n = 1000 (411 positive / 589 negative); `results/D06_early_autism/set_all/leaderboard.csv`. features per columns.yaml.

| model | ROC-AUC | balanced accuracy | sensitivity | specificity | F1 | bal. acc. @ val threshold | pooled OOF ROC-AUC |
|---|---|---|---|---|---|---|---|
| dummy_majority | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.500 ± 0.000 | 0.5 |
| logreg | 0.481 ± 0.065 | 0.503 ± 0.057 | 0.501 ± 0.096 | 0.504 ± 0.070 | 0.451 ± 0.071 | 0.485 ± 0.036 | 0.482 |
| random_forest | 0.466 ± 0.051 | 0.503 ± 0.037 | 0.273 ± 0.073 | 0.733 ± 0.046 | 0.326 ± 0.069 | 0.486 ± 0.025 | 0.466 |
| hist_gb | 0.459 ± 0.039 | 0.472 ± 0.037 | 0.397 ± 0.086 | 0.547 ± 0.055 | 0.385 ± 0.060 | 0.485 ± 0.026 | 0.46 |
| mlp | 0.449 ± 0.064 | 0.498 ± 0.026 | 0.143 ± 0.165 | 0.853 ± 0.190 | 0.168 ± 0.165 | 0.462 ± 0.053 | 0.468 |

## 4. Description (neutral)

- **D04**: logistic regression, random forest, gradient boosting and the MLP reach mean test ROC-AUCs of 0.88-0.91 against 0.50 for the majority baseline. At threshold 0.5 the models trade sensitivity against specificity differently (the positive class is 20% of rows); the validation-chosen threshold gives similar balanced accuracy across them. D04 contains the A1-A10 screening answers and `result`; Stage 2 found `result` is not a function of A1-A10 (kept as a feature).
- **D05 set_all**: upper bound only; see the rule comparison above.
- **D05 set_honest**: without the screening items and score, the remaining demographic/context features give mean ROC-AUCs of 0.59-0.61 (majority baseline 0.50).
- **D06**: no model exceeds the majority baseline: mean ROC-AUCs are 0.45-0.48. The descriptive single-feature AUCs below are all close to 0.5, i.e. this table shows no detectable association between its features and the label. Mean AUCs slightly below 0.5 are a known effect of cross-validation on data without signal and are not evidence of an inverse relation.

| feature | single-feature ROC-AUC (all rows) |
|---|---|
| Age | 0.516 |
| Social_Interaction_Score | 0.507 |
| Communication_Score | 0.522 |
| Repetitive_Behavior_Score | 0.501 |
| Gender | 0.503 |
| Jaundice | 0.525 |
| Family_ASD_History | 0.507 |
| Language_Delay | 0.506 |

None of these numbers is a diagnostic accuracy; the tables are small, single-source screening datasets.

## 5. Leakage and correctness tests (`tabular/tests/test_stage07.py`, `logs/stage07_pytest.txt`)

Tests: split sha256 = manifest; train/val/test disjoint by sample and by group in every fold, every sample tested exactly once; a corrupted split is rejected; set_honest contains none of A1-A10 / Qchat-10-Score (config, fitted pipeline and every saved run); D05 label == item rule; an audit hook records every file opened while running all experiments x models: no file under the raw dataset folder is opened (so D04 `test.csv` and `Autism_Screening_Data_Combined.csv` are never read); imputer statistics, scaler mean (== train-part mean, != all-data mean) and one-hot categories come from the train part only; MLP early stopping sees exactly the validation part; every saved threshold equals the Youden threshold of the saved validation predictions; all 200 runs complete and valid; torch never imported.

```
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0 -- /home/score/Documents/AI_project/asd_hybrid_10fold/.venv/bin/python
rootdir: /home/score/Documents/AI_project/asd_hybrid_10fold
configfile: pytest.ini
plugins: anyio-4.15.1
collecting ... collected 16 items

tabular/tests/test_stage07.py::test_split_hashes_match_manifest PASSED   [  6%]
tabular/tests/test_stage07.py::test_train_val_test_disjoint_every_fold[D04_autism_prediction] PASSED [ 12%]
tabular/tests/test_stage07.py::test_train_val_test_disjoint_every_fold[D05_toddler_screening] PASSED [ 18%]
tabular/tests/test_stage07.py::test_train_val_test_disjoint_every_fold[D06_early_autism] PASSED [ 25%]
tabular/tests/test_stage07.py::test_check_fold_detects_corrupted_split PASSED [ 31%]
tabular/tests/test_stage07.py::test_set_honest_has_no_items_or_score PASSED [ 37%]
tabular/tests/test_stage07.py::test_set_all_d05_label_is_the_item_rule PASSED [ 43%]
tabular/tests/test_stage07.py::test_raw_files_never_read PASSED          [ 50%]
tabular/tests/test_stage07.py::test_preprocessing_fitted_on_train_part_only[D04_autism_prediction-set_all] PASSED [ 56%]
tabular/tests/test_stage07.py::test_preprocessing_fitted_on_train_part_only[D05_toddler_screening-set_all] PASSED [ 62%]
tabular/tests/test_stage07.py::test_preprocessing_fitted_on_train_part_only[D05_toddler_screening-set_honest] PASSED [ 68%]
tabular/tests/test_stage07.py::test_preprocessing_fitted_on_train_part_only[D06_early_autism-set_all] PASSED [ 75%]
tabular/tests/test_stage07.py::test_mlp_early_stopping_uses_validation_part_only PASSED [ 81%]
tabular/tests/test_stage07.py::test_saved_thresholds_come_from_validation_predictions PASSED [ 87%]
tabular/tests/test_stage07.py::test_all_results_complete_and_valid PASSED [ 93%]
tabular/tests/test_stage07.py::test_torch_never_imported PASSED          [100%]

============================== 16 passed in 7.69s ==============================
```

## 6. Result checks (`tabular/stage7_checks.json`)

20/20 PASS: all 10 folds present per experiment x model, metric values valid, predictions = test ids of the fold exactly once, union over folds = every sample exactly once. One check needed a floating-point tolerance: sklearn's average precision returned 1.0000000000000002 for a perfectly separated D05 set_all fold (logreg fold 5); `tabular/run_stage7.py` clips values within 1e-9 of [0, 1] before the range check.

## 7. Run log (`logs/stage07_run.log`)

```
split hashes OK; CPU only (CUDA_VISIBLE_DEVICES=''), n_jobs 2, OMP_NUM_THREADS 2
D04_autism_prediction/set_all/dummy_majority: 10 folds done (0 s elapsed)
D04_autism_prediction/set_all/logreg: 10 folds done (1 s elapsed)
D04_autism_prediction/set_all/random_forest: 10 folds done (9 s elapsed)
D04_autism_prediction/set_all/hist_gb: 10 folds done (12 s elapsed)
D04_autism_prediction/set_all/mlp: 10 folds done (15 s elapsed)
D05_toddler_screening/set_all/dummy_majority: 10 folds done (15 s elapsed)
D05_toddler_screening/set_all/logreg: 10 folds done (16 s elapsed)
D05_toddler_screening/set_all/random_forest: 10 folds done (24 s elapsed)
D05_toddler_screening/set_all/hist_gb: 10 folds done (25 s elapsed)
D05_toddler_screening/set_all/mlp: 10 folds done (28 s elapsed)
D05_toddler_screening/set_honest/dummy_majority: 10 folds done (28 s elapsed)
D05_toddler_screening/set_honest/logreg: 10 folds done (28 s elapsed)
D05_toddler_screening/set_honest/random_forest: 10 folds done (37 s elapsed)
D05_toddler_screening/set_honest/hist_gb: 10 folds done (38 s elapsed)
D05_toddler_screening/set_honest/mlp: 10 folds done (42 s elapsed)
D06_early_autism/set_all/dummy_majority: 10 folds done (43 s elapsed)
D06_early_autism/set_all/logreg: 10 folds done (43 s elapsed)
D06_early_autism/set_all/random_forest: 10 folds done (52 s elapsed)
D06_early_autism/set_all/hist_gb: 10 folds done (53 s elapsed)
D06_early_autism/set_all/mlp: 10 folds done (54 s elapsed)
result checks: 19/20 PASS; total 55 s
split hashes OK; CPU only (CUDA_VISIBLE_DEVICES=''), n_jobs 2, OMP_NUM_THREADS 2
D04_autism_prediction/set_all/dummy_majority: 10 folds done (0 s elapsed)
D04_autism_prediction/set_all/logreg: 10 folds done (0 s elapsed)
D04_autism_prediction/set_all/random_forest: 10 folds done (0 s elapsed)
D04_autism_prediction/set_all/hist_gb: 10 folds done (0 s elapsed)
D04_autism_prediction/set_all/mlp: 10 folds done (0 s elapsed)
D05_toddler_screening/set_all/dummy_majority: 10 folds done (0 s elapsed)
D05_toddler_screening/set_all/logreg: 10 folds done (0 s elapsed)
D05_toddler_screening/set_all/random_forest: 10 folds done (0 s elapsed)
D05_toddler_screening/set_all/hist_gb: 10 folds done (0 s elapsed)
D05_toddler_screening/set_all/mlp: 10 folds done (0 s elapsed)
D05_toddler_screening/set_honest/dummy_majority: 10 folds done (0 s elapsed)
D05_toddler_screening/set_honest/logreg: 10 folds done (0 s elapsed)
D05_toddler_screening/set_honest/random_forest: 10 folds done (0 s elapsed)
D05_toddler_screening/set_honest/hist_gb: 10 folds done (0 s elapsed)
D05_toddler_screening/set_honest/mlp: 10 folds done (0 s elapsed)
D06_early_autism/set_all/dummy_majority: 10 folds done (0 s elapsed)
D06_early_autism/set_all/logreg: 10 folds done (0 s elapsed)
D06_early_autism/set_all/random_forest: 10 folds done (0 s elapsed)
D06_early_autism/set_all/hist_gb: 10 folds done (1 s elapsed)
D06_early_autism/set_all/mlp: 10 folds done (1 s elapsed)
result checks: 20/20 PASS; total 1 s
```

The first run reported 19/20 checks because of the PR-AUC tolerance issue above; the second invocation skipped all 200 DONE runs and only rebuilt the summaries and checks.

## 8. Files the S5 queue depends on were not touched

Comparison with `snapshots/stage_04/CHECKSUMS.sha256`:

| dir | changed | added | removed | added files |
|---|---|---|---|---|
| src | 0 | 0 | 0 |  |
| configs | 0 | 0 | 0 |  |
| splits | 0 | 0 | 0 |  |
| cache | 0 | 0 | 0 |  |
| scripts | 0 | 3 | 0 | scripts/stage5_launch.sh scripts/stage5_status.py scripts/stage5_stop.sh |
| tests | 0 | 0 | 0 |  |

scripts/ additions are the three Stage 5A files (written before Stage 7). src/, configs/, splits/ and cache/ are byte-identical to the stage_04 snapshot.

## 9. Open items

1. `src/training/metrics.is_valid_metric_block` (used by the deep-model gate `validate_model_results`) has the same strict [0, 1] check; a perfectly separated test fold could make it FAIL a valid deep-model run. src/ must not change while S5 runs: fix (tolerance 1e-9 + test) after S5 finishes and before S6.
2. Full pytest suite and `snapshots/stage_07/` after S5 finishes (the snapshot hashes src/; the stage_04 snapshot test currently fails because scripts/ gained the Stage 5A files).
3. Stage 8 statistics (bootstrap CIs, Friedman/Nemenyi) can use these summaries and predictions.

STAGE 7 DECISION: PASS
