# STAGE 6 REPORT - D02 facial images: 10 hybrid models x 10 folds

Screening-support research, not diagnosis. 2,940 facial images (1,470 autistic / 1,470 non-autistic, web-collected DFDK folder; original train/valid/test sub-folders pooled, DD-05), 2,929 near-duplicate groups (pHash Hamming <= 4, DD-07); group-wise stratified 10-fold CV, 294 test images per fold; protocol fixed in Stage 4 (`configs/training.yaml`), no tuning. Each image = one sample; the backbone's final feature map is the token sequence of the temporal encoder. Test folds were predicted once per run and never used for any choice. No git: recorded by `snapshots/stage_06/`.

## 1. Summary

- Queue: started 2026-10-14T11:02:45+05:30, ended 2026-10-14T14:31:58+05:30 (exit 0); 1 launch, no resume; 100/100 runs DONE; validate_model_results PASS for 10/10 models (0 flags).
- Split hashes re-verified; all 10 models predicted exactly the same 294 test ids in every fold; no near-duplicate group spans two folds (section 7).
- Highest mean test ROC-AUC: M04 (ConvNeXt + BiLSTM + Attn) 0.931 ± 0.015; range of model means 0.867-0.931.
- Total GPU-run time 3.48 h; 19 of 100 runs reached the 40-epoch limit (17 of them custom-CNN runs).

## 2. Run statistics (from run_info.json)

Total GPU-run time (sum of fold run seconds): 208.5 min (3.48 h); queue wall time 11:02:45 -> 14:31:58. Runs ended by early stopping: 83/100 (the other 17 reached max_epochs 40). Peak GPU memory over all runs: 1.69 GB. Fine-tuning policy: last backbone stage + head for the 7 pretrained models (backbone lr = head lr / 10), the whole network for the 3 custom CNNs.

| model | policy | GPU s (10 folds) | s / fold | epochs_min | epochs_median | epochs_max | runs at 40 | best_min | best_median | best_max | peak GB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M01 | last_stage (pretrained) | 1025.85 | 102.59 | 12 | 19.0 | 32 | 0 | 4 | 11.0 | 24 | 0.98 |
| M02 | full (custom CNN) | 1504.16 | 150.42 | 13 | 40.0 | 40 | 7 | 5 | 33.0 | 39 | 1.57 |
| M03 | last_stage (pretrained) | 1163.3 | 116.33 | 13 | 20.0 | 35 | 0 | 5 | 12.0 | 27 | 0.98 |
| M04 | last_stage (pretrained) | 1303.53 | 130.35 | 11 | 16.5 | 40 | 1 | 3 | 8.5 | 36 | 0.92 |
| M05 | full (custom CNN) | 1726.36 | 172.64 | 30 | 39.0 | 40 | 5 | 22 | 31.0 | 39 | 1.69 |
| M06 | last_stage (pretrained) | 974.77 | 97.48 | 11 | 16.5 | 29 | 0 | 3 | 8.5 | 21 | 0.77 |
| M07 | last_stage (pretrained) | 1536.79 | 153.68 | 13 | 16.0 | 27 | 0 | 5 | 8.0 | 19 | 1.04 |
| M08 | last_stage (pretrained) | 721.25 | 72.12 | 10 | 16.5 | 40 | 1 | 2 | 8.5 | 38 | 0.74 |
| M09 | full (custom CNN) | 1388.93 | 138.89 | 19 | 38.0 | 40 | 5 | 11 | 30.0 | 37 | 1.57 |
| M10 | last_stage (pretrained) | 1165.65 | 116.57 | 15 | 18.0 | 37 | 0 | 7 | 10.0 | 29 | 1.01 |

Best-epoch distribution over the 100 runs (epoch: count): 2: 1, 3: 5, 4: 4, 5: 6, 6: 4, 7: 8, 8: 2, 9: 4, 10: 6, 11: 6, 12: 3, 13: 3, 14: 1, 15: 2, 17: 2, 18: 1, 19: 2, 20: 1, 21: 1, 22: 2, 23: 2, 24: 3, 25: 2, 26: 1, 27: 2, 28: 1, 29: 5, 30: 1, 32: 2, 33: 6, 34: 1, 35: 3, 36: 1, 37: 2, 38: 1, 39: 3.

## 3. Main table - image level, test folds, threshold 0.5 (mean ± std over 10 folds, sorted by ROC-AUC)

| model | architecture | ROC-AUC | PR-AUC | accuracy | bal. acc. | sensitivity | specificity | F1 | Brier |
|---|---|---|---|---|---|---|---|---|---|
| M04 | ConvNeXt + BiLSTM + Attn | 0.931 ± 0.015 | 0.931 ± 0.016 | 0.845 ± 0.028 | 0.845 ± 0.028 | 0.848 ± 0.074 | 0.841 ± 0.095 | 0.845 ± 0.026 | 0.131 ± 0.023 |
| M07 | ViT + LSTM + Attn | 0.924 ± 0.015 | 0.919 ± 0.019 | 0.832 ± 0.033 | 0.832 ± 0.033 | 0.816 ± 0.085 | 0.848 ± 0.060 | 0.827 ± 0.042 | 0.136 ± 0.031 |
| M08 | EfficientNet + Transformer + Attn | 0.903 ± 0.011 | 0.896 ± 0.014 | 0.819 ± 0.028 | 0.819 ± 0.028 | 0.807 ± 0.036 | 0.831 ± 0.063 | 0.817 ± 0.025 | 0.152 ± 0.019 |
| M01 | EfficientNetB3 + BiLSTM + Attn | 0.901 ± 0.021 | 0.893 ± 0.020 | 0.812 ± 0.024 | 0.812 ± 0.024 | 0.789 ± 0.060 | 0.834 ± 0.034 | 0.806 ± 0.031 | 0.155 ± 0.030 |
| M06 | DenseNet121 + LSTM + Attn | 0.901 ± 0.011 | 0.897 ± 0.016 | 0.811 ± 0.022 | 0.811 ± 0.022 | 0.790 ± 0.047 | 0.832 ± 0.065 | 0.807 ± 0.021 | 0.144 ± 0.017 |
| M03 | ResNet50 + LSTM + Attn | 0.890 ± 0.010 | 0.883 ± 0.013 | 0.804 ± 0.014 | 0.804 ± 0.014 | 0.776 ± 0.027 | 0.832 ± 0.041 | 0.799 ± 0.012 | 0.147 ± 0.012 |
| M10 | ResNet50 + BiLSTM + Transformer | 0.885 ± 0.016 | 0.880 ± 0.017 | 0.789 ± 0.030 | 0.789 ± 0.030 | 0.721 ± 0.096 | 0.856 ± 0.051 | 0.771 ± 0.048 | 0.164 ± 0.024 |
| M05 | CNN + Transformer + MLP | 0.881 ± 0.016 | 0.866 ± 0.020 | 0.795 ± 0.022 | 0.795 ± 0.022 | 0.807 ± 0.081 | 0.783 ± 0.077 | 0.797 ± 0.028 | 0.147 ± 0.017 |
| M09 | CNN + GRU + Attn | 0.875 ± 0.028 | 0.860 ± 0.025 | 0.778 ± 0.042 | 0.778 ± 0.042 | 0.776 ± 0.073 | 0.780 ± 0.120 | 0.778 ± 0.033 | 0.157 ± 0.021 |
| M02 | CNN + BiLSTM + Attn | 0.867 ± 0.046 | 0.853 ± 0.051 | 0.785 ± 0.039 | 0.785 ± 0.039 | 0.787 ± 0.052 | 0.784 ± 0.083 | 0.786 ± 0.034 | 0.153 ± 0.023 |

### At the validation-chosen threshold (Youden J on each fold's validation images)

Pooled OOF ROC-AUC = one ROC-AUC over all 2,940 out-of-fold test predictions (each image predicted once, by the model of its own fold).

| model | val threshold (mean ± std) | accuracy | bal. acc. | sensitivity | specificity | F1 | pooled OOF ROC-AUC |
|---|---|---|---|---|---|---|---|
| M04 | 0.543 ± 0.405 | 0.852 ± 0.026 | 0.852 ± 0.026 | 0.826 ± 0.068 | 0.878 ± 0.029 | 0.846 ± 0.035 | 0.896 |
| M07 | 0.450 ± 0.349 | 0.846 ± 0.022 | 0.846 ± 0.022 | 0.851 ± 0.056 | 0.841 ± 0.051 | 0.846 ± 0.024 | 0.898 |
| M08 | 0.418 ± 0.307 | 0.820 ± 0.023 | 0.820 ± 0.023 | 0.803 ± 0.051 | 0.837 ± 0.031 | 0.816 ± 0.027 | 0.871 |
| M01 | 0.386 ± 0.241 | 0.813 ± 0.027 | 0.813 ± 0.027 | 0.822 ± 0.060 | 0.805 ± 0.043 | 0.814 ± 0.032 | 0.882 |
| M06 | 0.379 ± 0.250 | 0.814 ± 0.013 | 0.814 ± 0.013 | 0.825 ± 0.051 | 0.803 ± 0.054 | 0.816 ± 0.015 | 0.885 |
| M03 | 0.410 ± 0.202 | 0.795 ± 0.027 | 0.795 ± 0.027 | 0.797 ± 0.085 | 0.792 ± 0.066 | 0.793 ± 0.040 | 0.882 |
| M10 | 0.355 ± 0.219 | 0.799 ± 0.020 | 0.799 ± 0.020 | 0.774 ± 0.058 | 0.823 ± 0.054 | 0.793 ± 0.024 | 0.875 |
| M05 | 0.442 ± 0.166 | 0.794 ± 0.027 | 0.794 ± 0.027 | 0.839 ± 0.078 | 0.748 ± 0.082 | 0.802 ± 0.031 | 0.874 |
| M09 | 0.344 ± 0.235 | 0.793 ± 0.024 | 0.793 ± 0.024 | 0.844 ± 0.062 | 0.741 ± 0.055 | 0.802 ± 0.027 | 0.867 |
| M02 | 0.449 ± 0.134 | 0.786 ± 0.046 | 0.786 ± 0.046 | 0.797 ± 0.092 | 0.776 ± 0.058 | 0.787 ± 0.055 | 0.869 |

## 4. Per-fold test ROC-AUC (models x folds)

| model | f1 | f2 | f3 | f4 | f5 | f6 | f7 | f8 | f9 | f10 | min (fold) | max (fold) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M04 | 0.928 | 0.93 | 0.909 | 0.914 | 0.95 | 0.95 | 0.916 | 0.94 | 0.93 | 0.945 | 0.909 (f3) | 0.950 (f5) |
| M07 | 0.917 | 0.934 | 0.895 | 0.93 | 0.944 | 0.907 | 0.934 | 0.918 | 0.929 | 0.93 | 0.895 (f3) | 0.944 (f5) |
| M08 | 0.891 | 0.903 | 0.902 | 0.915 | 0.892 | 0.926 | 0.898 | 0.909 | 0.893 | 0.9 | 0.891 (f1) | 0.926 (f6) |
| M01 | 0.904 | 0.887 | 0.886 | 0.901 | 0.897 | 0.937 | 0.874 | 0.904 | 0.887 | 0.937 | 0.874 (f7) | 0.937 (f6) |
| M06 | 0.887 | 0.902 | 0.885 | 0.904 | 0.903 | 0.907 | 0.903 | 0.893 | 0.903 | 0.924 | 0.885 (f3) | 0.924 (f10) |
| M03 | 0.892 | 0.899 | 0.884 | 0.902 | 0.885 | 0.88 | 0.879 | 0.883 | 0.888 | 0.908 | 0.879 (f7) | 0.908 (f10) |
| M10 | 0.858 | 0.898 | 0.869 | 0.889 | 0.891 | 0.877 | 0.89 | 0.871 | 0.901 | 0.909 | 0.858 (f1) | 0.909 (f10) |
| M05 | 0.855 | 0.881 | 0.877 | 0.901 | 0.896 | 0.873 | 0.869 | 0.887 | 0.868 | 0.906 | 0.855 (f1) | 0.906 (f10) |
| M09 | 0.885 | 0.891 | 0.833 | 0.899 | 0.893 | 0.834 | 0.867 | 0.892 | 0.849 | 0.909 | 0.833 (f3) | 0.909 (f10) |
| M02 | 0.883 | 0.885 | 0.869 | 0.899 | 0.743 | 0.873 | 0.868 | 0.872 | 0.866 | 0.908 | 0.743 (f5) | 0.908 (f10) |

Mean over the 10 models per fold: f1 0.890, f2 0.901, f3 0.881, f4 0.905, f5 0.889, f6 0.897, f7 0.890, f8 0.897, f9 0.891, f10 0.918.

## 5. Flags

- folds with test ROC-AUC < 0.5: 0; models with mean ROC-AUC < 0.55: 0
- runs stopping at epoch <= 3: 0 (minimum possible with patience 8 is 9 epochs)
- runs hitting max_epochs 40: 19 of 100 (per model: {'M01': 0, 'M02': 7, 'M03': 0, 'M04': 1, 'M05': 5, 'M06': 0, 'M07': 0, 'M08': 1, 'M09': 5, 'M10': 0}); 17 of them are custom-CNN runs (M02, M05, M09, trained from scratch); 17 of the 19 ended without the early-stopping criterion being met (validation ROC-AUC still improved within the last 8 epochs), so those runs may not have converged under the fixed 40-epoch budget; the others early-stopped exactly at epoch 40
- runs whose best validation epoch was <= 3: 6 of 100 (per model: {'M01': 0, 'M02': 0, 'M03': 0, 'M04': 2, 'M05': 0, 'M06': 1, 'M07': 0, 'M08': 3, 'M09': 0, 'M10': 0})
- folds predicting a single class at threshold 0.5: 0
- folds predicting a single class at the validation threshold: 0
- validation (Youden) thresholds outside [0.05, 0.95]: 8 of 100 folds (per model: {'M01': 1, 'M02': 0, 'M03': 0, 'M04': 4, 'M05': 0, 'M06': 0, 'M07': 2, 'M08': 1, 'M09': 0, 'M10': 0}; overall range 0.001-0.999): validation probabilities of these runs are pushed towards 0/1, so the chosen threshold sits in a sparse tail and can move a lot between folds
- validate_model_results flags (e.g. undefined AUC): 0
- resumed runs: 0; queue launches: 1

## 6. Original dataset split (descriptive only)

The DFDK folder ships its own train/valid/test sub-folders; they were pooled in Stage 2 (DD-05) and kept only as `original_split`. Below, the out-of-fold test predictions of each model are split by that column: 300 images from the original 'test' folder vs the other 2640 (original train + valid). No model was fitted or selected on this; it is a description of where errors fall. Per-fold values on the original-'test' subset rest on about 30 images per fold and are very coarse (`logs/stage06_original_split_auc.csv`).

| model | orig 'test' pooled AUC | orig 'test' per-fold mean ± std | orig train+valid pooled AUC | orig train+valid per-fold mean ± std | all pooled AUC |
|---|---|---|---|---|---|
| M04 | 0.941 | 0.968 ± 0.023 | 0.89 | 0.927 ± 0.016 | 0.896 |
| M07 | 0.923 | 0.958 ± 0.033 | 0.895 | 0.921 ± 0.016 | 0.898 |
| M08 | 0.908 | 0.947 ± 0.038 | 0.867 | 0.898 ± 0.013 | 0.871 |
| M01 | 0.899 | 0.929 ± 0.056 | 0.88 | 0.899 ± 0.022 | 0.882 |
| M06 | 0.888 | 0.918 ± 0.042 | 0.885 | 0.900 ± 0.013 | 0.885 |
| M03 | 0.907 | 0.927 ± 0.037 | 0.879 | 0.888 ± 0.010 | 0.882 |
| M10 | 0.91 | 0.921 ± 0.032 | 0.87 | 0.882 ± 0.019 | 0.875 |
| M05 | 0.898 | 0.909 ± 0.063 | 0.871 | 0.879 ± 0.017 | 0.874 |
| M09 | 0.904 | 0.921 ± 0.046 | 0.863 | 0.871 ± 0.028 | 0.867 |
| M02 | 0.904 | 0.910 ± 0.053 | 0.864 | 0.861 ± 0.047 | 0.869 |

Pooled AUC on the original 'test' images minus pooled AUC on the rest: +0.003 to +0.050 over the 10 models (higher on the original 'test' images for 10 of 10).

## 7. Near-duplicate sanity check (`logs/stage06_verify.txt`, 1e)

- Groups (Stage 2 rule: pHash Hamming <= 4 -> same group): 0 groups span more than one outer fold; no md5 (byte-identical) duplicates exist.
- Recomputed from manifest.csv over all 4,320,330 image pairs: Hamming 0-4: 11 pairs, 0 across folds (rule holds). Below the rule: Hamming 5-8: 38 pairs, 34 across folds; Hamming 9-10: 146 pairs, 128 across folds. These pairs were NOT grouped by the frozen Stage 2 rule (DD-07) and are reported, not acted on; at most 68 of 2,940 images are in a cross-fold pair at Hamming 5-8. Whether any of them are the same child (re-crops / edits) is UNKNOWN without subject IDs; Stage 8 may add a sensitivity check that excludes them.

## 8. Limitations

These are image-level results. D02 has no subject identifiers, so the same child may appear in several images and in both training and test folds; grouping by pHash removes only near-identical copies, not different photos of one child. The images were collected from the web (DFDK dataset), with unknown diagnosis verification, age range, consent and acquisition conditions, and the two classes may differ in source, background, lighting or image processing, which a model can exploit instead of facial phenotype. The results therefore may not generalise to clinical settings or to other populations and must not be read as diagnostic accuracy. No result here is a diagnosis.

## 9. Verification (`logs/stage06_verify.txt`)

```
1a queue: last run_stage6.sh end line ('2026-10-14T14:31:58+05:30', '0'); 'QUEUE DONE D02_facial_images_A' in log: True; PID 225660 alive: False; pid file renamed to logs/stage6.pid.finished: True
1b registry: 100 D02 rows, status counts {'DONE': 100}, models ['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'], folds per model [10]
1b last.pt files left under results/D02_facial_images_A: 0
1c split hashes vs SPLITS_MANIFEST.json: OK
1c fold  1: 294 test ids; all 10 models predicted exactly these ids: True; fold-file sha256 recorded by the 10 runs identical: True; test_ids_sha256 identical: True
1c fold  2: 294 test ids; all 10 models predicted exactly these ids: True; fold-file sha256 recorded by the 10 runs identical: True; test_ids_sha256 identical: True
1c fold  3: 294 test ids; all 10 models predicted exactly these ids: True; fold-file sha256 recorded by the 10 runs identical: True; test_ids_sha256 identical: True
1c fold  4: 294 test ids; all 10 models predicted exactly these ids: True; fold-file sha256 recorded by the 10 runs identical: True; test_ids_sha256 identical: True
1c fold  5: 294 test ids; all 10 models predicted exactly these ids: True; fold-file sha256 recorded by the 10 runs identical: True; test_ids_sha256 identical: True
1c fold  6: 294 test ids; all 10 models predicted exactly these ids: True; fold-file sha256 recorded by the 10 runs identical: True; test_ids_sha256 identical: True
1c fold  7: 294 test ids; all 10 models predicted exactly these ids: True; fold-file sha256 recorded by the 10 runs identical: True; test_ids_sha256 identical: True
1c fold  8: 294 test ids; all 10 models predicted exactly these ids: True; fold-file sha256 recorded by the 10 runs identical: True; test_ids_sha256 identical: True
1c fold  9: 294 test ids; all 10 models predicted exactly these ids: True; fold-file sha256 recorded by the 10 runs identical: True; test_ids_sha256 identical: True
1c fold 10: 294 test ids; all 10 models predicted exactly these ids: True; fold-file sha256 recorded by the 10 runs identical: True; test_ids_sha256 identical: True
1c identical test ids per fold across models: True
1d validate_model_results M01: PASS; reasons 0; flags 0
1d validate_model_results M02: PASS; reasons 0; flags 0
1d validate_model_results M03: PASS; reasons 0; flags 0
1d validate_model_results M04: PASS; reasons 0; flags 0
1d validate_model_results M05: PASS; reasons 0; flags 0
1d validate_model_results M06: PASS; reasons 0; flags 0
1d validate_model_results M07: PASS; reasons 0; flags 0
1d validate_model_results M08: PASS; reasons 0; flags 0
1d validate_model_results M09: PASS; reasons 0; flags 0
1d validate_model_results M10: PASS; reasons 0; flags 0
1e groups: 2929 groups (11 with > 1 image); groups spanning > 1 outer fold: 0
1e byte-identical images (same md5): 0 md5 values with > 1 image; spanning > 1 outer fold: 0
1e pHash pairs recomputed from manifest.csv over all 4,320,330 image pairs (group rule: Hamming <= 4):
     Hamming 0-4: 11 pairs, of which in different outer folds: 0
     Hamming 5-8: 38 pairs, of which in different outer folds: 34
     Hamming 9-10: 146 pairs, of which in different outer folds: 128
2 summarize: summary.json for 10 models + results/D02_facial_images_A/leaderboard.csv written
3 original-split subsets: logs/stage06_original_split_auc.csv written (10 models; original 'test' images per model [300], rest [2640])
D02 result files (metrics/predictions/run_info/summary/leaderboard, 311 files) unchanged by 06b_verify.py (summaries re-written byte-identically)
```

## 10. Tests (`.venv/bin/python -m pytest -v`, `logs/stage06_pytest.txt`)

Run after the provisional stage_06 snapshot. `tests/test_stage02_snapshot.py` now picks the latest snapshot by `SNAPSHOT_INFO.json` datetime instead of by folder name: stage_07 (made early, before S6) sorts after stage_06 by name, so the name rule would keep checking the outdated stage_07 snapshot. No other test changed.

Result: 331 passed in 126.81s (0:02:06)

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

======================= 331 passed in 126.81s (0:02:06) ========================
```

</details>

## 11. Neutral description

- The ten models reach mean test ROC-AUCs between 0.867 and 0.931 with fold-to-fold standard deviations of 0.010-0.046 (sections 3, 4).
- The three custom CNNs trained from scratch have mean ROC-AUCs 0.867-0.881; the seven pretrained models 0.885-0.931. 17 of the 30 custom-CNN runs reached the 40-epoch limit (section 5).
- Pooled OOF ROC-AUC (all 2,940 predictions in one ranking) is below the mean per-fold ROC-AUC for 9 of 10 models (differences -0.036 to +0.002); the ten fold models do not share one probability scale, consistent with the spread of validation thresholds (section 5).
- Fold 3 has the lowest mean AUC over models (0.881); fold effects are shared across models (section 4).
- Differences between models have not been tested statistically; bootstrap CIs and Friedman/Nemenyi follow in Stage 8.
- No result is a diagnosis or a clinical accuracy estimate.

## 12. Open items

1. Stage 8: shortcut controls (D01 quality-only baseline; D02 near-duplicate / original-split sensitivity), bootstrap CIs, Friedman/Nemenyi.
2. Stage 9 figures and the final results report; Stage 10 final audit.

STAGE 6 DECISION: PASS
