# STAGE 8 REPORT - Shortcut controls and statistics

Screening-support research, not diagnosis. Stage 8 trains NO deep model: every analysis reads the saved out-of-fold predictions / metrics of Stages 5-7 (results/D0*/ read-only, verified byte-identical before and after, section 9) plus the Stage 2 caches. New code: `src/analysis/` (stats.py, quality.py, image_stats.py, data.py), `scripts/08a_preflight.py`, `08b_d01_quality.py`, `08c_d02_shortcuts.py`, `08d_statistics.py`, `08e_figures.py`, `08f_write_report.py`; outputs `results/stage8/`, `reports/figures/stage08_*.png`; logs `logs/stage08_*.txt`. CPU only (CUDA_VISIBLE_DEVICES=""). Bootstrap: 2,000 resamples, seed 42, stratified by class at the level of the resampling unit (child / near-duplicate group / duplicate-feature group), the same resamples for every model of a dataset (paired). Pooled AUC = one AUC over all out-of-fold predictions.

## 1. Summary

- **D01**: the tracking-quality-only baseline (logistic regression on 3 per-child quality features, fit per fold on training children only) reaches child-level pooled AUC 0.823 (1 - mean valid_ratio alone: 0.842); the deep models reach 0.574-0.832. No model is significantly better than the quality baseline (all lower CI bounds of the difference < 0); 4 are significantly worse (M02, M05, M09, M10). After 1:1 matching on valid_ratio (14 pairs) only M07 keeps a bootstrap CI above 0.5. By the pre-specified rule (section 3.6): adds information beyond tracking quality: M07; not distinguishable from the shortcut: M01, M02, M03, M04, M05, M06, M08, M09, M10.
- **D02**: an image-statistics baseline without any face model reaches AUC 0.796 ± 0.019 (image_stats_random_forest); **dataset artefacts alone separate the classes** (>= 0.75). Removing the 65 test images with a pHash <= 8 training neighbour changes model AUCs by -0.002 to -0.001.
- **Friedman** (10 deep models x 10 folds): D01 clip p = 0.012, D01 child p = 0.566, D02 p = 3.7e-10.
- Top model per dataset with 95% CI: section 5.1.

## 2. Pre-flight (A, `logs/stage08_preflight.txt`)

```
A1 reports/STAGE_05_REPORT.md last line: STAGE 5 DECISION: PASS
A1 reports/STAGE_06_REPORT.md last line: STAGE 6 DECISION: PASS
A1 reports/STAGE_07_REPORT.md last line: STAGE 7 DECISION: PASS
A2 snapshot stage_05 (2026-10-14T10:57:13): 425 recorded lines; CHECKSUMS file matches its code_hash: True; splits/cache/results files differing: 0; code/report files added or changed since (expected for later stages): 8 ['reports/STAGE_06_REPORT.md', 'scripts/06b_verify.py', 'scripts/06b_write_report.py', 'scripts/08a_preflight.py', 'scripts/stage6_launch.sh', 'scripts/stage6_status.py', 'scripts/stage6_stop.sh', 'tests/test_stage02_snapshot.py']
A2 snapshot stage_06 (2026-10-14T15:51:44): 1075 recorded lines; CHECKSUMS file matches its code_hash: True; splits/cache/results files differing: 0; code/report files added or changed since (expected for later stages): 1 ['scripts/08a_preflight.py']
A2 snapshot stage_07 (2026-10-14T10:57:15): 647 recorded lines; CHECKSUMS file matches its code_hash: True; splits/cache/results files differing: 0; code/report files added or changed since (expected for later stages): 8 ['reports/STAGE_06_REPORT.md', 'scripts/06b_verify.py', 'scripts/06b_write_report.py', 'scripts/08a_preflight.py', 'scripts/stage6_launch.sh', 'scripts/stage6_status.py', 'scripts/stage6_stop.sh', 'tests/test_stage02_snapshot.py']
A3 split hashes vs SPLITS_MANIFEST.json: OK
A4 recorded sha256 of 2407 files under results/D0*/ (no *.pt), incl. 800 predictions.csv / metrics.json -> results/stage8/preflight_results_manifest.sha256
A PRE-FLIGHT: OK
```

The snapshot comparison separates data/results files (splits, cache, hashed result files: all identical) from code/report files added after each snapshot (expected: later stages add scripts and reports).

## 3. B - D01 eye-tracking: tracking-quality shortcut (DD-12)

Quality features (`cache/D01_eye_tracking/quality_per_child.csv`, never model inputs): mean_valid_ratio, mean_blink_ratio, tracking_ratio_before_filtering (= overall tracking ratio). 57 children (27 ASD / 30 TD), 1,241 clips. Child score of a deep model = mean of its out-of-fold clip probabilities.

### 3.1 Quality-only baselines on the same 10 outer folds

| baseline | level | description | n | pooled AUC | per-fold AUC mean ± std |
|---|---|---|---|---|---|
| quality_LR3 | child | LR on mean_valid_ratio + mean_blink_ratio + tracking_ratio_before_filtering (fit per fold) | 57 | 0.823 | 0.850 ± 0.216 |
| quality_LR_valid_ratio | child | LR on mean_valid_ratio (fit per fold) | 57 | 0.828 | 0.850 ± 0.216 |
| one_minus_valid_ratio | child | 1 - mean valid_ratio (no fitting) | 57 | 0.842 | 0.850 ± 0.216 |
| clip_one_minus_valid_ratio | clip | each clip: 1 - its own valid_ratio (no fitting) | 1241 | 0.678 | 0.670 ± 0.131 |
| clip_LR_valid_blink | clip | LR on clip valid_ratio + blink_ratio (fit per fold on training clips) | 1241 | 0.651 | 0.665 ± 0.140 |

Fitted baselines are fit on the fold's training (train + val) children only and predict its test children (`results/stage8/d01_quality_baseline_folds.csv` lists both sets per fold; checked by tests). 1 - mean valid_ratio needs no fitting, but its direction was noticed in Stage 5 after looking at the data, so it is optimistic; the fitted LR on valid_ratio learns the direction per fold. Per-fold child AUCs rest on 5-6 test children; the three child baselines rank the test children identically in every fold, hence identical per-fold mean ± std. The primary comparator below is quality_LR3 (fitted, no look-ahead).

### 3.2 Model vs quality-only, child level (paired bootstrap over children + DeLong)

| model | child AUC [95% CI] | diff vs quality LR3 [95% CI] | DeLong p (LR3) | diff vs 1 - valid_ratio [95% CI] | DeLong p (1-vr) |
|---|---|---|---|---|---|
| M01 | 0.714 [0.569, 0.840] | -0.110 [-0.265, +0.046] | 0.161 | -0.128 [-0.281, +0.017] | 0.087 |
| M02 | 0.578 [0.425, 0.731] | -0.246 [-0.417, -0.084] | 0.003 | -0.264 [-0.435, -0.110] | 0.001 |
| M03 | 0.735 [0.604, 0.852] | -0.089 [-0.233, +0.054] | 0.230 | -0.107 [-0.242, +0.022] | 0.120 |
| M04 | 0.762 [0.628, 0.883] | -0.062 [-0.196, +0.069] | 0.351 | -0.080 [-0.210, +0.048] | 0.216 |
| M05 | 0.605 [0.443, 0.756] | -0.219 [-0.398, -0.058] | 0.008 | -0.237 [-0.416, -0.078] | 0.004 |
| M06 | 0.784 [0.656, 0.895] | -0.040 [-0.151, +0.065] | 0.478 | -0.058 [-0.164, +0.036] | 0.246 |
| M07 | 0.832 [0.722, 0.927] | +0.009 [-0.098, +0.120] | 0.876 | -0.010 [-0.111, +0.101] | 0.854 |
| M08 | 0.786 [0.658, 0.895] | -0.037 [-0.174, +0.107] | 0.590 | -0.056 [-0.188, +0.079] | 0.400 |
| M09 | 0.574 [0.427, 0.725] | -0.249 [-0.410, -0.090] | 0.002 | -0.268 [-0.425, -0.107] | 0.001 |
| M10 | 0.673 [0.521, 0.809] | -0.151 [-0.295, -0.007] | 0.036 | -0.169 [-0.315, -0.028] | 0.018 |

### 3.3 Dependence of model child scores on mean valid_ratio (Spearman)

| model | rho all (n=57) | rho ASD (n=27) | rho TD (n=30) |
|---|---|---|---|
| M01 | -0.44 (p 0.00057) | -0.48 (p 0.012) | -0.09 (p 0.646) |
| M02 | -0.30 (p 0.025) | -0.42 (p 0.029) | -0.07 (p 0.723) |
| M03 | -0.54 (p 1.4e-05) | -0.50 (p 0.008) | -0.32 (p 0.089) |
| M04 | -0.56 (p 7.3e-06) | -0.53 (p 0.004) | -0.15 (p 0.431) |
| M05 | -0.32 (p 0.017) | -0.38 (p 0.053) | -0.16 (p 0.412) |
| M06 | -0.71 (p 9.2e-10) | -0.71 (p 3.5e-05) | -0.36 (p 0.048) |
| M07 | -0.67 (p 1e-08) | -0.66 (p 0.00015) | -0.31 (p 0.092) |
| M08 | -0.59 (p 1.1e-06) | -0.60 (p 0.001) | -0.30 (p 0.113) |
| M09 | -0.27 (p 0.039) | -0.60 (p 0.001) | +0.12 (p 0.545) |
| M10 | -0.46 (p 0.00033) | -0.57 (p 0.002) | -0.12 (p 0.523) |

All 10 models have negative overall correlations (lower tracking quality -> higher P(ASD)); within ASD children rho is -0.71 to -0.38 (p < 0.05 for 9 of 10), within TD children -0.36 to +0.12 (p < 0.05 for 1 of 10).

### 3.4 Quality-controlled performance

(a) AUC within quality tertiles. Clip tertile cut points of clip valid_ratio: 0.905, 0.977; child cut points of mean valid_ratio: 0.872, 0.938. Class counts:

| level | tertile | n | n_pos | n_neg |
|---|---|---|---|---|
| clip | low | 414 | 226 | 188 |
| clip | mid | 413 | 123 | 290 |
| clip | high | 414 | 88 | 326 |
| child | low | 19 | 17 | 2 |
| child | mid | 19 | 7 | 12 |
| child | high | 19 | 3 | 16 |

Clip level (pooled out-of-fold clips within the tertile):

| model | low | mid | high |
|---|---|---|---|
| M01 | 0.592 | 0.584 | 0.564 |
| M02 | 0.560 | 0.494 | 0.512 |
| M03 | 0.570 | 0.535 | 0.548 |
| M04 | 0.616 | 0.621 | 0.550 |
| M05 | 0.590 | 0.535 | 0.600 |
| M06 | 0.618 | 0.592 | 0.516 |
| M07 | 0.653 | 0.586 | 0.609 |
| M08 | 0.680 | 0.624 | 0.604 |
| M09 | 0.577 | 0.500 | 0.476 |
| M10 | 0.641 | 0.577 | 0.461 |
| quality: 1 - valid_ratio | 0.606 | 0.512 | 0.516 |

Child level (tertiles are strongly class-imbalanced, e.g. the low tertile has 2 TD children; AUCs on 19 children are very coarse):

| model | low | mid | high |
|---|---|---|---|
| M01 | 0.735 | 0.714 | 0.500 |
| M02 | 0.412 | 0.524 | 0.354 |
| M03 | 0.882 | 0.655 | 0.521 |
| M04 | 0.794 | 0.762 | 0.438 |
| M05 | 0.294 | 0.440 | 0.625 |
| M06 | 0.882 | 0.571 | 0.479 |
| M07 | 0.853 | 0.714 | 0.604 |
| M08 | 0.618 | 0.607 | 0.812 |
| M09 | 0.471 | 0.464 | 0.312 |
| M10 | 0.618 | 0.571 | 0.438 |
| quality: 1 - valid_ratio | 0.735 | 0.560 | 0.458 |
| quality_LR3 | 0.441 | 0.595 | 0.438 |

(b) Quality-matched subset: greedy 1:1 nearest-neighbour matching of ASD to TD children on mean valid_ratio, caliper 0.05, without replacement: **14 pairs** (of 27 ASD / 30 TD; max within-pair difference 0.042). Mean valid_ratio ASD/TD 0.843/0.931 (SMD -1.33) before, 0.903/0.910 (SMD -0.16) after. CI: 2,000 bootstrap resamples of matched pairs, seed 42. The quality-only scores on the matched children are a balance check (residual imbalance).

| model | n_pairs | matched AUC [95% CI] | ci_excludes_0.5_above |
|---|---|---|---|
| M01 | 14 | 0.556 [0.337, 0.765] | False |
| M02 | 14 | 0.469 [0.281, 0.663] | False |
| M03 | 14 | 0.587 [0.362, 0.796] | False |
| M04 | 14 | 0.592 [0.337, 0.816] | False |
| M05 | 14 | 0.515 [0.311, 0.709] | False |
| M06 | 14 | 0.597 [0.367, 0.781] | False |
| M07 | 14 | 0.724 [0.551, 0.883] | True |
| M08 | 14 | 0.679 [0.469, 0.842] | False |
| M09 | 14 | 0.536 [0.316, 0.755] | False |
| M10 | 14 | 0.551 [0.347, 0.750] | False |
| quality: 1 - valid_ratio | 14 | 0.571 [0.454, 0.684] | False |
| quality_LR3 | 14 | 0.561 [0.449, 0.684] | False |

### 3.5 Incremental value (EXPLORATORY)

Per outer fold: logistic regression on the training children with [3 quality features] vs [3 quality features + the model's child score], evaluated on the fold's test children. Scores are out-of-fold, but the score of a TRAINING child of fold k comes from the model of that child's own fold, which was trained on data that included fold k's test children: a mild, indirect information path, acceptable only for an exploratory analysis.

| model | pooled AUC quality | pooled AUC quality + score | pooled diff | per-fold diff mean ± std | folds better / worse |
|---|---|---|---|---|---|
| M01 | 0.823 | 0.823 | +0.000 | +0.006 ± 0.076 | 2 / 1 |
| M02 | 0.823 | 0.814 | -0.010 | +0.000 ± 0.000 | 0 / 0 |
| M03 | 0.823 | 0.812 | -0.011 | -0.006 ± 0.067 | 1 / 1 |
| M04 | 0.823 | 0.825 | +0.001 | +0.011 ± 0.035 | 1 / 0 |
| M05 | 0.823 | 0.817 | -0.006 | +0.000 ± 0.000 | 0 / 0 |
| M06 | 0.823 | 0.820 | -0.004 | +0.011 ± 0.035 | 1 / 0 |
| M07 | 0.823 | 0.836 | +0.012 | -0.006 ± 0.067 | 1 / 1 |
| M08 | 0.823 | 0.821 | -0.002 | +0.006 ± 0.076 | 2 / 1 |
| M09 | 0.823 | 0.820 | -0.004 | +0.000 ± 0.000 | 0 / 0 |
| M10 | 0.823 | 0.807 | -0.016 | +0.000 ± 0.000 | 0 / 0 |

### 3.6 Interpretation rules (written before applying them) and their application

A model **adds information beyond tracking quality** only if (rule 1) the 95% CI of its child-level AUC difference vs the quality-only baseline (quality_LR3, section 3.2) lies entirely above 0, OR (rule 2) its matched-subset AUC (section 3.4b) has a bootstrap 95% CI entirely above 0.5. Otherwise its D01 results are **not distinguishable from the tracking-quality shortcut**. No adjustment for testing 10 models is part of the rule; this is noted in section 7.

| model | rule 1 | rule 2 | adds information | lower CI vs 1-valid_ratio |
|---|---|---|---|---|
| M01 | False | False | False | -0.281 |
| M02 | False | False | False | -0.435 |
| M03 | False | False | False | -0.242 |
| M04 | False | False | False | -0.21 |
| M05 | False | False | False | -0.416 |
| M06 | False | False | False | -0.164 |
| M07 | False | True | True | -0.111 |
| M08 | False | False | False | -0.188 |
| M09 | False | False | False | -0.425 |
| M10 | False | False | False | -0.315 |

Result: rule 1 holds for 0 models; rule 2 holds for 1 (M07).

## 4. C - D02 facial images: dataset shortcuts

### 4.1 Near-duplicate sensitivity (no retraining)

For each fold, test images whose nearest pHash neighbour among that fold's training images (train + val = all images the fold's model could use) is at Hamming distance <= 8: 65 of 2,940 (per fold [8, 5, 8, 6, 7, 5, 9, 5, 6, 6]; at <= 4: 0, as required by the Stage 2 groups). The nearest neighbour has the same label for 55 of 65.

| model | n_removed | fold-mean AUC all | fold-mean AUC without near-dups | diff (fold mean) | pooled AUC all | pooled AUC without | AUC on the removed images only |
|---|---|---|---|---|---|---|---|
| M01 | 65 | 0.901 | 0.899 | -0.002 | 0.882 | 0.880 | 0.977 |
| M02 | 65 | 0.867 | 0.865 | -0.002 | 0.869 | 0.867 | 0.956 |
| M03 | 65 | 0.890 | 0.889 | -0.002 | 0.882 | 0.881 | 0.936 |
| M04 | 65 | 0.931 | 0.929 | -0.002 | 0.896 | 0.894 | 0.957 |
| M05 | 65 | 0.881 | 0.879 | -0.002 | 0.874 | 0.872 | 0.973 |
| M06 | 65 | 0.901 | 0.899 | -0.002 | 0.885 | 0.883 | 0.977 |
| M07 | 65 | 0.924 | 0.923 | -0.001 | 0.898 | 0.897 | 0.950 |
| M08 | 65 | 0.903 | 0.901 | -0.002 | 0.871 | 0.870 | 0.919 |
| M09 | 65 | 0.875 | 0.874 | -0.002 | 0.867 | 0.865 | 0.957 |
| M10 | 65 | 0.885 | 0.884 | -0.002 | 0.875 | 0.873 | 0.930 |

### 4.2 Image-statistics baseline (no face model)

17 non-facial features per image from the raw files (read-only): width, height, aspect ratio, file size, JPEG quality estimate (IJG scale from the luminance quantisation table; all 2,940 files are JPEG), mean/std of R, G, B, brightness mean/std, mean R/G/B of the outer 10% border, sharpness = log(1 + variance of the Laplacian). Logistic regression (standardised) and random forest with fixed parameters, trained on each fold's train + val images, tested on its test images (`results/stage8/d02_image_stats.csv`, `d02_image_stats_oof.csv`).

| baseline | AUC mean ± std (10 folds) | pooled AUC | fold_aucs | params |
|---|---|---|---|---|
| image_stats_logreg | 0.782 ± 0.026 | 0.781 | 0.740 0.804 0.763 0.818 0.747 0.770 0.792 0.793 0.797 0.796 | {'C': 1.0, 'max_iter': 5000} |
| image_stats_random_forest | 0.796 ± 0.019 | 0.797 | 0.787 0.797 0.787 0.797 0.776 0.820 0.760 0.811 0.821 0.802 | {'n_estimators': 500, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'random_state': 42, 'n_jobs': 2} |

**The image-statistics baseline reaches >= 0.75: dataset artefacts alone (colour, brightness, image size/compression, sharpness) separate the two classes to a large degree, without looking at facial features.**

Single features (no fitting; signed AUC > 0.5 = higher in autistic images):

| feature | AUC (signed) | direction-free AUC | mean autistic | mean non-autistic |
|---|---|---|---|---|
| mean_r | 0.339 | 0.661 | 154.35 | 167.64 |
| std_r | 0.641 | 0.641 | 63.37 | 57.34 |
| std_g | 0.634 | 0.634 | 59.06 | 54.16 |
| brightness_mean | 0.372 | 0.628 | 134.48 | 143.66 |
| brightness_std | 0.628 | 0.628 | 58.80 | 54.10 |
| std_b | 0.627 | 0.627 | 59.09 | 54.12 |
| mean_g | 0.393 | 0.607 | 128.68 | 136.44 |
| border_r | 0.397 | 0.603 | 128.78 | 140.39 |
| mean_b | 0.433 | 0.567 | 112.20 | 117.87 |
| width | 0.564 | 0.564 | 332.98 | 311.83 |
| height | 0.560 | 0.560 | 384.79 | 354.36 |
| file_size_kb | 0.555 | 0.555 | 42.83 | 38.78 |
| border_g | 0.473 | 0.527 | 114.70 | 117.38 |
| sharpness_log_lapvar | 0.487 | 0.513 | 5.41 | 5.49 |
| border_b | 0.508 | 0.508 | 101.61 | 100.52 |
| aspect_ratio | 0.495 | 0.505 | 0.87 | 0.88 |
| jpeg_quality | 0.500 | 0.500 | 95.00 | 95.00 |

### 4.3 Original dataset split

AUC of each model's out-of-fold predictions on the images from the original 'test' folder vs the rest (original train + valid); CIs from 2,000 stratified resamples of near-duplicate groups (seed 42), the difference computed within each resample.

| model | n_orig_test | n_rest | AUC orig 'test' [95% CI] | AUC orig train+valid [95% CI] | difference [95% CI] |
|---|---|---|---|---|---|
| M01 | 300 | 2640 | 0.899 [0.861, 0.932] | 0.880 [0.867, 0.893] | +0.020 [-0.022, +0.054] |
| M02 | 300 | 2640 | 0.904 [0.867, 0.938] | 0.864 [0.851, 0.878] | +0.040 [+0.001, +0.075] |
| M03 | 300 | 2640 | 0.907 [0.872, 0.938] | 0.879 [0.866, 0.892] | +0.028 [-0.009, +0.061] |
| M04 | 300 | 2640 | 0.941 [0.910, 0.967] | 0.890 [0.877, 0.903] | +0.050 [+0.018, +0.079] |
| M05 | 300 | 2640 | 0.898 [0.861, 0.931] | 0.871 [0.858, 0.885] | +0.027 [-0.012, +0.063] |
| M06 | 300 | 2640 | 0.888 [0.846, 0.924] | 0.885 [0.872, 0.898] | +0.003 [-0.040, +0.042] |
| M07 | 300 | 2640 | 0.923 [0.890, 0.952] | 0.895 [0.883, 0.907] | +0.028 [-0.007, +0.058] |
| M08 | 300 | 2640 | 0.908 [0.872, 0.942] | 0.867 [0.852, 0.882] | +0.041 [+0.003, +0.076] |
| M09 | 300 | 2640 | 0.904 [0.868, 0.936] | 0.863 [0.850, 0.877] | +0.041 [+0.002, +0.074] |
| M10 | 300 | 2640 | 0.910 [0.874, 0.943] | 0.870 [0.857, 0.884] | +0.040 [+0.003, +0.074] |

The CI of the difference lies above 0 for 5 of 10 models (M02, M04, M08, M09, M10); descriptive only.

## 5. D - Statistics for all datasets

### 5.1 Pooled out-of-fold AUC with 95% bootstrap CIs

| dataset | level | top model (pooled OOF AUC) | pooled AUC [95% CI] | resampling unit |
|---|---|---|---|---|
| D01_eye_tracking | clip | M08 | 0.689 [0.594, 0.781] | child |
| D01_eye_tracking | child | M07 | 0.832 [0.722, 0.927] | child |
| D02_facial_images_A | image | M07 | 0.898 [0.887, 0.909] | near-duplicate group |
| D04_autism_prediction/set_all | row | mlp | 0.905 [0.882, 0.926] | duplicate-feature group |
| D05_toddler_screening/set_all | row | logreg | 1.000 [1.000, 1.000] | duplicate-feature group |
| D05_toddler_screening/set_honest | row | hist_gb | 0.610 [0.570, 0.652] | duplicate-feature group |
| D06_early_autism/set_all | row | logreg | 0.482 [0.446, 0.519] | duplicate-feature group |

D01 clip level (resampling children):

| model | pooled AUC [95% CI] | n | n_groups | resampling_unit | stratified |
|---|---|---|---|---|---|
| M08 | 0.689 [0.594, 0.781] | 1241 | 57 | child | True |
| M07 | 0.675 [0.610, 0.745] | 1241 | 57 | child | True |
| M04 | 0.657 [0.560, 0.747] | 1241 | 57 | child | True |
| M06 | 0.652 [0.576, 0.728] | 1241 | 57 | child | True |
| M01 | 0.634 [0.544, 0.717] | 1241 | 57 | child | True |
| M10 | 0.628 [0.509, 0.746] | 1241 | 57 | child | True |
| M03 | 0.625 [0.542, 0.707] | 1241 | 57 | child | True |
| M05 | 0.610 [0.456, 0.748] | 1241 | 57 | child | True |
| M02 | 0.570 [0.426, 0.700] | 1241 | 57 | child | True |
| M09 | 0.541 [0.398, 0.693] | 1241 | 57 | child | True |

D01 child level (resampling children; quality baselines included for reference):

| model | pooled AUC [95% CI] | n | n_groups | resampling_unit | stratified |
|---|---|---|---|---|---|
| baseline: 1 - valid_ratio | 0.842 [0.727, 0.938] | 57 | 57 | child | True |
| M07 | 0.832 [0.722, 0.927] | 57 | 57 | child | True |
| baseline: quality_LR3 | 0.823 [0.701, 0.928] | 57 | 57 | child | True |
| M08 | 0.786 [0.658, 0.895] | 57 | 57 | child | True |
| M06 | 0.784 [0.656, 0.895] | 57 | 57 | child | True |
| M04 | 0.762 [0.628, 0.883] | 57 | 57 | child | True |
| M03 | 0.735 [0.604, 0.852] | 57 | 57 | child | True |
| M01 | 0.714 [0.569, 0.840] | 57 | 57 | child | True |
| M10 | 0.673 [0.521, 0.809] | 57 | 57 | child | True |
| M05 | 0.605 [0.443, 0.756] | 57 | 57 | child | True |
| M02 | 0.578 [0.425, 0.731] | 57 | 57 | child | True |
| M09 | 0.574 [0.427, 0.725] | 57 | 57 | child | True |

D02 image level (resampling near-duplicate groups; image-statistics baselines included):

| model | pooled AUC [95% CI] | n | n_groups | resampling_unit | stratified |
|---|---|---|---|---|---|
| M07 | 0.898 [0.887, 0.909] | 2940 | 2929 | near-duplicate group | True |
| M04 | 0.896 [0.884, 0.908] | 2940 | 2929 | near-duplicate group | True |
| M06 | 0.885 [0.873, 0.897] | 2940 | 2929 | near-duplicate group | True |
| M03 | 0.882 [0.870, 0.894] | 2940 | 2929 | near-duplicate group | True |
| M01 | 0.882 [0.870, 0.894] | 2940 | 2929 | near-duplicate group | True |
| M10 | 0.875 [0.862, 0.887] | 2940 | 2929 | near-duplicate group | True |
| M05 | 0.874 [0.862, 0.887] | 2940 | 2929 | near-duplicate group | True |
| M08 | 0.871 [0.857, 0.884] | 2940 | 2929 | near-duplicate group | True |
| M02 | 0.869 [0.855, 0.882] | 2940 | 2929 | near-duplicate group | True |
| M09 | 0.867 [0.855, 0.880] | 2940 | 2929 | near-duplicate group | True |
| baseline: image_stats_random_forest | 0.797 [0.781, 0.812] | 2940 | 2929 | near-duplicate group | True |
| baseline: image_stats_logreg | 0.781 [0.765, 0.798] | 2940 | 2929 | near-duplicate group | True |

Tabular (Stage 7 models; resampling duplicate-feature groups, stratified where groups are label-pure):

| model | D04_autism_prediction/set_all | D05_toddler_screening/set_all | D05_toddler_screening/set_honest | D06_early_autism/set_all |
|---|---|---|---|---|
| dummy_majority | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] |
| hist_gb | 0.880 [0.852, 0.905] | 1.000 [1.000, 1.000] | 0.610 [0.570, 0.652] | 0.460 [0.422, 0.494] |
| logreg | 0.899 [0.872, 0.921] | 1.000 [1.000, 1.000] | 0.591 [0.544, 0.637] | 0.482 [0.446, 0.519] |
| mlp | 0.905 [0.882, 0.926] | 0.999 [0.998, 1.000] | 0.587 [0.546, 0.627] | 0.468 [0.432, 0.504] |
| random_forest | 0.901 [0.875, 0.922] | 1.000 [1.000, 1.000] | 0.607 [0.564, 0.650] | 0.466 [0.429, 0.502] |

D05 set_all is the label-definitional upper bound (label = A1-A10 sum rule; Stage 7 experiment a), not a result.

### 5.2 Model comparison across the 10 deep models

Friedman test on the 10 folds x 10 models matrix of per-fold test ROC-AUC (from metrics.json; rank 1 = best per fold), Nemenyi post-hoc (alpha 0.05, studentized range, infinite df), Nadeau-Bengio corrected resampled t-test of the model with the highest mean fold AUC vs each other model (df 9; correction 1/10 + n_test/n_train with the mean fold sizes) with Holm adjustment over the 9 comparisons.

| analysis | metric | Friedman chi2 | p | Iman-Davenport F (p) | Nemenyi CD | significant Nemenyi pairs (of 45) | best avg rank |
|---|---|---|---|---|---|---|---|
| D01_child | test_child/threshold_0.5/roc_auc per fold | 7.69 | 0.566 | 0.84 (0.581) | 4.28 | 0 | M04 |
| D01_clip | test/threshold_0.5/roc_auc per fold | 21.19 | 0.012 | 2.77 (0.007) | 4.28 | 0 | M07 |
| D02_image | test/threshold_0.5/roc_auc per fold | 62.92 | 3.7e-10 | 20.92 (9.7e-18) | 4.28 | 11 | M04 |

Average ranks:

| model | D01_child | D01_clip | D02_image |
|---|---|---|---|
| M01 | 5.45 | 5.40 | 4.80 |
| M02 | 6.75 | 7.50 | 8.75 |
| M03 | 6.45 | 6.50 | 6.25 |
| M04 | 4.00 | 3.90 | 1.40 |
| M05 | 5.20 | 4.60 | 7.90 |
| M06 | 5.35 | 6.10 | 4.20 |
| M07 | 4.75 | 3.30 | 2.10 |
| M08 | 5.30 | 4.10 | 4.40 |
| M09 | 5.75 | 7.30 | 7.90 |
| M10 | 6.00 | 6.30 | 7.30 |

- Nemenyi pairs with p < 0.05, D01_child: none
- Nemenyi pairs with p < 0.05, D01_clip: none
- Nemenyi pairs with p < 0.05, D02_image: M02-M04 (p 2.5e-06), M02-M06 (p 0.027), M02-M07 (p 4e-05), M02-M08 (p 0.043), M03-M04 (p 0.013), M04-M05 (p 6.9e-05), M04-M09 (p 6.9e-05), M04-M10 (p 0.00056), M05-M07 (p 0.00077), M07-M09 (p 0.00077), M07-M10 (p 0.005)

| analysis | best | other | mean AUC best | mean AUC other | mean diff | t | p | Holm p |
|---|---|---|---|---|---|---|---|---|
| D01_clip | M08 | M01 | 0.686 | 0.645 | +0.041 | 1.25 | 0.244 | 1.000 |
| D01_clip | M08 | M02 | 0.686 | 0.595 | +0.092 | 1.67 | 0.128 | 1.000 |
| D01_clip | M08 | M03 | 0.686 | 0.635 | +0.051 | 1.27 | 0.235 | 1.000 |
| D01_clip | M08 | M04 | 0.686 | 0.683 | +0.004 | 0.09 | 0.928 | 1.000 |
| D01_clip | M08 | M05 | 0.686 | 0.657 | +0.029 | 0.64 | 0.539 | 1.000 |
| D01_clip | M08 | M06 | 0.686 | 0.639 | +0.047 | 1.29 | 0.231 | 1.000 |
| D01_clip | M08 | M07 | 0.686 | 0.678 | +0.008 | 0.23 | 0.821 | 1.000 |
| D01_clip | M08 | M09 | 0.686 | 0.605 | +0.081 | 1.58 | 0.148 | 1.000 |
| D01_clip | M08 | M10 | 0.686 | 0.638 | +0.048 | 1.12 | 0.290 | 1.000 |
| D01_child | M04 | M01 | 0.856 | 0.761 | +0.094 | 1.03 | 0.330 | 1.000 |
| D01_child | M04 | M02 | 0.856 | 0.678 | +0.178 | 1.34 | 0.214 | 1.000 |
| D01_child | M04 | M03 | 0.856 | 0.744 | +0.111 | 1.75 | 0.115 | 1.000 |
| D01_child | M04 | M05 | 0.856 | 0.767 | +0.089 | 1.04 | 0.324 | 1.000 |
| D01_child | M04 | M06 | 0.856 | 0.783 | +0.072 | 1.09 | 0.302 | 1.000 |
| D01_child | M04 | M07 | 0.856 | 0.800 | +0.056 | 0.69 | 0.509 | 1.000 |
| D01_child | M04 | M08 | 0.856 | 0.828 | +0.028 | 0.24 | 0.814 | 1.000 |
| D01_child | M04 | M09 | 0.856 | 0.706 | +0.150 | 1.22 | 0.253 | 1.000 |
| D01_child | M04 | M10 | 0.856 | 0.761 | +0.094 | 1.39 | 0.199 | 1.000 |
| D02_image | M04 | M01 | 0.931 | 0.901 | +0.030 | 4.15 | 0.003 | 0.010 |
| D02_image | M04 | M02 | 0.931 | 0.867 | +0.065 | 2.65 | 0.027 | 0.053 |
| D02_image | M04 | M03 | 0.931 | 0.890 | +0.041 | 4.94 | 0.0008 | 0.006 |
| D02_image | M04 | M05 | 0.931 | 0.881 | +0.050 | 5.65 | 0.00031 | 0.003 |
| D02_image | M04 | M06 | 0.931 | 0.901 | +0.030 | 4.82 | 0.00095 | 0.006 |
| D02_image | M04 | M07 | 0.931 | 0.924 | +0.007 | 0.86 | 0.413 | 0.413 |
| D02_image | M04 | M08 | 0.931 | 0.903 | +0.028 | 3.53 | 0.006 | 0.019 |
| D02_image | M04 | M09 | 0.931 | 0.875 | +0.056 | 4.26 | 0.002 | 0.010 |
| D02_image | M04 | M10 | 0.931 | 0.885 | +0.046 | 5.10 | 0.00065 | 0.005 |

- Nadeau-Bengio Holm p < 0.05, D01_child (best M04): none
- Nadeau-Bengio Holm p < 0.05, D01_clip (best M08): none
- Nadeau-Bengio Holm p < 0.05, D02_image (best M04): M01, M03, M05, M06, M08, M09, M10

### 5.3 Calibration (pooled out-of-fold, ECE with 10 equal-width bins)

| model | ECE D01 child | ECE D01 clip | ECE D02 image | Brier D01 child | Brier D01 clip | Brier D02 image |
|---|---|---|---|---|---|---|
| M01 | 0.195 | 0.146 | 0.130 | 0.218 | 0.243 | 0.155 |
| M02 | 0.244 | 0.302 | 0.067 | 0.283 | 0.330 | 0.153 |
| M03 | 0.145 | 0.115 | 0.084 | 0.210 | 0.240 | 0.147 |
| M04 | 0.164 | 0.113 | 0.097 | 0.209 | 0.225 | 0.131 |
| M05 | 0.217 | 0.238 | 0.062 | 0.268 | 0.284 | 0.147 |
| M06 | 0.116 | 0.125 | 0.094 | 0.196 | 0.231 | 0.144 |
| M07 | 0.158 | 0.096 | 0.099 | 0.187 | 0.219 | 0.136 |
| M08 | 0.121 | 0.121 | 0.121 | 0.189 | 0.221 | 0.152 |
| M09 | 0.366 | 0.405 | 0.080 | 0.355 | 0.424 | 0.157 |
| M10 | 0.119 | 0.195 | 0.123 | 0.232 | 0.267 | 0.164 |

Tabular:

| dataset | model | n | brier | ece_10 |
|---|---|---|---|---|
| D04_autism_prediction/set_all | dummy_majority | 800 | 0.201 | 0.201 |
| D04_autism_prediction/set_all | logreg | 800 | 0.123 | 0.107 |
| D04_autism_prediction/set_all | random_forest | 800 | 0.100 | 0.043 |
| D04_autism_prediction/set_all | hist_gb | 800 | 0.124 | 0.096 |
| D04_autism_prediction/set_all | mlp | 800 | 0.096 | 0.018 |
| D05_toddler_screening/set_all | dummy_majority | 1054 | 0.309 | 0.309 |
| D05_toddler_screening/set_all | logreg | 1054 | 0.003 | 0.022 |
| D05_toddler_screening/set_all | random_forest | 1054 | 0.004 | 0.036 |
| D05_toddler_screening/set_all | hist_gb | 1054 | 0.000 | 0.000 |
| D05_toddler_screening/set_all | mlp | 1054 | 0.023 | 0.063 |
| D05_toddler_screening/set_honest | dummy_majority | 1054 | 0.309 | 0.309 |
| D05_toddler_screening/set_honest | logreg | 1054 | 0.242 | 0.178 |
| D05_toddler_screening/set_honest | random_forest | 1054 | 0.250 | 0.186 |
| D05_toddler_screening/set_honest | hist_gb | 1054 | 0.248 | 0.178 |
| D05_toddler_screening/set_honest | mlp | 1054 | 0.214 | 0.037 |
| D06_early_autism/set_all | dummy_majority | 1000 | 0.411 | 0.411 |
| D06_early_autism/set_all | logreg | 1000 | 0.253 | 0.098 |
| D06_early_autism/set_all | random_forest | 1000 | 0.270 | 0.146 |
| D06_early_autism/set_all | hist_gb | 1000 | 0.312 | 0.228 |
| D06_early_autism/set_all | mlp | 1000 | 0.249 | 0.087 |

### 5.4 Pretrained vs from-scratch (descriptive)

| analysis | group | n_models | mean of model fold-mean AUC | range | mean pooled OOF AUC |
|---|---|---|---|---|---|
| D01 clip | pretrained (M01,M03,M04,M06,M07,M08,M10) | 7 | 0.658 | 0.635-0.686 | 0.651 |
| D01 clip | from scratch (M02,M05,M09) | 3 | 0.619 | 0.595-0.657 | 0.573 |
| D01 child | pretrained (M01,M03,M04,M06,M07,M08,M10) | 7 | 0.790 | 0.744-0.856 | 0.755 |
| D01 child | from scratch (M02,M05,M09) | 3 | 0.717 | 0.678-0.767 | 0.586 |
| D02 image | pretrained (M01,M03,M04,M06,M07,M08,M10) | 7 | 0.905 | 0.885-0.931 | 0.884 |
| D02 image | from scratch (M02,M05,M09) | 3 | 0.874 | 0.867-0.881 | 0.870 |

## 6. Figures (`reports/figures/`), each opened and checked

- `stage08_cd_D01.png`: two CD diagrams; at clip level the Friedman p is 0.012 but one Nemenyi bar (CD 4.28) spans all ten models; at child level (p 0.566) all ten models are joined as well.
- `stage08_cd_D02.png`: M04 and M07 hold the best average ranks (1.40, 2.10), the three custom CNNs the worst three; overlapping bars, 11 of 45 pairs significantly different.
- `stage08_reliability_D01.png`: clip-level curves are flatter than the diagonal for all models (over-confident at both ends); M09 puts many predictions in the top bin with only ~40% ASD there and M02 has almost no low predictions; ECE 0.10-0.41.
- `stage08_reliability_D02.png`: U-shaped prediction histograms (most predictions in the 0-0.1 and 0.9-1 bins); curves lie above the diagonal in the lower and middle bins (more ASD than predicted) and slightly below it in the top bin; ECE 0.06-0.13.
- `stage08_d01_model_vs_quality_forest.png`: left, every model's AUC difference vs the quality baseline is negative or straddles 0 (M07 the only point estimate above 0, CI crossing 0); right, matched-subset AUCs with wide CIs, only M07's CI clear of 0.5, the quality-only band 0.45-0.68 overlapping most models.
- `stage08_d02_near_dup_sensitivity.png`: the removed-near-duplicate points lie just below the all-image points for every model (differences ~0.002); all models sit above the two image-statistics baseline lines (0.78, 0.80).

## 7. What can and cannot be claimed

Written from the numbers above and the rule in section 3.6.

**D01 eye-tracking**
- Can be claimed: per-child tracking quality alone predicts the label about as well as or better than every deep model (quality_LR3 pooled child AUC 0.823; models 0.574-0.832); no model is significantly better than this baseline, and 4 models (M02, M05, M09, M10) are significantly worse (CI of the difference below 0). Model child scores fall as tracking quality rises, also within the ASD group.
- By the pre-specified rule, M07 adds information beyond tracking quality, via rule 2 only: matched-subset AUC 0.724 [0.551, 0.883] on 14 pairs. This is weak evidence: rule 1 fails (difference vs quality_LR3 +0.009 [-0.098, +0.120]); 10 models were tested without multiplicity adjustment (with no signal beyond quality, the chance that at least one of 10 independent one-sided 97.5% bounds lies above 0.5 is 1 - 0.975^10 = 22%; the models are correlated, so this is only a guide); the matched set is small and keeps some residual imbalance (quality-only AUC on it 0.571, SMD -0.16); the exploratory incremental analysis gives +0.012 pooled AUC.
- For M01, M02, M03, M04, M05, M06, M08, M09, M10: the D01 results are **not distinguishable from the tracking-quality shortcut**.
- Cannot be claimed: that the D01 models learned gaze behaviour characteristic of autism; that any D01 model outperforms a recording-quality measure; any clinical accuracy.

**D02 facial images**
- Can be claimed: simple non-facial image statistics alone reach AUC 0.782-0.796, so the two classes differ systematically in image properties (colour, brightness, size/compression, sharpness); the deep models (0.867-0.931) score higher than this baseline, but how much of their performance comes from facial phenotype rather than such artefacts is not identified by these analyses.
- Near-duplicate leakage at pHash <= 8 does not explain the results (AUC change -0.002 to -0.001); different photos of the same child cannot be detected without subject IDs, so subject-level leakage is neither shown nor excluded.
- Models differ (Friedman p 3.7e-10); M04 has the highest mean fold AUC and is significantly better than 7 of 9 models after Holm (M01, M03, M05, M06, M08, M09, M10); pretrained backbones average higher than the custom CNNs.
- AUC is higher on the original 'test' images than on the rest for all 10 models (CI above 0 for 5): the original sub-folders are not exchangeable, a further sign of source heterogeneity.
- Cannot be claimed: that facial phenotype was learned; clinical validity; generalisation to clinical photos or other populations (web-collected images, no subject IDs, unknown diagnosis verification).

**Tabular and across datasets**
- D04: tabular models reach pooled AUC up to 0.905; D05 set_honest up to 0.610; D06 no model above 0.5 (max 0.482); D05 set_all = 1.000 is label-definitional.
- No result in this project is a diagnosis or a clinical accuracy estimate.

## 8. Tests (`.venv/bin/python -m pytest -v`, `logs/stage08_pytest.txt`)

New `tests/test_stage08.py` (10 tests): bootstrap reproducible with seed and cluster-preserving; DeLong vs a brute-force placement implementation on a known example (AUC 8/9) and with ties; Friedman chi2 = 8 on a toy matrix and Nemenyi q_alpha vs Demšar's table; Nadeau-Bengio / Holm / ECE / matching; the quality baseline uses only training children (test labels flipped -> identical predictions; overlap refused; fold log = split files); results/D0*/ unchanged since pre-flight.

Result: 341 passed in 135.32s (0:02:15)

<details><summary>full pytest output</summary>

```
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0 -- /home/score/Documents/AI_project/asd_hybrid_10fold/.venv/bin/python
rootdir: /home/score/Documents/AI_project/asd_hybrid_10fold
configfile: pytest.ini
testpaths: tests
plugins: anyio-4.15.1
collecting ... collected 341 items

tests/test_setup.py::test_project_dir_and_symlink PASSED                 [  0%]
tests/test_setup.py::test_paths_yaml_every_path_exists PASSED            [  0%]
tests/test_setup.py::test_datasets_dir_outside_project PASSED            [  0%]
tests/test_setup.py::test_no_absolute_paths_outside_paths_yaml PASSED    [  1%]
tests/test_setup.py::test_venv_imports PASSED                            [  1%]
tests/test_setup.py::test_cuda_available PASSED                          [  1%]
tests/test_setup.py::test_check_gpu_script_exits_zero PASSED             [  2%]
tests/test_setup.py::test_claude_md_sections PASSED                      [  2%]
tests/test_setup.py::test_raw_data_snapshot_matches PASSED               [  2%]
tests/test_setup.py::test_gitignore[.venv/] PASSED                       [  2%]
tests/test_setup.py::test_gitignore[cache/] PASSED                       [  3%]
tests/test_setup.py::test_gitignore[*.pt] PASSED                         [  3%]
tests/test_setup.py::test_gitignore[logs/*.log] PASSED                   [  3%]
tests/test_stage01.py::test_skeleton_dir_exists[configs] PASSED          [  4%]
tests/test_stage01.py::test_skeleton_dir_exists[src/data] PASSED         [  4%]
tests/test_stage01.py::test_skeleton_dir_exists[src/models] PASSED       [  4%]
tests/test_stage01.py::test_skeleton_dir_exists[src/training] PASSED     [  4%]
tests/test_stage01.py::test_skeleton_dir_exists[src/evaluation] PASSED   [  5%]
tests/test_stage01.py::test_skeleton_dir_exists[src/utils] PASSED        [  5%]
tests/test_stage01.py::test_skeleton_dir_exists[scripts] PASSED          [  5%]
tests/test_stage01.py::test_skeleton_dir_exists[tests] PASSED            [  6%]
tests/test_stage01.py::test_skeleton_dir_exists[data_registry] PASSED    [  6%]
tests/test_stage01.py::test_skeleton_dir_exists[splits] PASSED           [  6%]
tests/test_stage01.py::test_skeleton_dir_exists[cache] PASSED            [  7%]
tests/test_stage01.py::test_skeleton_dir_exists[results] PASSED          [  7%]
tests/test_stage01.py::test_skeleton_dir_exists[reports] PASSED          [  7%]
tests/test_stage01.py::test_skeleton_dir_exists[logs] PASSED             [  7%]
tests/test_stage01.py::test_results_readme_and_gitignore PASSED          [  8%]
tests/test_stage01.py::test_audit_json_loads_and_nonzero[D01_eye_tracking] PASSED [  8%]
tests/test_stage01.py::test_audit_json_loads_and_nonzero[D02_facial_images_A] PASSED [  8%]
tests/test_stage01.py::test_audit_json_loads_and_nonzero[D03_facial_images_B] PASSED [  9%]
tests/test_stage01.py::test_audit_json_loads_and_nonzero[D04_autism_prediction] PASSED [  9%]
tests/test_stage01.py::test_audit_json_loads_and_nonzero[D05_toddler_screening] PASSED [  9%]
tests/test_stage01.py::test_audit_json_loads_and_nonzero[D06_early_autism] PASSED [  9%]
tests/test_stage01.py::test_registry_exists PASSED                       [ 10%]
tests/test_stage01.py::test_raw_data_unmodified PASSED                   [ 10%]
tests/test_stage01.py::test_cuda_available PASSED                        [ 10%]
tests/test_stage02_d01.py::test_no_unidentified_rows PASSED              [ 11%]
tests/test_stage02_d01.py::test_no_duplicate_participant_time_trial PASSED [ 11%]
tests/test_stage02_d01.py::test_no_cars_anywhere_in_cache_or_splits PASSED [ 11%]
tests/test_stage02_d01.py::test_npy_shape_matches_index PASSED           [ 12%]
tests/test_stage02_d01.py::test_clip_rules PASSED                        [ 12%]
tests/test_stage02_d01.py::test_no_clip_spans_two_trials PASSED          [ 12%]
tests/test_stage02_d01.py::test_20_random_clips_rerendered_from_raw_are_identical PASSED [ 12%]
tests/test_stage02_d01.py::test_rebuilt_gaze_equals_cached PASSED        [ 13%]
tests/test_stage02_d01.py::test_quality_table_not_in_model_inputs PASSED [ 13%]
tests/test_stage02_d02.py::test_manifest_pooled_and_complete PASSED      [ 13%]
tests/test_stage02_d02.py::test_no_d03_path_anywhere PASSED              [ 14%]
tests/test_stage02_d02.py::test_near_duplicate_pairs_recomputed_lie_in_same_outer_fold PASSED [ 14%]
tests/test_stage02_d02.py::test_no_group_has_mixed_labels PASSED         [ 14%]
tests/test_stage02_snapshot.py::test_latest_snapshot_checksums_match_current_files PASSED [ 14%]
tests/test_stage02_snapshot.py::test_latest_snapshot_info_and_code_copy PASSED [ 15%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[1-D01_eye_tracking] PASSED [ 15%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[1-D02_facial_images_A] PASSED [ 15%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[1-D04_autism_prediction] PASSED [ 16%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[1-D05_toddler_screening] PASSED [ 16%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[1-D06_early_autism] PASSED [ 16%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[2-D01_eye_tracking] PASSED [ 17%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[2-D02_facial_images_A] PASSED [ 17%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[2-D04_autism_prediction] PASSED [ 17%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[2-D05_toddler_screening] PASSED [ 17%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[2-D06_early_autism] PASSED [ 18%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[3-D01_eye_tracking] PASSED [ 18%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[3-D02_facial_images_A] PASSED [ 18%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[3-D04_autism_prediction] PASSED [ 19%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[3-D05_toddler_screening] PASSED [ 19%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[3-D06_early_autism] PASSED [ 19%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[4-D01_eye_tracking] PASSED [ 19%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[4-D02_facial_images_A] PASSED [ 20%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[4-D04_autism_prediction] PASSED [ 20%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[4-D05_toddler_screening] PASSED [ 20%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[4-D06_early_autism] PASSED [ 21%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[5-D01_eye_tracking] PASSED [ 21%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[5-D02_facial_images_A] PASSED [ 21%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[5-D04_autism_prediction] PASSED [ 21%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[5-D05_toddler_screening] PASSED [ 22%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[5-D06_early_autism] PASSED [ 22%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[6-D01_eye_tracking] PASSED [ 22%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[6-D02_facial_images_A] PASSED [ 23%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[6-D04_autism_prediction] PASSED [ 23%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[6-D05_toddler_screening] PASSED [ 23%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[6-D06_early_autism] PASSED [ 24%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[7-D01_eye_tracking] PASSED [ 24%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[7-D02_facial_images_A] PASSED [ 24%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[7-D04_autism_prediction] PASSED [ 24%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[7-D05_toddler_screening] PASSED [ 25%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[7-D06_early_autism] PASSED [ 25%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[8-D01_eye_tracking] PASSED [ 25%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[8-D02_facial_images_A] PASSED [ 26%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[8-D04_autism_prediction] PASSED [ 26%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[8-D05_toddler_screening] PASSED [ 26%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[8-D06_early_autism] PASSED [ 26%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[9-D01_eye_tracking] PASSED [ 27%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[9-D02_facial_images_A] PASSED [ 27%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[9-D04_autism_prediction] PASSED [ 27%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[9-D05_toddler_screening] PASSED [ 28%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[9-D06_early_autism] PASSED [ 28%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[10-D01_eye_tracking] PASSED [ 28%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[10-D02_facial_images_A] PASSED [ 29%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[10-D04_autism_prediction] PASSED [ 29%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[10-D05_toddler_screening] PASSED [ 29%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[10-D06_early_autism] PASSED [ 29%]
tests/test_stage02_splits.py::test_every_sample_in_exactly_one_test_fold[D01_eye_tracking] PASSED [ 30%]
tests/test_stage02_splits.py::test_every_sample_in_exactly_one_test_fold[D02_facial_images_A] PASSED [ 30%]
tests/test_stage02_splits.py::test_every_sample_in_exactly_one_test_fold[D04_autism_prediction] PASSED [ 30%]
tests/test_stage02_splits.py::test_every_sample_in_exactly_one_test_fold[D05_toddler_screening] PASSED [ 31%]
tests/test_stage02_splits.py::test_every_sample_in_exactly_one_test_fold[D06_early_autism] PASSED [ 31%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[1-D01_eye_tracking] PASSED [ 31%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[1-D02_facial_images_A] PASSED [ 31%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[1-D04_autism_prediction] PASSED [ 32%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[1-D05_toddler_screening] PASSED [ 32%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[1-D06_early_autism] PASSED [ 32%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[2-D01_eye_tracking] PASSED [ 33%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[2-D02_facial_images_A] PASSED [ 33%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[2-D04_autism_prediction] PASSED [ 33%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[2-D05_toddler_screening] PASSED [ 34%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[2-D06_early_autism] PASSED [ 34%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[3-D01_eye_tracking] PASSED [ 34%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[3-D02_facial_images_A] PASSED [ 34%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[3-D04_autism_prediction] PASSED [ 35%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[3-D05_toddler_screening] PASSED [ 35%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[3-D06_early_autism] PASSED [ 35%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[4-D01_eye_tracking] PASSED [ 36%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[4-D02_facial_images_A] PASSED [ 36%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[4-D04_autism_prediction] PASSED [ 36%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[4-D05_toddler_screening] PASSED [ 36%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[4-D06_early_autism] PASSED [ 37%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[5-D01_eye_tracking] PASSED [ 37%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[5-D02_facial_images_A] PASSED [ 37%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[5-D04_autism_prediction] PASSED [ 38%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[5-D05_toddler_screening] PASSED [ 38%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[5-D06_early_autism] PASSED [ 38%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[6-D01_eye_tracking] PASSED [ 39%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[6-D02_facial_images_A] PASSED [ 39%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[6-D04_autism_prediction] PASSED [ 39%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[6-D05_toddler_screening] PASSED [ 39%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[6-D06_early_autism] PASSED [ 40%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[7-D01_eye_tracking] PASSED [ 40%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[7-D02_facial_images_A] PASSED [ 40%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[7-D04_autism_prediction] PASSED [ 41%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[7-D05_toddler_screening] PASSED [ 41%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[7-D06_early_autism] PASSED [ 41%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[8-D01_eye_tracking] PASSED [ 41%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[8-D02_facial_images_A] PASSED [ 42%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[8-D04_autism_prediction] PASSED [ 42%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[8-D05_toddler_screening] PASSED [ 42%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[8-D06_early_autism] PASSED [ 43%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[9-D01_eye_tracking] PASSED [ 43%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[9-D02_facial_images_A] PASSED [ 43%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[9-D04_autism_prediction] PASSED [ 43%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[9-D05_toddler_screening] PASSED [ 44%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[9-D06_early_autism] PASSED [ 44%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[10-D01_eye_tracking] PASSED [ 44%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[10-D02_facial_images_A] PASSED [ 45%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[10-D04_autism_prediction] PASSED [ 45%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[10-D05_toddler_screening] PASSED [ 45%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[10-D06_early_autism] PASSED [ 46%]
tests/test_stage02_splits.py::test_d01_test_folds_have_asd_and_td_children PASSED [ 46%]
tests/test_stage02_splits.py::test_splits_manifest_hashes_match PASSED   [ 46%]
tests/test_stage02_splits.py::test_rerun_make_splits_is_byte_identical PASSED [ 46%]
tests/test_stage02_tabular.py::test_d04_only_labelled_train_rows PASSED  [ 47%]
tests/test_stage02_tabular.py::test_d04_result_decision_recorded PASSED  [ 47%]
tests/test_stage02_tabular.py::test_d05_honest_set_excludes_items_and_score PASSED [ 47%]
tests/test_stage02_tabular.py::test_d05_label_rule_on_clean_data PASSED  [ 48%]
tests/test_stage02_tabular.py::test_d05_other_csv_never_read_by_stage2_code PASSED [ 48%]
tests/test_stage02_tabular.py::test_d06_empty_column_absent PASSED       [ 48%]
tests/test_stage02_tabular.py::test_identical_feature_rows_share_group PASSED [ 48%]
tests/test_stage02_tabular.py::test_every_column_has_a_role PASSED       [ 49%]
tests/test_stage02_tabular.py::test_raw_data_snapshot_unchanged PASSED   [ 49%]
tests/test_stage03_models.py::test_config_mapping[M01] PASSED            [ 49%]
tests/test_stage03_models.py::test_config_mapping[M02] PASSED            [ 50%]
tests/test_stage03_models.py::test_config_mapping[M03] PASSED            [ 50%]
tests/test_stage03_models.py::test_config_mapping[M04] PASSED            [ 50%]
tests/test_stage03_models.py::test_config_mapping[M05] PASSED            [ 51%]
tests/test_stage03_models.py::test_config_mapping[M06] PASSED            [ 51%]
tests/test_stage03_models.py::test_config_mapping[M07] PASSED            [ 51%]
tests/test_stage03_models.py::test_config_mapping[M08] PASSED            [ 51%]
tests/test_stage03_models.py::test_config_mapping[M09] PASSED            [ 52%]
tests/test_stage03_models.py::test_config_mapping[M10] PASSED            [ 52%]
tests/test_stage03_models.py::test_factory_rejects_unknown PASSED        [ 52%]
tests/test_stage03_models.py::test_forward[M01-sequence] PASSED          [ 53%]
tests/test_stage03_models.py::test_forward[M01-tokens] PASSED            [ 53%]
tests/test_stage03_models.py::test_forward[M02-sequence] PASSED          [ 53%]
tests/test_stage03_models.py::test_forward[M02-tokens] PASSED            [ 53%]
tests/test_stage03_models.py::test_forward[M03-sequence] PASSED          [ 54%]
tests/test_stage03_models.py::test_forward[M03-tokens] PASSED            [ 54%]
tests/test_stage03_models.py::test_forward[M04-sequence] PASSED          [ 54%]
tests/test_stage03_models.py::test_forward[M04-tokens] PASSED            [ 55%]
tests/test_stage03_models.py::test_forward[M05-sequence] PASSED          [ 55%]
tests/test_stage03_models.py::test_forward[M05-tokens] PASSED            [ 55%]
tests/test_stage03_models.py::test_forward[M06-sequence] PASSED          [ 56%]
tests/test_stage03_models.py::test_forward[M06-tokens] PASSED            [ 56%]
tests/test_stage03_models.py::test_forward[M07-sequence] PASSED          [ 56%]
tests/test_stage03_models.py::test_forward[M07-tokens] PASSED            [ 56%]
tests/test_stage03_models.py::test_forward[M08-sequence] PASSED          [ 57%]
tests/test_stage03_models.py::test_forward[M08-tokens] PASSED            [ 57%]
tests/test_stage03_models.py::test_forward[M09-sequence] PASSED          [ 57%]
tests/test_stage03_models.py::test_forward[M09-tokens] PASSED            [ 58%]
tests/test_stage03_models.py::test_forward[M10-sequence] PASSED          [ 58%]
tests/test_stage03_models.py::test_forward[M10-tokens] PASSED            [ 58%]
tests/test_stage03_models.py::test_backward_and_step[M01-sequence] PASSED [ 58%]
tests/test_stage03_models.py::test_backward_and_step[M01-tokens] PASSED  [ 59%]
tests/test_stage03_models.py::test_backward_and_step[M02-sequence] PASSED [ 59%]
tests/test_stage03_models.py::test_backward_and_step[M02-tokens] PASSED  [ 59%]
tests/test_stage03_models.py::test_backward_and_step[M03-sequence] PASSED [ 60%]
tests/test_stage03_models.py::test_backward_and_step[M03-tokens] PASSED  [ 60%]
tests/test_stage03_models.py::test_backward_and_step[M04-sequence] PASSED [ 60%]
tests/test_stage03_models.py::test_backward_and_step[M04-tokens] PASSED  [ 60%]
tests/test_stage03_models.py::test_backward_and_step[M05-sequence] PASSED [ 61%]
tests/test_stage03_models.py::test_backward_and_step[M05-tokens] PASSED  [ 61%]
tests/test_stage03_models.py::test_backward_and_step[M06-sequence] PASSED [ 61%]
tests/test_stage03_models.py::test_backward_and_step[M06-tokens] PASSED  [ 62%]
tests/test_stage03_models.py::test_backward_and_step[M07-sequence] PASSED [ 62%]
tests/test_stage03_models.py::test_backward_and_step[M07-tokens] PASSED  [ 62%]
tests/test_stage03_models.py::test_backward_and_step[M08-sequence] PASSED [ 63%]
tests/test_stage03_models.py::test_backward_and_step[M08-tokens] PASSED  [ 63%]
tests/test_stage03_models.py::test_backward_and_step[M09-sequence] PASSED [ 63%]
tests/test_stage03_models.py::test_backward_and_step[M09-tokens] PASSED  [ 63%]
tests/test_stage03_models.py::test_backward_and_step[M10-sequence] PASSED [ 64%]
tests/test_stage03_models.py::test_backward_and_step[M10-tokens] PASSED  [ 64%]
tests/test_stage03_models.py::test_determinism_eval[M01-sequence] PASSED [ 64%]
tests/test_stage03_models.py::test_determinism_eval[M01-tokens] PASSED   [ 65%]
tests/test_stage03_models.py::test_determinism_eval[M02-sequence] PASSED [ 65%]
tests/test_stage03_models.py::test_determinism_eval[M02-tokens] PASSED   [ 65%]
tests/test_stage03_models.py::test_determinism_eval[M03-sequence] PASSED [ 65%]
tests/test_stage03_models.py::test_determinism_eval[M03-tokens] PASSED   [ 66%]
tests/test_stage03_models.py::test_determinism_eval[M04-sequence] PASSED [ 66%]
tests/test_stage03_models.py::test_determinism_eval[M04-tokens] PASSED   [ 66%]
tests/test_stage03_models.py::test_determinism_eval[M05-sequence] PASSED [ 67%]
tests/test_stage03_models.py::test_determinism_eval[M05-tokens] PASSED   [ 67%]
tests/test_stage03_models.py::test_determinism_eval[M06-sequence] PASSED [ 67%]
tests/test_stage03_models.py::test_determinism_eval[M06-tokens] PASSED   [ 68%]
tests/test_stage03_models.py::test_determinism_eval[M07-sequence] PASSED [ 68%]
tests/test_stage03_models.py::test_determinism_eval[M07-tokens] PASSED   [ 68%]
tests/test_stage03_models.py::test_determinism_eval[M08-sequence] PASSED [ 68%]
tests/test_stage03_models.py::test_determinism_eval[M08-tokens] PASSED   [ 69%]
tests/test_stage03_models.py::test_determinism_eval[M09-sequence] PASSED [ 69%]
tests/test_stage03_models.py::test_determinism_eval[M09-tokens] PASSED   [ 69%]
tests/test_stage03_models.py::test_determinism_eval[M10-sequence] PASSED [ 70%]
tests/test_stage03_models.py::test_determinism_eval[M10-tokens] PASSED   [ 70%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M01-sequence] PASSED [ 70%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M01-tokens] PASSED [ 70%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M02-sequence] PASSED [ 71%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M02-tokens] PASSED [ 71%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M03-sequence] PASSED [ 71%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M03-tokens] PASSED [ 72%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M04-sequence] PASSED [ 72%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M04-tokens] PASSED [ 72%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M05-sequence] PASSED [ 73%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M05-tokens] PASSED [ 73%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M06-sequence] PASSED [ 73%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M06-tokens] PASSED [ 73%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M07-sequence] PASSED [ 74%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M07-tokens] PASSED [ 74%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M08-sequence] PASSED [ 74%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M08-tokens] PASSED [ 75%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M09-sequence] PASSED [ 75%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M09-tokens] PASSED [ 75%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M10-sequence] PASSED [ 75%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M10-tokens] PASSED [ 76%]
tests/test_stage03_models.py::test_feature_cache_equivalence[M01] PASSED [ 76%]
tests/test_stage03_models.py::test_feature_cache_equivalence[M03] PASSED [ 76%]
tests/test_stage03_models.py::test_feature_cache_equivalence[M04] PASSED [ 77%]
tests/test_stage03_models.py::test_feature_cache_equivalence[M06] PASSED [ 77%]
tests/test_stage03_models.py::test_feature_cache_equivalence[M07] PASSED [ 77%]
tests/test_stage03_models.py::test_feature_cache_equivalence[M08] PASSED [ 78%]
tests/test_stage03_models.py::test_feature_cache_equivalence[M10] PASSED [ 78%]
tests/test_stage03_models.py::test_last_stage_trains_part_of_backbone PASSED [ 78%]
tests/test_stage03_models.py::test_model_configs_have_batch_sizes PASSED [ 78%]
tests/test_stage04_engine.py::test_interrupt_leaves_running_and_checkpoint[M01] PASSED [ 79%]
tests/test_stage04_engine.py::test_resume_identical_history_and_predictions[M01] PASSED [ 79%]
tests/test_stage04_engine.py::test_resumed_run_info_and_cleanup[M01] PASSED [ 79%]
tests/test_stage04_engine.py::test_youden_threshold_comes_from_validation_predictions[M01] PASSED [ 80%]
tests/test_stage04_engine.py::test_done_run_is_skipped_unless_forced[M01] PASSED [ 80%]
tests/test_stage04_engine.py::test_checks_detect_incomplete_results[M01] PASSED [ 80%]
tests/test_stage04_engine.py::test_interrupt_leaves_running_and_checkpoint[M09] PASSED [ 80%]
tests/test_stage04_engine.py::test_resume_identical_history_and_predictions[M09] PASSED [ 81%]
tests/test_stage04_engine.py::test_resumed_run_info_and_cleanup[M09] PASSED [ 81%]
tests/test_stage04_engine.py::test_youden_threshold_comes_from_validation_predictions[M09] PASSED [ 81%]
tests/test_stage04_engine.py::test_done_run_is_skipped_unless_forced[M09] PASSED [ 82%]
tests/test_stage04_engine.py::test_checks_detect_incomplete_results[M09] PASSED [ 82%]
tests/test_stage04_engine.py::test_engine_refuses_corrupted_split PASSED [ 82%]
tests/test_stage04_engine.py::test_resume_refuses_changed_config PASSED  [ 82%]
tests/test_stage04_units.py::test_training_protocol_values_fixed PASSED  [ 83%]
tests/test_stage04_units.py::test_merge_config[M01-D01_eye_tracking-1-0.0005-0.0-cached_features] PASSED [ 83%]
tests/test_stage04_units.py::test_merge_config[M01-D02_facial_images_A-3-0.0005-5e-05-images] PASSED [ 83%]
tests/test_stage04_units.py::test_merge_config[M07-D02_facial_images_A-10-0.0005-5e-05-images] PASSED [ 84%]
tests/test_stage04_units.py::test_merge_config[M09-D01_eye_tracking-2-0.001-0.001-clips] PASSED [ 84%]
tests/test_stage04_units.py::test_merge_config[M05-D02_facial_images_A-4-0.001-0.001-images] PASSED [ 84%]
tests/test_stage04_units.py::test_model_config_lr_mirrors_training_yaml PASSED [ 85%]
tests/test_stage04_units.py::test_scheduler_warmup_and_cosine PASSED     [ 85%]
tests/test_stage04_units.py::test_d02_transforms_uint8 PASSED            [ 85%]
tests/test_stage04_units.py::test_real_splits_pass_check_fold[D01_eye_tracking] PASSED [ 85%]
tests/test_stage04_units.py::test_real_splits_pass_check_fold[D02_facial_images_A] PASSED [ 86%]
tests/test_stage04_units.py::test_leakage_asserts_trigger_on_corrupted_split[id_in_train_and_test-D01_eye_tracking] PASSED [ 86%]
tests/test_stage04_units.py::test_leakage_asserts_trigger_on_corrupted_split[id_in_train_and_test-D02_facial_images_A] PASSED [ 86%]
tests/test_stage04_units.py::test_leakage_asserts_trigger_on_corrupted_split[duplicate-D01_eye_tracking] PASSED [ 87%]
tests/test_stage04_units.py::test_leakage_asserts_trigger_on_corrupted_split[duplicate-D02_facial_images_A] PASSED [ 87%]
tests/test_stage04_units.py::test_leakage_asserts_trigger_on_corrupted_split[group_overlap-D01_eye_tracking] PASSED [ 87%]
tests/test_stage04_units.py::test_leakage_asserts_trigger_on_corrupted_split[group_overlap-D02_facial_images_A] PASSED [ 87%]
tests/test_stage04_units.py::test_leakage_asserts_trigger_on_corrupted_split[unknown_id-D01_eye_tracking] PASSED [ 88%]
tests/test_stage04_units.py::test_leakage_asserts_trigger_on_corrupted_split[unknown_id-D02_facial_images_A] PASSED [ 88%]
tests/test_stage04_units.py::test_test_set_must_match_folds_csv PASSED   [ 88%]
tests/test_stage04_units.py::test_verify_manifest_detects_modified_temp_copy PASSED [ 89%]
tests/test_stage04_units.py::test_assert_no_test_ids PASSED              [ 89%]
tests/test_stage04_units.py::test_binary_metrics_match_sklearn[0.5-0] PASSED [ 89%]
tests/test_stage04_units.py::test_binary_metrics_match_sklearn[0.5-1] PASSED [ 90%]
tests/test_stage04_units.py::test_binary_metrics_match_sklearn[0.5-2] PASSED [ 90%]
tests/test_stage04_units.py::test_binary_metrics_match_sklearn[0.3-0] PASSED [ 90%]
tests/test_stage04_units.py::test_binary_metrics_match_sklearn[0.3-1] PASSED [ 90%]
tests/test_stage04_units.py::test_binary_metrics_match_sklearn[0.3-2] PASSED [ 91%]
tests/test_stage04_units.py::test_binary_metrics_match_sklearn[0.72-0] PASSED [ 91%]
tests/test_stage04_units.py::test_binary_metrics_match_sklearn[0.72-1] PASSED [ 91%]
tests/test_stage04_units.py::test_binary_metrics_match_sklearn[0.72-2] PASSED [ 92%]
tests/test_stage04_units.py::test_binary_metrics_known_values PASSED     [ 92%]
tests/test_stage04_units.py::test_youden_threshold_brute_force_on_validation_only[0] PASSED [ 92%]
tests/test_stage04_units.py::test_youden_threshold_brute_force_on_validation_only[1] PASSED [ 92%]
tests/test_stage04_units.py::test_youden_threshold_brute_force_on_validation_only[2] PASSED [ 93%]
tests/test_stage04_units.py::test_youden_threshold_brute_force_on_validation_only[3] PASSED [ 93%]
tests/test_stage04_units.py::test_child_level_mean_probability PASSED    [ 93%]
tests/test_stage04_units.py::test_registry_skip_force_resume PASSED      [ 94%]
tests/test_stage04_units.py::test_registry_file_lock_no_lost_rows PASSED [ 94%]
tests/test_stage04_units.py::test_summary_aggregation_on_fake_folds PASSED [ 94%]
tests/test_stage05_checker.py::test_reference_block_is_valid PASSED      [ 95%]
tests/test_stage05_checker.py::test_perfect_pr_auc_passes PASSED         [ 95%]
tests/test_stage05_checker.py::test_auc_zero_passes PASSED               [ 95%]
tests/test_stage05_checker.py::test_value_above_one_fails PASSED         [ 95%]
tests/test_stage05_checker.py::test_negative_value_fails PASSED          [ 96%]
tests/test_stage05_checker.py::test_nan_is_flagged_never_silent PASSED   [ 96%]
tests/test_stage05_checker.py::test_nan_where_defined_fails PASSED       [ 96%]
tests/test_stage05_checker.py::test_missing_or_non_numeric_fails PASSED  [ 97%]
tests/test_stage08.py::test_bootstrap_reproducible_with_seed PASSED      [ 97%]
tests/test_stage08.py::test_cluster_bootstrap_keeps_groups_whole_and_both_classes PASSED [ 97%]
tests/test_stage08.py::test_delong_known_example PASSED                  [ 97%]
tests/test_stage08.py::test_delong_with_ties_matches_bruteforce_and_identical_scores_give_p1 PASSED [ 98%]
tests/test_stage08.py::test_friedman_toy_matrix PASSED                   [ 98%]
tests/test_stage08.py::test_nadeau_bengio_and_holm PASSED                [ 98%]
tests/test_stage08.py::test_ece_and_matching PASSED                      [ 99%]
tests/test_stage08.py::test_quality_baseline_uses_only_train_fold_children PASSED [ 99%]
tests/test_stage08.py::test_results_d0x_unchanged_since_preflight PASSED [ 99%]
tests/test_stage08.py::test_stage8_outputs_present PASSED                [100%]

======================= 341 passed in 135.32s (0:02:15) ========================
```

</details>

## 9. Read-only check of earlier results

```
results/D0*/ (no *.pt): 2407 files recorded at pre-flight, 2407 now; identical sha256 + size + path for all: True (incl. 800 predictions.csv / metrics.json)
```

## 10. Notes and deviations

- Bootstrap resampling is stratified by class at the resampling-unit level (guarantees both classes in every resample); stratified for 54 of 54 rows (column `stratified`; unstratified cluster resampling where some groups contain both labels: none).
- 'Training images' in C1 = the fold's train + val images (all non-test images available to that fold's model).
- Friedman / Nemenyi / Nadeau-Bengio use per-fold clip-level AUC (D01 primary) and per-fold child-level AUC (D01 secondary, 5-6 children per fold) as stored in metrics.json; the Nadeau-Bengio 'best' model is the one with the highest mean fold AUC, which can differ from the best average rank.
- The D01 quality-matched subset and the tertile analyses are small-sample; their CIs are wide.

STAGE 8 DECISION: PASS
