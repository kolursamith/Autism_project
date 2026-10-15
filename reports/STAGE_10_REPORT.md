# STAGE 10 REPORT - Final audit and freeze

Screening-support research, not diagnosis. Stage 10 trains nothing and changes no result value. Code: `scripts/10_audit.py` (fresh re-verification: sklearn / numpy / hashlib only, nothing imported from src.training, src.analysis or make_snapshot), `scripts/10b_finalize.py` (README, this report, freeze); `src/analysis/final.py` now also skips `results/FINAL_MANIFEST.json` in the Stage 9 read-only manifest (Stage 10 output, not a Stage 9 input).

## 1. Summary

- Wording fix applied to FINAL_RESULTS_REPORT.md (True), its HTML (True) and STAGE_09_REPORT.md (True). No report sentence claiming clinical accuracy, diagnosis or autism detection was found (section 2.2).
- Re-verification: metric mismatches 0 of 4000 (400 runs); master-table mismatches 0 of 800; leakage checks all passed: True; snapshots stage_01..09 intact: True; report numbers needing manual check: 0.
- Tests: 360 passed in 129.92s (0:02:09).
- Freeze: results/FINAL_MANIFEST.json, read-only results, splits, reports/figures/final, submission zip with 247 files (size and sha256 in `submission/SUBMISSION_INFO.json`, written after the zip because the zip contains this report).

## 2. Wording fix (A)

### 2.1 Matched-subset sentence

Old (abstract of FINAL_RESULTS_REPORT.md and the abstract quoted in STAGE_09_REPORT.md): "... only one model (M07) kept an AUC above 0.5 after matching children on tracking quality."

New (numbers read from `results/final/shortcut_table.csv` by `scripts/09d_write_report.py`, which also asserts that exactly one model's matched CI excludes 0.5): "only one model (M07) had a matched-subset AUC whose 95% bootstrap CI excluded 0.5 (0.724 [0.551, 0.883], 14 matched pairs); the other models' matched AUCs had CIs including 0.5."

The sentence occurs only in the abstract; sections 6 and 8 already used CI wording. The abstract grew from 183 to 197 words (Stage 9 limit 150-200); STAGE_09_REPORT.md's stated word count was updated accordingly. The HTML was regenerated with pandoc. `diff` of the regenerated FINAL_RESULTS_REPORT.md vs the Stage 9 version: only line 9 (abstract) differs.

### 2.2 Scan for clinical / diagnostic / detection claims (`logs/stage10_claims_scan.txt`)

```
reports/*.md lines matching 'diagnos|detect|clinical|clinically|identif(y|ies) (autism|asd)|predicts? autism|distinguish(es)? (autism|asd|children)': 46
   25  explicit negation / limitation (screening-support framing)
   16  pytest test name (software check, not about autism)
    3  dataset folder name (proper noun of the D02 source, cannot be renamed)
    2  data column name (D06 label column)
claims needing a rewrite: 0
reports/FINAL_RESULTS_REPORT.md:5 [explicit negation / limitation] > Screening-support research only. No result in this report is a diagnosis or an estimate of clinical accuracy.
reports/FINAL_RESULTS_REPORT.md:9 [explicit negation / limitation] We trained and compared ten hybrid deep-learning architectures (a pretrained or custom-CNN spatial encoder, a recurrent or Transformer temporal encode
reports/FINAL_RESULTS_REPORT.md:321 [explicit negation / limitation] - Cannot be claimed: that the D01 models learned gaze behaviour characteristic of autism; that any D01 model outperforms a recording-quality measure; 
reports/FINAL_RESULTS_REPORT.md:326 [explicit negation / limitation] - Near-duplicate leakage at pHash <= 8 does not explain the results (AUC change -0.002 to -0.001); different photos of the same child cannot be detect
reports/FINAL_RESULTS_REPORT.md:329 [explicit negation / limitation] - Cannot be claimed: that facial phenotype was learned; clinical validity; generalisation to clinical photos or other populations (web-collected image
reports/FINAL_RESULTS_REPORT.md:334 [explicit negation / limitation] - No result in this project is a diagnosis or a clinical accuracy estimate.
reports/FINAL_RESULTS_REPORT.md:342 [explicit negation / limitation] - D02 images were collected from the web with unknown diagnosis verification, age range, consent and acquisition conditions; image artefacts alone sep
reports/STAGE_01_REPORT.md:129 [dataset folder name] - **D02_facial_images_A** <- `Autism Spectrum Disorder Detection using Facial Traits Dual Framework DenseRes with K-Fold Cross Validation (DFDK) for E
reports/STAGE_01_REPORT.md:135 [dataset folder name] The folder truncated in the file manager is, in full: `Autism Spectrum Disorder Detection using Facial Traits Dual Framework DenseRes with K-Fold Cros
reports/STAGE_01_REPORT.md:140 [dataset folder name] | **D02_facial_images_A** | `Autism Spectrum Disorder Detection using Facial Traits Dual Framework DenseRes with K-Fold Cross Validation (DFDK) for En
reports/STAGE_02_REPORT.md:3 [explicit negation / limitation] Project: Reliability-Weighted Multimodal AI for Early Autism Risk Screening in Children (BITE308L). Screening-support research, not diagnosis. Seed 42
reports/STAGE_02_REPORT.md:176 [data column name] - excluded: `Unnamed: 10`, `Diagnosed_ASD`
reports/STAGE_02_REPORT.md:190 [data column name] | Diagnosed_ASD | label ['yes'] -> 1, ['no'] -> 0, stored as 'label' | 1000 |
reports/STAGE_03_REPORT.md:3 [explicit negation / limitation] Screening-support research, not diagnosis. No model was trained on real labels in this stage (tests use random inputs and random 0/1 labels only). No 
reports/STAGE_04_REPORT.md:3 [explicit negation / limitation] Screening-support research, not diagnosis. The values below come from 2-epoch smoke tests and are meaningless as performance estimates; they only show
reports/STAGE_04_REPORT.md:369 [pytest test name] tests/test_stage04_engine.py::test_checks_detect_incomplete_results[M01] PASSED [ 85%]
reports/STAGE_04_REPORT.md:375 [pytest test name] tests/test_stage04_engine.py::test_checks_detect_incomplete_results[M09] PASSED [ 86%]
reports/STAGE_04_REPORT.md:398 [pytest test name] tests/test_stage04_units.py::test_verify_manifest_detects_modified_temp_copy PASSED [ 94%]
reports/STAGE_05_REPORT.md:3 [explicit negation / limitation] Screening-support research, not diagnosis. 57 children (27 ASD / 30 TD), 1,241 clips of 8 x 1-s gaze windows; child-grouped 10-fold CV (5-6 test child
reports/STAGE_05_REPORT.md:587 [pytest test name] tests/test_stage04_engine.py::test_checks_detect_incomplete_results[M01] PASSED [ 83%]
reports/STAGE_05_REPORT.md:593 [pytest test name] tests/test_stage04_engine.py::test_checks_detect_incomplete_results[M09] PASSED [ 84%]
reports/STAGE_05_REPORT.md:616 [pytest test name] tests/test_stage04_units.py::test_verify_manifest_detects_modified_temp_copy PASSED [ 91%]
reports/STAGE_05_REPORT.md:656 [explicit negation / limitation] - No result is a diagnosis or a clinical accuracy estimate.
reports/STAGE_06_REPORT.md:3 [explicit negation / limitation] Screening-support research, not diagnosis. 2,940 facial images (1,470 autistic / 1,470 non-autistic, web-collected DFDK folder; original train/valid/t
reports/STAGE_06_REPORT.md:118 [explicit negation / limitation] These are image-level results. D02 has no subject identifiers, so the same child may appear in several images and in both training and test folds; gro
reports/STAGE_06_REPORT.md:450 [pytest test name] tests/test_stage04_engine.py::test_checks_detect_incomplete_results[M01] PASSED [ 83%]
reports/STAGE_06_REPORT.md:456 [pytest test name] tests/test_stage04_engine.py::test_checks_detect_incomplete_results[M09] PASSED [ 84%]
reports/STAGE_06_REPORT.md:479 [pytest test name] tests/test_stage04_units.py::test_verify_manifest_detects_modified_temp_copy PASSED [ 91%]
reports/STAGE_06_REPORT.md:520 [explicit negation / limitation] - No result is a diagnosis or a clinical accuracy estimate.
reports/STAGE_07_REPORT.md:3 [explicit negation / limitation] Screening-support research, not diagnosis. Run on 2026-10-14 BEFORE Stage 6, with the user's explicit approval, while the S5 GPU training was running.
reports/STAGE_07_REPORT.md:97 [explicit negation / limitation] - **D06**: no model exceeds the majority baseline: mean ROC-AUCs are 0.45-0.48. The descriptive single-feature AUCs below are all close to 0.5, i.e. t
reports/STAGE_07_REPORT.md:110 [explicit negation / limitation] None of these numbers is a diagnostic accuracy; the tables are small, single-source screening datasets.
reports/STAGE_07_REPORT.md:128 [pytest test name] tabular/tests/test_stage07.py::test_check_fold_detects_corrupted_split PASSED [ 31%]
reports/STAGE_08_REPORT.md:3 [explicit negation / limitation] Screening-support research, not diagnosis. Stage 8 trains NO deep model: every analysis reads the saved out-of-fold predictions / metrics of Stages 5-
reports/STAGE_08_REPORT.md:452 [explicit negation / limitation] - Cannot be claimed: that the D01 models learned gaze behaviour characteristic of autism; that any D01 model outperforms a recording-quality measure; 
reports/STAGE_08_REPORT.md:456 [explicit negation / limitation] - Near-duplicate leakage at pHash <= 8 does not explain the results (AUC change -0.002 to -0.001); different photos of the same child cannot be detect
reports/STAGE_08_REPORT.md:459 [explicit negation / limitation] - Cannot be claimed: that facial phenotype was learned; clinical validity; generalisation to clinical photos or other populations (web-collected image
reports/STAGE_08_REPORT.md:463 [explicit negation / limitation] - No result in this project is a diagnosis or a clinical accuracy estimate.
reports/STAGE_08_REPORT.md:756 [pytest test name] tests/test_stage04_engine.py::test_checks_detect_incomplete_results[M01] PASSED [ 80%]
reports/STAGE_08_REPORT.md:762 [pytest test name] tests/test_stage04_engine.py::test_checks_detect_incomplete_results[M09] PASSED [ 82%]
reports/STAGE_08_REPORT.md:785 [pytest test name] tests/test_stage04_units.py::test_verify_manifest_detects_modified_temp_copy PASSED [ 89%]
reports/STAGE_09_REPORT.md:3 [explicit negation / limitation] Screening-support research, not diagnosis. Stage 9 collects and presents; it trains nothing and recomputes no metric (results/D0*/ and results/stage8/
reports/STAGE_09_REPORT.md:62 [explicit negation / limitation] > We trained and compared ten hybrid deep-learning architectures (a pretrained or custom-CNN spatial encoder, a recurrent or Transformer temporal enco
reports/STAGE_09_REPORT.md:359 [pytest test name] tests/test_stage04_engine.py::test_checks_detect_incomplete_results[M01] PASSED [ 76%]
reports/STAGE_09_REPORT.md:365 [pytest test name] tests/test_stage04_engine.py::test_checks_detect_incomplete_results[M09] PASSED [ 78%]
reports/STAGE_09_REPORT.md:388 [pytest test name] tests/test_stage04_units.py::test_verify_manifest_detects_modified_temp_copy PASSED [ 84%]
```

## 3. Independent re-verification (B, `logs/stage10_audit.txt`)

```
B1 400 runs (200 deep, 200 tabular): 4000 metric checks (ROC-AUC, balanced accuracy, sensitivity, specificity at threshold 0.5 and at the stored validation threshold; D01 also child level) recomputed with sklearn from predictions.csv vs metrics.json, tol 1e-06: mismatches 0
B2 master_table.csv: 50 rows; 800 values recomputed from predictions.csv (7 metrics x mean/std over 10 folds, ddof=1; pooled AUC; ECE 10 bins), tol 1e-06: mismatches 0 (bootstrap CIs not recomputed)
B3 leakage: predictions == fold test ids for all 400 runs: True; every sample predicted exactly once per model: 40/40 models; train/val/test disjoint by sample AND group in all 50 dataset-folds: True
B3 groups in more than one test fold: {'D01_eye_tracking': 0, 'D02_facial_images_A': 0, 'D04_autism_prediction': 0, 'D05_toddler_screening': 0, 'D06_early_autism': 0} (D01 = children, D02 = near-duplicate groups)
B3 D02 pHash <= 4 pairs recomputed (fresh popcount): 11 pairs, across folds: 0
B3 D03: cache cells starting with the D03 folder 'AutismDataset/': 0; all D02 manifest paths under the D02 folder: True
B3 CARS: columns named *cars* in D01 caches / tabular feature definitions: none; training/model code referencing participants.csv or CARS: none
B3 D05 set_honest: forbidden (A1-A10, Qchat score) in feature set: none; in the 50 run configs: none; features used ['Age_Mons', 'Ethnicity', 'Family_mem_with_ASD', 'Jaundice', 'Sex', 'Who completed the test']
B3 D04: clean.csv rows = raw train.csv rows (ID + A1-A10, age, result identical): True; clean rows identical to a test.csv row with the same ID: 0; data file of all D04 runs: [['cache/D04_autism_prediction/clean.csv']]
B4 snapshots (recomputed sha256 of every recorded file; changes explained by the later snapshot that recorded the current version) -> logs/stage10_snapshot_changes.csv
   stage_01 (2026-10-13T16:56:45): integrity True; recorded 33, unchanged 32, changed 1, added 245, deleted 0; not recorded by any later snapshot: 0
   stage_02 (2026-10-13T17:18:53): integrity True; recorded 150, unchanged 148, changed 2, added 128, deleted 0; not recorded by any later snapshot: 0
   stage_03 (2026-10-14T09:10:29): integrity True; recorded 186, unchanged 174, changed 12, added 92, deleted 0; not recorded by any later snapshot: 0
   stage_04 (2026-10-14T09:43:34): integrity True; recorded 207, unchanged 202, changed 5, added 71, deleted 0; not recorded by any later snapshot: 0
   stage_05 (2026-10-14T10:57:13): integrity True; recorded 425, unchanged 423, changed 2, added 64, deleted 0; not recorded by any later snapshot: 0
   stage_07 (2026-10-14T10:57:15): integrity True; recorded 647, unchanged 645, changed 2, added 64, deleted 0; not recorded by any later snapshot: 0
   stage_06 (2026-10-14T15:51:44): integrity True; recorded 1075, unchanged 1074, changed 1, added 58, deleted 0; not recorded by any later snapshot: 0
   stage_08 (2026-10-14T16:10:41): integrity True; recorded 1126, unchanged 1125, changed 1, added 39, deleted 0; not recorded by any later snapshot: 0
   stage_09 (2026-10-14T16:25:26): integrity True; recorded 1178, unchanged 1172, changed 6, added 4, deleted 0; not recorded by any later snapshot: 0
   stage_10 (2026-10-14T16:44:42): integrity True; recorded 1183, unchanged 1183, changed 0, added 0, deleted 0; not recorded by any later snapshot: 0
B4 files whose current version is in no snapshot (Stage 10 edits): []
B4 deleted files: 0
B5 FINAL_RESULTS_REPORT.md: 1977 decimal numbers scanned (21 section numbers / cross-references skipped); 1973 match a value in results/final/*.csv (rounded as printed); not matched: 4 -> {'protocol constant: one-sided bound level (multiplicity note)': 2, 'derived (GPU time from reproducibility_table seconds)': 2}
```

First run of the audit (kept for the record, `logs/stage10_audit_first_run_note.txt`):

```
Stage 10 audit, first run (before exact CSV parsing), B1: 4000 checks, 23 mismatches, ALL at the stored validation
(Youden) threshold, none at threshold 0.5 and none in ROC-AUC:
- D05_toddler_screening/set_all/hist_gb folds 1-7, 10 (19 values: balanced accuracy + sensitivity): the predictions.csv text
  is 0.9999780982691833 (= the stored threshold) but pandas.read_csv's default C float parser returned 0.9999780982691832
  (1 ulp lower), so p >= threshold flipped for the tied positives.
- D01_eye_tracking/M02/fold_04 and M10/fold_07 (4 values: balanced accuracy + specificity): one test clip each has a float32
  probability equal to the float32 validation threshold; predictions.csv stores float32 values in shortest float32 form
  (0.9036349), which read as float64 lies just below the threshold 0.9036349058151245 (= float64 of the same float32).
Cause: the audit's own parsing, not the stored metrics. Fix in scripts/10_audit.py read_pred(): float_precision="round_trip"
and, for deep runs, cast back to float32. Second run: 0 mismatches. Stored files unchanged.
```

## 4. Snapshot check (B4, `logs/stage10_snapshot_changes.csv`)

| snapshot | datetime | checksums_integrity | recorded | unchanged | changed | added | deleted | current version in no later snapshot |
|---|---|---|---|---|---|---|---|---|
| stage_01 | 2026-10-13T16:56:45 | True | 33 | 32 | 1 | 245 | 0 | 0 |
| stage_02 | 2026-10-13T17:18:53 | True | 150 | 148 | 2 | 128 | 0 | 0 |
| stage_03 | 2026-10-14T09:10:29 | True | 186 | 174 | 12 | 92 | 0 | 0 |
| stage_04 | 2026-10-14T09:43:34 | True | 207 | 202 | 5 | 71 | 0 | 0 |
| stage_05 | 2026-10-14T10:57:13 | True | 425 | 423 | 2 | 64 | 0 | 0 |
| stage_07 | 2026-10-14T10:57:15 | True | 647 | 645 | 2 | 64 | 0 | 0 |
| stage_06 | 2026-10-14T15:51:44 | True | 1075 | 1074 | 1 | 58 | 0 | 0 |
| stage_08 | 2026-10-14T16:10:41 | True | 1126 | 1125 | 1 | 39 | 0 | 0 |
| stage_09 | 2026-10-14T16:25:26 | True | 1178 | 1172 | 6 | 4 | 0 | 0 |
| stage_10 | 2026-10-14T16:44:42 | True | 1183 | 1183 | 0 | 0 | 0 | 0 |

Every changed or added file is explained by the later snapshot that recorded its current version (the stage that legitimately edited it); `current version in no later snapshot` = 0 everywhere. Files changed or added in Stage 10 (stage_09 -> provisional stage_10, which was taken before this report): reports/FINAL_RESULTS_REPORT.html, reports/FINAL_RESULTS_REPORT.md, reports/STAGE_09_REPORT.md, reports/STAGE_10_REPORT.md, scripts/09d_write_report.py, scripts/10_audit.py, scripts/10b_finalize.py, scripts/10c_claims_scan.py, src/analysis/final.py, tests/test_stage04_units.py. After this report, the final stage_10 snapshot also records reports/STAGE_10_REPORT.md and results/FINAL_MANIFEST.json. No recorded file was deleted. The large 'added' counts of early snapshots are later stages' new code, reports and caches.

## 5. Tests (`.venv/bin/python -m pytest -v`, `logs/stage10_pytest.txt`, run WITH the read-only freeze in place)

Result: 360 passed in 129.92s (0:02:09)

Sequence: before the freeze 360 passed in 132.96s (0:02:12) (`logs/stage10_pytest_before_freeze.txt`). The first run after the freeze gave 1 failed, 359 passed in 131.25s (0:02:11) (`logs/stage10_pytest_after_freeze_first.txt`): `tests/test_stage04_units.py::test_verify_manifest_detects_modified_temp_copy` copies splits/ to a temp dir with shutil.copytree, which also copies the new read-only mode bits, so the test could not modify its own temp copy. Fix (test only): make the temp copy writable before modifying it. The real splits/ stay read-only and unchanged. Manifest, zip and snapshot were then regenerated.

<details><summary>full pytest output</summary>

```
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0 -- /home/score/Documents/AI_project/asd_hybrid_10fold/.venv/bin/python
rootdir: /home/score/Documents/AI_project/asd_hybrid_10fold
configfile: pytest.ini
testpaths: tests
plugins: anyio-4.15.1
collecting ... collected 360 items

tests/test_setup.py::test_project_dir_and_symlink PASSED                 [  0%]
tests/test_setup.py::test_paths_yaml_every_path_exists PASSED            [  0%]
tests/test_setup.py::test_datasets_dir_outside_project PASSED            [  0%]
tests/test_setup.py::test_no_absolute_paths_outside_paths_yaml PASSED    [  1%]
tests/test_setup.py::test_venv_imports PASSED                            [  1%]
tests/test_setup.py::test_cuda_available PASSED                          [  1%]
tests/test_setup.py::test_check_gpu_script_exits_zero PASSED             [  1%]
tests/test_setup.py::test_claude_md_sections PASSED                      [  2%]
tests/test_setup.py::test_raw_data_snapshot_matches PASSED               [  2%]
tests/test_setup.py::test_gitignore[.venv/] PASSED                       [  2%]
tests/test_setup.py::test_gitignore[cache/] PASSED                       [  3%]
tests/test_setup.py::test_gitignore[*.pt] PASSED                         [  3%]
tests/test_setup.py::test_gitignore[logs/*.log] PASSED                   [  3%]
tests/test_stage01.py::test_skeleton_dir_exists[configs] PASSED          [  3%]
tests/test_stage01.py::test_skeleton_dir_exists[src/data] PASSED         [  4%]
tests/test_stage01.py::test_skeleton_dir_exists[src/models] PASSED       [  4%]
tests/test_stage01.py::test_skeleton_dir_exists[src/training] PASSED     [  4%]
tests/test_stage01.py::test_skeleton_dir_exists[src/evaluation] PASSED   [  5%]
tests/test_stage01.py::test_skeleton_dir_exists[src/utils] PASSED        [  5%]
tests/test_stage01.py::test_skeleton_dir_exists[scripts] PASSED          [  5%]
tests/test_stage01.py::test_skeleton_dir_exists[tests] PASSED            [  5%]
tests/test_stage01.py::test_skeleton_dir_exists[data_registry] PASSED    [  6%]
tests/test_stage01.py::test_skeleton_dir_exists[splits] PASSED           [  6%]
tests/test_stage01.py::test_skeleton_dir_exists[cache] PASSED            [  6%]
tests/test_stage01.py::test_skeleton_dir_exists[results] PASSED          [  6%]
tests/test_stage01.py::test_skeleton_dir_exists[reports] PASSED          [  7%]
tests/test_stage01.py::test_skeleton_dir_exists[logs] PASSED             [  7%]
tests/test_stage01.py::test_results_readme_and_gitignore PASSED          [  7%]
tests/test_stage01.py::test_audit_json_loads_and_nonzero[D01_eye_tracking] PASSED [  8%]
tests/test_stage01.py::test_audit_json_loads_and_nonzero[D02_facial_images_A] PASSED [  8%]
tests/test_stage01.py::test_audit_json_loads_and_nonzero[D03_facial_images_B] PASSED [  8%]
tests/test_stage01.py::test_audit_json_loads_and_nonzero[D04_autism_prediction] PASSED [  8%]
tests/test_stage01.py::test_audit_json_loads_and_nonzero[D05_toddler_screening] PASSED [  9%]
tests/test_stage01.py::test_audit_json_loads_and_nonzero[D06_early_autism] PASSED [  9%]
tests/test_stage01.py::test_registry_exists PASSED                       [  9%]
tests/test_stage01.py::test_raw_data_unmodified PASSED                   [ 10%]
tests/test_stage01.py::test_cuda_available PASSED                        [ 10%]
tests/test_stage02_d01.py::test_no_unidentified_rows PASSED              [ 10%]
tests/test_stage02_d01.py::test_no_duplicate_participant_time_trial PASSED [ 10%]
tests/test_stage02_d01.py::test_no_cars_anywhere_in_cache_or_splits PASSED [ 11%]
tests/test_stage02_d01.py::test_npy_shape_matches_index PASSED           [ 11%]
tests/test_stage02_d01.py::test_clip_rules PASSED                        [ 11%]
tests/test_stage02_d01.py::test_no_clip_spans_two_trials PASSED          [ 11%]
tests/test_stage02_d01.py::test_20_random_clips_rerendered_from_raw_are_identical PASSED [ 12%]
tests/test_stage02_d01.py::test_rebuilt_gaze_equals_cached PASSED        [ 12%]
tests/test_stage02_d01.py::test_quality_table_not_in_model_inputs PASSED [ 12%]
tests/test_stage02_d02.py::test_manifest_pooled_and_complete PASSED      [ 13%]
tests/test_stage02_d02.py::test_no_d03_path_anywhere PASSED              [ 13%]
tests/test_stage02_d02.py::test_near_duplicate_pairs_recomputed_lie_in_same_outer_fold PASSED [ 13%]
tests/test_stage02_d02.py::test_no_group_has_mixed_labels PASSED         [ 13%]
tests/test_stage02_snapshot.py::test_latest_snapshot_checksums_match_current_files PASSED [ 14%]
tests/test_stage02_snapshot.py::test_latest_snapshot_info_and_code_copy PASSED [ 14%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[1-D01_eye_tracking] PASSED [ 14%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[1-D02_facial_images_A] PASSED [ 15%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[1-D04_autism_prediction] PASSED [ 15%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[1-D05_toddler_screening] PASSED [ 15%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[1-D06_early_autism] PASSED [ 15%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[2-D01_eye_tracking] PASSED [ 16%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[2-D02_facial_images_A] PASSED [ 16%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[2-D04_autism_prediction] PASSED [ 16%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[2-D05_toddler_screening] PASSED [ 16%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[2-D06_early_autism] PASSED [ 17%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[3-D01_eye_tracking] PASSED [ 17%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[3-D02_facial_images_A] PASSED [ 17%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[3-D04_autism_prediction] PASSED [ 18%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[3-D05_toddler_screening] PASSED [ 18%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[3-D06_early_autism] PASSED [ 18%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[4-D01_eye_tracking] PASSED [ 18%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[4-D02_facial_images_A] PASSED [ 19%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[4-D04_autism_prediction] PASSED [ 19%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[4-D05_toddler_screening] PASSED [ 19%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[4-D06_early_autism] PASSED [ 20%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[5-D01_eye_tracking] PASSED [ 20%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[5-D02_facial_images_A] PASSED [ 20%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[5-D04_autism_prediction] PASSED [ 20%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[5-D05_toddler_screening] PASSED [ 21%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[5-D06_early_autism] PASSED [ 21%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[6-D01_eye_tracking] PASSED [ 21%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[6-D02_facial_images_A] PASSED [ 21%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[6-D04_autism_prediction] PASSED [ 22%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[6-D05_toddler_screening] PASSED [ 22%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[6-D06_early_autism] PASSED [ 22%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[7-D01_eye_tracking] PASSED [ 23%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[7-D02_facial_images_A] PASSED [ 23%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[7-D04_autism_prediction] PASSED [ 23%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[7-D05_toddler_screening] PASSED [ 23%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[7-D06_early_autism] PASSED [ 24%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[8-D01_eye_tracking] PASSED [ 24%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[8-D02_facial_images_A] PASSED [ 24%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[8-D04_autism_prediction] PASSED [ 25%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[8-D05_toddler_screening] PASSED [ 25%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[8-D06_early_autism] PASSED [ 25%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[9-D01_eye_tracking] PASSED [ 25%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[9-D02_facial_images_A] PASSED [ 26%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[9-D04_autism_prediction] PASSED [ 26%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[9-D05_toddler_screening] PASSED [ 26%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[9-D06_early_autism] PASSED [ 26%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[10-D01_eye_tracking] PASSED [ 27%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[10-D02_facial_images_A] PASSED [ 27%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[10-D04_autism_prediction] PASSED [ 27%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[10-D05_toddler_screening] PASSED [ 28%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[10-D06_early_autism] PASSED [ 28%]
tests/test_stage02_splits.py::test_every_sample_in_exactly_one_test_fold[D01_eye_tracking] PASSED [ 28%]
tests/test_stage02_splits.py::test_every_sample_in_exactly_one_test_fold[D02_facial_images_A] PASSED [ 28%]
tests/test_stage02_splits.py::test_every_sample_in_exactly_one_test_fold[D04_autism_prediction] PASSED [ 29%]
tests/test_stage02_splits.py::test_every_sample_in_exactly_one_test_fold[D05_toddler_screening] PASSED [ 29%]
tests/test_stage02_splits.py::test_every_sample_in_exactly_one_test_fold[D06_early_autism] PASSED [ 29%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[1-D01_eye_tracking] PASSED [ 30%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[1-D02_facial_images_A] PASSED [ 30%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[1-D04_autism_prediction] PASSED [ 30%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[1-D05_toddler_screening] PASSED [ 30%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[1-D06_early_autism] PASSED [ 31%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[2-D01_eye_tracking] PASSED [ 31%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[2-D02_facial_images_A] PASSED [ 31%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[2-D04_autism_prediction] PASSED [ 31%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[2-D05_toddler_screening] PASSED [ 32%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[2-D06_early_autism] PASSED [ 32%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[3-D01_eye_tracking] PASSED [ 32%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[3-D02_facial_images_A] PASSED [ 33%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[3-D04_autism_prediction] PASSED [ 33%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[3-D05_toddler_screening] PASSED [ 33%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[3-D06_early_autism] PASSED [ 33%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[4-D01_eye_tracking] PASSED [ 34%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[4-D02_facial_images_A] PASSED [ 34%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[4-D04_autism_prediction] PASSED [ 34%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[4-D05_toddler_screening] PASSED [ 35%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[4-D06_early_autism] PASSED [ 35%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[5-D01_eye_tracking] PASSED [ 35%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[5-D02_facial_images_A] PASSED [ 35%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[5-D04_autism_prediction] PASSED [ 36%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[5-D05_toddler_screening] PASSED [ 36%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[5-D06_early_autism] PASSED [ 36%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[6-D01_eye_tracking] PASSED [ 36%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[6-D02_facial_images_A] PASSED [ 37%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[6-D04_autism_prediction] PASSED [ 37%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[6-D05_toddler_screening] PASSED [ 37%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[6-D06_early_autism] PASSED [ 38%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[7-D01_eye_tracking] PASSED [ 38%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[7-D02_facial_images_A] PASSED [ 38%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[7-D04_autism_prediction] PASSED [ 38%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[7-D05_toddler_screening] PASSED [ 39%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[7-D06_early_autism] PASSED [ 39%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[8-D01_eye_tracking] PASSED [ 39%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[8-D02_facial_images_A] PASSED [ 40%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[8-D04_autism_prediction] PASSED [ 40%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[8-D05_toddler_screening] PASSED [ 40%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[8-D06_early_autism] PASSED [ 40%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[9-D01_eye_tracking] PASSED [ 41%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[9-D02_facial_images_A] PASSED [ 41%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[9-D04_autism_prediction] PASSED [ 41%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[9-D05_toddler_screening] PASSED [ 41%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[9-D06_early_autism] PASSED [ 42%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[10-D01_eye_tracking] PASSED [ 42%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[10-D02_facial_images_A] PASSED [ 42%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[10-D04_autism_prediction] PASSED [ 43%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[10-D05_toddler_screening] PASSED [ 43%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[10-D06_early_autism] PASSED [ 43%]
tests/test_stage02_splits.py::test_d01_test_folds_have_asd_and_td_children PASSED [ 43%]
tests/test_stage02_splits.py::test_splits_manifest_hashes_match PASSED   [ 44%]
tests/test_stage02_splits.py::test_rerun_make_splits_is_byte_identical PASSED [ 44%]
tests/test_stage02_tabular.py::test_d04_only_labelled_train_rows PASSED  [ 44%]
tests/test_stage02_tabular.py::test_d04_result_decision_recorded PASSED  [ 45%]
tests/test_stage02_tabular.py::test_d05_honest_set_excludes_items_and_score PASSED [ 45%]
tests/test_stage02_tabular.py::test_d05_label_rule_on_clean_data PASSED  [ 45%]
tests/test_stage02_tabular.py::test_d05_other_csv_never_read_by_stage2_code PASSED [ 45%]
tests/test_stage02_tabular.py::test_d06_empty_column_absent PASSED       [ 46%]
tests/test_stage02_tabular.py::test_identical_feature_rows_share_group PASSED [ 46%]
tests/test_stage02_tabular.py::test_every_column_has_a_role PASSED       [ 46%]
tests/test_stage02_tabular.py::test_raw_data_snapshot_unchanged PASSED   [ 46%]
tests/test_stage03_models.py::test_config_mapping[M01] PASSED            [ 47%]
tests/test_stage03_models.py::test_config_mapping[M02] PASSED            [ 47%]
tests/test_stage03_models.py::test_config_mapping[M03] PASSED            [ 47%]
tests/test_stage03_models.py::test_config_mapping[M04] PASSED            [ 48%]
tests/test_stage03_models.py::test_config_mapping[M05] PASSED            [ 48%]
tests/test_stage03_models.py::test_config_mapping[M06] PASSED            [ 48%]
tests/test_stage03_models.py::test_config_mapping[M07] PASSED            [ 48%]
tests/test_stage03_models.py::test_config_mapping[M08] PASSED            [ 49%]
tests/test_stage03_models.py::test_config_mapping[M09] PASSED            [ 49%]
tests/test_stage03_models.py::test_config_mapping[M10] PASSED            [ 49%]
tests/test_stage03_models.py::test_factory_rejects_unknown PASSED        [ 50%]
tests/test_stage03_models.py::test_forward[M01-sequence] PASSED          [ 50%]
tests/test_stage03_models.py::test_forward[M01-tokens] PASSED            [ 50%]
tests/test_stage03_models.py::test_forward[M02-sequence] PASSED          [ 50%]
tests/test_stage03_models.py::test_forward[M02-tokens] PASSED            [ 51%]
tests/test_stage03_models.py::test_forward[M03-sequence] PASSED          [ 51%]
tests/test_stage03_models.py::test_forward[M03-tokens] PASSED            [ 51%]
tests/test_stage03_models.py::test_forward[M04-sequence] PASSED          [ 51%]
tests/test_stage03_models.py::test_forward[M04-tokens] PASSED            [ 52%]
tests/test_stage03_models.py::test_forward[M05-sequence] PASSED          [ 52%]
tests/test_stage03_models.py::test_forward[M05-tokens] PASSED            [ 52%]
tests/test_stage03_models.py::test_forward[M06-sequence] PASSED          [ 53%]
tests/test_stage03_models.py::test_forward[M06-tokens] PASSED            [ 53%]
tests/test_stage03_models.py::test_forward[M07-sequence] PASSED          [ 53%]
tests/test_stage03_models.py::test_forward[M07-tokens] PASSED            [ 53%]
tests/test_stage03_models.py::test_forward[M08-sequence] PASSED          [ 54%]
tests/test_stage03_models.py::test_forward[M08-tokens] PASSED            [ 54%]
tests/test_stage03_models.py::test_forward[M09-sequence] PASSED          [ 54%]
tests/test_stage03_models.py::test_forward[M09-tokens] PASSED            [ 55%]
tests/test_stage03_models.py::test_forward[M10-sequence] PASSED          [ 55%]
tests/test_stage03_models.py::test_forward[M10-tokens] PASSED            [ 55%]
tests/test_stage03_models.py::test_backward_and_step[M01-sequence] PASSED [ 55%]
tests/test_stage03_models.py::test_backward_and_step[M01-tokens] PASSED  [ 56%]
tests/test_stage03_models.py::test_backward_and_step[M02-sequence] PASSED [ 56%]
tests/test_stage03_models.py::test_backward_and_step[M02-tokens] PASSED  [ 56%]
tests/test_stage03_models.py::test_backward_and_step[M03-sequence] PASSED [ 56%]
tests/test_stage03_models.py::test_backward_and_step[M03-tokens] PASSED  [ 57%]
tests/test_stage03_models.py::test_backward_and_step[M04-sequence] PASSED [ 57%]
tests/test_stage03_models.py::test_backward_and_step[M04-tokens] PASSED  [ 57%]
tests/test_stage03_models.py::test_backward_and_step[M05-sequence] PASSED [ 58%]
tests/test_stage03_models.py::test_backward_and_step[M05-tokens] PASSED  [ 58%]
tests/test_stage03_models.py::test_backward_and_step[M06-sequence] PASSED [ 58%]
tests/test_stage03_models.py::test_backward_and_step[M06-tokens] PASSED  [ 58%]
tests/test_stage03_models.py::test_backward_and_step[M07-sequence] PASSED [ 59%]
tests/test_stage03_models.py::test_backward_and_step[M07-tokens] PASSED  [ 59%]
tests/test_stage03_models.py::test_backward_and_step[M08-sequence] PASSED [ 59%]
tests/test_stage03_models.py::test_backward_and_step[M08-tokens] PASSED  [ 60%]
tests/test_stage03_models.py::test_backward_and_step[M09-sequence] PASSED [ 60%]
tests/test_stage03_models.py::test_backward_and_step[M09-tokens] PASSED  [ 60%]
tests/test_stage03_models.py::test_backward_and_step[M10-sequence] PASSED [ 60%]
tests/test_stage03_models.py::test_backward_and_step[M10-tokens] PASSED  [ 61%]
tests/test_stage03_models.py::test_determinism_eval[M01-sequence] PASSED [ 61%]
tests/test_stage03_models.py::test_determinism_eval[M01-tokens] PASSED   [ 61%]
tests/test_stage03_models.py::test_determinism_eval[M02-sequence] PASSED [ 61%]
tests/test_stage03_models.py::test_determinism_eval[M02-tokens] PASSED   [ 62%]
tests/test_stage03_models.py::test_determinism_eval[M03-sequence] PASSED [ 62%]
tests/test_stage03_models.py::test_determinism_eval[M03-tokens] PASSED   [ 62%]
tests/test_stage03_models.py::test_determinism_eval[M04-sequence] PASSED [ 63%]
tests/test_stage03_models.py::test_determinism_eval[M04-tokens] PASSED   [ 63%]
tests/test_stage03_models.py::test_determinism_eval[M05-sequence] PASSED [ 63%]
tests/test_stage03_models.py::test_determinism_eval[M05-tokens] PASSED   [ 63%]
tests/test_stage03_models.py::test_determinism_eval[M06-sequence] PASSED [ 64%]
tests/test_stage03_models.py::test_determinism_eval[M06-tokens] PASSED   [ 64%]
tests/test_stage03_models.py::test_determinism_eval[M07-sequence] PASSED [ 64%]
tests/test_stage03_models.py::test_determinism_eval[M07-tokens] PASSED   [ 65%]
tests/test_stage03_models.py::test_determinism_eval[M08-sequence] PASSED [ 65%]
tests/test_stage03_models.py::test_determinism_eval[M08-tokens] PASSED   [ 65%]
tests/test_stage03_models.py::test_determinism_eval[M09-sequence] PASSED [ 65%]
tests/test_stage03_models.py::test_determinism_eval[M09-tokens] PASSED   [ 66%]
tests/test_stage03_models.py::test_determinism_eval[M10-sequence] PASSED [ 66%]
tests/test_stage03_models.py::test_determinism_eval[M10-tokens] PASSED   [ 66%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M01-sequence] PASSED [ 66%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M01-tokens] PASSED [ 67%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M02-sequence] PASSED [ 67%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M02-tokens] PASSED [ 67%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M03-sequence] PASSED [ 68%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M03-tokens] PASSED [ 68%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M04-sequence] PASSED [ 68%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M04-tokens] PASSED [ 68%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M05-sequence] PASSED [ 69%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M05-tokens] PASSED [ 69%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M06-sequence] PASSED [ 69%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M06-tokens] PASSED [ 70%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M07-sequence] PASSED [ 70%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M07-tokens] PASSED [ 70%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M08-sequence] PASSED [ 70%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M08-tokens] PASSED [ 71%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M09-sequence] PASSED [ 71%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M09-tokens] PASSED [ 71%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M10-sequence] PASSED [ 71%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M10-tokens] PASSED [ 72%]
tests/test_stage03_models.py::test_feature_cache_equivalence[M01] PASSED [ 72%]
tests/test_stage03_models.py::test_feature_cache_equivalence[M03] PASSED [ 72%]
tests/test_stage03_models.py::test_feature_cache_equivalence[M04] PASSED [ 73%]
tests/test_stage03_models.py::test_feature_cache_equivalence[M06] PASSED [ 73%]
tests/test_stage03_models.py::test_feature_cache_equivalence[M07] PASSED [ 73%]
tests/test_stage03_models.py::test_feature_cache_equivalence[M08] PASSED [ 73%]
tests/test_stage03_models.py::test_feature_cache_equivalence[M10] PASSED [ 74%]
tests/test_stage03_models.py::test_last_stage_trains_part_of_backbone PASSED [ 74%]
tests/test_stage03_models.py::test_model_configs_have_batch_sizes PASSED [ 74%]
tests/test_stage04_engine.py::test_interrupt_leaves_running_and_checkpoint[M01] PASSED [ 75%]
tests/test_stage04_engine.py::test_resume_identical_history_and_predictions[M01] PASSED [ 75%]
tests/test_stage04_engine.py::test_resumed_run_info_and_cleanup[M01] PASSED [ 75%]
tests/test_stage04_engine.py::test_youden_threshold_comes_from_validation_predictions[M01] PASSED [ 75%]
tests/test_stage04_engine.py::test_done_run_is_skipped_unless_forced[M01] PASSED [ 76%]
tests/test_stage04_engine.py::test_checks_detect_incomplete_results[M01] PASSED [ 76%]
tests/test_stage04_engine.py::test_interrupt_leaves_running_and_checkpoint[M09] PASSED [ 76%]
tests/test_stage04_engine.py::test_resume_identical_history_and_predictions[M09] PASSED [ 76%]
tests/test_stage04_engine.py::test_resumed_run_info_and_cleanup[M09] PASSED [ 77%]
tests/test_stage04_engine.py::test_youden_threshold_comes_from_validation_predictions[M09] PASSED [ 77%]
tests/test_stage04_engine.py::test_done_run_is_skipped_unless_forced[M09] PASSED [ 77%]
tests/test_stage04_engine.py::test_checks_detect_incomplete_results[M09] PASSED [ 78%]
tests/test_stage04_engine.py::test_engine_refuses_corrupted_split PASSED [ 78%]
tests/test_stage04_engine.py::test_resume_refuses_changed_config PASSED  [ 78%]
tests/test_stage04_units.py::test_training_protocol_values_fixed PASSED  [ 78%]
tests/test_stage04_units.py::test_merge_config[M01-D01_eye_tracking-1-0.0005-0.0-cached_features] PASSED [ 79%]
tests/test_stage04_units.py::test_merge_config[M01-D02_facial_images_A-3-0.0005-5e-05-images] PASSED [ 79%]
tests/test_stage04_units.py::test_merge_config[M07-D02_facial_images_A-10-0.0005-5e-05-images] PASSED [ 79%]
tests/test_stage04_units.py::test_merge_config[M09-D01_eye_tracking-2-0.001-0.001-clips] PASSED [ 80%]
tests/test_stage04_units.py::test_merge_config[M05-D02_facial_images_A-4-0.001-0.001-images] PASSED [ 80%]
tests/test_stage04_units.py::test_model_config_lr_mirrors_training_yaml PASSED [ 80%]
tests/test_stage04_units.py::test_scheduler_warmup_and_cosine PASSED     [ 80%]
tests/test_stage04_units.py::test_d02_transforms_uint8 PASSED            [ 81%]
tests/test_stage04_units.py::test_real_splits_pass_check_fold[D01_eye_tracking] PASSED [ 81%]
tests/test_stage04_units.py::test_real_splits_pass_check_fold[D02_facial_images_A] PASSED [ 81%]
tests/test_stage04_units.py::test_leakage_asserts_trigger_on_corrupted_split[id_in_train_and_test-D01_eye_tracking] PASSED [ 81%]
tests/test_stage04_units.py::test_leakage_asserts_trigger_on_corrupted_split[id_in_train_and_test-D02_facial_images_A] PASSED [ 82%]
tests/test_stage04_units.py::test_leakage_asserts_trigger_on_corrupted_split[duplicate-D01_eye_tracking] PASSED [ 82%]
tests/test_stage04_units.py::test_leakage_asserts_trigger_on_corrupted_split[duplicate-D02_facial_images_A] PASSED [ 82%]
tests/test_stage04_units.py::test_leakage_asserts_trigger_on_corrupted_split[group_overlap-D01_eye_tracking] PASSED [ 83%]
tests/test_stage04_units.py::test_leakage_asserts_trigger_on_corrupted_split[group_overlap-D02_facial_images_A] PASSED [ 83%]
tests/test_stage04_units.py::test_leakage_asserts_trigger_on_corrupted_split[unknown_id-D01_eye_tracking] PASSED [ 83%]
tests/test_stage04_units.py::test_leakage_asserts_trigger_on_corrupted_split[unknown_id-D02_facial_images_A] PASSED [ 83%]
tests/test_stage04_units.py::test_test_set_must_match_folds_csv PASSED   [ 84%]
tests/test_stage04_units.py::test_verify_manifest_detects_modified_temp_copy PASSED [ 84%]
tests/test_stage04_units.py::test_assert_no_test_ids PASSED              [ 84%]
tests/test_stage04_units.py::test_binary_metrics_match_sklearn[0.5-0] PASSED [ 85%]
tests/test_stage04_units.py::test_binary_metrics_match_sklearn[0.5-1] PASSED [ 85%]
tests/test_stage04_units.py::test_binary_metrics_match_sklearn[0.5-2] PASSED [ 85%]
tests/test_stage04_units.py::test_binary_metrics_match_sklearn[0.3-0] PASSED [ 85%]
tests/test_stage04_units.py::test_binary_metrics_match_sklearn[0.3-1] PASSED [ 86%]
tests/test_stage04_units.py::test_binary_metrics_match_sklearn[0.3-2] PASSED [ 86%]
tests/test_stage04_units.py::test_binary_metrics_match_sklearn[0.72-0] PASSED [ 86%]
tests/test_stage04_units.py::test_binary_metrics_match_sklearn[0.72-1] PASSED [ 86%]
tests/test_stage04_units.py::test_binary_metrics_match_sklearn[0.72-2] PASSED [ 87%]
tests/test_stage04_units.py::test_binary_metrics_known_values PASSED     [ 87%]
tests/test_stage04_units.py::test_youden_threshold_brute_force_on_validation_only[0] PASSED [ 87%]
tests/test_stage04_units.py::test_youden_threshold_brute_force_on_validation_only[1] PASSED [ 88%]
tests/test_stage04_units.py::test_youden_threshold_brute_force_on_validation_only[2] PASSED [ 88%]
tests/test_stage04_units.py::test_youden_threshold_brute_force_on_validation_only[3] PASSED [ 88%]
tests/test_stage04_units.py::test_child_level_mean_probability PASSED    [ 88%]
tests/test_stage04_units.py::test_registry_skip_force_resume PASSED      [ 89%]
tests/test_stage04_units.py::test_registry_file_lock_no_lost_rows PASSED [ 89%]
tests/test_stage04_units.py::test_summary_aggregation_on_fake_folds PASSED [ 89%]
tests/test_stage05_checker.py::test_reference_block_is_valid PASSED      [ 90%]
tests/test_stage05_checker.py::test_perfect_pr_auc_passes PASSED         [ 90%]
tests/test_stage05_checker.py::test_auc_zero_passes PASSED               [ 90%]
tests/test_stage05_checker.py::test_value_above_one_fails PASSED         [ 90%]
tests/test_stage05_checker.py::test_negative_value_fails PASSED          [ 91%]
tests/test_stage05_checker.py::test_nan_is_flagged_never_silent PASSED   [ 91%]
tests/test_stage05_checker.py::test_nan_where_defined_fails PASSED       [ 91%]
tests/test_stage05_checker.py::test_missing_or_non_numeric_fails PASSED  [ 91%]
tests/test_stage08.py::test_bootstrap_reproducible_with_seed PASSED      [ 92%]
tests/test_stage08.py::test_cluster_bootstrap_keeps_groups_whole_and_both_classes PASSED [ 92%]
tests/test_stage08.py::test_delong_known_example PASSED                  [ 92%]
tests/test_stage08.py::test_delong_with_ties_matches_bruteforce_and_identical_scores_give_p1 PASSED [ 93%]
tests/test_stage08.py::test_friedman_toy_matrix PASSED                   [ 93%]
tests/test_stage08.py::test_nadeau_bengio_and_holm PASSED                [ 93%]
tests/test_stage08.py::test_ece_and_matching PASSED                      [ 93%]
tests/test_stage08.py::test_quality_baseline_uses_only_train_fold_children PASSED [ 94%]
tests/test_stage08.py::test_results_d0x_unchanged_since_preflight PASSED [ 94%]
tests/test_stage08.py::test_stage8_outputs_present PASSED                [ 94%]
tests/test_stage09.py::test_master_table_every_number_matches_its_source PASSED [ 95%]
tests/test_stage09.py::test_long_tables_match_sources[shortcut_table.csv] PASSED [ 95%]
tests/test_stage09.py::test_long_tables_match_sources[stats_table.csv] PASSED [ 95%]
tests/test_stage09.py::test_long_tables_match_sources[fold_auc_matrix.csv] PASSED [ 95%]
tests/test_stage09.py::test_resolver_rejects_ambiguous_filters PASSED    [ 96%]
tests/test_stage09.py::test_figure_files_exist_nonempty_300dpi[F01_pipeline] PASSED [ 96%]
tests/test_stage09.py::test_figure_files_exist_nonempty_300dpi[F02_d01_auc_vs_quality] PASSED [ 96%]
tests/test_stage09.py::test_figure_files_exist_nonempty_300dpi[F03_d01_forest_vs_quality] PASSED [ 96%]
tests/test_stage09.py::test_figure_files_exist_nonempty_300dpi[F04_d01_matched_subset] PASSED [ 97%]
tests/test_stage09.py::test_figure_files_exist_nonempty_300dpi[F05_d02_auc_vs_image_stats] PASSED [ 97%]
tests/test_stage09.py::test_figure_files_exist_nonempty_300dpi[F06_d02_roc_top3] PASSED [ 97%]
tests/test_stage09.py::test_figure_files_exist_nonempty_300dpi[F07_cd_diagrams] PASSED [ 98%]
tests/test_stage09.py::test_figure_files_exist_nonempty_300dpi[F08_confusion_matrices] PASSED [ 98%]
tests/test_stage09.py::test_figure_files_exist_nonempty_300dpi[F09_d01_m07_attention] PASSED [ 98%]
tests/test_stage09.py::test_figure_files_exist_nonempty_300dpi[F10_tabular_auc] PASSED [ 98%]
tests/test_stage09.py::test_figure_files_exist_nonempty_300dpi[F11_pretrained_vs_scratch] PASSED [ 99%]
tests/test_stage09.py::test_figure_files_exist_nonempty_300dpi[F12_reliability_best] PASSED [ 99%]
tests/test_stage09.py::test_results_unchanged_since_stage9_preflight PASSED [ 99%]
tests/test_stage09.py::test_final_report_present_with_abstract_length PASSED [100%]

======================= 360 passed in 129.92s (0:02:09) ========================
```

</details>

## 6. Freeze (C)

- `results/FINAL_MANIFEST.json`: sha256 + size of every file in `results/`, `splits/`, `reports/`, `configs/`, `src/`, `tabular/`, `scripts/` (no *.pt, no cache/, no __pycache__; this report included), environment summary and total compute time; written after this report.
- Read-only after the manifest: `chmod -R a-w results splits reports/figures/final`; undo with `chmod -R u+w results splits reports/figures/final`.
- `submission/ASD_Hybrid_10fold_submission.zip`: 247 files (src/, tabular/, scripts/, configs/, tests/, splits/, results/final/, reports/*.md + *.html, reports/figures/final/, CLAUDE.md, README.md, requirements.txt, results/FINAL_MANIFEST.json; no raw data, no cache/, no *.pt). Size and sha256: `submission/SUBMISSION_INFO.json`.
- `README.md`: summary, reproduction commands, locations, five findings and the limitations copied verbatim from FINAL_RESULTS_REPORT.md.

STAGE 10 DECISION: PASS
