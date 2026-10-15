# STAGE 4 REPORT - training engine + smoke tests

Screening-support research, not diagnosis. The values below come from 2-epoch smoke tests and are meaningless as performance estimates; they only show that the engine produces valid outputs. No full experiment was run; smoke outputs are in `results_smoke/`, `results/` contains no run output (confirmed). No git: recorded by `snapshots/stage_04/`.

## 1. Summary

- `configs/training.yaml` fixes the training protocol before any real result exists (section 3).
- `src/training/` (data, metrics, engine, registry, summarize, checks) + `scripts/train.py`, `scripts/run_queue.py`, `scripts/run_stage5.sh`, `scripts/run_stage6.sh` (written, not run).
- Tests: 323 passed in 124.44s (0:02:04) (whole suite; Stage 4 adds 40 unit + 14 GPU engine tests).
- Smoke tests 1-4 on real data: all validity checks PASS; 0 test ids in any training/validation batch in every run.
- Resume: deterministic cuDNN -> bit-identical (max |diff| 0.0e+00); interruption test resumed at epoch 2 and finished DONE.
- Found and fixed while testing: a job started with `&` from a non-interactive shell (nohup, scripts) inherits SIGINT as IGNORED, so the first interruption attempt was not interrupted at all (section 7.1). `train.py` and `run_queue.py` now install explicit SIGINT/SIGTERM handlers.
- Estimated S5 (D01): 0.78 h at ~20 epochs, 1.47 h at 40; S6 (D02): 2.95 h at ~20 epochs, 5.74 h at 40 (section 8).

## 2. Pre-flight (`logs/stage04_preflight.txt`)

```
# Stage 4 pre-flight 2026-10-14T09:17:41+05:30
A1 STAGE_03_REPORT last line: STAGE 3 DECISION: PASS
A1 snapshots/stage_03 checksums match current files: test_stage02_snapshot PASSED (2/2)
A2 verify_manifest(): OK (all split file sha256 match SPLITS_MANIFEST.json)
A2 check_gpu: GPU CHECK: OK
A2 nvidia-smi compute processes: 0
A2 full pytest: ======================= 269 passed in 109.94s (0:01:49) ========================
```

## 3. Training protocol (fixed before any real result)

| item | value (configs/training.yaml) |
|---|---|
| optimizer | AdamW, weight_decay 0.01 |
| head_lr | 0.0005 (pretrained M01 M03 M04 M06 M07 M08 M10) / 0.001 (custom CNN M02 M05 M09) |
| backbone_lr | head_lr x 0.1 when last_stage (D02 pretrained); = head_lr for the custom CNN (full); none when frozen (D01 pretrained) |
| schedule | cosine per optimizer step to 0 at max_epochs, 2-epoch linear warm-up |
| max_epochs | 40 |
| early stopping | validation ROC-AUC (D01: clip level), patience 8, min_delta 0.0001; best-val-AUC checkpoint kept |
| loss | BCEWithLogits, pos_weight = n_neg / n_pos of the TRAIN part of the fold (val loss uses the same pos_weight) |
| gradient clipping | global norm 1.0 (after GradScaler.unscale_) |
| precision | torch.autocast('cuda', float16) + torch.amp.GradScaler |
| batch size | configs/models/Mxx.yaml (64 for all 20 model x mode combinations) |
| seed | 42 + fold (python, numpy, torch, cuda, per-epoch data-loader generators) |
| cuDNN | benchmark=True (golden rule 10; nondeterministic in the last digits, recorded in run_info) |
| thresholds | primary 0.5; secondary = max Youden J on the VALIDATION predictions of the best checkpoint (D01 child level: on validation children); never chosen on test |
| D02 train augmentation | RandomResizedCrop(224, scale 0.8-1.0), horizontal flip p 0.5, ColorJitter(0.1, 0.1, 0.1) |
| D02 eval | resize shorter side to 224 + center crop 224 |
| D01 | no augmentation; frozen pretrained backbones read cache/features (Stage 3); M02 M05 M09 train end to end on uint8 clips |
| data loaders | num_workers = min(8, cpu_count) = 8, pin_memory, re-created each epoch with a generator seeded by (seed, epoch, split) |

Model configs (`configs/models/Mxx.yaml` `lr`) were updated to mirror these learning rates; `training.yaml` is authoritative and is merged over the model config (test `test_model_config_lr_mirrors_training_yaml`). Decisions taken here: D01 early stopping uses the clip-level validation AUC (7-8 validation children give a very coarse child AUC; it is logged as `val_auc_child`); `configs/data/D02.yaml` says normalisation happens in the loader, but the Stage 3 model contract (newer) normalises inside the model, so loaders pass raw uint8.

## 4. Engine design

| file | role |
|---|---|
| `src/training/data.py` | datasets built only from `splits/<ds>/fold_XX.json` ids; labels/groups from the hash-verified `folds.csv`, cross-checked against the data manifest; D01 cached features (in RAM) or uint8 clips (memmap), D02 PIL images with train/eval transforms; items carry their index so every batch's ids are logged |
| `src/training/metrics.py` | binary metrics (checked against sklearn), validation-only Youden threshold, child-level aggregation, range checks |
| `src/training/engine.py` | `merge_config`, `check_fold` (leakage asserts), `train_one_fold` |
| `src/training/registry.py` | `run_registry.csv`, fcntl-locked read-modify-write, skip/force logic |
| `src/training/summarize.py` | `summary.json` per model, `leaderboard.csv` per dataset |
| `src/training/checks.py` | `validate_model_results` (queue gate) |
| `scripts/train.py` / `run_queue.py` | entry points (signal handling, subprocess per model, stop on FAIL) |

`train_one_fold` order: `verify_manifest()` -> `check_fold` (no shared sample id or group between train/val/test, no duplicates or unknown ids, both classes everywhere, test = folds.csv outer_fold) -> registry decision (DONE: skip unless --force; RUNNING + last.pt: resume; else fresh start in a clean fold folder) -> per epoch: train (AMP, clipping, per-step cosine), validation, history row, best_model.pt on improvement, last.pt (model, optimizer, scheduler, scaler, epoch, best AUC, patience counter, history, ids seen per split, batch counts, python/numpy/torch/cuda RNG states) -> load best_model.pt -> validation (Youden) + test predicted once -> assert no test id was ever in a train/val batch, every train/val batch id belongs to its split, test predictions cover the test ids exactly once -> predictions.csv, val_predictions.csv, attention.npy, metrics.json, run_info.json -> registry DONE, last.pt deleted.

Exact resume: data loaders are re-created every epoch with a generator seeded by (seed, epoch, split), so shuffle order and worker (augmentation) seeds depend only on the epoch; resume refuses a last.pt written with a different config hash. Ctrl-C / SIGINT / SIGTERM -> status stays RUNNING, last.pt kept; any other exception -> FAILED with the error in the registry. Checkpoints are written atomically (tmp + rename). Pretrained weights are re-hashed against `reports/stage03_weights.csv` at every run start (HF_HUB_OFFLINE=1 in train.py: no network). Logs: `logs/<dataset>/<model>/fold_XX.log` for `results/`, `logs/<out root>/<dataset>/<model>/fold_XX.log` otherwise, plus one line per epoch on stdout.

## 5. Tests (`.venv/bin/python -m pytest -v`, `logs/stage04_pytest.txt`)

| file | tests |
|---|---|
| tests/test_setup.py | 13 |
| tests/test_stage01.py | 24 |
| tests/test_stage02_d01.py | 9 |
| tests/test_stage02_d02.py | 4 |
| tests/test_stage02_snapshot.py | 2 |
| tests/test_stage02_splits.py | 108 |
| tests/test_stage02_tabular.py | 9 |
| tests/test_stage03_models.py | 100 |
| tests/test_stage04_engine.py | 14 |
| tests/test_stage04_units.py | 40 |

Stage 4 tests: protocol values; merged learning rates per policy; scheduler shape; D02 transforms; real splits pass `check_fold`; corrupted temp copies trigger `LeakageError` (id in train and test, duplicate, group overlap, unknown id, test set != folds.csv) for D01 and D02; `verify_manifest` rejects a modified temp copy of splits/; the engine refuses a corrupted split before starting; metrics == sklearn on random inputs at 3 thresholds; Youden threshold == brute force on validation data and only takes validation arrays; metrics.json threshold == Youden of val_predictions.csv; registry skip/force/resume and 4 concurrent writer processes lose no rows; summary mean/std/min/max/pooled on fake folds; resume (D01 M01 cached, D02 M09 images with augmentation and 8 workers): 3 epochs straight == 1 epoch + simulated kill + resume (history and test predictions within 1e-5), interrupted state = RUNNING + last.pt, DONE run skipped, resume refused after a config change, checks catch incomplete results.

<details><summary>full pytest output</summary>

```
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0 -- /home/score/Documents/AI_project/asd_hybrid_10fold/.venv/bin/python
rootdir: /home/score/Documents/AI_project/asd_hybrid_10fold
configfile: pytest.ini
testpaths: tests
plugins: anyio-4.15.1
collecting ... collected 323 items

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
tests/test_setup.py::test_gitignore[logs/*.log] PASSED                   [  4%]
tests/test_stage01.py::test_skeleton_dir_exists[configs] PASSED          [  4%]
tests/test_stage01.py::test_skeleton_dir_exists[src/data] PASSED         [  4%]
tests/test_stage01.py::test_skeleton_dir_exists[src/models] PASSED       [  4%]
tests/test_stage01.py::test_skeleton_dir_exists[src/training] PASSED     [  5%]
tests/test_stage01.py::test_skeleton_dir_exists[src/evaluation] PASSED   [  5%]
tests/test_stage01.py::test_skeleton_dir_exists[src/utils] PASSED        [  5%]
tests/test_stage01.py::test_skeleton_dir_exists[scripts] PASSED          [  6%]
tests/test_stage01.py::test_skeleton_dir_exists[tests] PASSED            [  6%]
tests/test_stage01.py::test_skeleton_dir_exists[data_registry] PASSED    [  6%]
tests/test_stage01.py::test_skeleton_dir_exists[splits] PASSED           [  7%]
tests/test_stage01.py::test_skeleton_dir_exists[cache] PASSED            [  7%]
tests/test_stage01.py::test_skeleton_dir_exists[results] PASSED          [  7%]
tests/test_stage01.py::test_skeleton_dir_exists[reports] PASSED          [  8%]
tests/test_stage01.py::test_skeleton_dir_exists[logs] PASSED             [  8%]
tests/test_stage01.py::test_results_readme_and_gitignore PASSED          [  8%]
tests/test_stage01.py::test_audit_json_loads_and_nonzero[D01_eye_tracking] PASSED [  8%]
tests/test_stage01.py::test_audit_json_loads_and_nonzero[D02_facial_images_A] PASSED [  9%]
tests/test_stage01.py::test_audit_json_loads_and_nonzero[D03_facial_images_B] PASSED [  9%]
tests/test_stage01.py::test_audit_json_loads_and_nonzero[D04_autism_prediction] PASSED [  9%]
tests/test_stage01.py::test_audit_json_loads_and_nonzero[D05_toddler_screening] PASSED [ 10%]
tests/test_stage01.py::test_audit_json_loads_and_nonzero[D06_early_autism] PASSED [ 10%]
tests/test_stage01.py::test_registry_exists PASSED                       [ 10%]
tests/test_stage01.py::test_raw_data_unmodified PASSED                   [ 11%]
tests/test_stage01.py::test_cuda_available PASSED                        [ 11%]
tests/test_stage02_d01.py::test_no_unidentified_rows PASSED              [ 11%]
tests/test_stage02_d01.py::test_no_duplicate_participant_time_trial PASSED [ 12%]
tests/test_stage02_d01.py::test_no_cars_anywhere_in_cache_or_splits PASSED [ 12%]
tests/test_stage02_d01.py::test_npy_shape_matches_index PASSED           [ 12%]
tests/test_stage02_d01.py::test_clip_rules PASSED                        [ 13%]
tests/test_stage02_d01.py::test_no_clip_spans_two_trials PASSED          [ 13%]
tests/test_stage02_d01.py::test_20_random_clips_rerendered_from_raw_are_identical PASSED [ 13%]
tests/test_stage02_d01.py::test_rebuilt_gaze_equals_cached PASSED        [ 13%]
tests/test_stage02_d01.py::test_quality_table_not_in_model_inputs PASSED [ 14%]
tests/test_stage02_d02.py::test_manifest_pooled_and_complete PASSED      [ 14%]
tests/test_stage02_d02.py::test_no_d03_path_anywhere PASSED              [ 14%]
tests/test_stage02_d02.py::test_near_duplicate_pairs_recomputed_lie_in_same_outer_fold PASSED [ 15%]
tests/test_stage02_d02.py::test_no_group_has_mixed_labels PASSED         [ 15%]
tests/test_stage02_snapshot.py::test_latest_snapshot_checksums_match_current_files PASSED [ 15%]
tests/test_stage02_snapshot.py::test_latest_snapshot_info_and_code_copy PASSED [ 16%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[1-D01_eye_tracking] PASSED [ 16%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[1-D02_facial_images_A] PASSED [ 16%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[1-D04_autism_prediction] PASSED [ 17%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[1-D05_toddler_screening] PASSED [ 17%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[1-D06_early_autism] PASSED [ 17%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[2-D01_eye_tracking] PASSED [ 17%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[2-D02_facial_images_A] PASSED [ 18%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[2-D04_autism_prediction] PASSED [ 18%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[2-D05_toddler_screening] PASSED [ 18%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[2-D06_early_autism] PASSED [ 19%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[3-D01_eye_tracking] PASSED [ 19%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[3-D02_facial_images_A] PASSED [ 19%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[3-D04_autism_prediction] PASSED [ 20%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[3-D05_toddler_screening] PASSED [ 20%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[3-D06_early_autism] PASSED [ 20%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[4-D01_eye_tracking] PASSED [ 21%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[4-D02_facial_images_A] PASSED [ 21%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[4-D04_autism_prediction] PASSED [ 21%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[4-D05_toddler_screening] PASSED [ 21%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[4-D06_early_autism] PASSED [ 22%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[5-D01_eye_tracking] PASSED [ 22%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[5-D02_facial_images_A] PASSED [ 22%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[5-D04_autism_prediction] PASSED [ 23%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[5-D05_toddler_screening] PASSED [ 23%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[5-D06_early_autism] PASSED [ 23%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[6-D01_eye_tracking] PASSED [ 24%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[6-D02_facial_images_A] PASSED [ 24%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[6-D04_autism_prediction] PASSED [ 24%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[6-D05_toddler_screening] PASSED [ 25%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[6-D06_early_autism] PASSED [ 25%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[7-D01_eye_tracking] PASSED [ 25%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[7-D02_facial_images_A] PASSED [ 26%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[7-D04_autism_prediction] PASSED [ 26%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[7-D05_toddler_screening] PASSED [ 26%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[7-D06_early_autism] PASSED [ 26%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[8-D01_eye_tracking] PASSED [ 27%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[8-D02_facial_images_A] PASSED [ 27%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[8-D04_autism_prediction] PASSED [ 27%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[8-D05_toddler_screening] PASSED [ 28%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[8-D06_early_autism] PASSED [ 28%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[9-D01_eye_tracking] PASSED [ 28%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[9-D02_facial_images_A] PASSED [ 29%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[9-D04_autism_prediction] PASSED [ 29%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[9-D05_toddler_screening] PASSED [ 29%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[9-D06_early_autism] PASSED [ 30%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[10-D01_eye_tracking] PASSED [ 30%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[10-D02_facial_images_A] PASSED [ 30%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[10-D04_autism_prediction] PASSED [ 30%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[10-D05_toddler_screening] PASSED [ 31%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[10-D06_early_autism] PASSED [ 31%]
tests/test_stage02_splits.py::test_every_sample_in_exactly_one_test_fold[D01_eye_tracking] PASSED [ 31%]
tests/test_stage02_splits.py::test_every_sample_in_exactly_one_test_fold[D02_facial_images_A] PASSED [ 32%]
tests/test_stage02_splits.py::test_every_sample_in_exactly_one_test_fold[D04_autism_prediction] PASSED [ 32%]
tests/test_stage02_splits.py::test_every_sample_in_exactly_one_test_fold[D05_toddler_screening] PASSED [ 32%]
tests/test_stage02_splits.py::test_every_sample_in_exactly_one_test_fold[D06_early_autism] PASSED [ 33%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[1-D01_eye_tracking] PASSED [ 33%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[1-D02_facial_images_A] PASSED [ 33%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[1-D04_autism_prediction] PASSED [ 34%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[1-D05_toddler_screening] PASSED [ 34%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[1-D06_early_autism] PASSED [ 34%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[2-D01_eye_tracking] PASSED [ 34%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[2-D02_facial_images_A] PASSED [ 35%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[2-D04_autism_prediction] PASSED [ 35%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[2-D05_toddler_screening] PASSED [ 35%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[2-D06_early_autism] PASSED [ 36%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[3-D01_eye_tracking] PASSED [ 36%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[3-D02_facial_images_A] PASSED [ 36%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[3-D04_autism_prediction] PASSED [ 37%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[3-D05_toddler_screening] PASSED [ 37%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[3-D06_early_autism] PASSED [ 37%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[4-D01_eye_tracking] PASSED [ 38%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[4-D02_facial_images_A] PASSED [ 38%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[4-D04_autism_prediction] PASSED [ 38%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[4-D05_toddler_screening] PASSED [ 39%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[4-D06_early_autism] PASSED [ 39%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[5-D01_eye_tracking] PASSED [ 39%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[5-D02_facial_images_A] PASSED [ 39%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[5-D04_autism_prediction] PASSED [ 40%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[5-D05_toddler_screening] PASSED [ 40%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[5-D06_early_autism] PASSED [ 40%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[6-D01_eye_tracking] PASSED [ 41%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[6-D02_facial_images_A] PASSED [ 41%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[6-D04_autism_prediction] PASSED [ 41%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[6-D05_toddler_screening] PASSED [ 42%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[6-D06_early_autism] PASSED [ 42%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[7-D01_eye_tracking] PASSED [ 42%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[7-D02_facial_images_A] PASSED [ 43%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[7-D04_autism_prediction] PASSED [ 43%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[7-D05_toddler_screening] PASSED [ 43%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[7-D06_early_autism] PASSED [ 43%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[8-D01_eye_tracking] PASSED [ 44%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[8-D02_facial_images_A] PASSED [ 44%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[8-D04_autism_prediction] PASSED [ 44%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[8-D05_toddler_screening] PASSED [ 45%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[8-D06_early_autism] PASSED [ 45%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[9-D01_eye_tracking] PASSED [ 45%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[9-D02_facial_images_A] PASSED [ 46%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[9-D04_autism_prediction] PASSED [ 46%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[9-D05_toddler_screening] PASSED [ 46%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[9-D06_early_autism] PASSED [ 47%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[10-D01_eye_tracking] PASSED [ 47%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[10-D02_facial_images_A] PASSED [ 47%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[10-D04_autism_prediction] PASSED [ 47%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[10-D05_toddler_screening] PASSED [ 48%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[10-D06_early_autism] PASSED [ 48%]
tests/test_stage02_splits.py::test_d01_test_folds_have_asd_and_td_children PASSED [ 48%]
tests/test_stage02_splits.py::test_splits_manifest_hashes_match PASSED   [ 49%]
tests/test_stage02_splits.py::test_rerun_make_splits_is_byte_identical PASSED [ 49%]
tests/test_stage02_tabular.py::test_d04_only_labelled_train_rows PASSED  [ 49%]
tests/test_stage02_tabular.py::test_d04_result_decision_recorded PASSED  [ 50%]
tests/test_stage02_tabular.py::test_d05_honest_set_excludes_items_and_score PASSED [ 50%]
tests/test_stage02_tabular.py::test_d05_label_rule_on_clean_data PASSED  [ 50%]
tests/test_stage02_tabular.py::test_d05_other_csv_never_read_by_stage2_code PASSED [ 51%]
tests/test_stage02_tabular.py::test_d06_empty_column_absent PASSED       [ 51%]
tests/test_stage02_tabular.py::test_identical_feature_rows_share_group PASSED [ 51%]
tests/test_stage02_tabular.py::test_every_column_has_a_role PASSED       [ 52%]
tests/test_stage02_tabular.py::test_raw_data_snapshot_unchanged PASSED   [ 52%]
tests/test_stage03_models.py::test_config_mapping[M01] PASSED            [ 52%]
tests/test_stage03_models.py::test_config_mapping[M02] PASSED            [ 52%]
tests/test_stage03_models.py::test_config_mapping[M03] PASSED            [ 53%]
tests/test_stage03_models.py::test_config_mapping[M04] PASSED            [ 53%]
tests/test_stage03_models.py::test_config_mapping[M05] PASSED            [ 53%]
tests/test_stage03_models.py::test_config_mapping[M06] PASSED            [ 54%]
tests/test_stage03_models.py::test_config_mapping[M07] PASSED            [ 54%]
tests/test_stage03_models.py::test_config_mapping[M08] PASSED            [ 54%]
tests/test_stage03_models.py::test_config_mapping[M09] PASSED            [ 55%]
tests/test_stage03_models.py::test_config_mapping[M10] PASSED            [ 55%]
tests/test_stage03_models.py::test_factory_rejects_unknown PASSED        [ 55%]
tests/test_stage03_models.py::test_forward[M01-sequence] PASSED          [ 56%]
tests/test_stage03_models.py::test_forward[M01-tokens] PASSED            [ 56%]
tests/test_stage03_models.py::test_forward[M02-sequence] PASSED          [ 56%]
tests/test_stage03_models.py::test_forward[M02-tokens] PASSED            [ 56%]
tests/test_stage03_models.py::test_forward[M03-sequence] PASSED          [ 57%]
tests/test_stage03_models.py::test_forward[M03-tokens] PASSED            [ 57%]
tests/test_stage03_models.py::test_forward[M04-sequence] PASSED          [ 57%]
tests/test_stage03_models.py::test_forward[M04-tokens] PASSED            [ 58%]
tests/test_stage03_models.py::test_forward[M05-sequence] PASSED          [ 58%]
tests/test_stage03_models.py::test_forward[M05-tokens] PASSED            [ 58%]
tests/test_stage03_models.py::test_forward[M06-sequence] PASSED          [ 59%]
tests/test_stage03_models.py::test_forward[M06-tokens] PASSED            [ 59%]
tests/test_stage03_models.py::test_forward[M07-sequence] PASSED          [ 59%]
tests/test_stage03_models.py::test_forward[M07-tokens] PASSED            [ 60%]
tests/test_stage03_models.py::test_forward[M08-sequence] PASSED          [ 60%]
tests/test_stage03_models.py::test_forward[M08-tokens] PASSED            [ 60%]
tests/test_stage03_models.py::test_forward[M09-sequence] PASSED          [ 60%]
tests/test_stage03_models.py::test_forward[M09-tokens] PASSED            [ 61%]
tests/test_stage03_models.py::test_forward[M10-sequence] PASSED          [ 61%]
tests/test_stage03_models.py::test_forward[M10-tokens] PASSED            [ 61%]
tests/test_stage03_models.py::test_backward_and_step[M01-sequence] PASSED [ 62%]
tests/test_stage03_models.py::test_backward_and_step[M01-tokens] PASSED  [ 62%]
tests/test_stage03_models.py::test_backward_and_step[M02-sequence] PASSED [ 62%]
tests/test_stage03_models.py::test_backward_and_step[M02-tokens] PASSED  [ 63%]
tests/test_stage03_models.py::test_backward_and_step[M03-sequence] PASSED [ 63%]
tests/test_stage03_models.py::test_backward_and_step[M03-tokens] PASSED  [ 63%]
tests/test_stage03_models.py::test_backward_and_step[M04-sequence] PASSED [ 64%]
tests/test_stage03_models.py::test_backward_and_step[M04-tokens] PASSED  [ 64%]
tests/test_stage03_models.py::test_backward_and_step[M05-sequence] PASSED [ 64%]
tests/test_stage03_models.py::test_backward_and_step[M05-tokens] PASSED  [ 65%]
tests/test_stage03_models.py::test_backward_and_step[M06-sequence] PASSED [ 65%]
tests/test_stage03_models.py::test_backward_and_step[M06-tokens] PASSED  [ 65%]
tests/test_stage03_models.py::test_backward_and_step[M07-sequence] PASSED [ 65%]
tests/test_stage03_models.py::test_backward_and_step[M07-tokens] PASSED  [ 66%]
tests/test_stage03_models.py::test_backward_and_step[M08-sequence] PASSED [ 66%]
tests/test_stage03_models.py::test_backward_and_step[M08-tokens] PASSED  [ 66%]
tests/test_stage03_models.py::test_backward_and_step[M09-sequence] PASSED [ 67%]
tests/test_stage03_models.py::test_backward_and_step[M09-tokens] PASSED  [ 67%]
tests/test_stage03_models.py::test_backward_and_step[M10-sequence] PASSED [ 67%]
tests/test_stage03_models.py::test_backward_and_step[M10-tokens] PASSED  [ 68%]
tests/test_stage03_models.py::test_determinism_eval[M01-sequence] PASSED [ 68%]
tests/test_stage03_models.py::test_determinism_eval[M01-tokens] PASSED   [ 68%]
tests/test_stage03_models.py::test_determinism_eval[M02-sequence] PASSED [ 69%]
tests/test_stage03_models.py::test_determinism_eval[M02-tokens] PASSED   [ 69%]
tests/test_stage03_models.py::test_determinism_eval[M03-sequence] PASSED [ 69%]
tests/test_stage03_models.py::test_determinism_eval[M03-tokens] PASSED   [ 69%]
tests/test_stage03_models.py::test_determinism_eval[M04-sequence] PASSED [ 70%]
tests/test_stage03_models.py::test_determinism_eval[M04-tokens] PASSED   [ 70%]
tests/test_stage03_models.py::test_determinism_eval[M05-sequence] PASSED [ 70%]
tests/test_stage03_models.py::test_determinism_eval[M05-tokens] PASSED   [ 71%]
tests/test_stage03_models.py::test_determinism_eval[M06-sequence] PASSED [ 71%]
tests/test_stage03_models.py::test_determinism_eval[M06-tokens] PASSED   [ 71%]
tests/test_stage03_models.py::test_determinism_eval[M07-sequence] PASSED [ 72%]
tests/test_stage03_models.py::test_determinism_eval[M07-tokens] PASSED   [ 72%]
tests/test_stage03_models.py::test_determinism_eval[M08-sequence] PASSED [ 72%]
tests/test_stage03_models.py::test_determinism_eval[M08-tokens] PASSED   [ 73%]
tests/test_stage03_models.py::test_determinism_eval[M09-sequence] PASSED [ 73%]
tests/test_stage03_models.py::test_determinism_eval[M09-tokens] PASSED   [ 73%]
tests/test_stage03_models.py::test_determinism_eval[M10-sequence] PASSED [ 73%]
tests/test_stage03_models.py::test_determinism_eval[M10-tokens] PASSED   [ 74%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M01-sequence] PASSED [ 74%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M01-tokens] PASSED [ 74%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M02-sequence] PASSED [ 75%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M02-tokens] PASSED [ 75%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M03-sequence] PASSED [ 75%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M03-tokens] PASSED [ 76%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M04-sequence] PASSED [ 76%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M04-tokens] PASSED [ 76%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M05-sequence] PASSED [ 77%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M05-tokens] PASSED [ 77%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M06-sequence] PASSED [ 77%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M06-tokens] PASSED [ 78%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M07-sequence] PASSED [ 78%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M07-tokens] PASSED [ 78%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M08-sequence] PASSED [ 78%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M08-tokens] PASSED [ 79%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M09-sequence] PASSED [ 79%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M09-tokens] PASSED [ 79%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M10-sequence] PASSED [ 80%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M10-tokens] PASSED [ 80%]
tests/test_stage03_models.py::test_feature_cache_equivalence[M01] PASSED [ 80%]
tests/test_stage03_models.py::test_feature_cache_equivalence[M03] PASSED [ 81%]
tests/test_stage03_models.py::test_feature_cache_equivalence[M04] PASSED [ 81%]
tests/test_stage03_models.py::test_feature_cache_equivalence[M06] PASSED [ 81%]
tests/test_stage03_models.py::test_feature_cache_equivalence[M07] PASSED [ 82%]
tests/test_stage03_models.py::test_feature_cache_equivalence[M08] PASSED [ 82%]
tests/test_stage03_models.py::test_feature_cache_equivalence[M10] PASSED [ 82%]
tests/test_stage03_models.py::test_last_stage_trains_part_of_backbone PASSED [ 82%]
tests/test_stage03_models.py::test_model_configs_have_batch_sizes PASSED [ 83%]
tests/test_stage04_engine.py::test_interrupt_leaves_running_and_checkpoint[M01] PASSED [ 83%]
tests/test_stage04_engine.py::test_resume_identical_history_and_predictions[M01] PASSED [ 83%]
tests/test_stage04_engine.py::test_resumed_run_info_and_cleanup[M01] PASSED [ 84%]
tests/test_stage04_engine.py::test_youden_threshold_comes_from_validation_predictions[M01] PASSED [ 84%]
tests/test_stage04_engine.py::test_done_run_is_skipped_unless_forced[M01] PASSED [ 84%]
tests/test_stage04_engine.py::test_checks_detect_incomplete_results[M01] PASSED [ 85%]
tests/test_stage04_engine.py::test_interrupt_leaves_running_and_checkpoint[M09] PASSED [ 85%]
tests/test_stage04_engine.py::test_resume_identical_history_and_predictions[M09] PASSED [ 85%]
tests/test_stage04_engine.py::test_resumed_run_info_and_cleanup[M09] PASSED [ 86%]
tests/test_stage04_engine.py::test_youden_threshold_comes_from_validation_predictions[M09] PASSED [ 86%]
tests/test_stage04_engine.py::test_done_run_is_skipped_unless_forced[M09] PASSED [ 86%]
tests/test_stage04_engine.py::test_checks_detect_incomplete_results[M09] PASSED [ 86%]
tests/test_stage04_engine.py::test_engine_refuses_corrupted_split PASSED [ 87%]
tests/test_stage04_engine.py::test_resume_refuses_changed_config PASSED  [ 87%]
tests/test_stage04_units.py::test_training_protocol_values_fixed PASSED  [ 87%]
tests/test_stage04_units.py::test_merge_config[M01-D01_eye_tracking-1-0.0005-0.0-cached_features] PASSED [ 88%]
tests/test_stage04_units.py::test_merge_config[M01-D02_facial_images_A-3-0.0005-5e-05-images] PASSED [ 88%]
tests/test_stage04_units.py::test_merge_config[M07-D02_facial_images_A-10-0.0005-5e-05-images] PASSED [ 88%]
tests/test_stage04_units.py::test_merge_config[M09-D01_eye_tracking-2-0.001-0.001-clips] PASSED [ 89%]
tests/test_stage04_units.py::test_merge_config[M05-D02_facial_images_A-4-0.001-0.001-images] PASSED [ 89%]
tests/test_stage04_units.py::test_model_config_lr_mirrors_training_yaml PASSED [ 89%]
tests/test_stage04_units.py::test_scheduler_warmup_and_cosine PASSED     [ 90%]
tests/test_stage04_units.py::test_d02_transforms_uint8 PASSED            [ 90%]
tests/test_stage04_units.py::test_real_splits_pass_check_fold[D01_eye_tracking] PASSED [ 90%]
tests/test_stage04_units.py::test_real_splits_pass_check_fold[D02_facial_images_A] PASSED [ 91%]
tests/test_stage04_units.py::test_leakage_asserts_trigger_on_corrupted_split[id_in_train_and_test-D01_eye_tracking] PASSED [ 91%]
tests/test_stage04_units.py::test_leakage_asserts_trigger_on_corrupted_split[id_in_train_and_test-D02_facial_images_A] PASSED [ 91%]
tests/test_stage04_units.py::test_leakage_asserts_trigger_on_corrupted_split[duplicate-D01_eye_tracking] PASSED [ 91%]
tests/test_stage04_units.py::test_leakage_asserts_trigger_on_corrupted_split[duplicate-D02_facial_images_A] PASSED [ 92%]
tests/test_stage04_units.py::test_leakage_asserts_trigger_on_corrupted_split[group_overlap-D01_eye_tracking] PASSED [ 92%]
tests/test_stage04_units.py::test_leakage_asserts_trigger_on_corrupted_split[group_overlap-D02_facial_images_A] PASSED [ 92%]
tests/test_stage04_units.py::test_leakage_asserts_trigger_on_corrupted_split[unknown_id-D01_eye_tracking] PASSED [ 93%]
tests/test_stage04_units.py::test_leakage_asserts_trigger_on_corrupted_split[unknown_id-D02_facial_images_A] PASSED [ 93%]
tests/test_stage04_units.py::test_test_set_must_match_folds_csv PASSED   [ 93%]
tests/test_stage04_units.py::test_verify_manifest_detects_modified_temp_copy PASSED [ 94%]
tests/test_stage04_units.py::test_assert_no_test_ids PASSED              [ 94%]
tests/test_stage04_units.py::test_binary_metrics_match_sklearn[0.5-0] PASSED [ 94%]
tests/test_stage04_units.py::test_binary_metrics_match_sklearn[0.5-1] PASSED [ 95%]
tests/test_stage04_units.py::test_binary_metrics_match_sklearn[0.5-2] PASSED [ 95%]
tests/test_stage04_units.py::test_binary_metrics_match_sklearn[0.3-0] PASSED [ 95%]
tests/test_stage04_units.py::test_binary_metrics_match_sklearn[0.3-1] PASSED [ 95%]
tests/test_stage04_units.py::test_binary_metrics_match_sklearn[0.3-2] PASSED [ 96%]
tests/test_stage04_units.py::test_binary_metrics_match_sklearn[0.72-0] PASSED [ 96%]
tests/test_stage04_units.py::test_binary_metrics_match_sklearn[0.72-1] PASSED [ 96%]
tests/test_stage04_units.py::test_binary_metrics_match_sklearn[0.72-2] PASSED [ 97%]
tests/test_stage04_units.py::test_binary_metrics_known_values PASSED     [ 97%]
tests/test_stage04_units.py::test_youden_threshold_brute_force_on_validation_only[0] PASSED [ 97%]
tests/test_stage04_units.py::test_youden_threshold_brute_force_on_validation_only[1] PASSED [ 98%]
tests/test_stage04_units.py::test_youden_threshold_brute_force_on_validation_only[2] PASSED [ 98%]
tests/test_stage04_units.py::test_youden_threshold_brute_force_on_validation_only[3] PASSED [ 98%]
tests/test_stage04_units.py::test_child_level_mean_probability PASSED    [ 99%]
tests/test_stage04_units.py::test_registry_skip_force_resume PASSED      [ 99%]
tests/test_stage04_units.py::test_registry_file_lock_no_lost_rows PASSED [ 99%]
tests/test_stage04_units.py::test_summary_aggregation_on_fake_folds PASSED [100%]

======================= 323 passed in 124.44s (0:02:04) ========================
```

</details>

## 6. Smoke tests (real data, max 2 epochs, `results_smoke/`)

| smoke | dataset | model | fold | input | epochs | best_epoch | s/epoch (each) | peak GB | run s | resumed | files | batches tr/val/test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | D01_eye_tracking | M01 | 1 | frozen backbone + cached features | 2 | 2 | 1.1 / 0.5 | 0.12 | 2.4 | False | 8 | 32/6/2 |
| 1 | D01_eye_tracking | M01 | 2 | frozen backbone + cached features | 2 | 1 | 0.5 / 0.4 | 0.12 | 1.7 | False | 8 | 30/12/2 |
| 2 | D01_eye_tracking | M09 | 1 | custom CNN end to end on clips | 2 | 1 | 12.1 / 3.2 | 3.17 | 16.2 | False | 8 | 32/6/2 |
| 2 | D01_eye_tracking | M09 | 2 | custom CNN end to end on clips | 2 | 2 | 6.2 / 3.1 | 2.73 | 10.7 | False | 8 | 30/12/2 |
| 3 | D02_facial_images_A | M07 | 1 | ViT-B/16 | 2 | 2 | 9.0 / 8.0 | 1.04 | 20.8 | False | 8 | 72/18/5 |
| 3 | D02_facial_images_A | M07 | 2 | ViT-B/16 | 2 | 2 | 7.8 / 8.0 | 1.04 | 19.5 | False | 8 | 72/18/5 |
| 4 | D02_facial_images_A | M03 | 1 | interruption test (SIGINT after epoch 1 | 3 | 3 | 10.3 / 10.5 / 4.9 | 0.98 | 29.9 | True | 8 | 108/24/5 |

Validity checks (`validate_model_results` adapted to the smoke folds):

- smoke 1 D01_eye_tracking x M01 folds [1, 2]: **PASS** 
- smoke 2 D01_eye_tracking x M09 folds [1, 2]: **PASS** 
- smoke 3 D02_facial_images_A x M07 folds [1, 2]: **PASS** 
- smoke 4 D02_facial_images_A x M03 folds [1]: **PASS** 

Test metrics (2 epochs; only checked to be valid numbers):

| run | val thr (Youden) | AUC | PR-AUC | bal acc @0.5 | bal acc @val thr | sens/spec @0.5 | brier | CM @0.5 | child AUC | child CM @0.5 |
|---|---|---|---|---|---|---|---|---|---|---|
| M01 D01 f1 | 0.5802 | 0.5514 | 0.4625 | 0.5771 | 0.5796 | 0.306/0.848 | 0.2483 | [[67, 12], [34, 15]] | 0.4444 | [[2, 1], [2, 1]] |
| M01 D01 f2 | 0.5205 | 0.6574 | 0.6461 | 0.5792 | 0.5825 | 0.709/0.449 | 0.2391 | [[31, 38], [16, 39]] | 0.7778 | [[1, 2], [1, 2]] |
| M09 D01 f1 | 0.6539 | 0.5172 | 0.4194 | 0.5 | 0.5209 | 1.000/0.000 | 0.3093 | [[0, 79], [0, 49]] | 0.3333 | [[0, 3], [0, 3]] |
| M09 D01 f2 | 0.8426 | 0.5249 | 0.5401 | 0.5 | 0.5299 | 1.000/0.000 | 0.4044 | [[0, 69], [0, 55]] | 0.6667 | [[0, 3], [0, 3]] |
| M07 D02 f1 | 0.4291 | 0.8957 | 0.8918 | 0.8231 | 0.8333 | 0.796/0.850 | 0.1315 | [[125, 22], [30, 117]] | n/a | n/a |
| M07 D02 f2 | 0.4547 | 0.9114 | 0.9172 | 0.8197 | 0.8095 | 0.878/0.762 | 0.1292 | [[112, 35], [18, 129]] | n/a | n/a |
| M03 D02 f1 | 0.4869 | 0.8502 | 0.8479 | 0.7585 | 0.7687 | 0.701/0.816 | 0.1609 | [[120, 27], [44, 103]] | n/a | n/a |

### history.csv

`D01_eye_tracking/M01/fold_01/history.csv`

| epoch | train_loss | val_loss | val_auc | lr | seconds | peak_gpu_gb | val_auc_child |
|---|---|---|---|---|---|---|---|
| 1.0 | 0.88456 | 0.81112 | 0.72549 | 0.00027 | 1.09227 | 0.10517 | 0.875 |
| 2.0 | 0.81683 | 0.67932 | 0.79503 | 0.0005 | 0.47464 | 0.10517 | 1.0 |

`D01_eye_tracking/M01/fold_02/history.csv`

| epoch | train_loss | val_loss | val_auc | lr | seconds | peak_gpu_gb | val_auc_child |
|---|---|---|---|---|---|---|---|
| 1.0 | 0.93066 | 0.97009 | 0.72371 | 0.00027 | 0.47038 | 0.10455 | 0.8125 |
| 2.0 | 0.87149 | 0.92779 | 0.71155 | 0.0005 | 0.40605 | 0.10455 | 0.875 |

`D01_eye_tracking/M09/fold_01/history.csv`

| epoch | train_loss | val_loss | val_auc | lr | seconds | peak_gpu_gb | val_auc_child |
|---|---|---|---|---|---|---|---|
| 1.0 | 0.87398 | 0.96714 | 0.7943 | 0.00053 | 12.06015 | 3.17154 | 1.0 |
| 2.0 | 0.8375 | 1.02908 | 0.77832 | 0.001 | 3.18076 | 2.72707 | 1.0 |

`D01_eye_tracking/M09/fold_02/history.csv`

| epoch | train_loss | val_loss | val_auc | lr | seconds | peak_gpu_gb | val_auc_child |
|---|---|---|---|---|---|---|---|
| 1.0 | 0.93353 | 1.00058 | 0.56832 | 0.00053 | 6.22279 | 2.72641 | 0.4375 |
| 2.0 | 0.87522 | 1.20039 | 0.61663 | 0.001 | 3.09595 | 2.72536 | 0.8125 |

`D02_facial_images_A/M07/fold_01/history.csv`

| epoch | train_loss | val_loss | val_auc | lr | seconds | peak_gpu_gb |
|---|---|---|---|---|---|---|
| 1.0 | 0.646 | 0.48657 | 0.86856 | 0.00026 | 8.97323 | 1.04143 |
| 2.0 | 0.4584 | 0.34476 | 0.93827 | 0.0005 | 8.00134 | 1.04012 |

`D02_facial_images_A/M07/fold_02/history.csv`

| epoch | train_loss | val_loss | val_auc | lr | seconds | peak_gpu_gb |
|---|---|---|---|---|---|---|
| 1.0 | 0.63442 | 0.47565 | 0.8654 | 0.00026 | 7.84992 | 1.04176 |
| 2.0 | 0.41776 | 0.3772 | 0.91285 | 0.0005 | 8.04575 | 1.04176 |

`D02_facial_images_A/M03/fold_01/history.csv`

| epoch | train_loss | val_loss | val_auc | lr | seconds | peak_gpu_gb |
|---|---|---|---|---|---|---|
| 1.0 | 0.68288 | 0.64366 | 0.78405 | 0.00026 | 10.33523 | 0.98471 |
| 2.0 | 0.53434 | 0.48513 | 0.86856 | 0.0005 | 10.47081 | 0.98073 |
| 3.0 | 0.45133 | 0.43275 | 0.89033 | 0.0 | 4.86444 | 0.72005 |

### Result folder tree (`results_smoke/`, bytes)

```
        818  results_smoke/run_registry.csv
          0  results_smoke/run_registry.csv.lock
        801  results_smoke/D01_eye_tracking/leaderboard.csv
      9,564  results_smoke/D01_eye_tracking/M01/summary.json
      4,224  results_smoke/D01_eye_tracking/M01/fold_01/attention.npy
 46,660,795  results_smoke/D01_eye_tracking/M01/fold_01/best_model.pt
      1,601  results_smoke/D01_eye_tracking/M01/fold_01/config.yaml
        280  results_smoke/D01_eye_tracking/M01/fold_01/history.csv
      3,229  results_smoke/D01_eye_tracking/M01/fold_01/metrics.json
      6,994  results_smoke/D01_eye_tracking/M01/fold_01/predictions.csv
      2,609  results_smoke/D01_eye_tracking/M01/fold_01/run_info.json
      6,288  results_smoke/D01_eye_tracking/M01/fold_01/val_predictions.csv
      4,096  results_smoke/D01_eye_tracking/M01/fold_02/attention.npy
 46,660,795  results_smoke/D01_eye_tracking/M01/fold_02/best_model.pt
      1,601  results_smoke/D01_eye_tracking/M01/fold_02/config.yaml
        292  results_smoke/D01_eye_tracking/M01/fold_02/history.csv
      3,295  results_smoke/D01_eye_tracking/M01/fold_02/metrics.json
      6,739  results_smoke/D01_eye_tracking/M01/fold_02/predictions.csv
      2,610  results_smoke/D01_eye_tracking/M01/fold_02/run_info.json
     11,739  results_smoke/D01_eye_tracking/M01/fold_02/val_predictions.csv
      9,224  results_smoke/D01_eye_tracking/M09/summary.json
      4,224  results_smoke/D01_eye_tracking/M09/fold_01/attention.npy
  2,498,221  results_smoke/D01_eye_tracking/M09/fold_01/best_model.pt
      1,552  results_smoke/D01_eye_tracking/M09/fold_01/config.yaml
        273  results_smoke/D01_eye_tracking/M09/fold_01/history.csv
      3,040  results_smoke/D01_eye_tracking/M09/fold_01/metrics.json
      6,925  results_smoke/D01_eye_tracking/M09/fold_01/predictions.csv
      2,181  results_smoke/D01_eye_tracking/M09/fold_01/run_info.json
      6,241  results_smoke/D01_eye_tracking/M09/fold_01/val_predictions.csv
      4,096  results_smoke/D01_eye_tracking/M09/fold_02/attention.npy
  2,498,221  results_smoke/D01_eye_tracking/M09/fold_02/best_model.pt
      1,552  results_smoke/D01_eye_tracking/M09/fold_02/config.yaml
        290  results_smoke/D01_eye_tracking/M09/fold_02/history.csv
      3,119  results_smoke/D01_eye_tracking/M09/fold_02/metrics.json
      6,728  results_smoke/D01_eye_tracking/M09/fold_02/predictions.csv
      2,183  results_smoke/D01_eye_tracking/M09/fold_02/run_info.json
     11,684  results_smoke/D01_eye_tracking/M09/fold_02/val_predictions.csv
        528  results_smoke/D02_facial_images_A/leaderboard.csv
      4,871  results_smoke/D02_facial_images_A/M03/summary.json
     57,752  results_smoke/D02_facial_images_A/M03/fold_01/attention.npy
 97,389,895  results_smoke/D02_facial_images_A/M03/fold_01/best_model.pt
      1,739  results_smoke/D02_facial_images_A/M03/fold_01/config.yaml
        362  results_smoke/D02_facial_images_A/M03/fold_01/history.csv
      1,800  results_smoke/D02_facial_images_A/M03/fold_01/metrics.json
     18,803  results_smoke/D02_facial_images_A/M03/fold_01/predictions.csv
      2,436  results_smoke/D02_facial_images_A/M03/fold_01/run_info.json
     24,186  results_smoke/D02_facial_images_A/M03/fold_01/val_predictions.csv
      5,226  results_smoke/D02_facial_images_A/M07/summary.json
    230,624  results_smoke/D02_facial_images_A/M07/fold_01/attention.npy
345,117,543  results_smoke/D02_facial_images_A/M07/fold_01/best_model.pt
      1,757  results_smoke/D02_facial_images_A/M07/fold_01/config.yaml
        260  results_smoke/D02_facial_images_A/M07/fold_01/history.csv
      1,805  results_smoke/D02_facial_images_A/M07/fold_01/metrics.json
     18,779  results_smoke/D02_facial_images_A/M07/fold_01/predictions.csv
      2,291  results_smoke/D02_facial_images_A/M07/fold_01/run_info.json
     24,170  results_smoke/D02_facial_images_A/M07/fold_01/val_predictions.csv
    230,624  results_smoke/D02_facial_images_A/M07/fold_02/attention.npy
345,117,543  results_smoke/D02_facial_images_A/M07/fold_02/best_model.pt
      1,757  results_smoke/D02_facial_images_A/M07/fold_02/config.yaml
        261  results_smoke/D02_facial_images_A/M07/fold_02/history.csv
      1,802  results_smoke/D02_facial_images_A/M07/fold_02/metrics.json
     18,809  results_smoke/D02_facial_images_A/M07/fold_02/predictions.csv
      2,287  results_smoke/D02_facial_images_A/M07/fold_02/run_info.json
     24,099  results_smoke/D02_facial_images_A/M07/fold_02/val_predictions.csv
        202  results_smoke/resume_reference/run_registry.csv
          0  results_smoke/resume_reference/run_registry.csv.lock
     57,752  results_smoke/resume_reference/D02_facial_images_A/M03/fold_01/attention.npy
 97,389,895  results_smoke/resume_reference/D02_facial_images_A/M03/fold_01/best_model.pt
      1,739  results_smoke/resume_reference/D02_facial_images_A/M03/fold_01/config.yaml
        359  results_smoke/resume_reference/D02_facial_images_A/M03/fold_01/history.csv
      1,803  results_smoke/resume_reference/D02_facial_images_A/M03/fold_01/metrics.json
     18,800  results_smoke/resume_reference/D02_facial_images_A/M03/fold_01/predictions.csv
      2,265  results_smoke/resume_reference/D02_facial_images_A/M03/fold_01/run_info.json
     24,201  results_smoke/resume_reference/D02_facial_images_A/M03/fold_01/val_predictions.csv
```

`summary.json` and `leaderboard.csv` were produced by `scripts/run_queue.py` / `summarize.py` on the smoke folds (`logs/stage04_queue_smoke.log`: the queue skipped the DONE folds, validated them and summarised).

## 7. Interruption / resume evidence

### 7.1 First attempt: SIGINT was ignored (bug found and fixed)

`logs/stage04_smoke_interrupt_attempt1_sigint_ignored.log`: `kill -INT` was sent after epoch 1, but the process trained on to epoch 3 and finished DONE. Cause: POSIX shells start background jobs (`cmd &`) of a non-interactive shell with SIGINT ignored, and Python then never raises KeyboardInterrupt. The same applies to any nohup / script-launched S5/S6 run. Fix: explicit SIGINT/SIGTERM handlers in `train.py` and `run_queue.py` (the queue waits up to 15 s for its child, then forwards SIGINT). The M03 smoke outputs of that attempt were deleted and the test was re-run.

### 7.2 Interruption test (`scripts/04_smoke_interrupt.sh`, `logs/stage04_smoke_interrupt.log`)

```
### 2026-10-14T09:32:35+05:30 step 1: start scripts/train.py --dataset D02 --model M03 --folds 1 --max-epochs 3 --out results_smoke
[D02_facial_images_A M03 fold 1] START (new) input=images policy=last_stage n_train=2268 n_val=378 n_test=294 pos_weight=1.0000 bs=64 max_epochs=3
[D02_facial_images_A M03 fold 1] epoch 1/3 train_loss 0.6829 val_loss 0.6437 val_auc 0.7840 lr 2.57e-04 10.3s peak 0.98GB *best*
### 2026-10-14T09:32:48+05:30 epoch 1 finished -> kill -INT 109882
[D02_facial_images_A M03 fold 1] INTERRUPTED (signal SIGINT); status stays RUNNING, last.pt kept for resume
train.py interrupted (signal SIGINT): re-run the same command to resume
### exit code after SIGINT: 130
### step 2: state after the interruption
D02_facial_images_A,M03,1,RUNNING,2026-10-14T09:32:37,,,,,,,,interrupted 2026-10-14T09:32:49 signal SIGINT
total 313208
drwxrwxr-x 2 score score      4096 Oct 14 09:32 .
drwxrwxr-x 3 score score      4096 Oct 14 09:32 ..
-rw-rw-r-- 1 score score  97389895 Oct 14 09:32 best_model.pt
-rw-rw-r-- 1 score score      1739 Oct 14 09:32 config.yaml
-rw-rw-r-- 1 score score       169 Oct 14 09:32 history.csv
-rw-rw-r-- 1 score score 223311137 Oct 14 09:32 last.pt
-rw-rw-r-- 1 score score      1737 Oct 14 09:32 run_info.json
### 2026-10-14T09:32:50+05:30 step 3: restart the same command
[D02_facial_images_A M03 fold 1] RESUME from last.pt: epoch 1 done, continuing at epoch 2 (best val AUC 0.7840 @ 1, patience 0/8)
[D02_facial_images_A M03 fold 1] epoch 2/3 train_loss 0.5343 val_loss 0.4851 val_auc 0.8686 lr 5.00e-04 10.5s peak 0.98GB *best*
[D02_facial_images_A M03 fold 1] epoch 3/3 train_loss 0.4513 val_loss 0.4328 val_auc 0.8903 lr 0.00e+00 4.9s peak 0.72GB *best*
[D02_facial_images_A M03 fold 1] DONE epochs 3 best_epoch 3 val_auc 0.8903 test_auc 0.8502 peak 0.98GB 30s batches {'train': 108, 'val': 24, 'test': 5} test ids in train/val batches: 0
### exit code after restart: 0
### fold log:
2026-10-14 09:32:38 [D02_facial_images_A M03 fold 1] START (new) input=images policy=last_stage n_train=2268 n_val=378 n_test=294 pos_weight=1.0000 bs=64 max_epochs=3
2026-10-14 09:32:48 [D02_facial_images_A M03 fold 1] epoch 1/3 train_loss 0.6829 val_loss 0.6437 val_auc 0.7840 lr 2.57e-04 10.3s peak 0.98GB *best*
2026-10-14 09:32:49 [D02_facial_images_A M03 fold 1] INTERRUPTED (signal SIGINT); status stays RUNNING, last.pt kept for resume
2026-10-14 09:32:53 [D02_facial_images_A M03 fold 1] RESUME from last.pt: epoch 1 done, continuing at epoch 2 (best val AUC 0.7840 @ 1, patience 0/8)
2026-10-14 09:33:03 [D02_facial_images_A M03 fold 1] epoch 2/3 train_loss 0.5343 val_loss 0.4851 val_auc 0.8686 lr 5.00e-04 10.5s peak 0.98GB *best*
2026-10-14 09:33:08 [D02_facial_images_A M03 fold 1] epoch 3/3 train_loss 0.4513 val_loss 0.4328 val_auc 0.8903 lr 0.00e+00 4.9s peak 0.72GB *best*
2026-10-14 09:33:11 [D02_facial_images_A M03 fold 1] DONE epochs 3 best_epoch 3 val_auc 0.8903 test_auc 0.8502 peak 0.98GB 30s batches {'train': 108, 'val': 24, 'test': 5} test ids in train/val batches: 0
### step 4: straight reference run
[D02_facial_images_A M03 fold 1] START (new) input=images policy=last_stage n_train=2268 n_val=378 n_test=294 pos_weight=1.0000 bs=64 max_epochs=3
[D02_facial_images_A M03 fold 1] epoch 1/3 train_loss 0.6829 val_loss 0.6437 val_auc 0.7841 lr 2.57e-04 10.4s peak 0.98GB *best*
[D02_facial_images_A M03 fold 1] epoch 2/3 train_loss 0.5343 val_loss 0.4853 val_auc 0.8685 lr 5.00e-04 4.8s peak 0.73GB *best*
[D02_facial_images_A M03 fold 1] epoch 3/3 train_loss 0.4516 val_loss 0.4331 val_auc 0.8903 lr 0.00e+00 4.8s peak 0.73GB *best*
[D02_facial_images_A M03 fold 1] DONE epochs 3 best_epoch 3 val_auc 0.8903 test_auc 0.8502 peak 0.98GB 23s batches {'train': 108, 'val': 24, 'test': 5} test ids in train/val batches: 0
INTERRUPT_TEST_END
```

Resumed run_info sessions: [(1, 1, True), (2, 3, False)] (start epoch, end epoch, interrupted).

### 7.3 Exactness

Deterministic cuDNN settings (`logs/stage04_resume_deterministic_diff.txt`, same comparison as the pytest resume test):

```
deterministic cuDNN (benchmark off), subset 12/class of fold 1, 3 epochs: straight vs 1 epoch + simulated kill + resume
D01_eye_tracking M01 max |diff|: {'train_loss': 0.0, 'val_loss': 0.0, 'val_auc': 0.0, 'lr': 0.0, 'test_y_prob': 0.0}
D02_facial_images_A M09 max |diff|: {'train_loss': 0.0, 'val_loss': 0.0, 'val_auc': 0.0, 'lr': 0.0, 'test_y_prob': 0.0}
```

Real setting (`cudnn.benchmark=True`), interrupted D02 x M03 run vs. a straight run of the same fold (`results_smoke/resume_reference/`): max |diff| train_loss 3.0e-04, val_loss 3.3e-04, val_auc 5.6e-05, lr 0.0e+00. Two straight runs differ by the same order (attempt 1, which was effectively straight, logged train_loss 0.4513 / val_loss 0.4327 at epoch 3 vs. 0.4516 / 0.4331 for the reference), so this is cuDNN autotuning nondeterminism, not the resume logic; it is recorded in every run_info.json.

### 7.4 Queue interruption (`logs/stage04_queue_sigterm_check.log`)

`run_queue.py` started in its own session on a scratch output folder (outside the project, deleted afterwards; D01 x M09 fold 3, max 8 epochs), SIGTERM sent to the queue after epoch 1: the queue waited, forwarded SIGINT, train.py saved and exited (registry RUNNING, last.pt present), the queue exited 130; re-running the same command resumed at epoch 6, finished DONE and validate_model_results gave PASS.

## 8. Time estimates for S5 and S6 (`scripts/04_time_estimate.py`, `reports/stage04_time_estimate.csv`)

Method: measured smoke epochs (data loading, validation, checkpointing included) give a per-input-type overhead; every model = overhead + steps/epoch x its Stage 3 profiled s/step; per-fold overhead = first-epoch extra (cuDNN autotune, cold page cache) + model build / weight hash / final prediction. 10 folds x 10 models; ~20 epochs = assumed average with early stopping (patience 8, so a run lasts at least 9 epochs), 40 = max_epochs worst case. Process start-up (~5 s per model) is not included.

Measured smoke timing (`reports/stage04_smoke_timing.csv`):

| run | input | steady_epoch_s | first_epoch_extra_s | fixed_s | overhead_epoch_s |
|---|---|---|---|---|---|
| D01_eye_tracking/M01/fold_01 | cached_features | 0.475 | 0.618 | 0.869 | 0.431 |
| D01_eye_tracking/M01/fold_02 | cached_features | 0.406 | 0.064 | 0.851 | 0.363 |
| D01_eye_tracking/M09/fold_01 | clips | 3.181 | 8.879 | 1.008 | 0.779 |
| D01_eye_tracking/M09/fold_02 | clips | 3.096 | 3.127 | 1.413 | 0.694 |
| D02_facial_images_A/M07/fold_01 | images | 8.001 | 0.972 | 3.867 | 1.388 |
| D02_facial_images_A/M07/fold_02 | images | 8.046 | 0.0 | 3.65 | 1.433 |
| resume_reference/D02_facial_images_A/M03/fold_01 | images | 4.779 | 5.635 | 3.283 | 1.241 |

Overheads per input type (`reports/stage04_overheads.csv`):

| input | overhead_epoch_s | overhead_fold_s | n_runs |
|---|---|---|---|
| cached_features | 0.397 | 1.201 | 2 |
| clips | 0.737 | 7.214 | 2 |
| images | 1.354 | 5.802 | 3 |

| stage | model_id | input | measured_in_smoke | est_epoch_s | est_fold_min_E20 | est_fold_min_E40 | est_model_h_E20 | est_model_h_E40 |
|---|---|---|---|---|---|---|---|---|
| S5 | M01 | cached_features | True | 0.44 | 0.17 | 0.31 | 0.028 | 0.052 |
| S5 | M02 | clips | False | 3.11 | 1.16 | 2.19 | 0.193 | 0.365 |
| S5 | M03 | cached_features | False | 0.43 | 0.16 | 0.31 | 0.027 | 0.051 |
| S5 | M04 | cached_features | False | 0.44 | 0.17 | 0.31 | 0.028 | 0.052 |
| S5 | M05 | clips | False | 3.15 | 1.17 | 2.22 | 0.195 | 0.37 |
| S5 | M06 | cached_features | False | 0.43 | 0.16 | 0.31 | 0.027 | 0.052 |
| S5 | M07 | cached_features | False | 0.43 | 0.16 | 0.31 | 0.027 | 0.052 |
| S5 | M08 | cached_features | False | 0.46 | 0.17 | 0.33 | 0.029 | 0.055 |
| S5 | M09 | clips | True | 3.14 | 1.17 | 2.21 | 0.194 | 0.369 |
| S5 | M10 | cached_features | False | 0.47 | 0.18 | 0.34 | 0.03 | 0.056 |
| S6 | M01 | images | False | 4.85 | 1.71 | 3.33 | 0.286 | 0.555 |
| S6 | M02 | images | False | 4.27 | 1.52 | 2.94 | 0.253 | 0.491 |
| S6 | M03 | images | True | 4.89 | 1.73 | 3.36 | 0.288 | 0.56 |
| S6 | M04 | images | False | 5.65 | 1.98 | 3.86 | 0.33 | 0.644 |
| S6 | M05 | images | False | 4.89 | 1.73 | 3.36 | 0.288 | 0.559 |
| S6 | M06 | images | False | 4.93 | 1.74 | 3.38 | 0.29 | 0.564 |
| S6 | M07 | images | True | 7.97 | 2.75 | 5.41 | 0.459 | 0.901 |
| S6 | M08 | images | False | 3.32 | 1.2 | 2.31 | 0.201 | 0.385 |
| S6 | M09 | images | False | 4.27 | 1.52 | 2.94 | 0.253 | 0.491 |
| S6 | M10 | images | False | 5.18 | 1.82 | 3.55 | 0.304 | 0.592 |

| stage | ~20 epochs (early stopping) | 40 epochs (worst case) |
|---|---|---|
| S5 (D01, 100 runs) | 0.78 h | 1.47 h |
| S6 (D02, 100 runs) | 2.95 h | 5.74 h |

Uncertainty: only 4 of the 20 model x dataset combinations were timed directly; light D02 models may be bounded by JPEG decoding rather than by the GPU (their estimate could be low by ~1-2 s per epoch).

## 9. How to start S5 (after the user's go)

tmux is NOT installed on this machine (`which tmux` found nothing; checked 2026-10-14). Either install it (`sudo apt install tmux`) and run:

```bash
cd /home/score/Documents/AI_project/asd_hybrid_10fold
tmux new -s stage5 'bash scripts/run_stage5.sh'     # detach: Ctrl-b d
tmux attach -t stage5                                # re-attach; Ctrl-c stops gracefully (resumable)
tail -f logs/stage5_queue.log                        # progress from any terminal
```

or, without tmux:

```bash
cd /home/score/Documents/AI_project/asd_hybrid_10fold
nohup bash scripts/run_stage5.sh > /dev/null 2>&1 &    # log: logs/stage5_queue.log
tail -f logs/stage5_queue.log
pkill -TERM -f 'run_queue.py --dataset D01'          # graceful stop; re-run the same command to resume
```

Both scripts run `check_gpu.py` first, append to `logs/stage{5,6}_queue.log`, hold a lock (one copy at a time), skip DONE folds, resume RUNNING folds and stop at the first model that fails `validate_model_results`. `cat results/run_registry.csv` shows the status of every run.

## 10. Open items for Stage 5

1. Get the user's go; then `bash scripts/run_stage5.sh` (D01, M01 -> M10, sequential, gate after each model).
2. D01 validation sets have 7-8 children: the clip-level validation AUC used for early stopping is noisy; report child-level test metrics as the main D01 result (Stage 2 open item 3).
3. cuDNN benchmark nondeterminism: re-running a fold reproduces results only to ~1e-3 in the losses; the registry never re-runs DONE folds, so the stored results are the reference.
4. Stage 8 must still add the shortcut controls (quality-only baseline, stimulus-aware checks) before any D01 result is interpreted.

Checks for the decision: pytest OK; smoke checks OK; interruption/resume OK; deterministic resume OK; results/ clean OK.

STAGE 4 DECISION: PASS
