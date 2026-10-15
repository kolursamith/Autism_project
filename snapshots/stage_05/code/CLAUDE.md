# CLAUDE.md — ASD hybrid 10-fold project (project memory, read at the start of every session)

## 1. Project goal

- Course BITE308L Artificial Intelligence (VIT). Team: Samith Kolur (24BIT0117), Iniya K, Chandru Vel Kumar.
- Project: "Reliability-Weighted Multimodal AI for Early Autism Risk Screening in Children".
- This phase: implement, train and compare 10 hybrid deep-learning architectures (suggested by our faculty) with
  10-fold cross-validation on the local NVIDIA GPU; every result stored in a structured, reproducible way.
- **Screening-support research, not diagnosis.** No output of this project is a diagnosis; say so in every report.

## 2. Paths and environment

- All absolute paths live in ONE file: `configs/paths.yaml` (`project_dir`, `datasets_dir`, `legacy_symlink`,
  `datasets: {dataset_id: folder name}`). Code loads it via `src/utils/paths.py`
  (`PROJECT_DIR`, `DATASETS_DIR`, `DATASETS`, `dataset_dir(id)`). No other file may contain an absolute path
  (enforced by `tests/test_setup.py`).
- Project dir: `/home/score/Documents/AI_project/asd_hybrid_10fold`. Old path `~/asd_hybrid_10fold` is a symlink to it.
- Raw data: `/home/score/Documents/AI_project/dataset` (6 folders) — outside the project, READ-ONLY.
- `/home/score/Documents/AI_project/Stage1_results/` is a frozen copy of Stage 1 outputs made on 2026-10-12 by the
  previous agent (for sharing). Not the working project; do not edit it.
- venv: `.venv` inside the project, created with `python3 -m venv --system-site-packages .venv` and
  `.venv/bin/python -m pip install -r requirements.txt -c logs/pip_constraints.txt`.
  numpy 1.26.4, scipy 1.11.4, matplotlib 3.6.3, pillow 10.2.0, pyyaml 6.0.1 come from the system (apt) Python —
  matplotlib 3.6.3 has no Python 3.12 wheel. Full list: `logs/setup_pip_freeze.txt`.
- Activate: `source .venv/bin/activate` (or call `.venv/bin/python` directly).
- Hardware (verified 2026-10-13, `logs/setup_gpu_check.txt`): NVIDIA RTX A2000 12GB (11.618 GiB visible to torch),
  driver 580.178.04, driver CUDA 13.0, torch 2.14.1+cu130 (CUDA 13.0, cuDNN 92400), Python 3.12.3, 24 CPU threads.
- No git (user decision 2026-10-13). Versions are recorded by local snapshots in `snapshots/stage_NN/`
  (`scripts/make_snapshot.py`). Nothing is uploaded or pushed anywhere; everything stays inside the project dir.

## 3. Golden rules

1. Work in strict stages. Never start stage N+1 until stage N has: passing tests, a written report
   `reports/STAGE_NN_REPORT.md` ending in `STAGE N DECISION: PASS`, a local snapshot, AND the user explicitly says go.
   No git: every stage ends with a local snapshot in `snapshots/stage_NN/` (code copy + CHECKSUMS.sha256 +
   SNAPSHOT_INFO.json), made with `python scripts/make_snapshot.py --stage N --decision PASS|FAIL`.
   Wherever a commit hash would be recorded, use `code_hash` (SNAPSHOT_INFO.json, or `make_snapshot.code_hash()`).
2. Raw data is read-only. Never write, move, rename, extract or copy anything from `datasets_dir` into the project.
   `tests/` compare a live snapshot with `data_registry/raw_snapshot_start.tsv` (path, kind, size, mtime_ns).
3. Never invent facts. Every number in a report comes from code that was run and whose output is saved
   (logs/, data_registry/, results/). Unknown = UNKNOWN.
4. Seed 42 everywhere (python, numpy, torch, sklearn). One config file per run, saved next to its results.
5. No data leakage: splits are made ONCE in Stage 2, saved to `splits/`, and reused unchanged by every model.
   Split assertions (no subject/group in two folds, both classes in every test fold) run before every training run.
6. Test folds are never used for early stopping, model selection, threshold tuning or hyperparameters. The
   validation set = 15% of the TRAINING groups of each fold (grouped by child/near-dup group, stratified by class).
7. All fitted transforms (normalisation stats, imputers, scalers, encoders) are fit on the training part of each
   fold only.
8. A model that cannot run on a dataset is marked BLOCKED with the reason — never faked, never silently skipped.
9. Never merge datasets or combine predictions across datasets/children (DD-11). CARS is never used (DD-03).
10. GPU rules (all training code):
    - `device = "cuda"`; if `torch.cuda.is_available()` is False, raise an error and STOP. Never fall back to CPU.
    - Mixed precision for training (`torch.autocast("cuda")` + `torch.amp.GradScaler`), `cudnn.benchmark = True`.
    - DataLoader: `pin_memory=True`, `num_workers = min(8, os.cpu_count())`.
    - 12 GB GPU: pick batch sizes so peak memory stays under ~10 GB; log `torch.cuda.max_memory_allocated()` per run.
    - Determinism where possible: seed 42 for python/numpy/torch (+ worker seeds); record any nondeterministic ops
      used (cudnn.benchmark itself is a source of nondeterminism — note it in run_info.json).
    - Every run writes GPU name, driver, CUDA and torch versions into its `run_info.json`.
    - Long jobs run under `nohup` or `tmux`, logs in `logs/`; training resumes from the last checkpoint if
      interrupted (implemented in Stage 4).
    - `scripts/check_gpu.py` must exit 0 before any training session.

## 4. Architectures

Exact ids — never rename. "CNN" = custom CNN trained from scratch; all others = ImageNet-pretrained `timm` backbones.

| id | architecture | spatial encoder | temporal encoder | head |
|---|---|---|---|---|
| M01 | EfficientNetB3 + BiLSTM + Attention | EfficientNet-B3 (pretrained) | BiLSTM | temporal attention |
| M02 | CNN + BiLSTM + Attention | custom CNN | BiLSTM | temporal attention |
| M03 | ResNet50 + LSTM + Attention | ResNet-50 (pretrained) | LSTM | temporal attention |
| M04 | ConvNeXt + BiLSTM + Attention | ConvNeXt (pretrained) | BiLSTM | temporal attention |
| M05 | CNN + Transformer + MLP | custom CNN | Transformer encoder | MLP head |
| M06 | DenseNet121 + LSTM + Attention | DenseNet-121 (pretrained) | LSTM | temporal attention |
| M07 | ViT + LSTM + Attention | ViT (pretrained) | LSTM | temporal attention |
| M08 | EfficientNet + Transformer + Attention | EfficientNet (pretrained) | Transformer encoder | temporal attention |
| M09 | CNN + GRU + Attention | custom CNN | GRU | temporal attention |
| M10 | ResNet50 + BiLSTM + Transformer | ResNet-50 (pretrained) | BiLSTM → Transformer | mean pooling |

Shared design: per-step spatial encoder (pretrained backbone or custom CNN) → linear projection → temporal encoder
(LSTM / BiLSTM / GRU / Transformer) → temporal attention (MLP head for M05, mean pooling for M10) → classifier →
P(ASD).

How each dataset feeds the temporal model:
- **D01 eye-tracking**: each sample = a clip of consecutive 1-second gaze windows, each rendered as an image
  (scan path, fixation heat map, time channel) → the sequence is over TIME.
- **D02 facial images**: each sample = one image; the backbone's final feature map (e.g. 7×7 grid) is flattened
  into a sequence of spatial tokens → the sequence is over SPATIAL REGIONS.
- Exact preprocessing parameters (window length, clip length, image size, augmentations, variants of
  EfficientNet/ConvNeXt/ViT) are decided and documented in Stage 2/3, not here.

### Model contracts (Stage 3 — `src/models/`)

- `from src.models.factory import build_model, param_groups, count_params`;
  `model = build_model(model_id, mode, cfg=None, policy=None)` → `HybridModel` on CPU (call `.cuda()`);
  `model_id` ∈ M01..M10, `mode` ∈ `"sequence"` (D01) | `"tokens"` (D02); cfg = `configs/models/<id>.yaml`.
- Input: sequence `x` uint8 (B, 8, 3, 224, 224); tokens `x` uint8 (B, 3, 224, 224). Normalisation/resizing happen
  inside the model (`spatial.preprocess`); loaders pass raw uint8.
- Output: `model(x)` → `{"logit": (B,), "attn": (B, L) or None (M05, M10), "features": (B, d)}`; P(ASD) = sigmoid(logit).
  L = 8 (sequence) or 49 / 196 tokens (196 for ViT and the custom CNN).
- Policy (`cfg["policy"][mode]`): D01 `frozen`, D02 `last_stage`, custom CNN always `full`. Optimizer:
  `torch.optim.AdamW(param_groups(model))` (backbone_lr = head_lr / 10 for pretrained backbones).
  Batch size: `cfg["batch_size"][mode]` (chosen by `scripts/03_profile_models.py`, peak < 10 GB; = 64 for all 20
  model x mode combinations, profile in `reports/stage03_model_profile.csv`).
- D01 + frozen backbone: train the head on cached features: `load_cache(timm_name)` →
  `model.forward_features(feats)` with feats (B, 8, C) float (from `cache/features/<timm_name>_D01.npy`, row = row_in_npy).
  Caches for all 6 pretrained backbones are built (`scripts/03b_build_feature_cache.py`, fp16, 1,241 clips; M03 and
  M10 share resnet50); `load_cache` refuses a cache whose clips_index.csv hash changed. Rebuild if a backbone changes.
- Pinned weights (timm tags, sha256): `reports/stage03_weights.csv`. Custom CNN = 4 x [Conv3x3-BN-ReLU-MaxPool2],
  32-64-128-256, from scratch (112 px in sequence mode, 224 px in tokens mode).
- Stage 3 scripts: `03_profile_models.py` (profile + batch sizes + weights csv), `03b_build_feature_cache.py`,
  `03c_write_report.py` (report from saved outputs only). Training-time estimates assume 40 epochs (Stage 4 decides).

### Training contracts (Stage 4 — `src/training/`, protocol `configs/training.yaml`, FIXED: never tune on test)

- Protocol: AdamW wd 0.01; head_lr 5e-4 (pretrained) / 1e-3 (custom CNN); backbone_lr = head_lr/10 for last_stage,
  = head_lr for the custom CNN, none when frozen; cosine per step with 2-epoch linear warm-up; max_epochs 40; early
  stopping on validation ROC-AUC (D01 clip level), patience 8, min_delta 1e-4; BCEWithLogits pos_weight =
  n_neg/n_pos of the fold's TRAIN part; grad clip 1.0; AMP; batch size from Mxx.yaml (64); seed = 42 + fold;
  thresholds 0.5 + Youden J on VALIDATION (D01 child level on validation children). D02 train aug:
  RandomResizedCrop(224, 0.8-1.0), hflip 0.5, ColorJitter(0.1,0.1,0.1); eval resize 224 + center crop. D01: no aug;
  frozen backbones use cache/features, M02/M05/M09 train end to end on clips. `merge_config` merges training.yaml
  over Mxx.yaml (Mxx.yaml `lr` mirrors it; training.yaml wins).
- Commands: `python scripts/train.py --dataset D01 --model M01 --folds 1-10 [--max-epochs N] [--out results_smoke]
  [--force]`; `python scripts/run_queue.py --dataset D01 --models M01-M10` (subprocess per model, then
  `checks.validate_model_results`; STOPS on FAIL; refreshes summary.json / leaderboard.csv);
  `bash scripts/run_stage5.sh` / `run_stage6.sh` (check_gpu + queue, log appended to logs/stage{5,6}_queue.log,
  flock: one copy). tmux is NOT installed (2026-10-14): `sudo apt install tmux` or use the nohup form in
  reports/STAGE_04_REPORT.md §9.
- Layout: `results/README.md` (fold folder: config.yaml, history.csv, best_model.pt, last.pt only while RUNNING,
  predictions.csv + val_predictions.csv [sample_id, group_id, y_true, y_prob, fold], attention.npy, metrics.json,
  run_info.json). Registry `results/run_registry.csv` (dataset, model, fold, status PENDING/RUNNING/DONE/FAILED/
  BLOCKED, start, end, epochs, best_epoch, val_auc, test_auc, peak_gpu_gb, seconds, error), fcntl-locked.
  Smoke outputs: `results_smoke/` only. Fold logs: `logs/<dataset>/<model>/fold_XX.log` (`logs/<out>/...` otherwise).
- Resume: DONE -> skipped unless --force; RUNNING + last.pt -> continues exactly (model, optimizer, scheduler,
  scaler, epoch, best AUC, patience, history, ids seen, RNG states; loaders re-seeded per (seed, epoch, split));
  refuses if the config hash changed; FAILED / RUNNING without last.pt -> fresh start of that fold folder.
  SIGINT/SIGTERM (explicit handlers: background jobs of non-interactive shells inherit SIGINT as ignored) -> status
  stays RUNNING; stop a queue with Ctrl-c in its terminal or `pkill -TERM -f 'run_queue.py --dataset D01'`, then
  re-run the same command. Deterministic cuDNN: resume bit-identical; benchmark=True (default): ~1e-4 drift, the
  same as between two straight runs (recorded in run_info).
- Leakage guards per run: verify_manifest, check_fold (no shared id/group, both classes, test = folds.csv), every
  batch's ids logged and asserted (no test id in train/val batches), test predicted once from best_model.pt.

## 5. Datasets

Folder names: `configs/paths.yaml`. Evidence and full decisions: `data_registry/DATASET_DECISIONS.md` (DD-01…DD-12);
summary table: `data_registry/DATASET_REGISTRY.md`; per-dataset audits: `data_registry/<id>_audit.{json,md}`.

| id | folder | role | key facts (from audits, re-verified 2026-10-13 in `logs/setup_fact_check.txt`) |
|---|---|---|---|
| D01_eye_tracking | `Eye-Tracking Dataset` | PRIMARY, 10 hybrids | 57 children with gaze (27 ASD / 30 TD); IDs 12, 16 no gaze; 40% rows "Unidentified" (dropped); cross-file duplicates (numeric DD-02 key: identified rows 1,351,157 → 1,103,660); CARS = label leak; ~60 Hz |
| D02_facial_images_A | DFDK folder (long name) | SECONDARY, 10 hybrids | 2,940 images, balanced; pool train/valid/test; 11 pHash≤4 pairs → group; no subject IDs (image-level only) |
| D03_facial_images_B | `AutismDataset` | EXCLUDED | copy of D02 (2,541 byte-identical + 399 re-saved) |
| D04_autism_prediction | `Autism_prediction dataset` | tabular baselines | 800 labelled rows (639/161); test.csv unlabelled → unused |
| D05_toddler_screening | `Autism_dataset` | tabular baselines | Toddler label == (A1..A10 sum ≥ 4): exp (a) all features = "label-definitional upper bound", exp (b) without A1–A10/Qchat score; Combined file EXCLUDED (728 wrong labels) |
| D06_early_autism | `early_autism` | tabular baselines | 1,000 rows (589/411); drop empty `Unnamed: 10` |

CV (Stage 2, all datasets): outer StratifiedGroupKFold(10, shuffle=True, random_state=42) by group (D01 child,
D02 near-dup group, tabular identical-feature-row group); inner val = first split of
StratifiedGroupKFold(7, shuffle=True, random_state=42+k) on the training groups of fold k.

### Data contracts (Stage 2 — every later stage must use exactly these)

- **Before any training run**: `from src.data.splits import verify_manifest, load_fold`; call `verify_manifest()`
  (raises on any hash mismatch), then `load_fold(ds, k)` → `{"train": [...], "val": [...], "test": [...]}` sample ids.
- **D01**: `cache/D01_eye_tracking/clips_uint8.npy` (np.load mmap_mode="r"), uint8, shape (N, 8, 3, 224, 224) =
  clip × 1-s window × [scan path, fixation density, time] × H × W. Row = `clips_index.csv.row_in_npy`;
  sample_id = `clip_id`; child = `participant_id` (group `pXX`); label 1 = ASD. Child-level metric = mean clip
  probability per child. `quality_per_child.csv` = DD-12 shortcut features ONLY (never model inputs).
- **D02**: `cache/D02_facial_images_A/manifest.csv` (sample_id = `image_id`; image = datasets_dir/`rel_path`; label
  1 = autistic; `group_id` = near-dup group). No stored pixels: loader resizes to 224, ImageNet-normalises, augments
  train only. Backbone feature map → spatial token sequence.
- **D04/D05/D06**: `cache/<ds>/clean.csv` + `columns.yaml` (roles + `feature_sets`; D05 has `set_all` and
  `set_honest`). Encoders/scalers/imputers are fit per fold on the train part only (Stage 7).
- Split files: `splits/<ds>/folds.csv`, `fold_01..10.json`, `split_summary.csv`; `splits/SPLITS_MANIFEST.json`.
  Regenerate only with `scripts/02d_make_splits.py` (deterministic); never edit by hand.

## 6. Stage plan

Update the status column at the end of every stage.

| stage | content | status |
|---|---|---|
| S1 | Workspace, environment, dataset audit | PASS (git requirement replaced by local snapshot, 2026-10-13) |
| S2 | Data preparation + saved 10-fold split manifests + leakage tests | PASS (2026-10-13, snapshots/stage_02) |
| S3 | Implement the 10 architectures + unit tests (shapes, forward/backward on GPU, param counts) | PASS (2026-10-14, snapshots/stage_03; resumed after interruption) |
| S4 | Training engine (fold loop, early stopping, AMP, checkpoint/resume, metrics, result writer) + smoke test: 1 model × 2 folds × 2 epochs | PASS (2026-10-14, snapshots/stage_04) |
| S5 | D01 eye-tracking: M01 → M10 sequentially, 10 folds each (100 runs) | PASS (2026-10-14, snapshots/stage_05) |
| S6 | D02 facial images: M01 → M10 sequentially, 10 folds each (100 runs) | not started |
| S7 | Tabular baselines on D04, D05 (both experiments), D06 — 10-fold | PASS (done EARLY 2026-10-14, before S6, user-approved; CPU-only alongside S5; `tabular/`; snapshots/stage_07) |
| S8 | Shortcut controls + statistics (quality-only baseline, bootstrap CIs, Friedman/Nemenyi) | not started |
| S9 | Figures, leaderboards, final results report | not started |
| S10 | Final audit: re-run all tests, verify every result file, freeze | not started |

Sequential rule inside S5/S6: M01 must finish all 10 folds and pass its checks before M02 starts, and so on.
Stage 2 open items are listed in `reports/STAGE_01_REPORT.md` §8.

## 7. Results layout

Authoritative spec: `results/README.md`.

```
results/
  run_registry.csv          dataset, model, fold, status, start, end, epochs, best_epoch, val_auc, test_auc,
                            peak_gpu_gb, seconds, error
  <dataset_id>/
    leaderboard.csv
    <model_id>/
      summary.json          mean ± std of every test metric over the 10 folds
      fold_01/ … fold_10/
        config.yaml  metrics.json  predictions.csv  val_predictions.csv  history.csv  attention.npy
        best_model.pt (never snapshotted)  last.pt (only while RUNNING)  run_info.json
```

- Tabular baselines (Stage 7): `results/<D04|D05|D06>/<experiment>/<model>/fold_XX/` (experiment = set_all /
  set_honest; models dummy_majority, logreg, random_forest, hist_gb, mlp); code `tabular/`, registry
  `tabular/run_registry_stage7.csv`.
- Metrics per fold: accuracy, balanced accuracy, precision, recall/sensitivity, specificity, F1, ROC-AUC, PR-AUC,
  confusion matrix, Brier score. D01 also child-level metrics (child probability = mean clip probability).
- `predictions.csv`: `sample_id, group_id, y_true, y_prob, fold`. `history.csv`: `epoch, train_loss, val_loss,
  val_auc, lr, seconds, peak_gpu_gb` (+ `val_auc_child` for D01). Status ∈ PENDING / RUNNING / DONE / FAILED / BLOCKED
  (Stage 4 formats; details in `results/README.md`).

## 8. Commands

```bash
cd /home/score/Documents/AI_project/asd_hybrid_10fold
source .venv/bin/activate
python scripts/check_gpu.py                          # must print "GPU CHECK: OK" (exit 1 if no CUDA)
python -m pytest -v                                  # whole suite (pytest.ini -> tests/)
python scripts/setup_verify_stage1_facts.py          # re-check audit facts
python scripts/00_snapshot_raw.py /tmp/snap.tsv      # live raw-data snapshot (compare with data_registry/)
nohup python <train script> ... > logs/<name>.log 2>&1 &   # long jobs (Stage 4+)
```

- Stage reports: `reports/STAGE_NN_REPORT.md` (last line `STAGE N DECISION: PASS|FAIL`).
  Setup report: `reports/SETUP_REPORT.md`.
- Logs: `logs/`. Snapshots: `snapshots/stage_NN/` (cache/ is hashed in CHECKSUMS, never copied).

## 9. Session log (details: `reports/STAGE_NN_REPORT.md`; keep entries short - CLAUDE.md max 260 lines)

- 2026-10-12 — S1 (previous agent): skeleton, venv, audit of 6 datasets, DD-01…DD-12; FAIL (git not installed).
- 2026-10-13 — Setup (Claude): project moved into AI_project (+ symlink), .venv, `configs/paths.yaml`, check_gpu,
  test_setup, this file. User override: NO GIT -> S1 + Setup PASS (`snapshots/stage_01/`).
- 2026-10-13 — S2: D01 1,241 clips / 57 children; D02 2,940 images / 2,929 groups; D04-D06 cleaned; splits +
  SPLITS_MANIFEST; leakage tests. PASS.
- 2026-10-13/14 — S3: interrupted during profiling, resumed (`logs/stage03_recovery_checklist.md`); 10 models,
  6 feature caches, batch size 64 everywhere. PASS.
- 2026-10-14 — S4: fixed protocol, engine, queue, run_stage5/6.sh, smoke tests; fixed the SIGINT-ignored-in-background
  bug. PASS.
- 2026-10-14 — S5 launched 10:01:09 with nohup/setsid (tmux not installable), PID 119087, via
  `scripts/stage5_{launch.sh,stop.sh,status.py}`; finished 10:45:38, exit 0, 100/100 DONE; pid file moved to
  `logs/stage5.pid.finished`.
- 2026-10-14 — S7 run EARLY (user-approved, before S6) in parallel with S5: CPU only, `tabular/`, 200 runs. PASS.
- 2026-10-14 — S5B: checker fix (inclusive range +-1e-9, NaN only flagged; validation code only, 630 result files
  byte-identical), verify + `reports/STAGE_05_REPORT.md` PASS; snapshots stage_05 + stage_07 (hash result files).
  S6 NOT launched: user reviews S5 first.
