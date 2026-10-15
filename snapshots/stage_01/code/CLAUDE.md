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

CV: D01 StratifiedGroupKFold(10, shuffle=True, random_state=42) by child; D02 same, by near-dup group;
D04/D05/D06 StratifiedKFold(10, shuffle=True, random_state=42).

## 6. Stage plan

Update the status column at the end of every stage.

| stage | content | status |
|---|---|---|
| S1 | Workspace, environment, dataset audit | PASS (git requirement replaced by local snapshot, 2026-10-13) |
| S2 | Data preparation + saved 10-fold split manifests + leakage tests | not started |
| S3 | Implement the 10 architectures + unit tests (shapes, forward/backward on GPU, param counts) | not started |
| S4 | Training engine (fold loop, early stopping, AMP, checkpoint/resume, metrics, result writer) + smoke test: 1 model × 2 folds × 2 epochs | not started |
| S5 | D01 eye-tracking: M01 → M10 sequentially, 10 folds each (100 runs) | not started |
| S6 | D02 facial images: M01 → M10 sequentially, 10 folds each (100 runs) | not started |
| S7 | Tabular baselines on D04, D05 (both experiments), D06 — 10-fold | not started |
| S8 | Shortcut controls + statistics (quality-only baseline, bootstrap CIs, Friedman/Nemenyi) | not started |
| S9 | Figures, leaderboards, final results report | not started |
| S10 | Final audit: re-run all tests, verify every result file, freeze | not started |

Sequential rule inside S5/S6: M01 must finish all 10 folds and pass its checks before M02 starts, and so on.
Stage 2 open items are listed in `reports/STAGE_01_REPORT.md` §8.

## 7. Results layout

Authoritative spec: `results/README.md`.

```
results/
  run_registry.csv          dataset_id, model_id, fold, status, start_time, end_time, val_auc, test_auc,
                            gpu_mem_gb, seconds, run_dir
  <dataset_id>/
    leaderboard.csv
    <model_id>/
      summary.json          mean ± std of every test metric over the 10 folds
      fold_01/ … fold_10/
        config.yaml  metrics.json  predictions.csv  history.csv  best_model.pt (never snapshotted)  run_info.json
```

- Metrics per fold: accuracy, balanced accuracy, precision, recall/sensitivity, specificity, F1, ROC-AUC, PR-AUC,
  confusion matrix, Brier score. D01 also child-level metrics (child probability = mean clip probability).
- `predictions.csv`: `sample_id, subject_id, y_true, y_prob, fold`. `history.csv`: `epoch, train_loss, val_loss,
  val_auc, lr, seconds`. `run_info.json` status ∈ running / completed / failed / blocked.

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

## 9. Session log

- 2026-10-12 — Stage 1 (previous agent): skeleton, venv, audit of 6 datasets, DD-01…DD-12, 24 tests pass.
  STAGE 1 DECISION: FAIL (git not installed).
- 2026-10-13 — Setup (Claude): moved project into AI_project (+ symlink), fresh .venv, `configs/paths.yaml`,
  `scripts/check_gpu.py`, `tests/test_setup.py`, 37/37 Stage 1 facts confirmed, DD-12 D02 clause added, this file.
  Git was never installed → no commits.
- 2026-10-13 — User override: NO GIT. Stage 1 + Setup reports regenerated → PASS; recorded in
  `snapshots/stage_01/`. Stage 2 started.
