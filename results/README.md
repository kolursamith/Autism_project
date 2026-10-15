# Results layout (used by ALL training stages; updated in Stage 4 to the engine's actual formats)

Written by `src/training/` (engine, registry, summarize). Real experiments (S5, S6) write here; smoke tests write to
`results_smoke/` with the same layout. Protocol: `configs/training.yaml`.

```
results/
  run_registry.csv                         one row per (dataset, model, fold) run with status (file-locked writes)
  <dataset_id>/
    leaderboard.csv                        all models for that dataset (sorted by mean test ROC-AUC)
    <model_id>/
      summary.json                         mean, std (ddof=1), min, max over folds of every test metric +
                                           pooled out-of-fold metrics (sample level; child level for D01)
      fold_01/ … fold_10/
        config.yaml                        exact merged config (configs/models/Mxx.yaml + configs/training.yaml)
        history.csv                        epoch, train_loss, val_loss, val_auc, lr, seconds, peak_gpu_gb
                                           (+ val_auc_child for D01)
        best_model.pt                      model state_dict at the best validation AUC (never snapshotted)
        last.pt                            resume checkpoint; exists only while a run is RUNNING (deleted when DONE)
        predictions.csv                    test fold, best checkpoint, predicted once
        val_predictions.csv                validation part (source of the Youden threshold)
        attention.npy                      test attention weights (N_test, L), predictions.csv row order
                                           (attention models only; not M05, M10)
        metrics.json                       test metrics at 0.5 and at the validation-chosen threshold
        run_info.json                      status, times, versions, hashes, batch counts, peak GPU memory
```

Conventions
- `<dataset_id>`: `D01_eye_tracking` or `D02_facial_images_A`. Fold folders `fold_01` … `fold_10`.
- `predictions.csv` / `val_predictions.csv` columns, in this order: `sample_id, group_id, y_true, y_prob, fold`
  (`group_id` = child `pXX` for D01, near-duplicate group for D02). `y_prob` = predicted P(ASD).
- `metrics.json`: `test.threshold_0.5` and `test.threshold_val_youden` blocks (the threshold maximises Youden's J on
  the VALIDATION predictions; never chosen on test), each with accuracy, balanced_accuracy, precision,
  recall (= sensitivity), specificity, f1, roc_auc, pr_auc (average precision), brier, confusion_matrix
  [[TN, FP], [FN, TP]], n, n_pos, n_neg. D01 adds `test_child` (child probability = mean clip probability; child
  threshold chosen on validation children). Also `val_at_best` (validation metrics of the best checkpoint).
- `run_registry.csv` columns: `dataset, model, fold, status, start, end, epochs, best_epoch, val_auc, test_auc,
  peak_gpu_gb, seconds, error`; status ∈ {PENDING, RUNNING, DONE, FAILED, BLOCKED}. `run_info.json` uses the same
  statuses. A DONE run is never re-run unless `--force`; a RUNNING run with `last.pt` resumes exactly.
- `run_info.json` records GPU name, driver, CUDA, cuDNN, torch, timm, python, `code_hash_src`, `config_sha256`,
  sha256 of the split files and of the pretrained weights / feature cache, seed, cuDNN settings, sessions (resume
  history), batch counts per split and the test-leakage counter (must be 0).
- Seed = 42 + fold. `*.pt` files are never snapshotted.
