"""Stage 4 - write reports/STAGE_04_REPORT.md from saved outputs only (results_smoke/, logs/, reports/, configs/).
The decision line is computed: PASS only if pytest passed completely, every smoke check passes, the interruption test
resumed at epoch 2 and finished DONE, deterministic resume diffs are <= 1e-5 and results/ holds no run output.
Usage: .venv/bin/python scripts/04c_write_report.py"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.data.common import LOG_DIR, PROJECT_DIR, md_table  # noqa: E402
from src.training.checks import validate_model_results  # noqa: E402

REP = os.path.join(PROJECT_DIR, "reports")
SMOKE = os.path.join(PROJECT_DIR, "results_smoke")
SMOKE_RUNS = [("1", "D01_eye_tracking", "M01", [1, 2], "frozen backbone + cached features"),
              ("2", "D01_eye_tracking", "M09", [1, 2], "custom CNN end to end on clips"),
              ("3", "D02_facial_images_A", "M07", [1, 2], "ViT-B/16, last_stage fine-tuning (heaviest)"),
              ("4", "D02_facial_images_A", "M03", [1], "interruption test (SIGINT after epoch 1, restart)")]


def R(p):
    with open(p) as fh:
        return fh.read().rstrip("\n")


def tree(root):
    out = []
    for dp, dns, fns in sorted(os.walk(root)):
        dns.sort()
        for fn in sorted(fns):
            p = os.path.join(dp, fn)
            out.append(f"{os.path.getsize(p):>11,}  {os.path.relpath(p, PROJECT_DIR)}")
    return "\n".join(out)


def main():
    t = yaml.safe_load(open(os.path.join(PROJECT_DIR, "configs", "training.yaml")))
    pyt = R(os.path.join(LOG_DIR, "stage04_pytest.txt"))
    last = pyt.splitlines()[-1]
    py_ok = re.search(r"(\d+) (failed|error)", last) is None and "passed" in last
    counts = pd.Series(re.findall(r"^(tests/\S+\.py)::", pyt, re.M)).value_counts().sort_index()
    est = pd.read_csv(os.path.join(REP, "stage04_time_estimate.csv"))
    ov = pd.read_csv(os.path.join(REP, "stage04_overheads.csv"))
    timing = pd.read_csv(os.path.join(REP, "stage04_smoke_timing.csv"))
    interrupt = R(os.path.join(LOG_DIR, "stage04_smoke_interrupt.log"))
    det = R(os.path.join(LOG_DIR, "stage04_resume_deterministic_diff.txt"))
    det_vals = [float(x) for x in re.findall(r"': ([0-9.e+-]+)", det)]
    det_ok = len(det_vals) == 10 and max(det_vals) <= 1e-5

    # ---------------- smoke runs
    rows, hist_md, met_md, checks, check_ok = [], [], [], [], True
    for n, ds, m, folds, what in SMOKE_RUNS:
        verdict, reasons = validate_model_results(ds, m, SMOKE, folds)
        check_ok &= verdict == "PASS"
        checks.append(f"- smoke {n} {ds} x {m} folds {folds}: **{verdict}** {'; '.join(reasons)}")
        for k in folds:
            d = os.path.join(SMOKE, ds, m, f"fold_{k:02d}")
            h = pd.read_csv(os.path.join(d, "history.csv"))
            ri = json.load(open(os.path.join(d, "run_info.json")))
            mt = json.load(open(os.path.join(d, "metrics.json")))
            rows.append({"smoke": n, "dataset": ds, "model": m, "fold": k, "input": what.split(",")[0],
                         "epochs": ri["epochs_run"], "best_epoch": ri["best_epoch"],
                         "s/epoch (each)": " / ".join(f"{s:.1f}" for s in h.seconds),
                         "peak GB": round(ri["peak_gpu_gb"], 2), "run s": round(ri["seconds"], 1),
                         "resumed": ri["resumed"], "files": len(os.listdir(d)),
                         "batches tr/val/test": "/".join(str(ri["batch_counts"][s]) for s in ["train", "val", "test"])})
            hist_md += [f"`{ds}/{m}/fold_{k:02d}/history.csv`", "", md_table(h.round(5)), ""]
            te, tv = mt["test"]["threshold_0.5"], mt["test"]["threshold_val_youden"]
            r = {"run": f"{m} {ds[:3]} f{k}", "val thr (Youden)": round(mt["val_threshold_youden"], 4),
                 "AUC": round(te["roc_auc"], 4), "PR-AUC": round(te["pr_auc"], 4),
                 "bal acc @0.5": round(te["balanced_accuracy"], 4), "bal acc @val thr": round(tv["balanced_accuracy"], 4),
                 "sens/spec @0.5": f"{te['sensitivity']:.3f}/{te['specificity']:.3f}", "brier": round(te["brier"], 4),
                 "CM @0.5": te["confusion_matrix"]}
            if "test_child" in mt:
                r["child AUC"] = round(mt["test_child"]["threshold_0.5"]["roc_auc"], 4)
                r["child CM @0.5"] = mt["test_child"]["threshold_0.5"]["confusion_matrix"]
            met_md.append(r)
    smoke = pd.DataFrame(rows)
    ri4 = json.load(open(os.path.join(SMOKE, "D02_facial_images_A", "M03", "fold_01", "run_info.json")))
    resume_ok = ("continuing at epoch 2" in interrupt and "exit code after SIGINT: 130" in interrupt
                 and ri4["status"] == "DONE" and [s["start_epoch"] for s in ri4["sessions"]] == [1, 2])
    res_dirs = [p for p in glob.glob(os.path.join(PROJECT_DIR, "results", "*")) if os.path.isdir(p)]
    results_clean = not res_dirs and not os.path.exists(os.path.join(PROJECT_DIR, "results", "run_registry.csv"))
    ref = os.path.join(SMOKE, "resume_reference", "D02_facial_images_A", "M03", "fold_01")
    ha = pd.read_csv(os.path.join(SMOKE, "D02_facial_images_A", "M03", "fold_01", "history.csv"))
    hb = pd.read_csv(os.path.join(ref, "history.csv"))
    bench = {c: float((ha[c] - hb[c]).abs().max()) for c in ["train_loss", "val_loss", "val_auc", "lr"]}

    s5 = est[est.stage == "S5"]
    s6 = est[est.stage == "S6"]
    proto = [("optimizer", f"AdamW, weight_decay {t['optimizer']['weight_decay']}"),
             ("head_lr", f"{t['lr']['head_lr_pretrained']} (pretrained M01 M03 M04 M06 M07 M08 M10) / "
                         f"{t['lr']['head_lr_custom_cnn']} (custom CNN M02 M05 M09)"),
             ("backbone_lr", f"head_lr x {t['lr']['backbone_lr_factor_last_stage']} when last_stage (D02 pretrained); "
                             f"= head_lr for the custom CNN (full); none when frozen (D01 pretrained)"),
             ("schedule", f"cosine per optimizer step to 0 at max_epochs, {t['schedule']['warmup_epochs']}-epoch linear "
                          "warm-up"),
             ("max_epochs", t["max_epochs"]),
             ("early stopping", f"validation ROC-AUC (D01: clip level), patience {t['early_stopping']['patience']}, "
                                f"min_delta {t['early_stopping']['min_delta']}; best-val-AUC checkpoint kept"),
             ("loss", "BCEWithLogits, pos_weight = n_neg / n_pos of the TRAIN part of the fold (val loss uses the "
                      "same pos_weight)"),
             ("gradient clipping", f"global norm {t['grad_clip_norm']} (after GradScaler.unscale_)"),
             ("precision", "torch.autocast('cuda', float16) + torch.amp.GradScaler"),
             ("batch size", "configs/models/Mxx.yaml (64 for all 20 model x mode combinations)"),
             ("seed", f"{t['seed_base']} + fold (python, numpy, torch, cuda, per-epoch data-loader generators)"),
             ("cuDNN", "benchmark=True (golden rule 10; nondeterministic in the last digits, recorded in run_info)"),
             ("thresholds", "primary 0.5; secondary = max Youden J on the VALIDATION predictions of the best "
                            "checkpoint (D01 child level: on validation children); never chosen on test"),
             ("D02 train augmentation", "RandomResizedCrop(224, scale 0.8-1.0), horizontal flip p 0.5, "
                                        "ColorJitter(0.1, 0.1, 0.1)"),
             ("D02 eval", "resize shorter side to 224 + center crop 224"),
             ("D01", "no augmentation; frozen pretrained backbones read cache/features (Stage 3); M02 M05 M09 train "
                     "end to end on uint8 clips"),
             ("data loaders", "num_workers = min(8, cpu_count) = 8, pin_memory, re-created each epoch with a "
                              "generator seeded by (seed, epoch, split)")]
    proto = pd.DataFrame(proto, columns=["item", "value (configs/training.yaml)"])

    L = ["# STAGE 4 REPORT - training engine + smoke tests", "",
         "Screening-support research, not diagnosis. The values below come from 2-epoch smoke tests and are "
         "meaningless as performance estimates; they only show that the engine produces valid outputs. No full "
         "experiment was run; smoke outputs are in `results_smoke/`, `results/` contains no run output "
         f"({'confirmed' if results_clean else 'NOT CLEAN: ' + str(res_dirs)}). No git: recorded by `snapshots/stage_04/`.",
         "", "## 1. Summary", "",
         "- `configs/training.yaml` fixes the training protocol before any real result exists (section 3).",
         "- `src/training/` (data, metrics, engine, registry, summarize, checks) + `scripts/train.py`, "
         "`scripts/run_queue.py`, `scripts/run_stage5.sh`, `scripts/run_stage6.sh` (written, not run).",
         f"- Tests: {last.strip('= ')} (whole suite; Stage 4 adds "
         f"{int(counts.get('tests/test_stage04_units.py', 0))} unit + "
         f"{int(counts.get('tests/test_stage04_engine.py', 0))} GPU engine tests).",
         f"- Smoke tests 1-4 on real data: all validity checks {'PASS' if check_ok else 'FAIL'}; 0 test ids in any "
         "training/validation batch in every run.",
         f"- Resume: deterministic cuDNN -> bit-identical (max |diff| {max(det_vals):.1e}); interruption test "
         f"{'resumed at epoch 2 and finished DONE' if resume_ok else 'FAILED'}.",
         "- Found and fixed while testing: a job started with `&` from a non-interactive shell (nohup, scripts) "
         "inherits SIGINT as IGNORED, so the first interruption attempt was not interrupted at all (section 7.1). "
         "`train.py` and `run_queue.py` now install explicit SIGINT/SIGTERM handlers.",
         f"- Estimated S5 (D01): {s5.est_model_h_E20.sum():.2f} h at ~20 epochs, {s5.est_model_h_E40.sum():.2f} h at 40; "
         f"S6 (D02): {s6.est_model_h_E20.sum():.2f} h at ~20 epochs, {s6.est_model_h_E40.sum():.2f} h at 40 (section 8).",
         "", "## 2. Pre-flight (`logs/stage04_preflight.txt`)", "", "```", R(os.path.join(LOG_DIR,
                                                                                     "stage04_preflight.txt")), "```",
         "", "## 3. Training protocol (fixed before any real result)", "", md_table(proto), "",
         "Model configs (`configs/models/Mxx.yaml` `lr`) were updated to mirror these learning rates; "
         "`training.yaml` is authoritative and is merged over the model config (test "
         "`test_model_config_lr_mirrors_training_yaml`). Decisions taken here: D01 early stopping uses the clip-level "
         "validation AUC (7-8 validation children give a very coarse child AUC; it is logged as `val_auc_child`); "
         "`configs/data/D02.yaml` says normalisation happens in the loader, but the Stage 3 model contract (newer) "
         "normalises inside the model, so loaders pass raw uint8.", "",
         "## 4. Engine design", "",
         "| file | role |", "|---|---|",
         "| `src/training/data.py` | datasets built only from `splits/<ds>/fold_XX.json` ids; labels/groups from the "
         "hash-verified `folds.csv`, cross-checked against the data manifest; D01 cached features (in RAM) or uint8 "
         "clips (memmap), D02 PIL images with train/eval transforms; items carry their index so every batch's ids "
         "are logged |",
         "| `src/training/metrics.py` | binary metrics (checked against sklearn), validation-only Youden threshold, "
         "child-level aggregation, range checks |",
         "| `src/training/engine.py` | `merge_config`, `check_fold` (leakage asserts), `train_one_fold` |",
         "| `src/training/registry.py` | `run_registry.csv`, fcntl-locked read-modify-write, skip/force logic |",
         "| `src/training/summarize.py` | `summary.json` per model, `leaderboard.csv` per dataset |",
         "| `src/training/checks.py` | `validate_model_results` (queue gate) |",
         "| `scripts/train.py` / `run_queue.py` | entry points (signal handling, subprocess per model, stop on FAIL) |",
         "", "`train_one_fold` order: `verify_manifest()` -> `check_fold` (no shared sample id or group between "
         "train/val/test, no duplicates or unknown ids, both classes everywhere, test = folds.csv outer_fold) -> "
         "registry decision (DONE: skip unless --force; RUNNING + last.pt: resume; else fresh start in a clean fold "
         "folder) -> per epoch: train (AMP, clipping, per-step cosine), validation, history row, best_model.pt on "
         "improvement, last.pt (model, optimizer, scheduler, scaler, epoch, best AUC, patience counter, history, "
         "ids seen per split, batch counts, python/numpy/torch/cuda RNG states) -> load best_model.pt -> validation "
         "(Youden) + test predicted once -> assert no test id was ever in a train/val batch, every train/val batch id "
         "belongs to its split, test predictions cover the test ids exactly once -> predictions.csv, "
         "val_predictions.csv, attention.npy, metrics.json, run_info.json -> registry DONE, last.pt deleted.", "",
         "Exact resume: data loaders are re-created every epoch with a generator seeded by (seed, epoch, split), so "
         "shuffle order and worker (augmentation) seeds depend only on the epoch; resume refuses a last.pt written "
         "with a different config hash. Ctrl-C / SIGINT / SIGTERM -> status stays RUNNING, last.pt kept; any other "
         "exception -> FAILED with the error in the registry. Checkpoints are written atomically (tmp + rename). "
         "Pretrained weights are re-hashed against `reports/stage03_weights.csv` at every run start "
         "(HF_HUB_OFFLINE=1 in train.py: no network). Logs: `logs/<dataset>/<model>/fold_XX.log` for `results/`, "
         "`logs/<out root>/<dataset>/<model>/fold_XX.log` otherwise, plus one line per epoch on stdout.", "",
         "## 5. Tests (`.venv/bin/python -m pytest -v`, `logs/stage04_pytest.txt`)", "",
         md_table(counts.rename_axis("file").reset_index(name="tests")), "",
         "Stage 4 tests: protocol values; merged learning rates per policy; scheduler shape; D02 transforms; real "
         "splits pass `check_fold`; corrupted temp copies trigger `LeakageError` (id in train and test, duplicate, "
         "group overlap, unknown id, test set != folds.csv) for D01 and D02; `verify_manifest` rejects a modified "
         "temp copy of splits/; the engine refuses a corrupted split before starting; metrics == sklearn on random "
         "inputs at 3 thresholds; Youden threshold == brute force on validation data and only takes validation "
         "arrays; metrics.json threshold == Youden of val_predictions.csv; registry skip/force/resume and 4 "
         "concurrent writer processes lose no rows; summary mean/std/min/max/pooled on fake folds; resume (D01 M01 "
         "cached, D02 M09 images with augmentation and 8 workers): 3 epochs straight == 1 epoch + simulated kill + "
         "resume (history and test predictions within 1e-5), interrupted state = RUNNING + last.pt, DONE run skipped, "
         "resume refused after a config change, checks catch incomplete results.", "",
         "<details><summary>full pytest output</summary>", "", "```", pyt, "```", "", "</details>", "",
         "## 6. Smoke tests (real data, max 2 epochs, `results_smoke/`)", "",
         md_table(smoke), "",
         "Validity checks (`validate_model_results` adapted to the smoke folds):", "", *checks, "",
         "Test metrics (2 epochs; only checked to be valid numbers):", "", md_table(pd.DataFrame(met_md).fillna("n/a")), "",
         "### history.csv", "", *hist_md,
         "### Result folder tree (`results_smoke/`, bytes)", "", "```", tree(SMOKE), "```", "",
         "`summary.json` and `leaderboard.csv` were produced by `scripts/run_queue.py` / `summarize.py` on the smoke "
         "folds (`logs/stage04_queue_smoke.log`: the queue skipped the DONE folds, validated them and summarised).", "",
         "## 7. Interruption / resume evidence", "",
         "### 7.1 First attempt: SIGINT was ignored (bug found and fixed)", "",
         "`logs/stage04_smoke_interrupt_attempt1_sigint_ignored.log`: `kill -INT` was sent after epoch 1, but the "
         "process trained on to epoch 3 and finished DONE. Cause: POSIX shells start background jobs (`cmd &`) of a "
         "non-interactive shell with SIGINT ignored, and Python then never raises KeyboardInterrupt. The same applies "
         "to any nohup / script-launched S5/S6 run. Fix: explicit SIGINT/SIGTERM handlers in `train.py` and "
         "`run_queue.py` (the queue waits up to 15 s for its child, then forwards SIGINT). The M03 smoke outputs of "
         "that attempt were deleted and the test was re-run.", "",
         "### 7.2 Interruption test (`scripts/04_smoke_interrupt.sh`, `logs/stage04_smoke_interrupt.log`)", "",
         "```", interrupt, "```", "",
         f"Resumed run_info sessions: {[(s['start_epoch'], s['end_epoch'], s.get('interrupted', False)) for s in ri4['sessions']]} "
         f"(start epoch, end epoch, interrupted).", "",
         "### 7.3 Exactness", "",
         "Deterministic cuDNN settings (`logs/stage04_resume_deterministic_diff.txt`, same comparison as the pytest "
         "resume test):", "", "```", det, "```", "",
         "Real setting (`cudnn.benchmark=True`), interrupted D02 x M03 run vs. a straight run of the same fold "
         f"(`results_smoke/resume_reference/`): max |diff| " + ", ".join(f"{k} {v:.1e}" for k, v in bench.items())
         + ". Two straight runs differ by the same order (attempt 1, which was effectively straight, logged "
         "train_loss 0.4513 / val_loss 0.4327 at epoch 3 vs. 0.4516 / 0.4331 for the reference), so this is cuDNN "
         "autotuning nondeterminism, not the resume logic; it is recorded in every run_info.json.", "",
         "### 7.4 Queue interruption (`logs/stage04_queue_sigterm_check.log`)", "",
         "`run_queue.py` started in its own session on a scratch output folder (outside the project, deleted "
         "afterwards; D01 x M09 fold 3, max 8 epochs), SIGTERM sent to the queue after epoch 1: the queue waited, "
         "forwarded SIGINT, train.py saved and exited (registry RUNNING, last.pt present), the queue exited 130; "
         "re-running the same command resumed at epoch 6, finished DONE and validate_model_results gave PASS.", "",
         "## 8. Time estimates for S5 and S6 (`scripts/04_time_estimate.py`, `reports/stage04_time_estimate.csv`)", "",
         "Method: measured smoke epochs (data loading, validation, checkpointing included) give a per-input-type "
         "overhead; every model = overhead + steps/epoch x its Stage 3 profiled s/step; per-fold overhead = first-epoch "
         "extra (cuDNN autotune, cold page cache) + model build / weight hash / final prediction. 10 folds x 10 models; "
         "~20 epochs = assumed average with early stopping (patience 8, so a run lasts at least 9 epochs), 40 = "
         "max_epochs worst case. Process start-up (~5 s per model) is not included.", "",
         "Measured smoke timing (`reports/stage04_smoke_timing.csv`):", "", md_table(timing), "",
         "Overheads per input type (`reports/stage04_overheads.csv`):", "", md_table(ov), "",
         md_table(est[["stage", "model_id", "input", "measured_in_smoke", "est_epoch_s", "est_fold_min_E20",
                       "est_fold_min_E40", "est_model_h_E20", "est_model_h_E40"]]), "",
         f"| stage | ~20 epochs (early stopping) | 40 epochs (worst case) |", "|---|---|---|",
         f"| S5 (D01, 100 runs) | {s5.est_model_h_E20.sum():.2f} h | {s5.est_model_h_E40.sum():.2f} h |",
         f"| S6 (D02, 100 runs) | {s6.est_model_h_E20.sum():.2f} h | {s6.est_model_h_E40.sum():.2f} h |", "",
         "Uncertainty: only 4 of the 20 model x dataset combinations were timed directly; light D02 models may be "
         "bounded by JPEG decoding rather than by the GPU (their estimate could be low by ~1-2 s per epoch).", "",
         "## 9. How to start S5 (after the user's go)", "",
         "tmux is NOT installed on this machine (`which tmux` found nothing; checked 2026-10-14). Either install it "
         "(`sudo apt install tmux`) and run:", "", "```bash",
         f"cd {PROJECT_DIR}", "tmux new -s stage5 'bash scripts/run_stage5.sh'     # detach: Ctrl-b d",
         "tmux attach -t stage5                                # re-attach; Ctrl-c stops gracefully (resumable)",
         "tail -f logs/stage5_queue.log                        # progress from any terminal", "```", "",
         "or, without tmux:", "", "```bash", f"cd {PROJECT_DIR}",
         "nohup bash scripts/run_stage5.sh > /dev/null 2>&1 &    # log: logs/stage5_queue.log",
         "tail -f logs/stage5_queue.log",
         "pkill -TERM -f 'run_queue.py --dataset D01'          # graceful stop; re-run the same command to resume",
         "```", "",
         "Both scripts run `check_gpu.py` first, append to `logs/stage{5,6}_queue.log`, hold a lock (one copy at a "
         "time), skip DONE folds, resume RUNNING folds and stop at the first model that fails "
         "`validate_model_results`. `cat results/run_registry.csv` shows the status of every run.", "",
         "## 10. Open items for Stage 5", "",
         "1. Get the user's go; then `bash scripts/run_stage5.sh` (D01, M01 -> M10, sequential, gate after each model).",
         "2. D01 validation sets have 7-8 children: the clip-level validation AUC used for early stopping is noisy; "
         "report child-level test metrics as the main D01 result (Stage 2 open item 3).",
         "3. cuDNN benchmark nondeterminism: re-running a fold reproduces results only to ~1e-3 in the losses; the "
         "registry never re-runs DONE folds, so the stored results are the reference.",
         "4. Stage 8 must still add the shortcut controls (quality-only baseline, stimulus-aware checks) before any "
         "D01 result is interpreted.", "",
         f"Checks for the decision: pytest {'OK' if py_ok else 'FAILED'}; smoke checks {'OK' if check_ok else 'FAILED'}; "
         f"interruption/resume {'OK' if resume_ok else 'FAILED'}; deterministic resume {'OK' if det_ok else 'FAILED'}; "
         f"results/ clean {'OK' if results_clean else 'FAILED'}.", ""]
    ok = py_ok and check_ok and resume_ok and det_ok and results_clean
    reasons = [n for n, v in [("pytest", py_ok), ("smoke checks", check_ok), ("interruption/resume", resume_ok),
                              ("deterministic resume", det_ok), ("results/ not clean", results_clean)] if not v]
    L.append("STAGE 4 DECISION: PASS" if ok else "STAGE 4 DECISION: FAIL - " + ", ".join(reasons))
    with open(os.path.join(REP, "STAGE_04_REPORT.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(L[-1])


if __name__ == "__main__":
    main()
