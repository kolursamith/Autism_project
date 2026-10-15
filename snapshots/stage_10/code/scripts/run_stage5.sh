#!/usr/bin/env bash
# Stage 5: D01, models M01 -> M10 sequentially, 10 folds each, outputs in results/ (NOT run in Stage 4).
# Safe to re-run: DONE folds are skipped, an interrupted (RUNNING) fold resumes from its last.pt, and the queue stops
# at the first model that fails validate_model_results. Only one copy can run at a time (flock).
# Start inside tmux:  tmux new -s stage5 'bash scripts/run_stage5.sh'   (detach: Ctrl-b d; re-attach: tmux attach -t stage5)
# Log (appended): logs/stage5_queue.log   per fold: logs/<dataset>/<model>/fold_XX.log
set -o pipefail
cd "$(dirname "$0")/.."
LOG=logs/stage5_queue.log
exec 9>logs/.stage5_queue.lock
flock -n 9 || { echo "stage 5 queue is already running (logs/.stage5_queue.lock)"; exit 1; }
{
  echo "===== $(date -Iseconds) run_stage5.sh start"
  .venv/bin/python scripts/check_gpu.py || { echo "check_gpu failed - STOP"; exit 1; }
  .venv/bin/python scripts/run_queue.py --dataset D01 --models M01-M10 --folds 1-10 --out results
  rc=$?
  echo "===== $(date -Iseconds) run_stage5.sh end (exit $rc)"
  exit $rc
} 2>&1 | tee -ia "$LOG"   # -i: tee ignores Ctrl-C so the graceful shutdown messages still reach the log
