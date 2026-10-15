#!/usr/bin/env bash
# Stage 4 smoke test 4 (interruption): D02 x M03 fold 1, max 3 epochs, outputs in results_smoke/ only.
#  1. start training; as soon as the fold log shows "epoch 1/3", send SIGINT to the process
#  2. check: registry status RUNNING, last.pt present
#  3. restart the SAME command -> must log "RESUME ... continuing at epoch 2" and finish DONE
#  4. reference: the same fold run straight (no interruption) into results_smoke/resume_reference/ for comparison
# Usage: bash scripts/04_smoke_interrupt.sh   (log: logs/stage04_smoke_interrupt.log)
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
CMD=(scripts/train.py --dataset D02 --model M03 --folds 1 --max-epochs 3 --out results_smoke)
FOLD_LOG=logs/results_smoke/D02_facial_images_A/M03/fold_01.log
FOLD_DIR=results_smoke/D02_facial_images_A/M03/fold_01

echo "### $(date -Iseconds) step 1: start ${CMD[*]}"
start_lines=$( [ -f "$FOLD_LOG" ] && wc -l < "$FOLD_LOG" || echo 0 )
$PY "${CMD[@]}" &
PID=$!
until tail -n +"$((start_lines + 1))" "$FOLD_LOG" 2>/dev/null | grep -q "epoch 1/3"; do
  kill -0 $PID 2>/dev/null || { echo "process ended before epoch 1"; exit 1; }
  sleep 0.2
done
echo "### $(date -Iseconds) epoch 1 finished -> kill -INT $PID"
kill -INT $PID
wait $PID
echo "### exit code after SIGINT: $?"
echo "### step 2: state after the interruption"
grep "M03,1," results_smoke/run_registry.csv
ls -la "$FOLD_DIR"
echo "### $(date -Iseconds) step 3: restart the same command"
$PY "${CMD[@]}"
echo "### exit code after restart: $?"
echo "### fold log:"
cat "$FOLD_LOG"
echo "### step 4: straight reference run"
$PY scripts/train.py --dataset D02 --model M03 --folds 1 --max-epochs 3 --out results_smoke/resume_reference
echo "INTERRUPT_TEST_END"
