#!/usr/bin/env bash
# Stop the Stage 5 queue cleanly: SIGTERM to the whole process group of the PID in logs/stage5.pid (run_stage5.sh,
# run_queue.py, train.py, data-loader workers). train.py turns SIGTERM into its graceful interrupt: the current fold
# keeps status RUNNING and its last.pt (last completed epoch) for resume. Waits up to 120 s, then reports whether the
# interrupted fold's last.pt was saved. Restart with: bash scripts/stage5_launch.sh
# Usage: bash scripts/stage5_stop.sh        (STAGE5_PIDFILE / STAGE5_OUT override the defaults, used for testing)
set -u
cd "$(dirname "$0")/.."
PIDFILE=${STAGE5_PIDFILE:-logs/stage5.pid}
OUT=${STAGE5_OUT:-results}
DS=D01_eye_tracking
PY=.venv/bin/python

if [ ! -f "$PIDFILE" ]; then echo "no $PIDFILE - nothing to stop"; exit 0; fi
PID=$(cat "$PIDFILE")
if ! kill -0 "$PID" 2>/dev/null; then echo "PID $PID is not running - nothing to stop"; rm -f "$PIDFILE"; exit 0; fi
PGID=$(ps -o pgid= -p "$PID" | tr -d ' ')
running_rows() {  # dataset,model,fold of RUNNING rows in the registry
  [ -f "$OUT/run_registry.csv" ] && awk -F, -v ds="$DS" '$1==ds && $4=="RUNNING" {print $2","$3}' "$OUT/run_registry.csv"
}
BEFORE=$(running_rows)
echo "$(date -Iseconds) sending SIGTERM to process group $PGID (leader PID $PID); RUNNING before: ${BEFORE:-none}"
kill -TERM -- -"$PGID"
for i in $(seq 1 120); do
  if ! pgrep -g "$PGID" > /dev/null; then echo "all processes of group $PGID exited after ${i} s"; break; fi
  sleep 1
done
if pgrep -g "$PGID" > /dev/null; then
  echo "WARNING: processes still alive after 120 s:"; pgrep -ag "$PGID"
  echo "They may still be saving. Check again, or force with: kill -KILL -- -$PGID (last.pt is written atomically)"
  exit 1
fi
rm -f "$PIDFILE"
for mf in $BEFORE; do
  M=${mf%,*}; F=${mf#*,}
  D="$OUT/$DS/$M/fold_$(printf %02d "$F")"
  ST=$(awk -F, -v ds="$DS" -v m="$M" -v f="$F" '$1==ds && $2==m && $3==f {print $4}' "$OUT/run_registry.csv")
  echo "fold $M fold $F: registry status now $ST"
  if [ -f "$D/last.pt" ]; then
    $PY - "$D/last.pt" <<'EOF'
import os, sys, time, torch
p = sys.argv[1]
ck = torch.load(p, map_location="cpu", weights_only=False)
st = ck["state"]
print(f"  last.pt SAVED: {p} ({os.path.getsize(p) / 1e6:.1f} MB, written {time.ctime(os.path.getmtime(p))}), "
      f"epoch {st['epoch']} completed, best val AUC {st['best_auc']:.4f} @ epoch {st['best_epoch']} -> resumes at epoch "
      f"{st['epoch'] + 1}")
EOF
  elif [ "$ST" = "DONE" ]; then
    echo "  fold finished (DONE) before the stop - nothing to resume"
  else
    echo "  NO last.pt (stopped during its first epoch) - this fold restarts from epoch 1 on relaunch"
  fi
done
echo "stopped. Restart / resume with: bash scripts/stage5_launch.sh"
