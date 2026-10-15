#!/usr/bin/env bash
# Stage 5 launcher (nohup, no tmux): starts scripts/run_stage5.sh fully detached (own session via setsid), so it
# survives closing the terminal / Claude Code. Safe to re-run after a stop, crash or reboot: DONE folds are skipped and
# a RUNNING fold resumes from its last.pt (scripts/run_queue.py + src/training/engine.py).
#  - refuses to start if logs/stage5.pid holds a live run_stage5.sh process (or any D01 queue is running)
#  - run_stage5.sh itself appends everything to logs/stage5_queue.log (tee -ia), so nohup's own stdout goes to
#    logs/stage5_nohup.out (a second copy; also catches messages printed outside the tee block)
#  - writes the PID (= process-group / session id) to logs/stage5.pid, start time + command to logs/stage5_launch.json
#    and appends the same record to logs/stage5_launch_history.jsonl
# Usage: bash scripts/stage5_launch.sh
set -u
cd "$(dirname "$0")/.."
PIDFILE=logs/stage5.pid
CMD="setsid nohup bash scripts/run_stage5.sh >> logs/stage5_nohup.out 2>&1 < /dev/null &"

if [ -f "$PIDFILE" ]; then
  OLD=$(cat "$PIDFILE")
  if kill -0 "$OLD" 2>/dev/null && tr '\0' ' ' < /proc/"$OLD"/cmdline 2>/dev/null | grep -q "run_stage5.sh"; then
    echo "REFUSED: stage 5 queue is already running (PID $OLD from $PIDFILE). Check: python scripts/stage5_status.py"
    exit 1
  fi
  echo "note: stale $PIDFILE (PID $OLD is not a running run_stage5.sh) - starting a new queue"
fi
if pgrep -f "run_queue.py --dataset D01" > /dev/null; then
  echo "REFUSED: a D01 run_queue.py process is running without a matching $PIDFILE:"
  pgrep -af "run_queue.py --dataset D01"
  exit 1
fi

START=$(date -Iseconds)
setsid nohup bash scripts/run_stage5.sh >> logs/stage5_nohup.out 2>&1 < /dev/null &
PID=$!
sleep 2
if ! kill -0 "$PID" 2>/dev/null; then
  echo "FAILED: run_stage5.sh (PID $PID) exited immediately - see logs/stage5_nohup.out and logs/stage5_queue.log"
  tail -5 logs/stage5_nohup.out
  exit 1
fi
PGID=$(ps -o pgid= -p "$PID" | tr -d ' ')
SID=$(ps -o sid= -p "$PID" | tr -d ' ')
echo "$PID" > "$PIDFILE"
REC=$(printf '{"start_time": "%s", "pid": %s, "pgid": %s, "sid": %s, "command": "%s", "cwd": "%s", "host": "%s", "queue_log": "logs/stage5_queue.log", "nohup_out": "logs/stage5_nohup.out"}' \
  "$START" "$PID" "$PGID" "$SID" "$CMD" "$(pwd)" "$(hostname)")
echo "$REC" > logs/stage5_launch.json
echo "$REC" >> logs/stage5_launch_history.jsonl
echo "Stage 5 queue started at $START: PID $PID (process group $PGID, session $SID)"
echo "  progress : .venv/bin/python scripts/stage5_status.py"
echo "  live log : tail -f logs/stage5_queue.log"
echo "  stop     : bash scripts/stage5_stop.sh"
