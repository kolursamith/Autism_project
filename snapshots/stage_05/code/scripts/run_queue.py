"""Run models sequentially on one dataset (S5 / S6). For each model: scripts/train.py in a subprocess for all folds
(resume / skip DONE), then checks.validate_model_results; the queue STOPS at the first FAIL (or failed subprocess).
After every model the summaries (summary.json, leaderboard.csv) are refreshed. Safe to re-run.
Usage: .venv/bin/python scripts/run_queue.py --dataset D01 --models M01-M10 [--folds 1-10] [--out results]
       [--max-epochs N]"""
import argparse
import datetime
import os
import signal
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.common import PROJECT_DIR  # noqa: E402
from src.training.checks import validate_model_results  # noqa: E402
from src.training.data import ALIASES  # noqa: E402
from src.training.summarize import leaderboard, summarize_model  # noqa: E402
from train import install_signal_handlers, parse_range  # noqa: E402


def say(s):
    print(f"{datetime.datetime.now().isoformat(timespec='seconds')} {s}", flush=True)


def run_child(cmd):
    """Run train.py; on Ctrl-C / SIGTERM forward the signal and WAIT until the child has saved its state."""
    p = subprocess.Popen(cmd, cwd=PROJECT_DIR)
    try:
        return p.wait()
    except KeyboardInterrupt:
        say("interrupt: waiting for train.py to save last.pt and exit")
        try:  # Ctrl-C in a terminal already reached the child (same process group); SIGTERM to the queue did not
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            p.send_signal(signal.SIGINT)
            p.wait()
        raise


def main(argv=None):
    install_signal_handlers()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--models", default="M01-M10")
    ap.add_argument("--folds", default="1-10")
    ap.add_argument("--out", default="results")
    ap.add_argument("--max-epochs", type=int, default=None)
    a = ap.parse_args(argv)
    ds = ALIASES[a.dataset]
    out_root = os.path.join(PROJECT_DIR, a.out)
    folds = parse_range(a.folds)
    models = [f"M{i:02d}" for i in parse_range(a.models, prefix="M")]
    say(f"QUEUE {ds} models {models} folds {folds} -> {a.out}/")
    for m in models:
        cmd = [sys.executable, os.path.join(PROJECT_DIR, "scripts", "train.py"), "--dataset", ds, "--model", m,
               "--folds", a.folds, "--out", a.out] + (["--max-epochs", str(a.max_epochs)] if a.max_epochs else [])
        say(f"{m}: start {' '.join(cmd[1:])}")
        try:
            rc = run_child(cmd)
        except KeyboardInterrupt:
            say(f"QUEUE INTERRUPTED at {m}: re-run the same command to resume")
            return 130
        verdict, reasons = validate_model_results(ds, m, out_root, folds)
        say(f"{m}: train.py exit {rc}; validate_model_results: {verdict}")
        for r in reasons:
            say(f"  - {r}")
        if rc != 0 or verdict != "PASS":
            say(f"QUEUE STOPPED at {m} (fix the cause, then re-run this command: DONE folds are skipped, "
                f"RUNNING folds resume)")
            return 1
        summarize_model(out_root, ds, m)
        leaderboard(out_root, ds)
    say(f"QUEUE DONE {ds}: {models}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
