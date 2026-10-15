"""Train one model on one dataset for a range of folds (resumes RUNNING folds, skips DONE folds unless --force).
Usage: .venv/bin/python scripts/train.py --dataset D01 --model M01 --folds 1-10 [--max-epochs N] [--out results_smoke]
       [--force]
Per-fold log: logs/<dataset>/<model>/fold_XX.log (logs/<out>/<dataset>/... when --out is not results)."""
import argparse
import os
import signal
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # use the pinned, sha256-verified cached weights only (no network)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.common import PROJECT_DIR  # noqa: E402
from src.training.data import ALIASES  # noqa: E402
from src.training.engine import merge_config, train_one_fold  # noqa: E402


def parse_range(s, prefix=""):
    """'1-10' / '1,3,5' / 'M01-M10' -> list of ints."""
    out = []
    for part in s.split(","):
        a, _, b = part.replace(prefix, "").partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def _graceful(signum, frame):
    raise KeyboardInterrupt(f"signal {signal.Signals(signum).name}")


def install_signal_handlers():
    """SIGINT / SIGTERM -> KeyboardInterrupt -> the engine keeps status RUNNING and last.pt for resume. Explicit,
    because a job started with '&' from a non-interactive shell (nohup, scripts) inherits SIGINT as IGNORED."""
    signal.signal(signal.SIGINT, _graceful)
    signal.signal(signal.SIGTERM, _graceful)


def main(argv=None):
    install_signal_handlers()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="D01 | D02 (or the full dataset id)")
    ap.add_argument("--model", required=True, help="M01..M10")
    ap.add_argument("--folds", default="1-10")
    ap.add_argument("--max-epochs", type=int, default=None, help="override max_epochs (smoke tests only)")
    ap.add_argument("--out", default="results", help="output root inside the project (default results)")
    ap.add_argument("--force", action="store_true", help="re-run folds that are already DONE")
    a = ap.parse_args(argv)
    ds = ALIASES[a.dataset]
    out_root = os.path.join(PROJECT_DIR, a.out)
    try:
        for k in parse_range(a.folds):
            ov = {"training.max_epochs": a.max_epochs} if a.max_epochs else {}
            train_one_fold(ds, a.model, k, cfg=merge_config(ds, a.model, k, ov), out_root=out_root, force=a.force)
    except KeyboardInterrupt as e:
        print(f"train.py interrupted ({e}): re-run the same command to resume", flush=True)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
