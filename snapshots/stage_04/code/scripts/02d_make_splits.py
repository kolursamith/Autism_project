"""Stage 2 - 10-fold outer + inner validation splits for D01, D02, D04, D05, D06 (F1-F4) and SPLITS_MANIFEST.json.
Deterministic: re-running reproduces byte-identical split files. Log: logs/stage02_splits.log
Usage: .venv/bin/python scripts/02d_make_splits.py [--out DIR]   (default: splits/)"""
import argparse
import datetime
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import pandas as pd  # noqa: E402

import make_snapshot  # noqa: E402
from src.data import splits as sp  # noqa: E402
from src.data.common import SPLITS_DIR, get_logger, md_table, set_seed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=SPLITS_DIR)
    a = ap.parse_args()
    set_seed(42)
    log = get_logger("stage02_splits" if os.path.abspath(a.out) == os.path.abspath(SPLITS_DIR) else "stage02_splits_rerun")
    for ds in sp.DATASETS:
        out = sp.make(ds, a.out)
        summ = pd.read_csv(os.path.join(out, "split_summary.csv"))
        log.info(f"===== {ds} -> {out}\n" + md_table(summ))
    man = sp.write_manifest(make_snapshot.code_hash(), datetime.datetime.now().isoformat(timespec="seconds"), a.out)
    log.info(f"SPLITS_MANIFEST.json written; code_hash_at_split_time = {man['code_hash_at_split_time']}")


if __name__ == "__main__":
    main()
