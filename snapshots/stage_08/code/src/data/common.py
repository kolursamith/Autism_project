"""Stage 2 shared helpers: paths (from configs/paths.yaml), configs, seeding, hashing, deterministic writers, logging."""
import hashlib
import json
import logging
import os
import random
import sys

import numpy as np
import yaml

from src.utils.paths import DATASETS, DATASETS_DIR, PROJECT_DIR, dataset_dir  # noqa: F401

SEED = 42
CACHE_DIR = os.path.join(PROJECT_DIR, "cache")
SPLITS_DIR = os.path.join(PROJECT_DIR, "splits")
FIG_DIR = os.path.join(PROJECT_DIR, "reports", "figures")
LOG_DIR = os.path.join(PROJECT_DIR, "logs")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_cfg(dataset_short):
    """configs/data/<D01|D02|D04|D05|D06>.yaml"""
    with open(os.path.join(PROJECT_DIR, "configs", "data", f"{dataset_short}.yaml")) as fh:
        return yaml.safe_load(fh)


def cache_dir(dataset_id):
    d = os.path.join(CACHE_DIR, dataset_id)
    os.makedirs(d, exist_ok=True)
    return d


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(df, path):
    """Deterministic CSV: no index, '\\n' line ends, floats in shortest exact round-trip form (no precision loss)."""
    df.to_csv(path, index=False, lineterminator="\n")


def write_json(obj, path):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True, default=_jdefault)
        fh.write("\n")


def write_yaml(obj, path):
    with open(path, "w") as fh:
        yaml.safe_dump(obj, fh, sort_keys=False, allow_unicode=True, width=120)


def _jdefault(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def get_logger(name):
    """Logger writing to stdout and logs/<name>.log (overwritten on every run)."""
    os.makedirs(LOG_DIR, exist_ok=True)
    log = logging.getLogger(name)
    log.handlers.clear()
    log.setLevel(logging.INFO)
    log.propagate = False
    fmt = logging.Formatter("%(message)s")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(LOG_DIR, f"{name}.log"), mode="w")):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


def md_table(df):
    """Markdown table from a DataFrame (for logs and reports)."""
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)
