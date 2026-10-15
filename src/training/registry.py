"""<out_root>/run_registry.csv: one row per (dataset, model, fold). Every read-modify-write holds an exclusive fcntl lock
on run_registry.csv.lock, so concurrent writers cannot lose rows. Status: PENDING / RUNNING / DONE / FAILED / BLOCKED.
A DONE run is never re-run unless force=True."""
import contextlib
import fcntl
import os

import pandas as pd

COLUMNS = ["dataset", "model", "fold", "status", "start", "end", "epochs", "best_epoch", "val_auc", "test_auc",
           "peak_gpu_gb", "seconds", "error"]
STATUSES = {"PENDING", "RUNNING", "DONE", "FAILED", "BLOCKED"}


def registry_path(out_root):
    return os.path.join(out_root, "run_registry.csv")


@contextlib.contextmanager
def _locked(out_root):
    os.makedirs(out_root, exist_ok=True)
    with open(registry_path(out_root) + ".lock", "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def _read(out_root):
    p = registry_path(out_root)
    if not os.path.isfile(p):
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(p, dtype={"error": str, "start": str, "end": str}, keep_default_na=False)


def read_registry(out_root):
    with _locked(out_root):
        return _read(out_root)


def get_status(out_root, dataset, model, fold):
    df = read_registry(out_root)
    r = df[(df.dataset == dataset) & (df.model == model) & (df.fold.astype(int) == int(fold))]
    return None if r.empty else r.iloc[0]["status"]


def update(out_root, dataset, model, fold, **fields):
    """Insert or update the (dataset, model, fold) row; unknown columns are rejected."""
    bad = set(fields) - set(COLUMNS)
    if bad:
        raise KeyError(bad)
    if "status" in fields and fields["status"] not in STATUSES:
        raise ValueError(fields["status"])
    with _locked(out_root):
        df = _read(out_root)
        mask = (df.dataset == dataset) & (df.model == model) & (df.fold.astype(int) == int(fold))
        if not mask.any():
            row = {c: "" for c in COLUMNS}
            row.update(dataset=dataset, model=model, fold=int(fold), status="PENDING")
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            mask = (df.dataset == dataset) & (df.model == model) & (df.fold.astype(int) == int(fold))
        for k, v in fields.items():
            df[k] = df[k].astype(object)
            df.loc[mask, k] = "" if v is None else v
        df["fold"] = df["fold"].astype(int)
        df = df.sort_values(["dataset", "model", "fold"]).reset_index(drop=True)
        tmp = registry_path(out_root) + ".tmp"
        df[COLUMNS].to_csv(tmp, index=False, lineterminator="\n")
        os.replace(tmp, registry_path(out_root))


def should_run(out_root, dataset, model, fold, force=False):
    """(run?, reason). DONE -> skip unless force; RUNNING -> resume; anything else -> (re)start."""
    st = get_status(out_root, dataset, model, fold)
    if st == "DONE" and not force:
        return False, "DONE (use --force to re-run)"
    return True, {None: "new", "DONE": "forced re-run"}.get(st, f"status {st}")
