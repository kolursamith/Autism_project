"""Stage 9 helpers: traceable values (every number in results/final/ carries a source reference that `resolve` reads
back), the results/ manifest (read-only check), and the shared figure style (one colour per model in all figures).

Source reference format (separator '::'):
  JSON: '<path>.json::key::key::...'      e.g. results/D01_eye_tracking/M01/summary.json::per_fold_metrics::test/threshold_0.5/roc_auc::mean
  CSV : '<path>.csv::col=val&col=val::column'   the filter must select exactly one row (string comparison)"""
import hashlib
import json
import os

import pandas as pd

from src.data.common import PROJECT_DIR

FINAL_DIR = os.path.join(PROJECT_DIR, "results", "final")
FIG_FINAL = os.path.join(PROJECT_DIR, "reports", "figures", "final")
_CSV_CACHE = {}


def resolve(ref, root=PROJECT_DIR):
    """Read the number a source reference points to (float; bool -> 0/1)."""
    path, *rest = ref.split("::")
    full = os.path.join(root, path)
    if path.endswith(".json"):
        with open(full) as fh:
            v = json.load(fh)
        for k in rest:
            v = v[k]
    elif path.endswith(".csv"):
        if full not in _CSV_CACHE:
            _CSV_CACHE[full] = pd.read_csv(full, dtype=str, keep_default_na=False)
        df = _CSV_CACHE[full]
        flt, col = rest
        sel = pd.Series(True, index=df.index)
        for cond in filter(None, flt.split("&")):
            c, val = cond.split("=", 1)
            sel &= df[c] == val
        if sel.sum() != 1:
            raise KeyError(f"{ref}: filter selects {int(sel.sum())} rows")
        v = df.loc[sel, col].iloc[0]
        v = {"True": 1.0, "False": 0.0}.get(v, v)
    else:
        raise ValueError(ref)
    return float(v)


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def results_manifest_lines(root=PROJECT_DIR):
    """sha256 + size + path of every file under results/ except results/final/ (Stage 9 output), results/FINAL_MANIFEST.json
    (Stage 10 output) and *.pt (large model weights: size + mtime_ns instead of a hash)."""
    out = []
    base = os.path.join(root, "results")
    for dp, dns, fns in os.walk(base):
        dns[:] = sorted(d for d in dns if not (dp == base and d == "final"))
        for fn in sorted(fns):
            if dp == base and fn == "FINAL_MANIFEST.json":
                continue
            p = os.path.join(dp, fn)
            rel = os.path.relpath(p, root)
            if fn.endswith(".pt"):
                st = os.stat(p)
                out.append(f"pt:{st.st_mtime_ns}  {st.st_size}  {rel}")
            else:
                out.append(f"{sha256_file(p)}  {os.path.getsize(p)}  {rel}")
    return sorted(out, key=lambda l: l.split("  ", 2)[2])


# ---------------------------------------------------------------------------------------------- figure style
MODEL_COLORS = {f"M{i:02d}": c for i, c in enumerate(["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
                                                         "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"], start=1)}
TAB_COLORS = {"dummy_majority": "#9e9e9e", "logreg": "#1b9e77", "random_forest": "#d95f02", "hist_gb": "#7570b3",
              "mlp": "#e7298a"}
BASELINE_COLOR = "#000000"


def apply_style(plt):
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.bbox": "tight", "svg.fonttype": "none"})
