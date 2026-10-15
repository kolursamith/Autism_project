"""Stage 8 pre-flight (A): earlier reports PASS, snapshot checksums of stage_05/06/07, split hashes, and the sha256
manifest of all results/D0*/ files except *.pt (Stage 8 must leave them identical; tests/test_stage08.py re-checks).
Writes results/stage8/preflight_results_manifest.sha256 + results/stage8/preflight.json.
Usage: .venv/bin/python scripts/08a_preflight.py | tee logs/stage08_preflight.txt"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_snapshot  # noqa: E402

from src.data.common import PROJECT_DIR, write_json  # noqa: E402
from src.data.splits import verify_manifest  # noqa: E402

OUT = os.path.join(PROJECT_DIR, "results", "stage8")
MANIFEST = os.path.join(OUT, "preflight_results_manifest.sha256")
CODE_PREFIXES = tuple(d + "/" for d in make_snapshot.CODE_DIRS + ["reports", "tabular"])


def results_manifest_lines(root=PROJECT_DIR):
    """'<sha256>  <size>  <path>' for every file under results/D0*/ except *.pt (and the registry lock)."""
    lines = []
    for d in sorted(glob.glob(os.path.join(root, "results", "D0*"))):
        for dp, dns, fns in os.walk(d):
            dns.sort()
            for fn in sorted(fns):
                if fn.endswith(".pt"):
                    continue
                p = os.path.join(dp, fn)
                lines.append(f"{make_snapshot.sha256_file(p)}  {os.path.getsize(p)}  {os.path.relpath(p, root)}")
    return sorted(lines, key=lambda l: l.split("  ", 2)[2])


def main():
    os.makedirs(OUT, exist_ok=True)
    out = {"reports": {}, "snapshots": {}}
    for s in ["05", "06", "07"]:
        last = open(os.path.join(PROJECT_DIR, "reports", f"STAGE_{s}_REPORT.md")).read().rstrip().splitlines()[-1]
        out["reports"][s] = last
        print(f"A1 reports/STAGE_{s}_REPORT.md last line: {last}")
    for s in ["05", "06", "07"]:
        snap = os.path.join(PROJECT_DIR, "snapshots", f"stage_{s}")
        info = json.load(open(os.path.join(snap, "SNAPSHOT_INFO.json")))
        rec = set(open(os.path.join(snap, "CHECKSUMS.sha256")).read().splitlines())
        cur = set(make_snapshot.snapshot_lines(info.get("extra_dirs", []), info.get("extra_hash_globs", [])))
        path = lambda l: l.split("  ", 2)[2]  # noqa: E731
        diff = {path(l) for l in rec ^ cur}
        data_diff = sorted(p for p in diff if not p.startswith(CODE_PREFIXES))
        code_diff = sorted(p for p in diff if p.startswith(CODE_PREFIXES))
        code_hash_ok = info["code_hash"] == make_snapshot.sha256_file(os.path.join(snap, "CHECKSUMS.sha256"))
        out["snapshots"][s] = {"n_recorded": len(rec), "data_results_mismatch": data_diff,
                               "code_report_files_changed_since": code_diff, "code_hash_ok": code_hash_ok}
        print(f"A2 snapshot stage_{s} ({info['datetime']}): {len(rec)} recorded lines; CHECKSUMS file matches its "
              f"code_hash: {code_hash_ok}; splits/cache/results files differing: {len(data_diff)}"
              + (f" {data_diff[:10]}" if data_diff else "")
              + f"; code/report files added or changed since (expected for later stages): {len(code_diff)}"
              + (f" {code_diff}" if code_diff else ""))
    verify_manifest()
    print("A3 split hashes vs SPLITS_MANIFEST.json: OK")
    lines = results_manifest_lines()
    with open(MANIFEST, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    n_pm = sum(path_ in l for l in lines for path_ in ["/predictions.csv", "/metrics.json"])
    print(f"A4 recorded sha256 of {len(lines)} files under results/D0*/ (no *.pt), incl. {n_pm} predictions.csv / "
          f"metrics.json -> {os.path.relpath(MANIFEST, PROJECT_DIR)}")
    out["n_results_files"] = len(lines)
    out["n_predictions_metrics"] = n_pm
    ok = all(v.endswith("PASS") for v in out["reports"].values()) and \
        all(not v["data_results_mismatch"] and v["code_hash_ok"] for v in out["snapshots"].values())
    out["preflight_ok"] = ok
    write_json(out, os.path.join(OUT, "preflight.json"))
    print(f"A PRE-FLIGHT: {'OK' if ok else 'PROBLEM'}")


if __name__ == "__main__":
    main()
