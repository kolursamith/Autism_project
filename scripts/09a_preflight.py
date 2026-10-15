"""Stage 9 pre-flight (A): STAGE_05..08 reports PASS, snapshot checksums of stage_05..08 (data/results part), split
hashes, and the manifest of all results/ files (except results/final/) that must be identical at the end.
Writes results/final/preflight_results_manifest.sha256 + results/final/preflight.json.
Usage: .venv/bin/python scripts/09a_preflight.py | tee logs/stage09_preflight.txt"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_snapshot  # noqa: E402

from src.analysis.final import FINAL_DIR, results_manifest_lines  # noqa: E402
from src.data.common import PROJECT_DIR, write_json  # noqa: E402
from src.data.splits import verify_manifest  # noqa: E402

CODE_PREFIXES = tuple(d + "/" for d in make_snapshot.CODE_DIRS + ["reports", "tabular"])
STAGES = ["05", "06", "07", "08"]


def main():
    os.makedirs(FINAL_DIR, exist_ok=True)
    out = {"reports": {}, "snapshots": {}}
    for s in STAGES:
        last = open(os.path.join(PROJECT_DIR, "reports", f"STAGE_{s}_REPORT.md")).read().rstrip().splitlines()[-1]
        out["reports"][s] = last
        print(f"A1 reports/STAGE_{s}_REPORT.md last line: {last}")
    for s in STAGES:
        snap = os.path.join(PROJECT_DIR, "snapshots", f"stage_{s}")
        info = json.load(open(os.path.join(snap, "SNAPSHOT_INFO.json")))
        rec = set(open(os.path.join(snap, "CHECKSUMS.sha256")).read().splitlines())
        cur = set(make_snapshot.snapshot_lines(info.get("extra_dirs", []), info.get("extra_hash_globs", [])))
        diff = {l.split("  ", 2)[2] for l in rec ^ cur}
        data_diff = sorted(p for p in diff if not p.startswith(CODE_PREFIXES))
        code_diff = sorted(p for p in diff if p.startswith(CODE_PREFIXES))
        ok_hash = info["code_hash"] == make_snapshot.sha256_file(os.path.join(snap, "CHECKSUMS.sha256"))
        out["snapshots"][s] = {"n_recorded": len(rec), "code_hash": info["code_hash"], "code_hash_ok": ok_hash,
                               "data_results_mismatch": data_diff, "code_report_files_changed_since": code_diff}
        print(f"A2 snapshot stage_{s} ({info['datetime']}): {len(rec)} lines; CHECKSUMS matches code_hash: {ok_hash}; "
              f"splits/cache/results files differing: {len(data_diff)}" + (f" {data_diff[:10]}" if data_diff else "")
              + f"; code/report files added or changed since: {len(code_diff)}" + (f" {code_diff}" if code_diff else ""))
    verify_manifest()
    print("A3 split hashes vs SPLITS_MANIFEST.json: OK")
    lines = results_manifest_lines()
    with open(os.path.join(FINAL_DIR, "preflight_results_manifest.sha256"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    n_pt = sum(l.startswith("pt:") for l in lines)
    print(f"A4 recorded {len(lines)} files under results/ (except results/final/): {len(lines) - n_pt} sha256, {n_pt} "
          "*.pt by size + mtime -> results/final/preflight_results_manifest.sha256")
    ok = all(v.endswith("PASS") for v in out["reports"].values()) and \
        all(v["code_hash_ok"] and not v["data_results_mismatch"] for v in out["snapshots"].values())
    out.update({"n_results_files": len(lines), "n_pt": n_pt, "preflight_ok": ok})
    write_json(out, os.path.join(FINAL_DIR, "preflight.json"))
    print(f"A PRE-FLIGHT: {'OK' if ok else 'PROBLEM'}")


if __name__ == "__main__":
    main()
