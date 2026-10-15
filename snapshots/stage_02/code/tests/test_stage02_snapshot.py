"""Local snapshot (replaces git): the latest snapshots/stage_NN/CHECKSUMS.sha256 matches the current files."""
import glob
import json
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_DIR, "scripts"))
import make_snapshot  # noqa: E402


def _latest():
    snaps = sorted(glob.glob(os.path.join(PROJECT_DIR, "snapshots", "stage_*")))
    assert snaps, "no snapshot"
    return snaps[-1]


def test_latest_snapshot_checksums_match_current_files():
    snap = _latest()
    with open(os.path.join(snap, "CHECKSUMS.sha256")) as fh:
        recorded = fh.read().splitlines()
    current = make_snapshot.checksum_lines(make_snapshot.HASH_DIRS)
    rec, cur = set(recorded), set(current)
    assert rec == cur, {"changed_or_missing": sorted(rec - cur)[:10], "new_or_changed": sorted(cur - rec)[:10]}


def test_latest_snapshot_info_and_code_copy():
    snap = _latest()
    with open(os.path.join(snap, "SNAPSHOT_INFO.json")) as fh:
        info = json.load(fh)
    assert info["code_hash"] == make_snapshot.sha256_file(os.path.join(snap, "CHECKSUMS.sha256"))
    assert info["decision"] in {"PASS", "FAIL"}
    for d in make_snapshot.CODE_DIRS:
        assert os.path.isdir(os.path.join(snap, "code", d))
    assert os.path.isfile(os.path.join(snap, "code", "CLAUDE.md"))
