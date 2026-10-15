"""Local stage snapshot (replaces git). Creates snapshots/stage_NN/ with
  code/                 copy of src/, scripts/, configs/, tests/ and CLAUDE.md
  CHECKSUMS.sha256      "<sha256>  <size_bytes>  <path>" for every file in src/ scripts/ configs/ tests/ splits/
                        reports/ and cache/ (cache is hashed, never copied), sorted by path
  SNAPSHOT_INFO.json    date-time, stage, decision, python/torch versions, GPU name,
                        code_hash = sha256 of the CHECKSUMS file
Usage: .venv/bin/python scripts/make_snapshot.py --stage 2 --decision PASS"""
import argparse
import datetime
import hashlib
import json
import os
import shutil
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIRS = ["src", "scripts", "configs", "tests"]
HASH_DIRS = CODE_DIRS + ["splits", "reports", "cache"]
SKIP_DIRS = {"__pycache__", ".pytest_cache"}


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def list_files(dirs, root=PROJECT_DIR):
    out = []
    for d in dirs:
        for dp, dns, fns in os.walk(os.path.join(root, d)):
            dns[:] = sorted(x for x in dns if x not in SKIP_DIRS)
            for fn in fns:
                if fn.endswith(".pyc"):
                    continue
                out.append(os.path.relpath(os.path.join(dp, fn), root))
    return sorted(out)


def checksum_lines(dirs, root=PROJECT_DIR):
    return [f"{sha256_file(os.path.join(root, p))}  {os.path.getsize(os.path.join(root, p))}  {p}"
            for p in list_files(dirs, root)]


def code_hash(dirs=CODE_DIRS, root=PROJECT_DIR):
    """sha256 over the sorted checksum lines of the given dirs (used before a stage starts / in manifests)."""
    return hashlib.sha256(("\n".join(checksum_lines(dirs, root)) + "\n").encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--decision", choices=["PASS", "FAIL"], required=True)
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    out = os.path.join(PROJECT_DIR, "snapshots", f"stage_{a.stage:02d}")
    if os.path.isdir(out):
        shutil.rmtree(out)
    code = os.path.join(out, "code")
    for d in CODE_DIRS:
        shutil.copytree(os.path.join(PROJECT_DIR, d), os.path.join(code, d),
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(os.path.join(PROJECT_DIR, "CLAUDE.md"), os.path.join(code, "CLAUDE.md"))

    lines = checksum_lines(HASH_DIRS)
    ck = os.path.join(out, "CHECKSUMS.sha256")
    with open(ck, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    import torch
    info = {
        "datetime": datetime.datetime.now().isoformat(timespec="seconds"),
        "stage": a.stage,
        "decision": a.decision,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "UNKNOWN (CUDA unavailable)",
        "n_files_checksummed": len(lines),
        "hashed_dirs": HASH_DIRS,
        "copied_dirs": CODE_DIRS + ["CLAUDE.md"],
        "code_hash": sha256_file(ck),
        "note": a.note,
    }
    with open(os.path.join(out, "SNAPSHOT_INFO.json"), "w") as fh:
        json.dump(info, fh, indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
