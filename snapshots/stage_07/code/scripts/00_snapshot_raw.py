"""Record a read-only listing (relative path, type, size, mtime_ns) of every entry under DATASETS_DIR.
Usage: python scripts/00_snapshot_raw.py <out.tsv>"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.paths import DATASETS_DIR, dataset_dir  # noqa: E402  (configs/paths.yaml)

def snapshot(root=DATASETS_DIR):
    rows = []
    for dp, dns, fns in os.walk(root):
        dns.sort()
        for n in sorted(dns) + sorted(fns):
            p = os.path.join(dp, n)
            st = os.lstat(p)
            kind = "d" if os.path.isdir(p) else "f"
            rows.append((os.path.relpath(p, root), kind, 0 if kind == "d" else st.st_size, st.st_mtime_ns))
    return sorted(rows)

if __name__ == "__main__":
    rows = snapshot()
    with open(sys.argv[1], "w") as f:
        f.write("path\tkind\tsize\tmtime_ns\n")
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")
    print(f"wrote {len(rows)} entries to {sys.argv[1]}")
