"""Stage 10 A - scan every report for wording that could claim clinical accuracy, diagnosis or autism detection, and
classify each match. Usage: .venv/bin/python scripts/10c_claims_scan.py | tee logs/stage10_claims_scan.txt"""
import glob
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAT = re.compile(r"diagnos|detect|clinical|clinically|identif(y|ies) (autism|asd)|predicts? autism|distinguish(es)? (autism|asd|children)", re.I)
RULES = [("dataset folder name (proper noun of the D02 source, cannot be renamed)", r"Disorder Detection using Facial Traits"),
         ("data column name (D06 label column)", r"Diagnosed_ASD"),
         ("pytest test name (software check, not about autism)", r"::test_\w*detect"),
         ("explicit negation / limitation (screening-support framing)",
          r"[Nn]o result|[Nn]ot (a |an )?diagnos|Cannot be claimed|cannot be detected|unknown diagnosis|not clinical accuracy|"
          r"is not a diagnosis|diagnostic accuracy;|neither shown|no detectable|[Nn]one of these numbers is a diagnostic"),
         ("statement of what may NOT be generalised (limitation)", r"may not generalise to clinical")]
n, rows = 0, []
for f in sorted(glob.glob(os.path.join(ROOT, "reports", "*.md"))):
    for i, line in enumerate(open(f).read().splitlines(), 1):
        if not PAT.search(line):
            continue
        n += 1
        cat = next((c for c, r in RULES if re.search(r, line)), "CLAIM - rewrite needed")
        rows.append((os.path.relpath(f, ROOT), i, cat, line.strip()[:150]))
cats = {}
for r in rows:
    cats[r[2]] = cats.get(r[2], 0) + 1
print(f"reports/*.md lines matching {PAT.pattern!r}: {n}")
for c, k in sorted(cats.items(), key=lambda x: -x[1]):
    print(f"  {k:3d}  {c}")
print("claims needing a rewrite:", sum(r[2].startswith("CLAIM") for r in rows))
for r in rows:
    print(f"{r[0]}:{r[1]} [{r[2].split(' (')[0]}] {r[3]}")
