"""Stage 10 C/D - finalize: `readme` (README.md, findings / limitations copied verbatim from the final report),
`report` (reports/STAGE_10_REPORT.md from logs/stage10_*), `freeze` (results/FINAL_MANIFEST.json -> chmod a-w on results/,
splits/, reports/figures/final/ -> submission zip + submission/SUBMISSION_INFO.json).
Usage: .venv/bin/python scripts/10b_finalize.py readme|report|freeze"""
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import zipfile
from datetime import datetime

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_DIRS = ["results", "splits", "reports", "configs", "src", "tabular", "scripts"]
READ_ONLY = ["results", "splits", "reports/figures/final"]
ZIP = os.path.join(ROOT, "submission", "ASD_Hybrid_10fold_submission.zip")
SKIP_DIRS = {"__pycache__", ".pytest_cache"}
OLD_SENTENCE = "kept an AUC above 0.5 after matching"


def P(*a):
    return os.path.join(ROOT, *a)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def walk(d, skip_pt=True):
    out = []
    for dp, dns, fns in os.walk(P(d)):
        dns[:] = sorted(x for x in dns if x not in SKIP_DIRS)
        for f in sorted(fns):
            if f.endswith(".pyc") or (skip_pt and f.endswith(".pt")):
                continue
            out.append(os.path.relpath(os.path.join(dp, f), ROOT))
    return out


def zip_members():
    files = []
    for d in ["src", "tabular", "scripts", "configs", "tests", "splits", "results/final", "reports/figures/final"]:
        files += walk(d)
    files += sorted(os.path.relpath(f, ROOT) for f in map(lambda x: P("reports", x), os.listdir(P("reports")))
                    if f.endswith((".md", ".html")) and os.path.isfile(f))
    files += ["reports/STAGE_10_REPORT.md", "CLAUDE.md", "README.md", "requirements.txt", "results/FINAL_MANIFEST.json"]
    return sorted(dict.fromkeys(files))


def report_text():
    return open(P("reports", "FINAL_RESULTS_REPORT.md")).read()


# ---------------------------------------------------------------------------------------------- README
def readme():
    t = report_text()
    abstract = t.split("## 1. Abstract", 1)[1].split("## 2.", 1)[0].strip()
    lines = t.splitlines()
    pick = ["- Can be claimed: per-child tracking quality alone", "- By the pre-specified rule,",
            "- Can be claimed: simple non-facial image statistics", "- Near-duplicate leakage at pHash", "- Models differ (Friedman"]
    findings = [next(l for l in lines if l.startswith(p)) for p in pick]
    lim = t.split("## 9. Limitations", 1)[1].split("## 10.", 1)[0].strip()
    L = ["# ASD hybrid 10-fold - Reliability-Weighted Multimodal AI for Early Autism Risk Screening", "",
         "BITE308L Artificial Intelligence (VIT). Team: Samith Kolur (24BIT0117), Iniya K, Chandru Vel Kumar.", "",
         "> Screening-support research only. Nothing in this repository is a diagnosis or a clinical accuracy estimate.", "",
         "## Summary", "", abstract, "",
         "## Main findings (copied verbatim from `reports/FINAL_RESULTS_REPORT.md`, section 8)", "", *findings, "",
         "## Limitations (copied from the report, section 9)", "", lim, "",
         "## Where things are", "",
         "| path | content |", "|---|---|",
         "| `reports/FINAL_RESULTS_REPORT.md` / `.html` | final results report (the HTML is self-contained) |",
         "| `reports/STAGE_NN_REPORT.md` | one report per stage (S1-S10), each ending in `STAGE N DECISION` |",
         "| `results/final/` | final tables (`master_table`, `shortcut_table`, `stats_table`, `fold_auc_matrix`, ...; .csv + .md; every number has a source reference) |",
         "| `reports/figures/final/` | final figures F01-F12 (300 dpi PNG + SVG) |",
         "| `results/<dataset>/<model>/fold_XX/` | per-run outputs: predictions.csv, metrics.json, history.csv, run_info.json, config.yaml (best_model.pt not in the submission) |",
         "| `results/stage8/` | shortcut-control and statistics outputs |",
         "| `results/FINAL_MANIFEST.json` | sha256 of every file in results/, splits/, reports/, configs/, src/, tabular/, scripts/ |",
         "| `splits/` | frozen 10-fold split files + SPLITS_MANIFEST.json |",
         "| `src/`, `scripts/`, `tabular/`, `configs/`, `tests/` | code, configuration, tests |",
         "| `snapshots/stage_NN/` | local snapshots (no git): code copy + CHECKSUMS.sha256 |", "",
         "## How to reproduce", "",
         "Raw datasets are not included; set their location in `configs/paths.yaml` (the only file with absolute paths).", "",
         "```bash",
         "python3 -m venv --system-site-packages .venv && .venv/bin/python -m pip install -r requirements.txt",
         "source .venv/bin/activate",
         "python scripts/check_gpu.py                      # must print GPU CHECK: OK",
         "python -m pytest -v                              # full test suite",
         "# S1 audit:   python scripts/01_audit_datasets.py; python scripts/01_build_registry.py",
         "# S2 data:    python scripts/02a_prepare_d01.py; python scripts/02b_prepare_d02.py; python scripts/02c_prepare_tabular.py",
         "#             python scripts/02d_make_splits.py   (deterministic; splits are frozen in splits/)",
         "# S3 models:  python scripts/03_profile_models.py; python scripts/03b_build_feature_cache.py",
         "# S4 engine:  python scripts/train.py --dataset D01 --model M01 --folds 1-2 --max-epochs 2 --out results_smoke",
         "# S5 D01:     bash scripts/stage5_launch.sh     (nohup; status: python scripts/stage5_status.py)",
         "# S6 D02:     bash scripts/stage6_launch.sh     (nohup; status: python scripts/stage6_status.py); python scripts/06b_verify.py",
         "# S7 tabular: CUDA_VISIBLE_DEVICES= python tabular/run_stage7.py",
         "# S8:         python scripts/08a_preflight.py ... scripts/08f_write_report.py",
         "# S9:         python scripts/09a_preflight.py; 09b_tables.py; 09c_figures.py; 09d_write_report.py; 09e_write_stage_report.py",
         "# S10 audit:  CUDA_VISIBLE_DEVICES= python scripts/10_audit.py; python scripts/10b_finalize.py report|freeze",
         "```", "",
         "Training is deterministic up to cuDNN benchmark effects (~1e-3 in the losses); the stored results are the reference. "
         "Training resumes from checkpoints; DONE folds are never re-run unless `--force`.", "",
         "## Frozen state", "",
         "After Stage 10, `results/`, `splits/` and `reports/figures/final/` are read-only (`chmod -R a-w`). To undo: "
         "`chmod -R u+w results splits reports/figures/final`.", ""]
    open(P("README.md"), "w").write("\n".join(L))
    print(f"README.md written ({len(L)} lines; 5 findings + limitations copied from the report)")


# ---------------------------------------------------------------------------------------------- STAGE_10_REPORT
def report():
    A = json.load(open(P("logs", "stage10_audit.json")))
    audit_txt = open(P("logs", "stage10_audit.txt")).read().rstrip()
    first = open(P("logs", "stage10_audit_first_run_note.txt")).read().rstrip()
    pyt = open(P("logs", "stage10_pytest.txt")).read().rstrip()
    last = pyt.splitlines()[-1]
    py_ok = re.search(r"(\d+) (failed|error)", last) is None and "passed" in last
    claims = open(P("logs", "stage10_claims_scan.txt")).read().rstrip()
    t = report_text()
    new_ok = "had a matched-subset AUC whose 95% bootstrap CI excluded 0.5" in t and OLD_SENTENCE not in t
    s9 = open(P("reports", "STAGE_09_REPORT.md")).read()
    s9_ok = "had a matched-subset AUC whose 95% bootstrap CI excluded 0.5" in s9 and OLD_SENTENCE not in s9
    html = open(P("reports", "FINAL_RESULTS_REPORT.html")).read().replace("\n", " ")
    html_ok = "had a matched-subset AUC whose 95% bootstrap CI excluded 0.5" in html and OLD_SENTENCE not in html
    sentence = re.search(r"only one model \(M\d+\) had a matched-subset AUC[^;]*; the other models' matched AUCs had CIs including 0\.5\.", t).group(0)
    b3 = A["b3"]
    leak_ok = (b3["predictions_equal_fold_test_ids"] and b3["each_sample_predicted_once_per_model"][1] == 0
               and b3["train_val_test_disjoint_sample_and_group"] and not any(b3["groups_in_more_than_one_test_fold"].values())
               and b3["D02_phash_le4_pairs_total_and_cross_fold"][1] == 0 and not b3["D03_paths_in_cache"]
               and b3["D02_manifest_all_under_D02_folder"] and not b3["CARS_in_inputs"]
               and not b3["training_code_reading_participants_or_CARS"] and not b3["D05_honest_forbidden_in_set"]
               and not b3["D05_honest_forbidden_in_run_configs"] and b3["D04_clean_rows_equal_train_csv"]
               and b3["D04_rows_identical_to_test_csv_rows"] == 0)
    snaps = pd.DataFrame(A["b4"]["summary"])
    snap_ok = bool(snaps.checksums_integrity.all()) and not A["b4"]["deleted"]
    num_ok = not A["b5"]["manual"]
    ver_ok = not A["b1"]["mismatches"] and not A["b2"]["mismatches"]
    members = zip_members()
    ch = pd.read_csv(P("logs", "stage10_snapshot_changes.csv"))
    s10_files = sorted(ch[(ch.snapshot == "stage_09") & ch.explanation.str.contains("stage_10")].path)
    md = lambda df: "\n".join(["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]  # noqa: E731
                              + ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)])
    L = ["# STAGE 10 REPORT - Final audit and freeze", "",
         "Screening-support research, not diagnosis. Stage 10 trains nothing and changes no result value. Code: "
         "`scripts/10_audit.py` (fresh re-verification: sklearn / numpy / hashlib only, nothing imported from src.training, "
         "src.analysis or make_snapshot), `scripts/10b_finalize.py` (README, this report, freeze); `src/analysis/final.py` now "
         "also skips `results/FINAL_MANIFEST.json` in the Stage 9 read-only manifest (Stage 10 output, not a Stage 9 input).", "",
         "## 1. Summary", "",
         f"- Wording fix applied to FINAL_RESULTS_REPORT.md ({new_ok}), its HTML ({html_ok}) and STAGE_09_REPORT.md ({s9_ok}). "
         "No report sentence claiming clinical accuracy, diagnosis or autism detection was found (section 2.2).",
         f"- Re-verification: metric mismatches {len(A['b1']['mismatches'])} of {A['b1']['checks']} (400 runs); master-table "
         f"mismatches {len(A['b2']['mismatches'])} of {A['b2']['checks']}; leakage checks all passed: {leak_ok}; snapshots "
         f"stage_01..09 intact: {snap_ok}; report numbers needing manual check: {len(A['b5']['manual'])}.",
         f"- Tests: {last.strip('= ')}.",
         f"- Freeze: results/FINAL_MANIFEST.json, read-only {', '.join(READ_ONLY)}, submission zip with {len(members)} files "
         "(size and sha256 in `submission/SUBMISSION_INFO.json`, written after the zip because the zip contains this report).", "",
         "## 2. Wording fix (A)", "",
         "### 2.1 Matched-subset sentence", "",
         f"Old (abstract of FINAL_RESULTS_REPORT.md and the abstract quoted in STAGE_09_REPORT.md): \"... only one model (M07) "
         f"{OLD_SENTENCE} children on tracking quality.\"", "",
         f"New (numbers read from `results/final/shortcut_table.csv` by `scripts/09d_write_report.py`, which also asserts that "
         f"exactly one model's matched CI excludes 0.5): \"{sentence}\"", "",
         "The sentence occurs only in the abstract; sections 6 and 8 already used CI wording. The abstract grew from 183 to 197 "
         "words (Stage 9 limit 150-200); STAGE_09_REPORT.md's stated word count was updated accordingly. The HTML was "
         "regenerated with pandoc. `diff` of the regenerated FINAL_RESULTS_REPORT.md vs the Stage 9 version: only line 9 "
         "(abstract) differs.", "",
         "### 2.2 Scan for clinical / diagnostic / detection claims (`logs/stage10_claims_scan.txt`)", "", "```", claims, "```", "",
         "## 3. Independent re-verification (B, `logs/stage10_audit.txt`)", "", "```", audit_txt, "```", "",
         "First run of the audit (kept for the record, `logs/stage10_audit_first_run_note.txt`):", "", "```", first, "```", "",
         "## 4. Snapshot check (B4, `logs/stage10_snapshot_changes.csv`)", "",
         md(snaps.rename(columns={"stage10_or_unexplained": "current version in no later snapshot"})), "",
         "Every changed or added file is explained by the later snapshot that recorded its current version (the stage that "
         "legitimately edited it); `current version in no later snapshot` = 0 everywhere. Files changed or added in Stage 10 "
         f"(stage_09 -> provisional stage_10, which was taken before this report): {', '.join(s10_files)}. After this report, "
         "the final stage_10 snapshot also records reports/STAGE_10_REPORT.md and results/FINAL_MANIFEST.json. No recorded "
         "file was deleted. The large 'added' counts of early snapshots are later stages' new code, reports and caches.", "",
         "## 5. Tests (`.venv/bin/python -m pytest -v`, `logs/stage10_pytest.txt`, run WITH the read-only freeze in place)", "",
         f"Result: {last.strip('= ')}", "",
         f"Sequence: before the freeze {open(P('logs', 'stage10_pytest_before_freeze.txt')).read().rstrip().splitlines()[-1].strip('= ')} "
         f"(`logs/stage10_pytest_before_freeze.txt`). The first run after the freeze gave "
         f"{open(P('logs', 'stage10_pytest_after_freeze_first.txt')).read().rstrip().splitlines()[-1].strip('= ')} "
         "(`logs/stage10_pytest_after_freeze_first.txt`): `tests/test_stage04_units.py::test_verify_manifest_detects_modified_"
         "temp_copy` copies splits/ to a temp dir with shutil.copytree, which also copies the new read-only mode bits, so the "
         "test could not modify its own temp copy. Fix (test only): make the temp copy writable before modifying it. The real "
         "splits/ stay read-only and unchanged. Manifest, zip and snapshot were then regenerated.", "",
         "<details><summary>full pytest output</summary>", "", "```", pyt, "```", "", "</details>", "",
         "## 6. Freeze (C)", "",
         "- `results/FINAL_MANIFEST.json`: sha256 + size of every file in " + ", ".join(f"`{d}/`" for d in MANIFEST_DIRS)
         + " (no *.pt, no cache/, no __pycache__; this report included), environment summary and total compute time; "
         "written after this report.",
         f"- Read-only after the manifest: `chmod -R a-w {' '.join(READ_ONLY)}`; undo with `chmod -R u+w {' '.join(READ_ONLY)}`.",
         f"- `submission/ASD_Hybrid_10fold_submission.zip`: {len(members)} files (src/, tabular/, scripts/, configs/, tests/, "
         "splits/, results/final/, reports/*.md + *.html, reports/figures/final/, CLAUDE.md, README.md, requirements.txt, "
         "results/FINAL_MANIFEST.json; no raw data, no cache/, no *.pt). Size and sha256: `submission/SUBMISSION_INFO.json`.",
         "- `README.md`: summary, reproduction commands, locations, five findings and the limitations copied verbatim from "
         "FINAL_RESULTS_REPORT.md.", ""]
    ok = new_ok and s9_ok and html_ok and ver_ok and leak_ok and snap_ok and num_ok and py_ok
    why = [n for n, v in [("wording", new_ok and s9_ok and html_ok), ("re-verification", ver_ok), ("leakage", leak_ok),
                          ("snapshots", snap_ok), ("report numbers", num_ok), ("pytest", py_ok)] if not v]
    L.append("STAGE 10 DECISION: PASS" if ok else "STAGE 10 DECISION: FAIL - " + ", ".join(why))
    open(P("reports", "STAGE_10_REPORT.md"), "w").write("\n".join(L) + "\n")
    print(L[-1], f"| zip members planned: {len(members)}")


# ---------------------------------------------------------------------------------------------- freeze
def freeze():
    rep = pd.read_csv(P("results", "final", "reproducibility_table.csv"))
    env = dict(zip(rep["item"], rep["value"]))
    files = {}
    for d in MANIFEST_DIRS:
        for f in walk(d):
            if f == "results/FINAL_MANIFEST.json":
                continue
            files[f] = {"sha256": sha(P(f)), "size": os.path.getsize(P(f))}
    gpu = float(env["S5 D01 GPU seconds (sum over 100 fold runs)"]) + float(env["S6 D02 GPU seconds (sum over 100 fold runs)"])
    man = {"created": datetime.now().isoformat(timespec="seconds"), "project": "asd_hybrid_10fold",
           "scope": MANIFEST_DIRS, "excluded": ["*.pt", "cache/", "__pycache__/", "*.pyc", "results/FINAL_MANIFEST.json (self)"],
           "n_files": len(files),
           "environment": {k: env[k] for k in ["python", "torch", "cuda", "cudnn", "gpu_name", "driver", "timm", "tabular sklearn",
                                               "tabular numpy", "tabular pandas"]} | {"platform": platform.platform()},
           "compute": {"S5_D01_gpu_seconds": float(env["S5 D01 GPU seconds (sum over 100 fold runs)"]),
                       "S6_D02_gpu_seconds": float(env["S6 D02 GPU seconds (sum over 100 fold runs)"]),
                       "total_gpu_hours": round(gpu / 3600, 3),
                       "S7_tabular_cpu_seconds": float(env["S7 tabular CPU seconds (sum over fold runs)"])},
           "undo_read_only": f"chmod -R u+w {' '.join(READ_ONLY)}", "files": files}
    with open(P("results", "FINAL_MANIFEST.json"), "w") as fh:
        json.dump(man, fh, indent=1)
    msha = sha(P("results", "FINAL_MANIFEST.json"))
    print(f"FINAL_MANIFEST.json: {len(files)} files, sha256 {msha}")
    members = zip_members()
    missing = [m for m in members if not os.path.isfile(P(m))]
    assert not missing, missing
    os.makedirs(os.path.dirname(ZIP), exist_ok=True)
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for m in members:
            z.write(P(m), arcname=f"asd_hybrid_10fold/{m}")
    for d in READ_ONLY:
        subprocess.run(["chmod", "-R", "a-w", P(d)], check=True)
    writable = [f for d in READ_ONLY for f in walk(d, skip_pt=False) if os.access(P(f), os.W_OK)]
    info = {"zip": os.path.relpath(ZIP, ROOT), "size_bytes": os.path.getsize(ZIP), "size_mb": round(os.path.getsize(ZIP) / 2 ** 20, 2),
            "n_files": len(members), "sha256": sha(ZIP), "final_manifest_sha256": msha, "read_only": READ_ONLY,
            "writable_files_left_in_read_only_dirs": len(writable), "undo_read_only": f"chmod -R u+w {' '.join(READ_ONLY)}",
            "created": datetime.now().isoformat(timespec="seconds")}
    with open(P("submission", "SUBMISSION_INFO.json"), "w") as fh:
        json.dump(info, fh, indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    {"readme": readme, "report": report, "freeze": freeze}[sys.argv[1]]()
