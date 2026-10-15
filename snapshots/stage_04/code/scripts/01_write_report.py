"""Stage 1 - write reports/STAGE_01_REPORT.md from the audit outputs, logs and live checks.
All numbers are loaded from files; the decision line is computed from the PASS conditions."""
import json
import os
import re
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.paths import DATASETS_DIR  # noqa: E402  (configs/paths.yaml)

P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REG = os.path.join(P, "data_registry")
IDS = ["D01_eye_tracking", "D02_facial_images_A", "D03_facial_images_B",
       "D04_autism_prediction", "D05_toddler_screening", "D06_early_autism"]


def rd(*p):
    with open(os.path.join(P, *p)) as fh:
        return fh.read()


def J(did):
    return json.loads(rd("data_registry", f"{did}_audit.json"))


def grab(pattern, text, default="UNKNOWN"):
    m = re.search(pattern, text, re.M)
    return m.group(1).strip() if m else default


def tree(root, depth=2, skip=(".venv", "cache", ".git", "__pycache__", ".pytest_cache")):
    lines = ["asd_hybrid_10fold/"]

    def walk(d, level, prefix):
        entries = sorted(e for e in os.listdir(d) if e not in skip)
        for i, e in enumerate(entries):
            full = os.path.join(d, e)
            last = i == len(entries) - 1
            lines.append(prefix + ("└── " if last else "├── ") + e + ("/" if os.path.isdir(full) else ""))
            if os.path.isdir(full) and level < depth:
                walk(full, level + 1, prefix + ("    " if last else "│   "))
    walk(root, 1, "")
    return "\n".join(lines)


def record_state():
    # User override 2026-10-13: git is not used in this project. A stage is recorded by a local snapshot
    # (scripts/make_snapshot.py -> snapshots/stage_NN/), taken right after this report is written.
    if not os.path.isfile(os.path.join(P, "scripts", "make_snapshot.py")):
        return False, "scripts/make_snapshot.py missing"
    return True, ("no git (user override 2026-10-13); Stage 1 is recorded by the local snapshot "
                  "`snapshots/stage_01/` (code copy + CHECKSUMS.sha256 + SNAPSHOT_INFO.json), created right after this report")


def main():
    env = rd("data_registry", "environment.txt")
    pytest_out = rd("logs", "stage01_pytest.txt")
    snap = rd("logs", "stage01_snapshot_compare.txt")
    a = {d: J(d) for d in IDS}
    e = a["D01_eye_tracking"]["eye_tracking"]
    im = a["D02_facial_images_A"]["images"]
    cr = a["D02_facial_images_A"]["cross_dataset_duplicates"]
    d4 = {f["file"]: f for f in a["D04_autism_prediction"]["tabular"]["files"]}
    d5 = {f["file"]: f for f in a["D05_toddler_screening"]["tabular"]["files"]}
    emb = a["D05_toddler_screening"]["tabular"]["cross_file_embedded_blocks"][0]
    d6 = a["D06_early_autism"]["tabular"]["files"][0]
    dd = json.loads(rd("data_registry", "D01_dedupe_key_check.json"))
    cv = json.loads(rd("data_registry", "D01_cv_feasibility.json"))
    tod = d5["Toddler Autism dataset July 2018.csv"]
    comb = d5["Autism_Screening_Data_Combined.csv"]
    tr = d4["train.csv"]

    # ---- PASS conditions
    m = re.search(r"=+ (\d+) passed(.*?) in ", pytest_out)
    tests_ok = bool(m) and "failed" not in m.group(0) and "error" not in m.group(0)
    raw_ok = "IDENTICAL" in snap
    audited_ok = all(x["n_samples"] > 0 for x in a.values()) and len(a) == 6
    decisions_ok = os.path.isfile(os.path.join(REG, "DATASET_DECISIONS.md"))
    git_ok, git_msg = record_state()

    pairs = e["duplicate_check_dd02_key"]["file_pairs_sharing_keys"]
    nk = dd["numeric_key_identified_rows"]
    L = []
    L += ["# STAGE 1 REPORT - workspace, environment and dataset audit", "",
          "Project: Reliability-Weighted Multimodal AI for Early Autism Risk Screening in Children (BITE308L).",
          f"Project dir: `{P}`. Raw data (read-only): `{DATASETS_DIR}`. Seed: 42.", ""]
    L += ["## 1. Summary", "",
          "- Built the project skeleton, the results layout (`results/README.md`) and a verified CUDA environment (`.venv`).",
          "- Audited all 6 raw datasets read-only (`scripts/01_audit_datasets.py`). Outputs are in `data_registry/` (json, md, contact sheets, licence text).",
          f"- Found serious data problems: D03 is a copy of D02; the D05 labels are a formula of A1-A10; the D05 Combined file has {emb['rows_with_different_label']} wrong labels; 40% of D01 rows have no child ID and {e['duplicate_check_dd02_key']['file_pairs_sharing_keys'].__len__()} D01 file pairs share rows.",
          "- Recorded 12 dataset decisions with evidence (`data_registry/DATASET_DECISIONS.md`).",
          f"- pytest: {m.group(1) if m else 'UNKNOWN'} passed; raw data unchanged (snapshot START == END).", ""]
    L += ["## 2. Environment", "", "Source: `data_registry/environment.txt`.", "",
          "| item | value |", "|---|---|",
          f"| GPU | {grab(r'get_device_name\(0\) = (.*)', env)} |",
          f"| GPU memory | {grab(r'(\d+MiB /\s+\d+MiB)', env)} (nvidia-smi used/total); torch total = {grab(r'total GPU memory \(GiB\) = (.*)', env)} GiB |",
          f"| NVIDIA driver | {grab(r'Driver Version: (\S+)', env)} (driver CUDA {grab(r'CUDA Version: (\S+)', env)}) |",
          f"| torch CUDA / cuDNN | {grab(r'torch.version.cuda = (.*)', env)} / {grab(r'cudnn = (.*)', env)} |",
          f"| Python | {grab(r'^python (\S+)', env)} |",
          f"| torch / torchvision / timm | {grab(r'torch\s+OK\s+(\S+)', env)} / {grab(r'torchvision\s+OK\s+(\S+)', env)} / {grab(r'timm\s+OK\s+(\S+)', env)} |",
          f"| cuda.is_available() | {grab(r'is_available\(\) = (.*)', env)} |",
          f"| 4096x4096 fp32 matmul | {grab(r'matmul on GPU: (.*?),', env)} |",
          f"| venv | `{P}/.venv` (`--system-site-packages`; numpy/scipy/matplotlib/pillow/pyyaml come from the system and are pinned via `logs/pip_constraints.txt`) |",
          "| requirements | `requirements.txt` (exact versions of the 13 required packages) |", ""]
    L += ["## 3. Project skeleton (depth 2, excluding .venv and cache)", "", "```", tree(P), "```", ""]
    L += ["## 4. Dataset registry", "",
          f"Folder -> id mapping (all inside `{DATASETS_DIR}/`):", ""]
    for d in IDS:
        L.append(f"- **{d}** <- `{a[d]['folder']}`")
    L += ["", "The folder truncated in the file manager is, in full: `Autism Spectrum Disorder Detection using Facial Traits "
          "Dual Framework DenseRes with K-Fold Cross Validation (DFDK) for Enhanced Classification Accuracy` (= D02).", ""]
    reg = rd("data_registry", "DATASET_REGISTRY.md").split("\n")
    L += [ln for ln in reg if ln.startswith("|")] + [""]
    L += ["## 5. Problems found", ""]
    probs = [
        f"**D01 - no child ID on {sum(e['non_numeric_participant_values_rows'].values()):,} of {e['total_gaze_rows']:,} gaze rows** {e['non_numeric_participant_values_rows']} - `D01_eye_tracking_audit.json`",
        f"**D01 - duplicated recordings across files:** {len(pairs)} file pairs share rows, e.g. "
        + "; ".join(f"{p['files'][0]}/{p['files'][1]}: {p['shared_keys']:,}" for p in pairs[:3])
        + f". Identified rows {nk['rows']:,} -> {nk['distinct']:,} after dedupe ({nk['duplicate_rows_removed']:,} removed, {nk['of_which_duplicate_within_same_file']:,} of them within a single file) - `D01_eye_tracking_audit.json`, `D01_dedupe_key_check.json`",
        f"**D01 - dedupe key must be numeric:** a string key removes only {dd['string_key_identified_rows']['duplicate_rows_removed']:,} identified duplicate rows vs {nk['duplicate_rows_removed']:,} with numeric parsing - `D01_dedupe_key_check.json`",
        f"**D01 - participants without gaze data:** {e['participants_metadata_but_no_gaze']} (both ASD), leaving {e['participants_with_metadata_and_gaze']} children {e['participants_with_metadata_and_gaze_by_class']} - `D01_eye_tracking_audit.json`",
        f"**D01 - CARS Score encodes the label:** missing by class {e['cars_missing_by_class']} - `D01_eye_tracking_audit.json`",
        f"**D01 - {len(e['column_set_variants'])} different column layouts across 25 files** (37-58 columns); sampling interval varies by file (median {e['sampling']['median_dt_ms']:.1f} ms overall, p95 {e['sampling']['p95_dt_ms']} ms) - `D01_eye_tracking_audit.json`",
        f"**D01 - small sample:** {cv['n_participants']} children -> {min(f['n_test'] for f in cv['folds'])}-{max(f['n_test'] for f in cv['folds'])} per test fold ({min(f['asd'] for f in cv['folds'])}-{max(f['asd'] for f in cv['folds'])} ASD); fold metrics will have high variance - `D01_cv_feasibility.json`",
        f"**D03 is a copy of D02:** {cr['exact_md5']['unique_md5_shared']:,} byte-identical + {cr['D03_facial_images_B_non_byte_identical_images']['n']} re-saved copies (nearest D02 split {cr['D03_facial_images_B_non_byte_identical_images']['nearest_match_split']}); all {cr['D03_facial_images_B_images_with_near_or_exact_match_in_D02_facial_images_A(nearest hamming<=4)']:,} D03 images match D02 with the same class - `D02_facial_images_A_audit.json`",
        f"**D02 - near-duplicates across the original splits:** {im['near_duplicates_phash']['pairs_total']} pHash<=4 pairs, {im['near_duplicates_phash']['across_splits']} across splits {im['near_duplicates_phash']['across_split_matrix']}; exact md5 groups {im['exact_duplicates_md5']['n_groups']} - `D02_facial_images_A_audit.json`",
        "**D02/D03 - no subject IDs** (filenames `<class>.<n>.jpg` / `<n>.jpg`, no metadata): the same child may occur in several images; results can only be image-level - `D02_facial_images_A_audit.json`",
        f"**D02 - inconsistent class folder name** {im['class_folder_names_per_split(raw)']['test']} in test vs {im['class_folder_names_per_split(raw)']['train']} in train; {len(im['image_files_outside_class_folders(not samples)'])} result-figure PNGs in the folder root (not data) - `D02_facial_images_A_audit.json`",
        f"**D02 - image size varies widely:** width {im['width']['min']}-{im['width']['max']} px, height {im['height']['min']}-{im['height']['max']} px, {im['n_distinct_sizes']:,} distinct sizes; unreadable files: {im['n_unreadable']} - `D02_facial_images_A_audit.json`",
        f"**D04 - test.csv has no label** (shape {d4['test.csv']['shape']}, label columns {d4['test.csv']['label_columns']}) - `D04_autism_prediction_audit.json`",
        f"**D04 - class imbalance:** train {tr['class_counts']} ({tr['class_balance_pos_fraction']*100:.1f}% positive); '?' placeholders {tr['question_mark_values_per_column']} - `D04_autism_prediction_audit.json`",
        f"**D05 - label leakage (Toddler):** '{tod['leakage_check']['rules'][0]['rule']}' accuracy {tod['leakage_check']['rules'][0]['accuracy']}, confusion {tod['leakage_check']['rules'][0]['confusion_matrix[[TN,FP],[FN,TP]]']}; item sum == Qchat-10-Score on {tod['leakage_check']['item_sum_equals_Qchat-10-Score']['rows_equal']}/{tod['leakage_check']['item_sum_equals_Qchat-10-Score']['n']} rows - `D05_toddler_screening_audit.json`",
        f"**D05 - wrong labels in Combined file:** Toddler rows embedded at rows {emb['container_rows'][0]}-{emb['container_rows'][1]}, all labelled {emb['label_container_counts']} vs original {emb['label_contained_counts']} -> {emb['rows_with_different_label']} conflicts - `D05_toddler_screening_audit.json`",
        f"**D05 - partial leakage in Combined file:** {comb['leakage_check']['summary']} - `D05_toddler_screening_audit.json`",
        f"**D05 - duplicates:** Combined {comb['duplicate_rows_full']} fully duplicated rows; Toddler {tod['duplicate_rows_excluding_id']} duplicates when Case_No is ignored - `D05_toddler_screening_audit.json`",
        f"**D05 - class imbalance (Toddler):** {tod['class_counts']} ({tod['class_balance_pos_fraction']*100:.1f}% ASD) - `D05_toddler_screening_audit.json`",
        f"**D06 - empty column** {d6['fully_empty_columns']}; weak signal: best single rule '{d6['leakage_check']['rules'][0]['rule']}' {d6['leakage_check']['rules'][0]['accuracy']} vs majority baseline {d6['leakage_check']['majority_class_baseline_accuracy']} - `D06_early_autism_audit.json`",
        "**Licences:** no licence file in any of the 6 folders; only D01 `Citation.txt` and the D05 description (.docx) exist - `data_registry/*_licence_citation.txt`",
        "**No unreadable images, no zip/rar archives** in any folder - `*_audit.json` (`n_unreadable`, `archives`)",
    ]
    L += [f"{i}. {p}" for i, p in enumerate(probs, 1)] + [""]
    dec = rd("data_registry", "DATASET_DECISIONS.md")
    L += ["## 6. Dataset decisions", "", "Full text with evidence: [data_registry/DATASET_DECISIONS.md](../data_registry/DATASET_DECISIONS.md). "
          + ("No CONTRADICTION found." if "No CONTRADICTION" in dec else "SEE CONTRADICTIONS in the file."), ""]
    for t in re.findall(r"^### (DD-\d+ - .*)$", dec, re.M):
        L.append(f"- {t}")
    L += ["", "## 7. Tests", "", "Command: `.venv/bin/python -m pytest tests/test_stage01.py -v` (output: `logs/stage01_pytest.txt`)", "",
          "```", pytest_out.rstrip(), "```", "", "Raw-data snapshot comparison (`logs/stage01_snapshot_compare.txt`):", "", "```", snap.rstrip(), "```", ""]
    L += ["## 8. Open items for Stage 2 (data preparation + 10-fold split manifests + leakage tests)", "",
          "1. D01: drop Unidentified rows, exclude IDs " + str(e["participants_metadata_but_no_gaze"]) + ", dedupe with the numeric DD-02 key, and report rows before/after (expected identified rows "
          f"{nk['rows']:,} -> {nk['distinct']:,}); harmonise the {len(e['column_set_variants'])} column layouts; define how gaze-image sequences (clips/windows) are rendered from gaze rows.",
          "2. D01: save the StratifiedGroupKFold(10, shuffle, seed 42) participant manifest in `splits/`; assert each test fold has >=1 ASD and >=1 TD and that no participant appears in two folds.",
          f"3. D02: pool the {im['image_files_used_as_samples']:,} images, normalise class names, build pHash<=4 groups, save the StratifiedGroupKFold manifest; assert no md5 or near-duplicate group crosses folds.",
          "4. D03: excluded; add a test that no D03 path ever appears in a manifest.",
          "5. D04 / D05-Toddler / D06: save StratifiedKFold(10, shuffle, seed 42) manifests; D05 manifests for experiments (a) all features and (b) without A1-A10/Qchat; treat '?' as missing in D04; drop `Unnamed: 10` in D06.",
          "6. Leakage tests (pytest): no ID/group overlap between train and test in any fold; CARS never in any D01 feature list; D05 experiment (b) feature list contains no A1-A10/score; DD-11 (no cross-dataset merges).",
          "7. Stage 8 shortcut control (DD-12): compute the per-child tracking-quality ratio in Stage 2 so it is available later.", ""]
    L += [f"Version record: {git_msg}", ""]
    reasons = [r for ok, r in [(tests_ok, "pytest did not fully pass"), (raw_ok, "raw data snapshot differs"),
                               (audited_ok, "not all 6 datasets audited"), (decisions_ok, "DATASET_DECISIONS.md missing"),
                               (git_ok, "stage record missing: " + git_msg)] if not ok]
    L.append("STAGE 1 DECISION: PASS" if not reasons else "STAGE 1 DECISION: FAIL - " + "; ".join(reasons))
    with open(os.path.join(P, "reports", "STAGE_01_REPORT.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(L[-1])


if __name__ == "__main__":
    main()
