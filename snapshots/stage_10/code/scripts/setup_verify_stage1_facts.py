"""Setup session - re-verify the Stage 1 facts quoted in the setup brief against the audit JSON files.
Prints one line per fact: CONFIRMED / CONTRADICTED, the claim, and the value loaded from the file.
Usage: .venv/bin/python scripts/setup_verify_stage1_facts.py > logs/setup_fact_check.txt"""
import json
import os
import subprocess
import sys

import torch

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REG = os.path.join(PROJECT_DIR, "data_registry")


def J(name):
    with open(os.path.join(REG, name)) as fh:
        return json.load(fh)


def tab(audit, fname):
    return next(f for f in audit["tabular"]["files"] if f["file"] == fname)


def rule(f, prefix):
    return next(r for r in f["leakage_check"]["rules"] if r["rule"].startswith(prefix))


rows = []


def check(ds, claim, value, ok, source):
    rows.append((ds, claim, value, ok, source))


# ---- D01 eye-tracking
e = J("D01_eye_tracking_audit.json")["eye_tracking"]
s1 = "D01_eye_tracking_audit.json"
check("D01", "2,252,943 gaze rows", e["total_gaze_rows"], e["total_gaze_rows"] == 2252943, s1)
check("D01", "25 gaze CSVs", e["n_csv_files_gaze"], e["n_csv_files_gaze"] == 25, s1)
nl = len(e["column_set_variants"])
check("D01", "9 column layouts", nl, nl == 9, s1)
check("D01", "metadata 59 participants (29 ASD / 30 TD)",
      (e["metadata_n_participants"], e["metadata_class_counts"]),
      e["metadata_n_participants"] == 59 and e["metadata_class_counts"] == {"TD": 30, "ASD": 29}, s1)
check("D01", "57 with gaze rows", e["participants_with_metadata_and_gaze"],
      e["participants_with_metadata_and_gaze"] == 57, s1)
check("D01", "IDs 12 and 16 have no gaze rows", e["participants_metadata_but_no_gaze"],
      sorted(e["participants_metadata_but_no_gaze"]) == [12, 16], s1)
unid = sum(e["non_numeric_participant_values_rows"].values())
frac = unid / e["total_gaze_rows"]
check("D01", "~40% of rows are 'Unidentified'", f"{unid} rows = {frac:.4f}", round(frac, 2) == 0.40, s1)
dc = e["duplicate_content_check"]
ndup = dc["rows_total"] - dc["distinct_row_keys_over_all_files"]
check("D01", "329,034 duplicated rows",
      f"{ndup} (= rows_total - distinct keys {dc['row_key']}, all files pooled)", ndup == 329034, s1)
pairs = {tuple(p["files"]): p for p in e["duplicate_check_dd02_key"]["file_pairs_sharing_keys"]}
p = pairs[("8.csv", "11.csv")]
check("D01", "files 8 & 11 share all rows", p, p["shared_keys"] == p["rows_a"] == p["rows_b"], s1)
p = pairs[("1.csv", "16.csv")]
check("D01", "file 1 is inside file 16", p, p["shared_keys"] == p["rows_a"] < p["rows_b"], s1)
cm = e["cars_missing_by_class"]
check("D01", "CARS only for ASD", cm,
      cm["ASD"]["cars_missing"] == 0 and cm["TD"]["cars_missing"] == cm["TD"]["n"], s1)
hz = e["sampling"]["implied_rate_hz"]
check("D01", "~60 Hz", hz, 58 <= hz <= 62, s1)

# ---- D02 facial images A
a2 = J("D02_facial_images_A_audit.json")
i2 = a2["images"]
s2 = "D02_facial_images_A_audit.json"
check("D02", "2,940 images", i2["image_files_used_as_samples"], i2["image_files_used_as_samples"] == 2940, s2)
split_n = {k: sum(v.values()) for k, v in i2["count_per_split_x_class"].items()}
check("D02", "train 2,386 / valid 254 / test 300", split_n,
      split_n == {"train": 2386, "valid": 254, "test": 300}, s2)
ct = i2["class_totals"]
check("D02", "balanced classes", ct, len(set(ct.values())) == 1, s2)
check("D02", "0 unreadable", i2["n_unreadable"], i2["n_unreadable"] == 0, s2)
check("D02", "0 exact duplicates", i2["exact_duplicates_md5"]["n_groups"],
      i2["exact_duplicates_md5"]["n_groups"] == 0, s2)
nd = i2["near_duplicates_phash"]
check("D02", "11 near-duplicate pairs, 5 across splits", (nd["pairs_total"], nd["across_splits"]),
      nd["pairs_total"] == 11 and nd["across_splits"] == 5, s2)
check("D02", "no subject IDs", i2["subject_ids"]["finding"][:40],
      i2["subject_ids"]["finding"].startswith("NO subject"), s2)

# ---- D03 = copy of D02
x = a2["cross_dataset_duplicates"]
nbi = x["D03_facial_images_B_non_byte_identical_images"]
n_match = x["D03_facial_images_B_images_with_near_or_exact_match_in_D02_facial_images_A(nearest hamming<=4)"]
check("D03", "same 2,940 images as D02", (x["exact_md5"]["D03_facial_images_B_images_total"], n_match),
      x["exact_md5"]["D03_facial_images_B_images_total"] == 2940 and n_match == 2940, s2)
check("D03", "2,541 byte-identical", x["exact_md5"]["unique_md5_shared"],
      x["exact_md5"]["unique_md5_shared"] == 2541, s2)
check("D03", "399 re-saved copies of D02 test/valid", (nbi["n"], nbi["nearest_match_split"]),
      nbi["n"] == 399 and set(nbi["nearest_match_split"]) <= {"test", "valid"}, s2)

# ---- D04
a4 = J("D04_autism_prediction_audit.json")
tr, te = tab(a4, "train.csv"), tab(a4, "test.csv")
s4 = "D04_autism_prediction_audit.json"
check("D04", "train 800 rows (639/161)", (tr["shape"][0], tr["class_counts"]),
      tr["shape"][0] == 800 and tr["class_counts"] == {"0": 639, "1": 161}, s4)
check("D04", "test.csv unlabelled", te["label_columns"], te["label_columns"] == [], s4)

# ---- D05
a5 = J("D05_toddler_screening_audit.json")
s5 = "D05_toddler_screening_audit.json"
td = tab(a5, "Toddler Autism dataset July 2018.csv")
r = rule(td, "ASD if sum(A1..A10) >= 4")
check("D05", "Toddler label == (A1..A10 sum >= 4), accuracy 100%", r["accuracy"], r["accuracy"] == 1.0, s5)
emb = a5["tabular"]["cross_file_embedded_blocks"][0]
check("D05", "Combined holds Toddler rows 5021-6074", emb["container_rows"], emb["container_rows"] == [5021, 6074], s5)
check("D05", "all 1,054 embedded labels = NO", emb["label_container_counts"],
      emb["label_container_counts"] == {"no": 1054}, s5)
check("D05", "728 wrong labels", emb["rows_with_different_label"], emb["rows_with_different_label"] == 728, s5)
cb = tab(a5, "Autism_Screening_Data_Combined.csv")
check("D05", "847 duplicate rows in Combined", cb["duplicate_rows_full"], cb["duplicate_rows_full"] == 847, s5)

# ---- D06
a6 = J("D06_early_autism_audit.json")
s6 = "D06_early_autism_audit.json"
f6 = tab(a6, "asd.csv")
check("D06", "1,000 rows (589/411)", (f6["shape"][0], f6["class_counts"]),
      f6["shape"][0] == 1000 and f6["class_counts"] == {"No": 589, "Yes": 411}, s6)
check("D06", "one fully empty column", f6["fully_empty_columns"], len(f6["fully_empty_columns"]) == 1, s6)
best = max(f6["leakage_check"]["rules"], key=lambda r: r["accuracy"])
maj = f6["leakage_check"]["majority_class_baseline_accuracy"]
check("D06", "no leakage: best rule 0.601 vs majority 0.589", (best["rule"], best["accuracy"], maj),
      best["accuracy"] == 0.601 and maj == 0.589, s6)

# ---- environment (live, not from the audit)
smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip()
smi_full = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
cuda_drv = smi_full.split("CUDA Version:")[1].split()[0] if "CUDA Version:" in smi_full else "UNKNOWN"
check("ENV", "NVIDIA RTX A2000 12GB, driver 580", smi, "RTX A2000 12GB" in smi and ", 580." in smi, "live nvidia-smi")
check("ENV", "driver CUDA 13.0", cuda_drv, cuda_drv == "13.0", "live nvidia-smi")
check("ENV", "Python 3.12.3", sys.version.split()[0], sys.version.split()[0] == "3.12.3", "live")
check("ENV", "torch 2.14.1+cu130, CUDA available", (torch.__version__, torch.cuda.is_available()),
      torch.__version__ == "2.14.1+cu130" and torch.cuda.is_available(), "live")
venv = os.path.join(PROJECT_DIR, ".venv")
check("ENV", ".venv inside project dir", sys.prefix, os.path.realpath(sys.prefix) == os.path.realpath(venv), "live")

print("| # | dataset | claim | value from file | verdict | source |")
print("|---|---|---|---|---|---|")
for k, (ds, claim, value, ok, src) in enumerate(rows, 1):
    print(f"| {k} | {ds} | {claim} | `{value}` | {'CONFIRMED' if ok else 'CONTRADICTED'} | {src} |")
n_bad = sum(not r[3] for r in rows)
print(f"\nTOTAL: {len(rows)} facts, {len(rows) - n_bad} CONFIRMED, {n_bad} CONTRADICTED")
