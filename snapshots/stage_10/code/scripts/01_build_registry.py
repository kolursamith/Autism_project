"""Stage 1 - build data_registry/DATASET_REGISTRY.md from the *_audit.json files (no hand-typed numbers)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.paths import DATASETS_DIR  # noqa: E402  (configs/paths.yaml)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REG = os.path.join(PROJECT_DIR, "data_registry")
IDS = ["D01_eye_tracking", "D02_facial_images_A", "D03_facial_images_B",
       "D04_autism_prediction", "D05_toddler_screening", "D06_early_autism"]
HYBRID = "10 hybrid architectures"
TABULAR = "tabular baselines only (LR, RF, GB, MLP) - NOT the CNN/ViT hybrids"


def load(did):
    with open(os.path.join(REG, f"{did}_audit.json")) as fh:
        return json.load(fh)


def row_eye(a):
    e = a["eye_tracking"]
    with open(os.path.join(REG, "D01_dedupe_key_check.json")) as fh:
        dd = json.load(fh)
    dup = e["duplicate_content_check"]
    n_dup = dup["rows_total"] - dup["distinct_row_keys_over_all_files"]
    shared = {f: v for f, v in dup["rows_whose_key_also_occurs_in_another_file_per_file"].items() if v}
    by_cls = e["participants_with_metadata_and_gaze_by_class"]
    one_class = len(e["metadata_class_counts"]) < 2
    return [
        "eye-tracking time-series CSV (gaze)",
        f"{e['total_gaze_rows']:,} gaze rows in {e['n_csv_files_gaze']} CSVs; "
        f"{sum(e['non_numeric_participant_values_rows'].values()):,} rows have no participant ID ({e['non_numeric_participant_values_rows']})",
        f"{e['metadata_n_participants']} in metadata; {e['participants_with_metadata_and_gaze']} with gaze rows "
        f"(no gaze: IDs {e['participants_metadata_but_no_gaze']})",
        f"metadata {e['metadata_class_counts']}; with gaze {by_cls}",
        f"{n_dup:,} rows repeat a (RecordingTime, TimeOfDay, Participant, Trial) key; {len(shared)} files share rows; "
        f"{len(e['duplicate_check_dd02_key']['file_pairs_sharing_keys'])} file pairs share DD-02-key rows. "
        f"DD-02 numeric key, identified rows: {dd['numeric_key_identified_rows']['rows']:,} -> {dd['numeric_key_identified_rows']['distinct']:,} "
        f"(D01_dedupe_key_check.json)",
        f"CARS Score missing/n by class {e['cars_missing_by_class']} -> CARS is a label proxy and must never be a feature. "
        f"Participant IDs recur across files -> split by participant.",
        "unusable (one class)" if one_class else f"{HYBRID} (gaze-image sequences), participant-level folds",
    ]


def row_img(a, did):
    im = a["images"]
    ex, nr = im["exact_duplicates_md5"], im["near_duplicates_phash"]
    cr = a["cross_dataset_duplicates"]
    other = [d for d in cr["datasets"] if d != did][0]
    n_match = cr["D03_facial_images_B_images_with_near_or_exact_match_in_D02_facial_images_A(nearest hamming<=4)"]
    split_txt = ", ".join(f"{s}: {c}" for s, c in im["count_per_split_x_class"].items())
    dup = (f"exact md5: {ex['n_groups']} groups; near (phash<=4): {nr['pairs_total']} pairs, "
           f"{nr['across_splits']} across splits {nr['across_split_matrix'] or ''}, {nr['different_class(label conflict)']} cross-class. "
           f"vs {other}: {cr['exact_md5']['unique_md5_shared']} byte-identical, {n_match}/2940 D03 images have a "
           f"phash<=4 match in D02 (same class: {cr['nearest_match_same_class']})")
    suit = HYBRID if len(im["class_totals"]) >= 2 else "unusable (one class)"
    if did == "D03_facial_images_B":
        suit += " in principle, BUT it is the same 2940 images as D02 -> use D02 only; never treat D03 as an independent dataset"
    return [
        "facial images (RGB jpg)",
        f"{im['image_files_used_as_samples']} images ({split_txt}); {im['n_unreadable']} unreadable; "
        f"size {im['width']['min']}-{im['width']['max']} x {im['height']['min']}-{im['height']['max']} px",
        "no subject IDs",
        str(im["class_totals"]),
        dup,
        "n/a (images). No subject IDs -> same child may fall in train and test; near-duplicates across splits listed.",
        suit,
    ]


def row_tab(a):
    fs = a["tabular"]["files"]
    samples, classes, dups, leaks, ids = [], [], [], [], []
    for f in fs:
        samples.append(f"`{f['file']}` {f['shape'][0]}x{f['shape'][1]}")
        classes.append(f"`{f['file']}`: {f.get('class_counts', 'NO LABEL (unlabelled)')}")
        dups.append(f"`{f['file']}`: {f['duplicate_rows_full']} full / {f['duplicate_rows_excluding_id']} excl. ID")
        ids.append(f"`{f['file']}`: {', '.join(f['id_columns']) or 'none'}")
        if "leakage_check" in f:
            leaks.append(f"`{f['file']}`: {f['leakage_check']['summary']}")
    for b in a["tabular"].get("cross_file_embedded_blocks", []):
        leaks.append(f"`{b['contained']}` is embedded (rows {b['container_rows'][0]}-{b['container_rows'][1]}) in "
                     f"`{b['container']}` with {b['rows_with_different_label']} labels changed")
    one_class = all(len(f.get("class_counts", {})) < 2 for f in fs)
    return [
        "questionnaire / screening table",
        "; ".join(samples),
        "no subject IDs beyond per-row ID columns (" + "; ".join(ids) + "); cannot tell whether rows are distinct people",
        "; ".join(classes),
        "; ".join(dups),
        " | ".join(leaks) or "none",
        "unusable (one class)" if one_class else TABULAR,
    ]


def main():
    hdr = ["id", "folder name", "modality", "n samples", "n subjects", "classes and counts",
           "duplicates found", "leakage findings", "suitable for"]
    lines = ["# DATASET REGISTRY (Stage 1)", "",
             "Generated by `scripts/01_build_registry.py` from `data_registry/*_audit.json` "
             f"(produced by `scripts/01_audit_datasets.py`). Raw data root: `{DATASETS_DIR}`.", "",
             "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for did in IDS:
        a = load(did)
        if did == "D01_eye_tracking":
            cells = row_eye(a)
        elif "images" in a:
            cells = row_img(a, did)
        else:
            cells = row_tab(a)
        lines.append("| " + " | ".join([f"**{did}**", f"`{a['folder']}`"] + [c.replace("|", "/") for c in cells]) + " |")
    lines += ["", "Suitability rules applied (fixed): image datasets with ASD vs non-ASD labels -> 10 hybrid architectures; "
              "eye-tracking with participant ASD/TD labels -> 10 hybrid architectures (gaze-image sequences); "
              "questionnaire tables -> tabular baselines only; a dataset with only one class -> unusable.", ""]
    with open(os.path.join(REG, "DATASET_REGISTRY.md"), "w") as fh:
        fh.write("\n".join(lines))
    print("wrote data_registry/DATASET_REGISTRY.md")


if __name__ == "__main__":
    main()
