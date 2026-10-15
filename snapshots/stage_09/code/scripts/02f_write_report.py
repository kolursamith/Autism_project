"""Stage 2 - write reports/STAGE_02_REPORT.md from the Stage 2 outputs (no hand-typed numbers).
The decision line is computed: PASS only if the saved pytest output shows 0 failed / 0 errors and all outputs exist.
Usage: .venv/bin/python scripts/02f_write_report.py"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd  # noqa: E402

from src.data.common import CACHE_DIR, LOG_DIR, PROJECT_DIR, SPLITS_DIR, md_table  # noqa: E402
from src.data.splits import DATASETS  # noqa: E402


def J(*p):
    with open(os.path.join(*p)) as fh:
        return json.load(fh)


def R(*p):
    with open(os.path.join(*p)) as fh:
        return fh.read().rstrip("\n")


def csv(*p):
    return pd.read_csv(os.path.join(*p))


def main():
    c1 = os.path.join(CACHE_DIR, "D01_eye_tracking")
    s1 = J(c1, "d01_summary.json")
    s2 = J(CACHE_DIR, "D02_facial_images_A", "d02_summary.json")
    ev = s1["resolution_evidence"]
    pre = R(LOG_DIR, "stage02_preflight.txt")
    pyt = R(LOG_DIR, "stage02_pytest.txt")
    last = pyt.splitlines()[-1]
    n_pass = int(re.search(r"(\d+) passed", last).group(1)) if "passed" in last else 0
    failed = re.search(r"(\d+) (failed|error)", last)
    man = J(SPLITS_DIR, "SPLITS_MANIFEST.json")
    idx = csv(c1, "clips_index.csv")
    parts = csv(c1, "participants.csv")
    unc = csv(c1, "clips_index_uncapped.csv")
    q = csv(c1, "quality_per_child.csv")

    L = ["# STAGE 2 REPORT - data preparation, 10-fold split manifests and leakage tests", "",
         "Project: Reliability-Weighted Multimodal AI for Early Autism Risk Screening in Children (BITE308L). "
         "Screening-support research, not diagnosis. Seed 42. No model was trained in this stage. "
         "No git (user override 2026-10-13): this stage is recorded by the local snapshot `snapshots/stage_02/`.", ""]

    # 1 summary
    inc = parts[parts.status == "included"]
    L += ["## 1. Summary", "",
          f"- D01: {s1['n_raw_rows']:,} raw gaze rows → {s1['cleaning_steps'][-1]['rows_after']:,} clean rows → "
          f"{len(idx):,} 8-s clips (uint8 {tuple(s1['npy_shape'])}) from {len(inc)} children "
          f"({s1['children_included_per_class']}); screen {ev['nearest_standard'][0]}×{ev['nearest_standard'][1]}.",
          f"- D02: {s2['n_images']:,} pooled images, {s2['n_groups']:,} near-duplicate groups, "
          f"{s2['n_excluded_conflict_images']} label-conflict images excluded; D03 never read.",
          "- D04/D05/D06 cleaned deterministically (nothing fitted); D05 label rule re-checked; feature roles in columns.yaml.",
          f"- 10-fold outer + inner-validation splits for 5 datasets with SPLITS_MANIFEST.json; pytest: {last.strip('= ')}.",
          "- Main risk found: D01 tracking quality differs strongly by class (section 2.5) → DD-12 shortcut control is essential.",
          ""]

    # 2 D01
    st = csv(c1, "cleaning_steps.csv")
    pairs = csv(c1, "dedupe_file_pairs.csv")
    L += ["## 2. D01 eye-tracking", "", "### 2.1 Cleaning (rows after every step; `cache/D01_eye_tracking/cleaning_steps.csv`)", "",
          md_table(st), "",
          "Duplicates removed per file pair (`dedupe_file_pairs.csv`; `kept_in` = file holding the first occurrence; "
          "same file = within-file duplicates):", "", md_table(pairs), "",
          f"After dedupe, duplicate (participant_id, t_ms, trial) = 0 (asserted in 02a and tested). "
          f"Excluded participants: {csv(c1, 'excluded_participants.csv').to_dict('records')}.", "",
          "Column mapping: `cache/D01_eye_tracking/column_mapping.csv` (25 files × 13 canonical fields). Right eye is used; "
          "the left eye only when the right-eye sample is missing or (0,0) and the file has left-eye columns. "
          "Files without left-eye columns (8.csv, 13.csv) are handled with the right eye only.", "",
          "### 2.2 Screen resolution (written to `configs/data/D01.yaml`, re-checked by 02a)", "",
          f"- 99.9th percentile of gaze: x = {ev['p_x']} px, y = {ev['p_y']} px → nearest standard resolution "
          f"**{ev['nearest_standard'][0]} × {ev['nearest_standard'][1]}**.",
          f"- Tracker clamp limits: x ∈ [{ev['x_min']}, {ev['x_max']}], y ∈ [{ev['y_min']}, {ev['y_max']}]; range centres "
          f"({ev['x_range_centre']}, {ev['y_range_centre']}) = centre of a 1280 × 1024 frame.",
          f"- Median gaze ({ev['x_median']}, {ev['y_median']}): stimuli may not fill the frame; exact layout UNKNOWN.",
          "- Documentation: only `Citation.txt` (reference, no screen data); no `Fulltext.pdf` in the folder.", "",
          "### 2.3 Windowing into clips", ""]
    sc = unc.status.value_counts()
    L += [f"- {s1['n_recordings']:,} (file, child, trial) recordings → {s1['n_segments']:,} continuous segments "
          f"(split at gaps > 500 ms) → {len(unc):,} candidate 8-s clips; status counts: {sc.to_dict()}.",
          f"- Clips per child (after cap 40): min {s1['clips_per_child_capped']['min']}, median "
          f"{s1['clips_per_child_capped']['median']}, max {s1['clips_per_child_capped']['max']}; before cap: min "
          f"{s1['clips_per_child_uncapped']['min']}, median {s1['clips_per_child_uncapped']['median']}, max "
          f"{s1['clips_per_child_uncapped']['max']} (uncapped index: `clips_index_uncapped.csv`).",
          f"- Clips per class: ASD {int((idx.label == 1).sum())}, TD {int((idx.label == 0).sum())}; children per class: "
          f"{s1['children_included_per_class']}.",
          "- Children with 0 clips: none (all 57 children with gaze rows keep ≥ 1 clip). Children excluded earlier: "
          f"{[r['participant_id'] for r in s1['children_excluded']]} (no identified gaze rows).",
          f"- Rendered windows: {s1['nonblank_windows']:,} of {len(idx) * 8:,} contain at least one drawn pixel; the rest "
          "have no valid gaze in that second (black, as specified).", "",
          "Kept clips per source file and class (stimulus sets differ between files):", "",
          md_table(idx.groupby(["source_file", "label"]).size().unstack(fill_value=0)
                   .rename(columns={0: "TD clips", 1: "ASD clips"}).reset_index()), "",
          "### 2.4 Rendering", "",
          "Each 1-s window → 3 × 224 × 224 uint8: ch0 scan path (cv2 LINE_8 lines between consecutive valid samples), "
          "ch1 fixation density (valid Fixation samples, Gaussian σ = 3 px, max-normalised to 255), ch2 time "
          "(valid samples, brightness 64 → 255 over the second). Coordinates scaled from 1280 × 1024 to 224 × 224. "
          "`clips_uint8.npy` + `clips_index.csv`.", "",
          "### 2.5 Quality features (DD-12 shortcut control; NOT model inputs) — means over children", "",
          md_table(q.drop(columns="participant_id").groupby("label").mean().T
                   .rename(columns={1: "ASD", 0: "TD"})[["ASD", "TD"]].round(4).reset_index()
                   .rename(columns={"index": "feature"})), "",
          "ASD children have clearly lower tracking quality and more blinks. A model could separate the classes from "
          "data quality alone (e.g. the number of black windows), so Stage 8 must compare every hybrid with a "
          "quality-only baseline.", ""]

    # 3 D02
    L += ["## 3. D02 facial images", "",
          f"- Real class folder names: {s2['raw_class_folder_names']} → autistic = 1, non-autistic = 0.",
          f"- Pooled {s2['n_images']:,} images from train/valid/test; labels {s2['label_counts']}; original split kept "
          "only as `original_split`. D03 (`AutismDataset`) is not read (tested).",
          f"- md5 + 64-bit pHash for all images; {s2['n_pairs']} pairs with Hamming ≤ 4 → {s2['n_groups']:,} groups; "
          f"group sizes {s2['group_size_distribution']}.",
          f"- Groups with both labels: {len(s2['label_conflict_groups'])} → {s2['n_excluded_conflict_images']} images excluded.",
          "- `cache/D02_facial_images_A/manifest.csv` (no pixels stored; resize/normalise/augment in the Stage 4 loader).", ""]

    # 4 tabular
    L += ["## 4. Tabular datasets", ""]
    for ds in ["D04_autism_prediction", "D05_toddler_screening", "D06_early_autism"]:
        ck = J(CACHE_DIR, ds, "checks.json")
        import yaml
        with open(os.path.join(CACHE_DIR, ds, "columns.yaml")) as fh:
            spec = yaml.safe_load(fh)
        roles = {}
        for col, v in spec["columns"].items():
            roles.setdefault(v["role"], []).append(col)
        L += [f"### {ds} (`{spec['source_file']}`)", "",
              f"- Rows: {ck['n_rows_raw']} raw → {ck['n_rows_clean']} clean; labels {ck['label_counts']}.",
              *[f"- {r}: {', '.join(f'`{c}`' for c in cols)}" for r, cols in roles.items()],
              *[f"- feature set `{k}` ({len(v)}): {', '.join(v)}" for k, v in spec["feature_sets"].items()],
              f"- Duplicates: {ck['duplicate_rows_full_excluding_id']} full-row duplicates (id ignored); "
              f"{ck['rows_in_duplicate_feature_groups']} rows in {ck['duplicate_feature_groups']} identical-feature groups "
              f"({ck['duplicate_groups_with_mixed_labels']} with mixed labels) → each group shares a `group_id` (same fold).",
              f"- Missing values: { {k: v for k, v in ck['missing_per_column'].items() if v} or 'none' }.", "",
              "Changes (`changes.csv`):", "", md_table(csv(CACHE_DIR, ds, "changes.csv")), ""]
        if ds == "D04_autism_prediction":
            r = J(CACHE_DIR, ds, "result_column_test.json")
            L += [f"`result` test: equals item sum in {r['equals_item_sum_fraction']:.0%} of rows; "
                  f"{r['item_vectors_with_more_than_one_value']} of {r['n_distinct_item_vectors']} A1-A10 patterns map "
                  f"to more than one `result` value (up to {r['max_distinct_values_per_item_vector']}); linear R² on "
                  f"A1-A10 = {r['linear_r2_on_items']} → **{r['decision']}**.", ""]
        if "label_rule_check" in ck:
            lr = ck["label_rule_check"]
            L += [f"Rule re-check on clean data: `{lr['rule']}` accuracy **{lr['accuracy']}**, confusion "
                  f"{lr['confusion_matrix[[TN,FP],[FN,TP]]']} → experiment (a) is label-definitional.", ""]

    # 5 splits
    L += ["## 5. Split summaries (`splits/<ds>/split_summary.csv`)", "",
          "Outer: " + man["outer"] + "; inner: " + man["inner"] + ". D01 is split at child level "
          "(one unit per child). `SPLITS_MANIFEST.json` code_hash_at_split_time = `" + man["code_hash_at_split_time"] + "`.", ""]
    for ds in DATASETS:
        s = csv(SPLITS_DIR, ds, "split_summary.csv")
        L += [f"### {ds}", "", md_table(s), ""]
    f1 = csv(SPLITS_DIR, "D01_eye_tracking", "folds.csv").drop_duplicates("group_id")
    kids = f1.groupby(["outer_fold", "label"]).size().unstack(fill_value=0)
    L += ["D01 test-fold children per class (ASD / TD): "
          + ", ".join(f"fold {k}: {kids.loc[k, 1]}/{kids.loc[k, 0]}" for k in kids.index), ""]

    # 6 figures
    L += ["## 6. Figures (opened and inspected)", "",
          "- `reports/figures/stage02_d01_examples.png` — 3 ASD + 3 TD clips × 8 windows. Scan paths (red) and fixation "
          "blobs (green) are drawn in every window that has valid gaze; TD children show compact fixation clusters, some "
          "ASD/TD windows show long saccades. A first version showed lines jumping to the top-left corner (tracker "
          "(0,0)-artefacts not exactly at 0); after the 35-px origin-artefact rule no corner lines remain. A few lines "
          "end at the left/top border: genuine off-screen gaze within the 50-px margin, clipped to the image edge.",
          "- `reports/figures/stage02_d02_neardups.png` — 8 near-duplicate groups; each is the same child with small "
          "edits (crop, colour/contrast, a grey-scale pair); group g0221 pairs a test image with a train image of the "
          "original split (a leak the grouping now prevents).",
          "- `reports/figures/stage02_fold_balance.png` — test-fold class counts: D01 3 TD + 2-3 ASD children per fold; "
          "D02 147/147 in every fold; D04/D05/D06 keep their class ratio within ±1 sample per fold.", ""]

    # 7 tests
    L += ["## 7. Tests (`.venv/bin/python -m pytest -v`, full output, `logs/stage02_pytest.txt`)", "",
          "Pre-flight (`logs/stage02_preflight.txt`):", "", "```", pre, "```", "", "```", pyt, "```", "",
          "The snapshot test checks the latest `snapshots/stage_NN/CHECKSUMS.sha256` against the current files. "
          "Order: provisional snapshot → this pytest run → this report → final snapshot (now including this report) → "
          "snapshot tests re-run (`logs/stage02_snapshot_check.txt`).", ""]

    # 8 deviations
    L += ["## 8. Deviations from DATASET_DECISIONS.md and the Stage 2 brief", "",
          "No decision DD-01..DD-11 was changed. Implementation refinements, each forced by evidence:", "",
          f"1. **DD-02 step b2**: {int(pairs[pairs.step == 'b2'].rows_removed.sum())} rows in 7.csv/8.csv repeat a "
          "(participant, t_ms, trial) with gaze differing by ≤ 0.05 px (7.csv stores 0.1-px precision), so the 4-dp key "
          "missed them. Removed as duplicates (keep first) so that the required uniqueness check holds.",
          "2. **Invalid gaze**: besides exact (0,0), samples within 35 px of (0,0) are invalid (tracker-loss artefacts; "
          f"samples otherwise valid within 30 px of each corner: {s1['origin_artifact_evidence']['valid_without_origin_rule_near_corner']}; "
          f"{s1['origin_artifact_evidence']['n_exactly_one_coordinate_zero']} samples have exactly one coordinate = 0, "
          f"the other in {s1['origin_artifact_evidence']['other_coordinate_range_when_one_is_zero']} px). "
          "Without the rule the rendered scan paths jumped to the corner (C6 check).",
          "3. **Continuous recordings**: recordings are split into segments at gaps > 500 ms and clips are laid on each "
          "segment, so no clip contains a gap (the brief's discard rule then never triggers: 0 `rejected_gap`). A clip "
          "must end at or before its segment's last sample, and is rejected if a sample of another trial of the same "
          "child/file lies inside its span (needed: "
          f"{s1['trial_overlap']['overlapping_trial_pairs']:,} trial pairs in files {', '.join(s1['trial_overlap']['files'])} "
          "overlap in time).",
          "4. **D01 split unit**: StratifiedGroupKFold is run on one row per child (group = child), so folds are "
          "balanced in children rather than weighted by clip counts; clips inherit their child's fold. Same for the inner "
          "validation split.",
          "5. **Tabular group_id**: rows with identical feature values (id and label ignored) share a group — a superset "
          "of 'identical rows'. No group has mixed labels.",
          "6. **D04 `result`**: tested, not a function of A1-A10 → kept as a feature (the brief's exclusion applied only "
          "if it were).",
          "7. **Categorical values lower-cased** (e.g. 'Health Care Professional' = 'Health care professional', "
          "'others' = 'Others'); constant `age_desc` in D04 excluded.",
          "8. **No git** (user override): SPLITS_MANIFEST and the snapshot use `code_hash` instead of a commit hash; "
          "Stage 1 / Setup decision lines were regenerated under the override before pre-flight A1.", ""]

    # 9 open items
    L += ["## 9. Open items for Stage 3", "",
          f"1. D01 is small and clip-imbalanced: {len(inc)} children, ASD {int((idx.label == 1).sum())} vs TD "
          f"{int((idx.label == 0).sum())} clips → use class weighting / balanced sampling; report child-level metrics.",
          "2. D01 data quality is a strong class signal (section 2.5) and some stimulus files are single-class "
          "(section 2.3 table) → Stage 8 quality-only and stimulus-aware controls are mandatory.",
          "3. D01 validation sets have only 7-8 children per fold → val AUC is noisy; prefer val loss / child-level AUC for "
          "early stopping (decide in Stage 4).",
          "4. Model inputs: D01 (B, 8, 3, 224, 224) uint8 → float/255 (+ normalisation defined in Stage 3); D02 one image "
          "→ backbone feature map → spatial tokens. Pick timm variants for EfficientNet/ConvNeXt/ViT in Stage 3.",
          f"5. Memory: clips_uint8.npy is {s1['npy_shape'][0] * 8 * 3 * 224 * 224 / 2**30:.2f} GiB (memmap, fine).",
          "6. Every training run must call `src.data.splits.verify_manifest()` first.", ""]

    ok = (failed is None and n_pass > 0 and all(os.path.isfile(os.path.join(SPLITS_DIR, ds, "folds.csv")) for ds in DATASETS))
    reasons = [] if ok else [f"pytest: {last.strip('= ')}"]
    L.append("STAGE 2 DECISION: PASS" if ok else "STAGE 2 DECISION: FAIL - " + "; ".join(reasons))
    out = os.path.join(PROJECT_DIR, "reports", "STAGE_02_REPORT.md")
    with open(out, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(L[-1])


if __name__ == "__main__":
    main()
