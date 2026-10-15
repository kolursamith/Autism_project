# Dataset decisions (Stage 1)

Every number below was loaded from the audit outputs, never typed from memory. The printout of these values is in
`logs/stage01_facts.txt`. Sources: `data_registry/<id>_audit.json` (from `scripts/01_audit_datasets.py`),
`data_registry/D01_dedupe_key_check.json` (`scripts/01_check_dd02_key.py`),
`data_registry/D01_cv_feasibility.json` (`scripts/01_check_cv_feasibility.py`).
Raw data root: `/home/score/Documents/AI_project/dataset`.

**No CONTRADICTION was found.** All 12 decisions are kept as specified. DD-02 adds an implementation requirement
that the evidence calls for, and does not change the decision.

---

### DD-01 - D01 eye-tracking is the PRIMARY dataset for the 10 hybrid architectures
- Unit of analysis is the **participant (child)**.
- Rows whose participant is "Unidentified" are dropped.
- Participants without gaze rows are excluded.

Evidence (`D01_eye_tracking_audit.json`):
- `total_gaze_rows` = 2,252,943 in 25 CSV files.
- `non_numeric_participant_values_rows` = {Unidentified(Neg): 878,362; Unidentified(Pos): 23,424}, a total of
  901,786 rows (40.0%) with no child ID. These rows cannot be assigned to a child or a label, so they are dropped.
- `metadata_n_participants` = 59 ({TD: 30, ASD: 29}).
- `participants_with_metadata_and_gaze` = 57 ({ASD: 27, TD: 30}).
- `participants_metadata_but_no_gaze` = **[12, 16]**, both ASD. These two are excluded.
- `participant_ids_gaze_but_no_metadata` = [].
- The analysis set is therefore **57 children (27 ASD, 30 TD)**.

### DD-02 - D01 duplicated rows across files are removed in Stage 2 BEFORE any clip or window is created
- Dedupe key = participant + recording timestamp + trial + gaze coordinates.
- Rows before and after dedupe must be reported.

Evidence:
- `D01_eye_tracking_audit.json` → `duplicate_check_dd02_key.file_pairs_sharing_keys` lists 10 file pairs that share
  rows. The largest:

  | file pair | shared keys | rows in each file |
  |---|---|---|
  | `8.csv` / `11.csv` | 114,285 | 114,285 and 114,285 (the same content, despite different column layouts: 39 vs 52 columns) |
  | `10.csv` / `25.csv` | 51,828 | 51,829 and 119,064 |
  | `1.csv` / `16.csv` | 34,630 | 34,630 and 110,967 (file 1 is entirely inside file 16) |
  | `21.csv` / `22.csv` | 24,266 | |
  | `21.csv` / `24.csv` | 19,822 | |
  | `14.csv` / `21.csv` | 10,469 | |
  | `4.csv` / `21.csv` | 9,728 | |
  | `5.csv` / `21.csv` | 104 | |
  | `9.csv` / `21.csv` | 93 | |
  | `18.csv` / `21.csv` | 93 | |

- `file_md5_identical_groups` = []. No two files are byte-identical; the duplication is in their content.
- `D01_dedupe_key_check.json`, identified rows only, using the key with numeric parsing: **1,351,157 rows before →
  1,103,660 after** (247,497 removed, 18,490 of them duplicates within the same file).

**Implementation requirement (from the evidence):** Stage 2 must compare the key **numerically**, not as text:
RecordingTime rounded to 3 decimal places, Point of Regard X/Y rounded to 4, and "-" treated as missing. Comparing
the raw strings removes only 205,275 identified rows (1,145,882 remain), which misses 42,222 true duplicates.

### DD-03 - D01 CARS Score is never a feature, filter or stratification variable
- From the metadata, only ParticipantID and Class are used.
- Age and Gender are kept for reporting only.

Evidence: `D01_eye_tracking_audit.json` → `cars_missing_by_class` = {ASD: 0 missing of 29; TD: 30 missing of 30}.
Whether CARS is present identifies the class perfectly, so it encodes the label.

### DD-04 - D01 10-fold CV = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
- Folds are grouped by participant and stratified by Class.
- Every test fold must contain at least one ASD child and one TD child.

Evidence: `D01_cv_feasibility.json` (a feasibility check only; no manifest was saved).
- Over the 57 children with gaze rows, every test fold has 5–6 children: 2–3 ASD and 3 TD.
- `every_fold_has_asd_and_td` = true.
- Stage 2 must re-assert this on the saved manifest.

### DD-05 - D02 facial images (DFDK folder) is the SECONDARY dataset for the 10 hybrid architectures
- The original train/valid/test folders are pooled (2,940 images) and re-split by 10-fold CV.

Evidence: `D02_facial_images_A_audit.json`
- `count_per_split_x_class`: train 1,193/1,193, valid 127/127, test 150/150, which gives 2,940 images
  (1,470 autistic, 1,470 non-autistic).
- `n_unreadable` = 0.

Two things Stage 2 must handle:
- The test class folder is named `Non-Autistic` while train and valid use `Non_Autistic`
  (`class_folder_names_per_split(raw)`). Both normalise to `nonautistic`.
- The 10 PNG files in the folder root (`CF.png`, `roc.png`, …) are result figures from the original paper and
  are not samples.

Full folder name: `Autism Spectrum Disorder Detection using Facial Traits Dual Framework DenseRes with K-Fold Cross
Validation (DFDK) for Enhanced Classification Accuracy`.

### DD-06 - D03 AutismDataset is EXCLUDED from training: it is a copy of D02
Evidence: `D02_facial_images_A_audit.json` → `cross_dataset_duplicates`
- `unique_md5_shared` = **2,541** byte-identical images.
- **399** D03 images are re-saved copies (not byte-identical):
  - nearest pHash distance is 0 for 398 of them and 2 for 1;
  - all 399 have the same pixel size as their D02 match;
  - their nearest D02 matches are 300 test images and 99 valid images.
- All 2,940 of 2,940 D03 images have a pHash ≤ 4 match in D02, and all 2,940 have the same class.

Using both datasets would put the same face in train and test.

### DD-07 - D02 near-duplicates are grouped, and CV uses StratifiedGroupKFold on those groups
- Near-duplicate images (pHash Hamming distance ≤ 4) are merged into one group (connected components).
- Near-duplicates therefore never cross folds.
- D02 has **no subject IDs**, so the same child may appear in different photos across folds. This is a documented
  limitation: results are image-level, not child-level.

Evidence: `D02_facial_images_A_audit.json` → `near_duplicates_phash`
- 11 pairs, forming 11 connected groups of 2.
- 6 pairs fall within one original split and 5 cross splits ({train/valid: 2, test/train: 3}).
- 0 pairs cross classes.
- `exact_duplicates_md5.n_groups` = 0.
- This gives 2,940 − 11 = 2,929 CV groups.
- `subject_ids.finding`: the filenames are only `<class>.<index>.jpg` or `<index>.jpg`, and the folder has no
  metadata files.

### DD-08 - D04 autism prediction: only the 800 labelled train rows are used
- test.csv (no labels) is not used.
- Tabular baselines only (LR, RF, GB, MLP), with 10-fold StratifiedKFold. D04 is not used for the CNN/ViT hybrids.

Evidence: `D04_autism_prediction_audit.json`
- `train.csv` shape [800, 22], label `Class/ASD` = {0: 639, 1: 161}, so 20.1% positive.
- `test.csv` shape [200, 21], `label_columns` = [], so it is unlabelled.
- No exact leakage: the best single rule is `sum(A1_Score..A10_Score) >= 10`, with accuracy 0.8625 against a
  majority baseline of 0.79875.
- `'?'` placeholders appear in train (ethnicity 203, relation 40). These are missing values and must be treated
  as such.

### DD-09 - D05 toddler screening: the label is defined by the A1–A10 score
Any model that sees A1–A10 is reproducing a formula, not predicting autism. Two tabular experiments:
- **(a)** all features, reported explicitly as "label-definitional / upper bound";
- **(b)** demographic and history features only, with A1–A10 and `Qchat-10-Score` removed. That leaves
  `Age_Mons, Sex, Ethnicity, Jaundice, Family_mem_with_ASD, Who completed the test`; `Case_No` is a row ID and
  is dropped.

The Combined file is EXCLUDED. Tabular baselines only.

Evidence (`D05_toddler_screening_audit.json`):
- `Toddler Autism dataset July 2018.csv` (1,054 rows; Yes 728 / No 326):
  - The rule **`ASD if sum(A1..A10) >= 4`** has accuracy **1.0**, confusion [[326, 0], [0, 728]].
  - `item_sum_equals_Qchat-10-Score`: 1,054 of 1,054 rows equal.
  - The dataset's own description (`D05_toddler_screening_licence_citation.txt`, copied from
    `Toddler data description.docx`) says (verbatim, including its typo): "the class varaible was assigned automatically based on the score obtained by
    the user".
- `Autism_Screening_Data_Combined.csv` (6,075 rows) is excluded:
  - `cross_file_embedded_blocks`: all 1,054 Toddler rows appear, in order, at rows **5021–6074**, compared on
    A1–A10 and Sex. They carry the label `NO` for **1,054 of 1,054** rows, while the original labels are
    326 No / 728 Yes, so **728 labels conflict**.
  - The file also has 847 fully duplicated rows.
  - Within the other age bands the label is exactly `item_sum >= 7`: accuracy 1.0 on 5,021 of 6,075 rows.
- The Toddler file has 79 rows that are duplicates once Case_No is ignored. Stage 2 must report them.

### DD-10 - D06 early autism: tabular baselines only, 10-fold StratifiedKFold; the empty column is dropped
Evidence: `D06_early_autism_audit.json`
- `asd.csv` shape [1000, 11], `Diagnosed_ASD` = {No: 589, Yes: 411}.
- `fully_empty_columns` = **[`Unnamed: 10`]**, missing in 1,000 of 1,000 rows.
- `Child_ID` is unique per row and is not a feature.
- No leakage: the best single rule is `Age >= 35`, with accuracy **0.601**, against a majority baseline of
  **0.589**.

### DD-11 - No datasets are merged, and no predictions are combined across children or datasets
Each dataset is trained and reported separately.

Evidence: D02 and D03 are the same images (DD-06), and the D05 Combined file is a relabelled merge (DD-09). Both
show how merging creates leakage and label conflicts. The datasets also do not share subjects or ID schemes:
- D01 uses ParticipantID 1–59.
- D02 has no IDs.
- D04, D05 and D06 have only per-row IDs.

### DD-12 - Shortcut control (in Stage 8): tracking quality as a baseline for D01; near-duplicate grouping effect for D02
- For D01, the per-child tracking-quality ratio (fraction of invalid or lost samples) is evaluated alone as a
  baseline classifier.
- Every hybrid model must be compared against it.

Evidence: `D01_eye_tracking_audit.json`
- `tracking_ratio_column_in_files` = 25 of 25 (`Tracking Ratio [%]` is present in every gaze file), and
  `Category Right` (e.g. Fixation, Blink, Separator) is present in all 9 column layouts.
- Sampling is a median of 16.7 ms between samples (≈59.9 Hz), with p95 at 44.0 ms, so dropped samples are
  measurable.
- Whether tracking quality differs by class is **UNKNOWN** at Stage 1. Measuring it is the purpose of DD-12.

D02 part (added in the setup session, 2026-10-13, to match the project brief):
- Stage 8 reports the effect of the DD-07 near-duplicate grouping on D02, i.e. results with grouped folds versus
  the same models on ungrouped folds.
- Evidence: `D02_facial_images_A_audit.json` → `near_duplicates_phash` = 11 pairs (11 groups of 2), 5 of them across
  the original splits. With so few pairs the effect may be small; its size is **UNKNOWN** until Stage 8 measures it.
