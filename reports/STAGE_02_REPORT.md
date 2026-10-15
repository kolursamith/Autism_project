# STAGE 2 REPORT - data preparation, 10-fold split manifests and leakage tests

Project: Reliability-Weighted Multimodal AI for Early Autism Risk Screening in Children (BITE308L). Screening-support research, not diagnosis. Seed 42. No model was trained in this stage. No git (user override 2026-10-13): this stage is recorded by the local snapshot `snapshots/stage_02/`.

## 1. Summary

- D01: 2,252,943 raw gaze rows → 1,103,427 clean rows → 1,241 8-s clips (uint8 (1241, 8, 3, 224, 224)) from 57 children ({'ASD': 27, 'TD': 30}); screen 1280×1024.
- D02: 2,940 pooled images, 2,929 near-duplicate groups, 0 label-conflict images excluded; D03 never read.
- D04/D05/D06 cleaned deterministically (nothing fitted); D05 label rule re-checked; feature roles in columns.yaml.
- 10-fold outer + inner-validation splits for 5 datasets with SPLITS_MANIFEST.json; pytest: 169 passed in 33.44s.
- Main risk found: D01 tracking quality differs strongly by class (section 2.5) → DD-12 shortcut control is essential.

## 2. D01 eye-tracking

### 2.1 Cleaning (rows after every step; `cache/D01_eye_tracking/cleaning_steps.csv`)

| step | rows_before | rows_after | removed |
|---|---|---|---|
| a. drop 'Unidentified' / missing participant | 2252943 | 1351157 | 901786 |
| b. remove duplicate rows (DD-02 key, keep first) | 1351157 | 1103660 | 247497 |
| b2. remove same (participant, t_ms, trial) with gaze within 0.05 px (export precision) | 1103660 | 1103427 | 233 |
| c. keep participants present in metadata | 1103427 | 1103427 | 0 |
| d. join label (ASD=1, TD=0); only id + class read from metadata (DD-03) | 1103427 | 1103427 | 0 |
| e. mark invalid gaze (rows kept): 243986 rows invalid | 1103427 | 1103427 | 0 |

Duplicates removed per file pair (`dedupe_file_pairs.csv`; `kept_in` = file holding the first occurrence; same file = within-file duplicates):

| kept_in | removed_from | rows_removed | step |
|---|---|---|---|
| 8.csv | 11.csv | 46894 | b |
| 10.csv | 25.csv | 46763 | b |
| 21.csv | 22.csv | 35719 | b |
| 21.csv | 24.csv | 30306 | b |
| 1.csv | 16.csv | 29865 | b |
| 14.csv | 21.csv | 22200 | b |
| 4.csv | 21.csv | 16376 | b |
| 17.csv | 17.csv | 9483 | b |
| 23.csv | 23.csv | 3109 | b |
| 20.csv | 20.csv | 2950 | b |
| 25.csv | 25.csv | 2943 | b |
| 18.csv | 21.csv | 297 | b |
| 5.csv | 21.csv | 262 | b |
| 9.csv | 21.csv | 247 | b |
| 7.csv | 11.csv | 39 | b |
| 7.csv | 8.csv | 39 | b |
| 21.csv | 21.csv | 5 | b |
| 7.csv | 8.csv | 233 | b2 |

After dedupe, duplicate (participant_id, t_ms, trial) = 0 (asserted in 02a and tested). Excluded participants: [{'participant_id': 12, 'reason': 'in metadata but no identified gaze rows'}, {'participant_id': 16, 'reason': 'in metadata but no identified gaze rows'}].

Column mapping: `cache/D01_eye_tracking/column_mapping.csv` (25 files × 13 canonical fields). Right eye is used; the left eye only when the right-eye sample is missing or (0,0) and the file has left-eye columns. Files without left-eye columns (8.csv, 13.csv) are handled with the right eye only.

### 2.2 Screen resolution (written to `configs/data/D01.yaml`, re-checked by 02a)

- 99.9th percentile of gaze: x = 1352.7 px, y = 1111.5 px → nearest standard resolution **1280 × 1024**.
- Tracker clamp limits: x ∈ [-126.8, 1407.1], y ∈ [-101.9, 1126.2]; range centres (640.2, 512.2) = centre of a 1280 × 1024 frame.
- Median gaze (627.9, 377.8): stimuli may not fill the frame; exact layout UNKNOWN.
- Documentation: only `Citation.txt` (reference, no screen data); no `Fulltext.pdf` in the folder.

### 2.3 Windowing into clips

- 2,789 (file, child, trial) recordings → 3,093 continuous segments (split at gaps > 500 ms) → 1,797 candidate 8-s clips; status counts: {'kept': 1241, 'rejected_valid_ratio': 368, 'dropped_by_cap': 188}.
- Clips per child (after cap 40): min 3, median 18.0, max 40; before cap: min 3, median 18.0, max 86 (uncapped index: `clips_index_uncapped.csv`).
- Clips per class: ASD 437, TD 804; children per class: {'ASD': 27, 'TD': 30}.
- Children with 0 clips: none (all 57 children with gaze rows keep ≥ 1 clip). Children excluded earlier: [12, 16] (no identified gaze rows).
- Rendered windows: 9,521 of 9,928 contain at least one drawn pixel; the rest have no valid gaze in that second (black, as specified).

Kept clips per source file and class (stimulus sets differ between files):

| source_file | TD clips | ASD clips |
|---|---|---|
| 1.csv | 31 | 28 |
| 10.csv | 71 | 41 |
| 12.csv | 68 | 48 |
| 13.csv | 17 | 10 |
| 14.csv | 0 | 27 |
| 15.csv | 22 | 0 |
| 16.csv | 40 | 23 |
| 17.csv | 96 | 32 |
| 2.csv | 26 | 4 |
| 21.csv | 87 | 47 |
| 22.csv | 34 | 0 |
| 23.csv | 48 | 16 |
| 24.csv | 6 | 7 |
| 25.csv | 19 | 21 |
| 3.csv | 56 | 11 |
| 4.csv | 19 | 21 |
| 5.csv | 28 | 9 |
| 7.csv | 30 | 20 |
| 8.csv | 51 | 14 |
| 9.csv | 55 | 58 |

### 2.4 Rendering

Each 1-s window → 3 × 224 × 224 uint8: ch0 scan path (cv2 LINE_8 lines between consecutive valid samples), ch1 fixation density (valid Fixation samples, Gaussian σ = 3 px, max-normalised to 255), ch2 time (valid samples, brightness 64 → 255 over the second). Coordinates scaled from 1280 × 1024 to 224 × 224. `clips_uint8.npy` + `clips_index.csv`.

### 2.5 Quality features (DD-12 shortcut control; NOT model inputs) — means over children

| feature | ASD | TD |
|---|---|---|
| n_clips | 16.1852 | 26.8 |
| mean_valid_ratio | 0.8427 | 0.9313 |
| mean_blink_ratio | 0.1289 | 0.0498 |
| tracking_ratio_before_filtering | 59.3765 | 83.8271 |
| valid_fraction_before_filtering | 0.6537 | 0.8803 |

ASD children have clearly lower tracking quality and more blinks. A model could separate the classes from data quality alone (e.g. the number of black windows), so Stage 8 must compare every hybrid with a quality-only baseline.

## 3. D02 facial images

- Real class folder names: {'test': ['Autistic', 'Non-Autistic'], 'train': ['Autistic', 'Non_Autistic'], 'valid': ['Autistic', 'Non_Autistic']} → autistic = 1, non-autistic = 0.
- Pooled 2,940 images from train/valid/test; labels {'0': 1470, '1': 1470}; original split kept only as `original_split`. D03 (`AutismDataset`) is not read (tested).
- md5 + 64-bit pHash for all images; 11 pairs with Hamming ≤ 4 → 2,929 groups; group sizes {'1': 2918, '2': 11}.
- Groups with both labels: 0 → 0 images excluded.
- `cache/D02_facial_images_A/manifest.csv` (no pixels stored; resize/normalise/augment in the Stage 4 loader).

## 4. Tabular datasets

### D04_autism_prediction (`train.csv`)

- Rows: 800 raw → 800 clean; labels {'0': 639, '1': 161}.
- id: `sample_id`, `group_id`, `ID`
- feature_numeric: `A1_Score`, `A2_Score`, `A3_Score`, `A4_Score`, `A5_Score`, `A6_Score`, `A7_Score`, `A8_Score`, `A9_Score`, `A10_Score`, `age`, `result`
- feature_categorical: `gender`, `ethnicity`, `jaundice`, `austim`, `contry_of_res`, `used_app_before`, `relation`
- label: `label`
- excluded: `age_desc`, `Class/ASD`
- feature set `set_all` (19): A1_Score, A2_Score, A3_Score, A4_Score, A5_Score, A6_Score, A7_Score, A8_Score, A9_Score, A10_Score, age, result, gender, ethnicity, jaundice, austim, contry_of_res, used_app_before, relation
- Duplicates: 0 full-row duplicates (id ignored); 0 rows in 0 identical-feature groups (0 with mixed labels) → each group shares a `group_id` (same fold).
- Missing values: {'ethnicity': 203, 'relation': 40}.

Changes (`changes.csv`):

| column | change | rows_affected |
|---|---|---|
| age_desc | column dropped: constant ('18 and more' in all rows) - carries no information | 800 |
| ethnicity | trimmed surrounding whitespace | 97 |
| ethnicity | missing token ['?'] -> missing | 203 |
| ethnicity | lower-cased values {'White-European': 'white-european', 'Middle Eastern': 'middle eastern', 'Pasifika': 'pasifika', 'Black': 'black', 'Others': 'others', 'Hispanic': 'hispanic', 'Asian': 'asian', 'Turkish': 'turkish', 'South Asian': 'south asian', 'Latino': 'latino'} | 594 |
| contry_of_res | lower-cased values {'Austria': 'austria', 'India': 'india', 'United States': 'united states', 'South Africa': 'south africa', 'Jordan': 'jordan', 'United Kingdom': 'united kingdom', 'Brazil': 'brazil', 'New Zealand': 'new zealand', 'Canada': 'canada', 'Kazakhstan': 'kazakhstan', 'United Arab Emirates': 'united arab emirates', 'Australia': 'australia', 'Ukraine': 'ukraine', 'Iraq': 'iraq', 'France': 'france', 'Malaysia': 'malaysia', 'Viet Nam': 'viet nam', 'Egypt': 'egypt', 'Netherlands': 'netherlands', 'Afghanistan': 'afghanistan', 'Oman': 'oman', 'Italy': 'italy', 'AmericanSamoa': 'americansamoa', 'Bahamas': 'bahamas', 'Saudi Arabia': 'saudi arabia', 'Ireland': 'ireland', 'Aruba': 'aruba', 'Sri Lanka': 'sri lanka', 'Russia': 'russia', 'Bolivia': 'bolivia', 'Azerbaijan': 'azerbaijan', 'Armenia': 'armenia', 'Serbia': 'serbia', 'Ethiopia': 'ethiopia', 'Sweden': 'sweden', 'Iceland': 'iceland', 'Hong Kong': 'hong kong', 'Angola': 'angola', 'China': 'china', 'Germany': 'germany', 'Spain': 'spain', 'Tonga': 'tonga', 'Pakistan': 'pakistan', 'Iran': 'iran', 'Argentina': 'argentina', 'Japan': 'japan', 'Mexico': 'mexico', 'Nicaragua': 'nicaragua', 'Sierra Leone': 'sierra leone', 'Czech Republic': 'czech republic', 'Niger': 'niger', 'Romania': 'romania', 'Cyprus': 'cyprus', 'Belgium': 'belgium', 'Burundi': 'burundi', 'Bangladesh': 'bangladesh'} | 800 |
| relation | missing token ['?'] -> missing | 40 |
| relation | lower-cased values {'Self': 'self', 'Relative': 'relative', 'Parent': 'parent', 'Others': 'others', 'Health care professional': 'health care professional'} | 760 |
| Class/ASD | label ['1'] -> 1, ['0'] -> 0, stored as 'label' | 800 |

`result` test: equals item sum in 0% of rows; 93 of 270 A1-A10 patterns map to more than one `result` value (up to 103); linear R² on A1-A10 = 0.2749 → **kept as a feature (NOT a direct function of A1-A10)**.

### D05_toddler_screening (`Toddler Autism dataset July 2018.csv`)

- Rows: 1054 raw → 1054 clean; labels {'0': 326, '1': 728}.
- id: `sample_id`, `group_id`, `Case_No`
- feature_numeric: `A1`, `A2`, `A3`, `A4`, `A5`, `A6`, `A7`, `A8`, `A9`, `A10`, `Age_Mons`, `Qchat-10-Score`
- feature_categorical: `Sex`, `Ethnicity`, `Jaundice`, `Family_mem_with_ASD`, `Who completed the test`
- label: `label`
- excluded: `Class/ASD Traits `
- feature set `set_all` (17): A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, Age_Mons, Qchat-10-Score, Sex, Ethnicity, Jaundice, Family_mem_with_ASD, Who completed the test
- feature set `set_honest` (6): Age_Mons, Sex, Ethnicity, Jaundice, Family_mem_with_ASD, Who completed the test
- Duplicates: 79 full-row duplicates (id ignored); 138 rows in 59 identical-feature groups (0 with mixed labels) → each group shares a `group_id` (same fold).
- Missing values: none.

Changes (`changes.csv`):

| column | change | rows_affected |
|---|---|---|
| Ethnicity | lower-cased values {'White European': 'white european', 'Hispanic': 'hispanic', 'Native Indian': 'native indian', 'Others': 'others', 'Latino': 'latino', 'Pacifica': 'pacifica'} | 446 |
| Who completed the test | lower-cased values {'Health Care Professional': 'health care professional', 'Health care professional': 'health care professional', 'Self': 'self', 'Others': 'others'} | 36 |
| Class/ASD Traits  | label ['yes'] -> 1, ['no'] -> 0, stored as 'label' | 1054 |

Rule re-check on clean data: `label == (sum(A1..A10) >= 4)` accuracy **1.0**, confusion [[326, 0], [0, 728]] → experiment (a) is label-definitional.

### D06_early_autism (`asd.csv`)

- Rows: 1000 raw → 1000 clean; labels {'0': 589, '1': 411}.
- id: `sample_id`, `group_id`, `Child_ID`
- feature_numeric: `Age`, `Social_Interaction_Score`, `Communication_Score`, `Repetitive_Behavior_Score`
- feature_categorical: `Gender`, `Jaundice`, `Family_ASD_History`, `Language_Delay`
- label: `label`
- excluded: `Unnamed: 10`, `Diagnosed_ASD`
- feature set `set_all` (8): Age, Social_Interaction_Score, Communication_Score, Repetitive_Behavior_Score, Gender, Jaundice, Family_ASD_History, Language_Delay
- Duplicates: 0 full-row duplicates (id ignored); 0 rows in 0 identical-feature groups (0 with mixed labels) → each group shares a `group_id` (same fold).
- Missing values: none.

Changes (`changes.csv`):

| column | change | rows_affected |
|---|---|---|
| Unnamed: 10 | column dropped: fully empty column (missing in all 1000 rows) | 1000 |
| Gender | lower-cased values {'Male': 'male', 'Female': 'female'} | 1000 |
| Jaundice | lower-cased values {'No': 'no', 'Yes': 'yes'} | 1000 |
| Family_ASD_History | lower-cased values {'No': 'no', 'Yes': 'yes'} | 1000 |
| Language_Delay | lower-cased values {'No': 'no', 'Yes': 'yes'} | 1000 |
| Diagnosed_ASD | label ['yes'] -> 1, ['no'] -> 0, stored as 'label' | 1000 |

## 5. Split summaries (`splits/<ds>/split_summary.csv`)

Outer: StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42); inner: first split of StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42+k) on the training groups of fold k. D01 is split at child level (one unit per child). `SPLITS_MANIFEST.json` code_hash_at_split_time = `fa421378addc1eed72622c9223476192f929117a1decb1a6fc3c235676acf2a5`.

### D01_eye_tracking

| fold | n_train_samples | n_train_groups | train_label_0 | train_label_1 | n_val_samples | n_val_groups | val_label_0 | val_label_1 | n_test_samples | n_test_groups | test_label_0 | test_label_1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 998 | 43 | 644 | 354 | 115 | 8 | 81 | 34 | 128 | 6 | 79 | 49 |
| 2 | 901 | 43 | 613 | 288 | 216 | 8 | 122 | 94 | 124 | 6 | 69 | 55 |
| 3 | 897 | 43 | 607 | 290 | 225 | 8 | 121 | 104 | 119 | 6 | 76 | 43 |
| 4 | 917 | 43 | 596 | 321 | 175 | 8 | 116 | 59 | 149 | 6 | 92 | 57 |
| 5 | 922 | 44 | 555 | 367 | 200 | 8 | 150 | 50 | 119 | 5 | 99 | 20 |
| 6 | 991 | 44 | 611 | 380 | 169 | 8 | 129 | 40 | 81 | 5 | 64 | 17 |
| 7 | 937 | 44 | 611 | 326 | 134 | 7 | 95 | 39 | 170 | 6 | 98 | 72 |
| 8 | 934 | 43 | 618 | 316 | 166 | 8 | 94 | 72 | 141 | 6 | 92 | 49 |
| 9 | 981 | 44 | 645 | 336 | 168 | 7 | 106 | 62 | 92 | 6 | 53 | 39 |
| 10 | 1000 | 44 | 628 | 372 | 123 | 8 | 94 | 29 | 118 | 5 | 82 | 36 |

### D02_facial_images_A

| fold | n_train_samples | n_train_groups | train_label_0 | train_label_1 | n_val_samples | n_val_groups | val_label_0 | val_label_1 | n_test_samples | n_test_groups | test_label_0 | test_label_1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2268 | 2260 | 1134 | 1134 | 378 | 376 | 189 | 189 | 294 | 293 | 147 | 147 |
| 2 | 2268 | 2260 | 1134 | 1134 | 378 | 376 | 189 | 189 | 294 | 293 | 147 | 147 |
| 3 | 2268 | 2260 | 1134 | 1134 | 378 | 376 | 189 | 189 | 294 | 293 | 147 | 147 |
| 4 | 2268 | 2260 | 1134 | 1134 | 378 | 376 | 189 | 189 | 294 | 293 | 147 | 147 |
| 5 | 2268 | 2260 | 1134 | 1134 | 378 | 376 | 189 | 189 | 294 | 293 | 147 | 147 |
| 6 | 2268 | 2260 | 1134 | 1134 | 378 | 376 | 189 | 189 | 294 | 293 | 147 | 147 |
| 7 | 2268 | 2260 | 1134 | 1134 | 378 | 376 | 189 | 189 | 294 | 293 | 147 | 147 |
| 8 | 2268 | 2260 | 1134 | 1134 | 378 | 376 | 189 | 189 | 294 | 293 | 147 | 147 |
| 9 | 2268 | 2261 | 1134 | 1134 | 378 | 376 | 189 | 189 | 294 | 292 | 147 | 147 |
| 10 | 2268 | 2260 | 1134 | 1134 | 378 | 376 | 189 | 189 | 294 | 293 | 147 | 147 |

### D04_autism_prediction

| fold | n_train_samples | n_train_groups | train_label_0 | train_label_1 | n_val_samples | n_val_groups | val_label_0 | val_label_1 | n_test_samples | n_test_groups | test_label_0 | test_label_1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 617 | 617 | 493 | 124 | 103 | 103 | 82 | 21 | 80 | 80 | 64 | 16 |
| 2 | 617 | 617 | 493 | 124 | 103 | 103 | 82 | 21 | 80 | 80 | 64 | 16 |
| 3 | 617 | 617 | 493 | 124 | 103 | 103 | 82 | 21 | 80 | 80 | 64 | 16 |
| 4 | 617 | 617 | 494 | 123 | 103 | 103 | 82 | 21 | 80 | 80 | 63 | 17 |
| 5 | 617 | 617 | 492 | 125 | 103 | 103 | 83 | 20 | 80 | 80 | 64 | 16 |
| 6 | 617 | 617 | 493 | 124 | 103 | 103 | 82 | 21 | 80 | 80 | 64 | 16 |
| 7 | 617 | 617 | 493 | 124 | 103 | 103 | 82 | 21 | 80 | 80 | 64 | 16 |
| 8 | 617 | 617 | 493 | 124 | 103 | 103 | 82 | 21 | 80 | 80 | 64 | 16 |
| 9 | 617 | 617 | 493 | 124 | 103 | 103 | 82 | 21 | 80 | 80 | 64 | 16 |
| 10 | 617 | 617 | 493 | 124 | 103 | 103 | 82 | 21 | 80 | 80 | 64 | 16 |

### D05_toddler_screening

| fold | n_train_samples | n_train_groups | train_label_0 | train_label_1 | n_val_samples | n_val_groups | val_label_0 | val_label_1 | n_test_samples | n_test_groups | test_label_0 | test_label_1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 812 | 751 | 251 | 561 | 136 | 126 | 42 | 94 | 106 | 98 | 33 | 73 |
| 2 | 813 | 752 | 252 | 561 | 136 | 126 | 42 | 94 | 105 | 97 | 32 | 73 |
| 3 | 813 | 753 | 252 | 561 | 136 | 125 | 42 | 94 | 105 | 97 | 32 | 73 |
| 4 | 813 | 752 | 252 | 561 | 136 | 126 | 42 | 94 | 105 | 97 | 32 | 73 |
| 5 | 812 | 752 | 251 | 561 | 136 | 125 | 42 | 94 | 106 | 98 | 33 | 73 |
| 6 | 812 | 751 | 251 | 561 | 136 | 126 | 42 | 94 | 106 | 98 | 33 | 73 |
| 7 | 812 | 751 | 251 | 561 | 136 | 126 | 42 | 94 | 106 | 98 | 33 | 73 |
| 8 | 813 | 753 | 251 | 562 | 136 | 125 | 42 | 94 | 105 | 97 | 33 | 72 |
| 9 | 813 | 752 | 251 | 562 | 136 | 126 | 42 | 94 | 105 | 97 | 33 | 72 |
| 10 | 813 | 751 | 252 | 561 | 136 | 126 | 42 | 94 | 105 | 98 | 32 | 73 |

### D06_early_autism

| fold | n_train_samples | n_train_groups | train_label_0 | train_label_1 | n_val_samples | n_val_groups | val_label_0 | val_label_1 | n_test_samples | n_test_groups | test_label_0 | test_label_1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 772 | 772 | 455 | 317 | 128 | 128 | 75 | 53 | 100 | 100 | 59 | 41 |
| 2 | 771 | 771 | 454 | 317 | 129 | 129 | 76 | 53 | 100 | 100 | 59 | 41 |
| 3 | 771 | 771 | 454 | 317 | 129 | 129 | 76 | 53 | 100 | 100 | 59 | 41 |
| 4 | 771 | 771 | 454 | 317 | 129 | 129 | 76 | 53 | 100 | 100 | 59 | 41 |
| 5 | 771 | 771 | 454 | 317 | 129 | 129 | 76 | 53 | 100 | 100 | 59 | 41 |
| 6 | 771 | 771 | 454 | 317 | 129 | 129 | 76 | 53 | 100 | 100 | 59 | 41 |
| 7 | 771 | 771 | 454 | 317 | 129 | 129 | 76 | 53 | 100 | 100 | 59 | 41 |
| 8 | 772 | 772 | 455 | 317 | 128 | 128 | 75 | 53 | 100 | 100 | 59 | 41 |
| 9 | 772 | 772 | 454 | 318 | 128 | 128 | 76 | 52 | 100 | 100 | 59 | 41 |
| 10 | 771 | 771 | 455 | 316 | 129 | 129 | 76 | 53 | 100 | 100 | 58 | 42 |

D01 test-fold children per class (ASD / TD): fold 1: 3/3, fold 2: 3/3, fold 3: 3/3, fold 4: 3/3, fold 5: 2/3, fold 6: 2/3, fold 7: 3/3, fold 8: 3/3, fold 9: 3/3, fold 10: 2/3

## 6. Figures (opened and inspected)

- `reports/figures/stage02_d01_examples.png` — 3 ASD + 3 TD clips × 8 windows. Scan paths (red) and fixation blobs (green) are drawn in every window that has valid gaze; TD children show compact fixation clusters, some ASD/TD windows show long saccades. A first version showed lines jumping to the top-left corner (tracker (0,0)-artefacts not exactly at 0); after the 35-px origin-artefact rule no corner lines remain. A few lines end at the left/top border: genuine off-screen gaze within the 50-px margin, clipped to the image edge.
- `reports/figures/stage02_d02_neardups.png` — 8 near-duplicate groups; each is the same child with small edits (crop, colour/contrast, a grey-scale pair); group g0221 pairs a test image with a train image of the original split (a leak the grouping now prevents).
- `reports/figures/stage02_fold_balance.png` — test-fold class counts: D01 3 TD + 2-3 ASD children per fold; D02 147/147 in every fold; D04/D05/D06 keep their class ratio within ±1 sample per fold.

## 7. Tests (`.venv/bin/python -m pytest -v`, full output, `logs/stage02_pytest.txt`)

Pre-flight (`logs/stage02_preflight.txt`):

```
# Stage 2 pre-flight 2026-10-13T16:56:50+05:30
A1:
STAGE 1 DECISION: PASS
SETUP DECISION: PASS - all setup items done and verified (37 passed, 37/37 facts CONFIRMED); the git commit requirement was withdrawn by the user on 2026-10-13 and replaced by the local snapshot snapshots/stage_01/
A2: check_gpu
python 3.12.3 | /home/score/Documents/AI_project/asd_hybrid_10fold/.venv/bin/python
torch.__version__ = 2.14.1+cu130
torch.version.cuda = 13.0
torch.backends.cudnn.version() = 92400
torch.cuda.is_available() = True
nvidia driver = 580.178.04
device name = NVIDIA RTX A2000 12GB
total memory = 11.618 GiB
4096x4096 fp32 matmul on cuda: 29.45 ms (4.67 TFLOP/s)
max_memory_allocated = 264.1 MiB
GPU CHECK: OK
check_gpu exit=0
A2: pytest
.....................................                                    [100%]
37 passed in 5.34s
A3: code_hash(src,scripts,configs,tests) before Stage 2 =
5bdabb7a29af177206a347cfc72f1f1c429c3fa692e221b27559df648fbb817c
```

```
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0 -- /home/score/Documents/AI_project/asd_hybrid_10fold/.venv/bin/python
rootdir: /home/score/Documents/AI_project/asd_hybrid_10fold
configfile: pytest.ini
testpaths: tests
plugins: anyio-4.15.1
collecting ... collected 169 items

tests/test_setup.py::test_project_dir_and_symlink PASSED                 [  0%]
tests/test_setup.py::test_paths_yaml_every_path_exists PASSED            [  1%]
tests/test_setup.py::test_datasets_dir_outside_project PASSED            [  1%]
tests/test_setup.py::test_no_absolute_paths_outside_paths_yaml PASSED    [  2%]
tests/test_setup.py::test_venv_imports PASSED                            [  2%]
tests/test_setup.py::test_cuda_available PASSED                          [  3%]
tests/test_setup.py::test_check_gpu_script_exits_zero PASSED             [  4%]
tests/test_setup.py::test_claude_md_sections PASSED                      [  4%]
tests/test_setup.py::test_raw_data_snapshot_matches PASSED               [  5%]
tests/test_setup.py::test_gitignore[.venv/] PASSED                       [  5%]
tests/test_setup.py::test_gitignore[cache/] PASSED                       [  6%]
tests/test_setup.py::test_gitignore[*.pt] PASSED                         [  7%]
tests/test_setup.py::test_gitignore[logs/*.log] PASSED                   [  7%]
tests/test_stage01.py::test_skeleton_dir_exists[configs] PASSED          [  8%]
tests/test_stage01.py::test_skeleton_dir_exists[src/data] PASSED         [  8%]
tests/test_stage01.py::test_skeleton_dir_exists[src/models] PASSED       [  9%]
tests/test_stage01.py::test_skeleton_dir_exists[src/training] PASSED     [ 10%]
tests/test_stage01.py::test_skeleton_dir_exists[src/evaluation] PASSED   [ 10%]
tests/test_stage01.py::test_skeleton_dir_exists[src/utils] PASSED        [ 11%]
tests/test_stage01.py::test_skeleton_dir_exists[scripts] PASSED          [ 11%]
tests/test_stage01.py::test_skeleton_dir_exists[tests] PASSED            [ 12%]
tests/test_stage01.py::test_skeleton_dir_exists[data_registry] PASSED    [ 13%]
tests/test_stage01.py::test_skeleton_dir_exists[splits] PASSED           [ 13%]
tests/test_stage01.py::test_skeleton_dir_exists[cache] PASSED            [ 14%]
tests/test_stage01.py::test_skeleton_dir_exists[results] PASSED          [ 14%]
tests/test_stage01.py::test_skeleton_dir_exists[reports] PASSED          [ 15%]
tests/test_stage01.py::test_skeleton_dir_exists[logs] PASSED             [ 15%]
tests/test_stage01.py::test_results_readme_and_gitignore PASSED          [ 16%]
tests/test_stage01.py::test_audit_json_loads_and_nonzero[D01_eye_tracking] PASSED [ 17%]
tests/test_stage01.py::test_audit_json_loads_and_nonzero[D02_facial_images_A] PASSED [ 17%]
tests/test_stage01.py::test_audit_json_loads_and_nonzero[D03_facial_images_B] PASSED [ 18%]
tests/test_stage01.py::test_audit_json_loads_and_nonzero[D04_autism_prediction] PASSED [ 18%]
tests/test_stage01.py::test_audit_json_loads_and_nonzero[D05_toddler_screening] PASSED [ 19%]
tests/test_stage01.py::test_audit_json_loads_and_nonzero[D06_early_autism] PASSED [ 20%]
tests/test_stage01.py::test_registry_exists PASSED                       [ 20%]
tests/test_stage01.py::test_raw_data_unmodified PASSED                   [ 21%]
tests/test_stage01.py::test_cuda_available PASSED                        [ 21%]
tests/test_stage02_d01.py::test_no_unidentified_rows PASSED              [ 22%]
tests/test_stage02_d01.py::test_no_duplicate_participant_time_trial PASSED [ 23%]
tests/test_stage02_d01.py::test_no_cars_anywhere_in_cache_or_splits PASSED [ 23%]
tests/test_stage02_d01.py::test_npy_shape_matches_index PASSED           [ 24%]
tests/test_stage02_d01.py::test_clip_rules PASSED                        [ 24%]
tests/test_stage02_d01.py::test_no_clip_spans_two_trials PASSED          [ 25%]
tests/test_stage02_d01.py::test_20_random_clips_rerendered_from_raw_are_identical PASSED [ 26%]
tests/test_stage02_d01.py::test_rebuilt_gaze_equals_cached PASSED        [ 26%]
tests/test_stage02_d01.py::test_quality_table_not_in_model_inputs PASSED [ 27%]
tests/test_stage02_d02.py::test_manifest_pooled_and_complete PASSED      [ 27%]
tests/test_stage02_d02.py::test_no_d03_path_anywhere PASSED              [ 28%]
tests/test_stage02_d02.py::test_near_duplicate_pairs_recomputed_lie_in_same_outer_fold PASSED [ 28%]
tests/test_stage02_d02.py::test_no_group_has_mixed_labels PASSED         [ 29%]
tests/test_stage02_snapshot.py::test_latest_snapshot_checksums_match_current_files PASSED [ 30%]
tests/test_stage02_snapshot.py::test_latest_snapshot_info_and_code_copy PASSED [ 30%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[1-D01_eye_tracking] PASSED [ 31%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[1-D02_facial_images_A] PASSED [ 31%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[1-D04_autism_prediction] PASSED [ 32%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[1-D05_toddler_screening] PASSED [ 33%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[1-D06_early_autism] PASSED [ 33%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[2-D01_eye_tracking] PASSED [ 34%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[2-D02_facial_images_A] PASSED [ 34%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[2-D04_autism_prediction] PASSED [ 35%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[2-D05_toddler_screening] PASSED [ 36%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[2-D06_early_autism] PASSED [ 36%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[3-D01_eye_tracking] PASSED [ 37%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[3-D02_facial_images_A] PASSED [ 37%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[3-D04_autism_prediction] PASSED [ 38%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[3-D05_toddler_screening] PASSED [ 39%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[3-D06_early_autism] PASSED [ 39%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[4-D01_eye_tracking] PASSED [ 40%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[4-D02_facial_images_A] PASSED [ 40%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[4-D04_autism_prediction] PASSED [ 41%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[4-D05_toddler_screening] PASSED [ 42%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[4-D06_early_autism] PASSED [ 42%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[5-D01_eye_tracking] PASSED [ 43%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[5-D02_facial_images_A] PASSED [ 43%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[5-D04_autism_prediction] PASSED [ 44%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[5-D05_toddler_screening] PASSED [ 44%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[5-D06_early_autism] PASSED [ 45%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[6-D01_eye_tracking] PASSED [ 46%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[6-D02_facial_images_A] PASSED [ 46%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[6-D04_autism_prediction] PASSED [ 47%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[6-D05_toddler_screening] PASSED [ 47%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[6-D06_early_autism] PASSED [ 48%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[7-D01_eye_tracking] PASSED [ 49%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[7-D02_facial_images_A] PASSED [ 49%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[7-D04_autism_prediction] PASSED [ 50%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[7-D05_toddler_screening] PASSED [ 50%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[7-D06_early_autism] PASSED [ 51%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[8-D01_eye_tracking] PASSED [ 52%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[8-D02_facial_images_A] PASSED [ 52%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[8-D04_autism_prediction] PASSED [ 53%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[8-D05_toddler_screening] PASSED [ 53%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[8-D06_early_autism] PASSED [ 54%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[9-D01_eye_tracking] PASSED [ 55%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[9-D02_facial_images_A] PASSED [ 55%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[9-D04_autism_prediction] PASSED [ 56%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[9-D05_toddler_screening] PASSED [ 56%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[9-D06_early_autism] PASSED [ 57%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[10-D01_eye_tracking] PASSED [ 57%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[10-D02_facial_images_A] PASSED [ 58%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[10-D04_autism_prediction] PASSED [ 59%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[10-D05_toddler_screening] PASSED [ 59%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[10-D06_early_autism] PASSED [ 60%]
tests/test_stage02_splits.py::test_every_sample_in_exactly_one_test_fold[D01_eye_tracking] PASSED [ 60%]
tests/test_stage02_splits.py::test_every_sample_in_exactly_one_test_fold[D02_facial_images_A] PASSED [ 61%]
tests/test_stage02_splits.py::test_every_sample_in_exactly_one_test_fold[D04_autism_prediction] PASSED [ 62%]
tests/test_stage02_splits.py::test_every_sample_in_exactly_one_test_fold[D05_toddler_screening] PASSED [ 62%]
tests/test_stage02_splits.py::test_every_sample_in_exactly_one_test_fold[D06_early_autism] PASSED [ 63%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[1-D01_eye_tracking] PASSED [ 63%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[1-D02_facial_images_A] PASSED [ 64%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[1-D04_autism_prediction] PASSED [ 65%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[1-D05_toddler_screening] PASSED [ 65%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[1-D06_early_autism] PASSED [ 66%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[2-D01_eye_tracking] PASSED [ 66%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[2-D02_facial_images_A] PASSED [ 67%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[2-D04_autism_prediction] PASSED [ 68%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[2-D05_toddler_screening] PASSED [ 68%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[2-D06_early_autism] PASSED [ 69%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[3-D01_eye_tracking] PASSED [ 69%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[3-D02_facial_images_A] PASSED [ 70%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[3-D04_autism_prediction] PASSED [ 71%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[3-D05_toddler_screening] PASSED [ 71%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[3-D06_early_autism] PASSED [ 72%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[4-D01_eye_tracking] PASSED [ 72%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[4-D02_facial_images_A] PASSED [ 73%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[4-D04_autism_prediction] PASSED [ 73%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[4-D05_toddler_screening] PASSED [ 74%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[4-D06_early_autism] PASSED [ 75%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[5-D01_eye_tracking] PASSED [ 75%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[5-D02_facial_images_A] PASSED [ 76%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[5-D04_autism_prediction] PASSED [ 76%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[5-D05_toddler_screening] PASSED [ 77%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[5-D06_early_autism] PASSED [ 78%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[6-D01_eye_tracking] PASSED [ 78%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[6-D02_facial_images_A] PASSED [ 79%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[6-D04_autism_prediction] PASSED [ 79%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[6-D05_toddler_screening] PASSED [ 80%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[6-D06_early_autism] PASSED [ 81%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[7-D01_eye_tracking] PASSED [ 81%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[7-D02_facial_images_A] PASSED [ 82%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[7-D04_autism_prediction] PASSED [ 82%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[7-D05_toddler_screening] PASSED [ 83%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[7-D06_early_autism] PASSED [ 84%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[8-D01_eye_tracking] PASSED [ 84%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[8-D02_facial_images_A] PASSED [ 85%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[8-D04_autism_prediction] PASSED [ 85%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[8-D05_toddler_screening] PASSED [ 86%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[8-D06_early_autism] PASSED [ 86%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[9-D01_eye_tracking] PASSED [ 87%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[9-D02_facial_images_A] PASSED [ 88%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[9-D04_autism_prediction] PASSED [ 88%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[9-D05_toddler_screening] PASSED [ 89%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[9-D06_early_autism] PASSED [ 89%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[10-D01_eye_tracking] PASSED [ 90%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[10-D02_facial_images_A] PASSED [ 91%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[10-D04_autism_prediction] PASSED [ 91%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[10-D05_toddler_screening] PASSED [ 92%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[10-D06_early_autism] PASSED [ 92%]
tests/test_stage02_splits.py::test_d01_test_folds_have_asd_and_td_children PASSED [ 93%]
tests/test_stage02_splits.py::test_splits_manifest_hashes_match PASSED   [ 94%]
tests/test_stage02_splits.py::test_rerun_make_splits_is_byte_identical PASSED [ 94%]
tests/test_stage02_tabular.py::test_d04_only_labelled_train_rows PASSED  [ 95%]
tests/test_stage02_tabular.py::test_d04_result_decision_recorded PASSED  [ 95%]
tests/test_stage02_tabular.py::test_d05_honest_set_excludes_items_and_score PASSED [ 96%]
tests/test_stage02_tabular.py::test_d05_label_rule_on_clean_data PASSED  [ 97%]
tests/test_stage02_tabular.py::test_d05_other_csv_never_read_by_stage2_code PASSED [ 97%]
tests/test_stage02_tabular.py::test_d06_empty_column_absent PASSED       [ 98%]
tests/test_stage02_tabular.py::test_identical_feature_rows_share_group PASSED [ 98%]
tests/test_stage02_tabular.py::test_every_column_has_a_role PASSED       [ 99%]
tests/test_stage02_tabular.py::test_raw_data_snapshot_unchanged PASSED   [100%]

============================= 169 passed in 33.44s =============================
```

The snapshot test checks the latest `snapshots/stage_NN/CHECKSUMS.sha256` against the current files. Order: provisional snapshot → this pytest run → this report → final snapshot (now including this report) → snapshot tests re-run (`logs/stage02_snapshot_check.txt`).

## 8. Deviations from DATASET_DECISIONS.md and the Stage 2 brief

No decision DD-01..DD-11 was changed. Implementation refinements, each forced by evidence:

1. **DD-02 step b2**: 233 rows in 7.csv/8.csv repeat a (participant, t_ms, trial) with gaze differing by ≤ 0.05 px (7.csv stores 0.1-px precision), so the 4-dp key missed them. Removed as duplicates (keep first) so that the required uniqueness check holds.
2. **Invalid gaze**: besides exact (0,0), samples within 35 px of (0,0) are invalid (tracker-loss artefacts; samples otherwise valid within 30 px of each corner: {'bottom_left': 53, 'bottom_right': 213, 'top_left': 9363, 'top_right': 8}; 8283 samples have exactly one coordinate = 0, the other in [-32.6, 32.1] px). Without the rule the rendered scan paths jumped to the corner (C6 check).
3. **Continuous recordings**: recordings are split into segments at gaps > 500 ms and clips are laid on each segment, so no clip contains a gap (the brief's discard rule then never triggers: 0 `rejected_gap`). A clip must end at or before its segment's last sample, and is rejected if a sample of another trial of the same child/file lies inside its span (needed: 2,042 trial pairs in files 6.csv, 16.csv, 17.csv, 20.csv, 21.csv, 22.csv overlap in time).
4. **D01 split unit**: StratifiedGroupKFold is run on one row per child (group = child), so folds are balanced in children rather than weighted by clip counts; clips inherit their child's fold. Same for the inner validation split.
5. **Tabular group_id**: rows with identical feature values (id and label ignored) share a group — a superset of 'identical rows'. No group has mixed labels.
6. **D04 `result`**: tested, not a function of A1-A10 → kept as a feature (the brief's exclusion applied only if it were).
7. **Categorical values lower-cased** (e.g. 'Health Care Professional' = 'Health care professional', 'others' = 'Others'); constant `age_desc` in D04 excluded.
8. **No git** (user override): SPLITS_MANIFEST and the snapshot use `code_hash` instead of a commit hash; Stage 1 / Setup decision lines were regenerated under the override before pre-flight A1.

## 9. Open items for Stage 3

1. D01 is small and clip-imbalanced: 57 children, ASD 437 vs TD 804 clips → use class weighting / balanced sampling; report child-level metrics.
2. D01 data quality is a strong class signal (section 2.5) and some stimulus files are single-class (section 2.3 table) → Stage 8 quality-only and stimulus-aware controls are mandatory.
3. D01 validation sets have only 7-8 children per fold → val AUC is noisy; prefer val loss / child-level AUC for early stopping (decide in Stage 4).
4. Model inputs: D01 (B, 8, 3, 224, 224) uint8 → float/255 (+ normalisation defined in Stage 3); D02 one image → backbone feature map → spatial tokens. Pick timm variants for EfficientNet/ConvNeXt/ViT in Stage 3.
5. Memory: clips_uint8.npy is 1.39 GiB (memmap, fine).
6. Every training run must call `src.data.splits.verify_manifest()` first.

STAGE 2 DECISION: PASS
