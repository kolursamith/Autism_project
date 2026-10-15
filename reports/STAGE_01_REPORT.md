# STAGE 1 REPORT - workspace, environment and dataset audit

Project: Reliability-Weighted Multimodal AI for Early Autism Risk Screening in Children (BITE308L).
Project dir: `/home/score/Documents/AI_project/asd_hybrid_10fold`. Raw data (read-only): `/home/score/Documents/AI_project/dataset`. Seed: 42.

## 1. Summary

- Built the project skeleton, the results layout (`results/README.md`) and a verified CUDA environment (`.venv`).
- Audited all 6 raw datasets read-only (`scripts/01_audit_datasets.py`). Outputs are in `data_registry/` (json, md, contact sheets, licence text).
- Found serious data problems: D03 is a copy of D02; the D05 labels are a formula of A1-A10; the D05 Combined file has 728 wrong labels; 40% of D01 rows have no child ID and 10 D01 file pairs share rows.
- Recorded 12 dataset decisions with evidence (`data_registry/DATASET_DECISIONS.md`).
- pytest: 24 passed; raw data unchanged (snapshot START == END).

## 2. Environment

Source: `data_registry/environment.txt`.

| item | value |
|---|---|
| GPU | NVIDIA RTX A2000 12GB |
| GPU memory | 536MiB /  12282MiB (nvidia-smi used/total); torch total = 11.618 GiB |
| NVIDIA driver | 580.178.04 (driver CUDA 13.0) |
| torch CUDA / cuDNN | 13.0 / 92400 |
| Python | 3.12.3 |
| torch / torchvision / timm | 2.14.1+cu130 / 0.29.1+cu130 / 1.0.30 |
| cuda.is_available() | True |
| 4096x4096 fp32 matmul | 29.83 ms (4.61 TFLOP/s) |
| venv | `/home/score/Documents/AI_project/asd_hybrid_10fold/.venv` (`--system-site-packages`; numpy/scipy/matplotlib/pillow/pyyaml come from the system and are pinned via `logs/pip_constraints.txt`) |
| requirements | `requirements.txt` (exact versions of the 13 required packages) |

## 3. Project skeleton (depth 2, excluding .venv and cache)

```
asd_hybrid_10fold/
├── .gitignore
├── CLAUDE.md
├── configs/
│   ├── .gitkeep
│   └── paths.yaml
├── data_registry/
│   ├── .gitkeep
│   ├── D01_cv_feasibility.json
│   ├── D01_dedupe_key_check.json
│   ├── D01_eye_tracking_audit.json
│   ├── D01_eye_tracking_audit.md
│   ├── D01_eye_tracking_licence_citation.txt
│   ├── D02_facial_images_A_audit.json
│   ├── D02_facial_images_A_audit.md
│   ├── D02_facial_images_A_licence_citation.txt
│   ├── D02_facial_images_A_samples.png
│   ├── D03_facial_images_B_audit.json
│   ├── D03_facial_images_B_audit.md
│   ├── D03_facial_images_B_licence_citation.txt
│   ├── D03_facial_images_B_samples.png
│   ├── D04_autism_prediction_audit.json
│   ├── D04_autism_prediction_audit.md
│   ├── D04_autism_prediction_licence_citation.txt
│   ├── D05_toddler_screening_audit.json
│   ├── D05_toddler_screening_audit.md
│   ├── D05_toddler_screening_licence_citation.txt
│   ├── D06_early_autism_audit.json
│   ├── D06_early_autism_audit.md
│   ├── D06_early_autism_licence_citation.txt
│   ├── DATASET_DECISIONS.md
│   ├── DATASET_REGISTRY.md
│   ├── environment.txt
│   ├── raw_snapshot_end.tsv
│   └── raw_snapshot_start.tsv
├── logs/
│   ├── .gitkeep
│   ├── pip_constraints.txt
│   ├── pip_install.log
│   ├── setup_fact_check.txt
│   ├── setup_gpu_check.txt
│   ├── setup_listing_after_move.tsv
│   ├── setup_listing_before_move.tsv
│   ├── setup_pip_freeze.txt
│   ├── setup_pip_install.log
│   ├── setup_pytest.txt
│   ├── stage01_audit.log
│   ├── stage01_facts.txt
│   ├── stage01_pytest.txt
│   └── stage01_snapshot_compare.txt
├── pytest.ini
├── reports/
│   ├── .gitkeep
│   ├── SETUP_REPORT.md
│   ├── STAGE_01_REPORT.md
│   ├── STAGE_01_REPORT.pdf
│   └── _pdf_build/
├── requirements.txt
├── results/
│   ├── .gitkeep
│   └── README.md
├── scripts/
│   ├── .gitkeep
│   ├── 00_snapshot_raw.py
│   ├── 01_audit_datasets.py
│   ├── 01_build_registry.py
│   ├── 01_check_cv_feasibility.py
│   ├── 01_check_dd02_key.py
│   ├── 01_check_environment.py
│   ├── 01_git_commit.sh
│   ├── 01_make_pdf.sh
│   ├── 01_write_report.py
│   ├── check_gpu.py
│   ├── make_snapshot.py
│   └── setup_verify_stage1_facts.py
├── splits/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── data/
│   ├── evaluation/
│   ├── models/
│   ├── training/
│   └── utils/
└── tests/
    ├── .gitkeep
    ├── test_setup.py
    └── test_stage01.py
```

## 4. Dataset registry

Folder -> id mapping (all inside `/home/score/Documents/AI_project/dataset/`):

- **D01_eye_tracking** <- `Eye-Tracking Dataset`
- **D02_facial_images_A** <- `Autism Spectrum Disorder Detection using Facial Traits Dual Framework DenseRes with K-Fold Cross Validation (DFDK) for Enhanced Classification Accuracy`
- **D03_facial_images_B** <- `AutismDataset`
- **D04_autism_prediction** <- `Autism_prediction dataset`
- **D05_toddler_screening** <- `Autism_dataset`
- **D06_early_autism** <- `early_autism`

The folder truncated in the file manager is, in full: `Autism Spectrum Disorder Detection using Facial Traits Dual Framework DenseRes with K-Fold Cross Validation (DFDK) for Enhanced Classification Accuracy` (= D02).

| id | folder name | modality | n samples | n subjects | classes and counts | duplicates found | leakage findings | suitable for |
|---|---|---|---|---|---|---|---|---|
| **D01_eye_tracking** | `Eye-Tracking Dataset` | eye-tracking time-series CSV (gaze) | 2,252,943 gaze rows in 25 CSVs; 901,786 rows have no participant ID ({'Unidentified(Neg)': 878362, 'Unidentified(Pos)': 23424}) | 59 in metadata; 57 with gaze rows (no gaze: IDs [12, 16]) | metadata {'TD': 30, 'ASD': 29}; with gaze {'ASD': 27, 'TD': 30} | 329,034 rows repeat a (RecordingTime, TimeOfDay, Participant, Trial) key; 15 files share rows; 10 file pairs share DD-02-key rows. DD-02 numeric key, identified rows: 1,351,157 -> 1,103,660 (D01_dedupe_key_check.json) | CARS Score missing/n by class {'ASD': {'cars_missing': 0, 'n': 29}, 'TD': {'cars_missing': 30, 'n': 30}} -> CARS is a label proxy and must never be a feature. Participant IDs recur across files -> split by participant. | 10 hybrid architectures (gaze-image sequences), participant-level folds |
| **D02_facial_images_A** | `Autism Spectrum Disorder Detection using Facial Traits Dual Framework DenseRes with K-Fold Cross Validation (DFDK) for Enhanced Classification Accuracy` | facial images (RGB jpg) | 2940 images (train: {'autistic': 1193, 'nonautistic': 1193}, valid: {'autistic': 127, 'nonautistic': 127}, test: {'nonautistic': 150, 'autistic': 150}); 0 unreadable; size 114-2471 x 126-2872 px | no subject IDs | {'autistic': 1470, 'nonautistic': 1470} | exact md5: 0 groups; near (phash<=4): 11 pairs, 5 across splits {'train/valid': 2, 'test/train': 3}, 0 cross-class. vs D03_facial_images_B: 2541 byte-identical, 2940/2940 D03 images have a phash<=4 match in D02 (same class: 2940) | n/a (images). No subject IDs -> same child may fall in train and test; near-duplicates across splits listed. | 10 hybrid architectures |
| **D03_facial_images_B** | `AutismDataset` | facial images (RGB jpg) | 2940 images (none: {'autistic': 1470, 'nonautistic': 1470}); 0 unreadable; size 114-2471 x 126-2872 px | no subject IDs | {'autistic': 1470, 'nonautistic': 1470} | exact md5: 0 groups; near (phash<=4): 11 pairs, 0 across splits , 0 cross-class. vs D02_facial_images_A: 2541 byte-identical, 2940/2940 D03 images have a phash<=4 match in D02 (same class: 2940) | n/a (images). No subject IDs -> same child may fall in train and test; near-duplicates across splits listed. | 10 hybrid architectures in principle, BUT it is the same 2940 images as D02 -> use D02 only; never treat D03 as an independent dataset |
| **D04_autism_prediction** | `Autism_prediction dataset` | questionnaire / screening table | `test.csv` 200x21; `train.csv` 800x22 | no subject IDs beyond per-row ID columns (`test.csv`: ID; `train.csv`: ID); cannot tell whether rows are distinct people | `test.csv`: NO LABEL (unlabelled); `train.csv`: {'0': 639, '1': 161} | `test.csv`: 0 full / 0 excl. ID; `train.csv`: 0 full / 0 excl. ID | `train.csv`: No exact single-feature rule over the whole file; best is 'ASD if sum(A1_Score..A10_Score) >= 10' accuracy 0.8625 (majority baseline 0.79875) | tabular baselines only (LR, RF, GB, MLP) - NOT the CNN/ViT hybrids |
| **D05_toddler_screening** | `Autism_dataset` | questionnaire / screening table | `Autism_Screening_Data_Combined.csv` 6075x15; `Toddler Autism dataset July 2018.csv` 1054x19 | no subject IDs beyond per-row ID columns (`Autism_Screening_Data_Combined.csv`: none; `Toddler Autism dataset July 2018.csv`: Case_No); cannot tell whether rows are distinct people | `Autism_Screening_Data_Combined.csv`: {'NO': 4271, 'YES': 1804}; `Toddler Autism dataset July 2018.csv`: {'Yes': 728, 'No': 326} | `Autism_Screening_Data_Combined.csv`: 847 full / 847 excl. ID; `Toddler Autism dataset July 2018.csv`: 0 full / 79 excl. ID | `Autism_Screening_Data_Combined.csv`: No exact single-feature rule over the whole file; best is 'ASD if sum(A1..A10) >= 7' accuracy 0.933827 (majority baseline 0.703045); PARTIAL LEAKAGE: within age bands the item-sum rule is exact for 5021 of 6075 rows / `Toddler Autism dataset July 2018.csv`: LEAKAGE: label reproduced exactly by 'ASD if sum(A1..A10) >= 4' (accuracy 1.0) / `Toddler Autism dataset July 2018.csv` is embedded (rows 5021-6074) in `Autism_Screening_Data_Combined.csv` with 728 labels changed | tabular baselines only (LR, RF, GB, MLP) - NOT the CNN/ViT hybrids |
| **D06_early_autism** | `early_autism` | questionnaire / screening table | `asd.csv` 1000x11 | no subject IDs beyond per-row ID columns (`asd.csv`: Child_ID); cannot tell whether rows are distinct people | `asd.csv`: {'No': 589, 'Yes': 411} | `asd.csv`: 0 full / 0 excl. ID | `asd.csv`: No exact single-feature rule over the whole file; best is 'ASD if Age >= 35' accuracy 0.601 (majority baseline 0.589) | tabular baselines only (LR, RF, GB, MLP) - NOT the CNN/ViT hybrids |

## 5. Problems found

1. **D01 - no child ID on 901,786 of 2,252,943 gaze rows** {'Unidentified(Neg)': 878362, 'Unidentified(Pos)': 23424} - `D01_eye_tracking_audit.json`
2. **D01 - duplicated recordings across files:** 10 file pairs share rows, e.g. 8.csv/11.csv: 114,285; 10.csv/25.csv: 51,828; 1.csv/16.csv: 34,630. Identified rows 1,351,157 -> 1,103,660 after dedupe (247,497 removed, 18,490 of them within a single file) - `D01_eye_tracking_audit.json`, `D01_dedupe_key_check.json`
3. **D01 - dedupe key must be numeric:** a string key removes only 205,275 identified duplicate rows vs 247,497 with numeric parsing - `D01_dedupe_key_check.json`
4. **D01 - participants without gaze data:** [12, 16] (both ASD), leaving 57 children {'ASD': 27, 'TD': 30} - `D01_eye_tracking_audit.json`
5. **D01 - CARS Score encodes the label:** missing by class {'ASD': {'cars_missing': 0, 'n': 29}, 'TD': {'cars_missing': 30, 'n': 30}} - `D01_eye_tracking_audit.json`
6. **D01 - 9 different column layouts across 25 files** (37-58 columns); sampling interval varies by file (median 16.7 ms overall, p95 44.0 ms) - `D01_eye_tracking_audit.json`
7. **D01 - small sample:** 57 children -> 5-6 per test fold (2-3 ASD); fold metrics will have high variance - `D01_cv_feasibility.json`
8. **D03 is a copy of D02:** 2,541 byte-identical + 399 re-saved copies (nearest D02 split {'valid': 99, 'test': 300}); all 2,940 D03 images match D02 with the same class - `D02_facial_images_A_audit.json`
9. **D02 - near-duplicates across the original splits:** 11 pHash<=4 pairs, 5 across splits {'train|valid': 2, 'test|train': 3}; exact md5 groups 0 - `D02_facial_images_A_audit.json`
10. **D02/D03 - no subject IDs** (filenames `<class>.<n>.jpg` / `<n>.jpg`, no metadata): the same child may occur in several images; results can only be image-level - `D02_facial_images_A_audit.json`
11. **D02 - inconsistent class folder name** ['Autistic', 'Non-Autistic'] in test vs ['Autistic', 'Non_Autistic'] in train; 10 result-figure PNGs in the folder root (not data) - `D02_facial_images_A_audit.json`
12. **D02 - image size varies widely:** width 114-2471 px, height 126-2872 px, 2,557 distinct sizes; unreadable files: 0 - `D02_facial_images_A_audit.json`
13. **D04 - test.csv has no label** (shape [200, 21], label columns []) - `D04_autism_prediction_audit.json`
14. **D04 - class imbalance:** train {'0': 639, '1': 161} (20.1% positive); '?' placeholders {'ethnicity': 203, 'relation': 40} - `D04_autism_prediction_audit.json`
15. **D05 - label leakage (Toddler):** 'ASD if sum(A1..A10) >= 4' accuracy 1.0, confusion [[326, 0], [0, 728]]; item sum == Qchat-10-Score on 1054/1054 rows - `D05_toddler_screening_audit.json`
16. **D05 - wrong labels in Combined file:** Toddler rows embedded at rows 5021-6074, all labelled {'no': 1054} vs original {'no': 326, 'yes': 728} -> 728 conflicts - `D05_toddler_screening_audit.json`
17. **D05 - partial leakage in Combined file:** No exact single-feature rule over the whole file; best is 'ASD if sum(A1..A10) >= 7' accuracy 0.933827 (majority baseline 0.703045); PARTIAL LEAKAGE: within age bands the item-sum rule is exact for 5021 of 6075 rows - `D05_toddler_screening_audit.json`
18. **D05 - duplicates:** Combined 847 fully duplicated rows; Toddler 79 duplicates when Case_No is ignored - `D05_toddler_screening_audit.json`
19. **D05 - class imbalance (Toddler):** {'Yes': 728, 'No': 326} (69.1% ASD) - `D05_toddler_screening_audit.json`
20. **D06 - empty column** ['Unnamed: 10']; weak signal: best single rule 'ASD if Age >= 35' 0.601 vs majority baseline 0.589 - `D06_early_autism_audit.json`
21. **Licences:** no licence file in any of the 6 folders; only D01 `Citation.txt` and the D05 description (.docx) exist - `data_registry/*_licence_citation.txt`
22. **No unreadable images, no zip/rar archives** in any folder - `*_audit.json` (`n_unreadable`, `archives`)

## 6. Dataset decisions

Full text with evidence: [data_registry/DATASET_DECISIONS.md](../data_registry/DATASET_DECISIONS.md). No CONTRADICTION found.

- DD-01 - D01 eye-tracking is the PRIMARY dataset for the 10 hybrid architectures
- DD-02 - D01 duplicated rows across files are removed in Stage 2 BEFORE any clip or window is created
- DD-03 - D01 CARS Score is never a feature, filter or stratification variable
- DD-04 - D01 10-fold CV = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
- DD-05 - D02 facial images (DFDK folder) is the SECONDARY dataset for the 10 hybrid architectures
- DD-06 - D03 AutismDataset is EXCLUDED from training: it is a copy of D02
- DD-07 - D02 near-duplicates are grouped, and CV uses StratifiedGroupKFold on those groups
- DD-08 - D04 autism prediction: only the 800 labelled train rows are used
- DD-09 - D05 toddler screening: the label is defined by the A1–A10 score
- DD-10 - D06 early autism: tabular baselines only, 10-fold StratifiedKFold; the empty column is dropped
- DD-11 - No datasets are merged, and no predictions are combined across children or datasets
- DD-12 - Shortcut control (in Stage 8): tracking quality as a baseline for D01; near-duplicate grouping effect for D02

## 7. Tests

Command: `.venv/bin/python -m pytest tests/test_stage01.py -v` (output: `logs/stage01_pytest.txt`)

```
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0 -- /home/score/Documents/AI_project/asd_hybrid_10fold/.venv/bin/python
rootdir: /home/score/Documents/AI_project/asd_hybrid_10fold
configfile: pytest.ini
plugins: anyio-4.15.1
collecting ... collected 24 items

tests/test_stage01.py::test_skeleton_dir_exists[configs] PASSED          [  4%]
tests/test_stage01.py::test_skeleton_dir_exists[src/data] PASSED         [  8%]
tests/test_stage01.py::test_skeleton_dir_exists[src/models] PASSED       [ 12%]
tests/test_stage01.py::test_skeleton_dir_exists[src/training] PASSED     [ 16%]
tests/test_stage01.py::test_skeleton_dir_exists[src/evaluation] PASSED   [ 20%]
tests/test_stage01.py::test_skeleton_dir_exists[src/utils] PASSED        [ 25%]
tests/test_stage01.py::test_skeleton_dir_exists[scripts] PASSED          [ 29%]
tests/test_stage01.py::test_skeleton_dir_exists[tests] PASSED            [ 33%]
tests/test_stage01.py::test_skeleton_dir_exists[data_registry] PASSED    [ 37%]
tests/test_stage01.py::test_skeleton_dir_exists[splits] PASSED           [ 41%]
tests/test_stage01.py::test_skeleton_dir_exists[cache] PASSED            [ 45%]
tests/test_stage01.py::test_skeleton_dir_exists[results] PASSED          [ 50%]
tests/test_stage01.py::test_skeleton_dir_exists[reports] PASSED          [ 54%]
tests/test_stage01.py::test_skeleton_dir_exists[logs] PASSED             [ 58%]
tests/test_stage01.py::test_results_readme_and_gitignore PASSED          [ 62%]
tests/test_stage01.py::test_audit_json_loads_and_nonzero[D01_eye_tracking] PASSED [ 66%]
tests/test_stage01.py::test_audit_json_loads_and_nonzero[D02_facial_images_A] PASSED [ 70%]
tests/test_stage01.py::test_audit_json_loads_and_nonzero[D03_facial_images_B] PASSED [ 75%]
tests/test_stage01.py::test_audit_json_loads_and_nonzero[D04_autism_prediction] PASSED [ 79%]
tests/test_stage01.py::test_audit_json_loads_and_nonzero[D05_toddler_screening] PASSED [ 83%]
tests/test_stage01.py::test_audit_json_loads_and_nonzero[D06_early_autism] PASSED [ 87%]
tests/test_stage01.py::test_registry_exists PASSED                       [ 91%]
tests/test_stage01.py::test_raw_data_unmodified PASSED                   [ 95%]
tests/test_stage01.py::test_cuda_available PASSED                        [100%]

============================== 24 passed in 1.08s ==============================
```

Raw-data snapshot comparison (`logs/stage01_snapshot_compare.txt`):

```
$ cmp data_registry/raw_snapshot_start.tsv data_registry/raw_snapshot_end.tsv
IDENTICAL: 5944 entries (path, kind, size, mtime_ns), sha256 start=51873a4ec225bc64 end=51873a4ec225bc64
```

## 8. Open items for Stage 2 (data preparation + 10-fold split manifests + leakage tests)

1. D01: drop Unidentified rows, exclude IDs [12, 16], dedupe with the numeric DD-02 key, and report rows before/after (expected identified rows 1,351,157 -> 1,103,660); harmonise the 9 column layouts; define how gaze-image sequences (clips/windows) are rendered from gaze rows.
2. D01: save the StratifiedGroupKFold(10, shuffle, seed 42) participant manifest in `splits/`; assert each test fold has >=1 ASD and >=1 TD and that no participant appears in two folds.
3. D02: pool the 2,940 images, normalise class names, build pHash<=4 groups, save the StratifiedGroupKFold manifest; assert no md5 or near-duplicate group crosses folds.
4. D03: excluded; add a test that no D03 path ever appears in a manifest.
5. D04 / D05-Toddler / D06: save StratifiedKFold(10, shuffle, seed 42) manifests; D05 manifests for experiments (a) all features and (b) without A1-A10/Qchat; treat '?' as missing in D04; drop `Unnamed: 10` in D06.
6. Leakage tests (pytest): no ID/group overlap between train and test in any fold; CARS never in any D01 feature list; D05 experiment (b) feature list contains no A1-A10/score; DD-11 (no cross-dataset merges).
7. Stage 8 shortcut control (DD-12): compute the per-child tracking-quality ratio in Stage 2 so it is available later.

Version record: no git (user override 2026-10-13); Stage 1 is recorded by the local snapshot `snapshots/stage_01/` (code copy + CHECKSUMS.sha256 + SNAPSHOT_INFO.json), created right after this report

STAGE 1 DECISION: PASS
