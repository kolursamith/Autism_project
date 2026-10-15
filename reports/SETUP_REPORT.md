# SETUP REPORT — move into AI_project, paths.yaml, GPU check, CLAUDE.md

Date: 2026-10-13. Scope: setup only. No Stage 2 work, no training, no preprocessing. No raw dataset file was
modified, moved, renamed or deleted (live snapshot == `data_registry/raw_snapshot_start.tsv`, see tests).

## 1. What I found

| item | finding |
|---|---|
| `AI_PROJECT_DIR` | `/home/score/Documents/AI_project` — contained `dataset/` and `Stage1_results/` (left unchanged) |
| `DATASETS_DIR` | `/home/score/Documents/AI_project/dataset` — 6 folders, names as in `configs/paths.yaml` |
| `STAGE1_PROJECT` | `/home/score/asd_hybrid_10fold` (69 files excl. `.venv`; `.venv` 29,535 files, 5.7 GB) |
| `Stage1_results/` | 3.8 MB copy of Stage 1 outputs made 2026-10-12 by the previous agent (its `README.txt`); no `.venv`/cache. Now stale for: `DATASET_DECISIONS.md` (DD-12), `STAGE_01_REPORT.md` (paths). Not touched. |
| git | **not installed** (`git: command not found`); no `.git` directory → no commits exist |

Stage 1 documents (2–4 line summaries):
- **reports/STAGE_01_REPORT.md** — skeleton, environment, registry, 22 numbered problems with sources, DD-01…DD-12,
  24/24 tests, Stage 2 open items. Ended `STAGE 1 DECISION: FAIL` solely because git is missing.
- **data_registry/DATASET_REGISTRY.md** — one table for the 6 datasets (modality, counts, duplicates, leakage, use),
  generated from the audit JSONs by `scripts/01_build_registry.py`.
- **data_registry/DATASET_DECISIONS.md** — DD-01…DD-12 with evidence; no CONTRADICTION. DD-02 adds that the dedupe
  key must be compared numerically (string key misses 42,222 duplicates).
- **D01 audit** — 25 gaze CSVs, 9 column layouts, 40% rows without child ID, file-pair duplication, CARS = label proxy.
- **D02 audit** — 2,940 balanced images, 0 unreadable, 0 md5 duplicates, 11 pHash≤4 pairs, no subject IDs,
  inconsistent `Non-Autistic`/`Non_Autistic` folder names, 10 paper-figure PNGs in the root.
- **D03 audit** — the same 2,940 images as D02 (2,541 byte-identical, 399 re-saved).
- **D04 audit** — train 800 rows (20.1% positive), test.csv unlabelled, `?` placeholders in ethnicity/relation.
- **D05 audit** — Toddler label is exactly `sum(A1..A10) >= 4`; Combined file embeds Toddler with 728 wrong labels;
  847 duplicate rows in Combined; 79 duplicates in Toddler ignoring Case_No.
- **D06 audit** — 1,000 rows (589/411), empty `Unnamed: 10`, no single-feature leakage (0.601 vs 0.589).
- **environment.txt** — nvidia-smi, torch 2.14.1+cu130 with CUDA, package versions; venv used `--system-site-packages`.
- **results/README.md** — results layout. **requirements.txt** — 13 pinned packages. **.gitignore** — cache, *.pt,
  .venv, pycache. **scripts/** — 00 snapshot, 01 audit/registry/report/feasibility/dedupe/env checks, git and PDF
  helpers. **tests/test_stage01.py** — 24 tests.

## 2. What I moved and changed

Move: `mv /home/score/asd_hybrid_10fold /home/score/Documents/AI_project/asd_hybrid_10fold`, then
`ln -s /home/score/Documents/AI_project/asd_hybrid_10fold /home/score/asd_hybrid_10fold`.
Before/after listings (path, size; excl. `.venv`): `logs/setup_listing_before_move.tsv`,
`logs/setup_listing_after_move.tsv`. `diff` exit code 0 (0 = identical, 69 files).

<details><summary>Listing before the move (path, bytes)</summary>

```
cache/.gitkeep	0
configs/.gitkeep	0
data_registry/D01_cv_feasibility.json	768
data_registry/D01_dedupe_key_check.json	1169
data_registry/D01_eye_tracking_audit.json	34343
data_registry/D01_eye_tracking_audit.md	5819
data_registry/D01_eye_tracking_licence_citation.txt	282
data_registry/D02_facial_images_A_audit.json	8013
data_registry/D02_facial_images_A_audit.md	4557
data_registry/D02_facial_images_A_licence_citation.txt	65
data_registry/D02_facial_images_A_samples.png	493181
data_registry/D03_facial_images_B_audit.json	6288
data_registry/D03_facial_images_B_audit.md	3985
data_registry/D03_facial_images_B_licence_citation.txt	65
data_registry/D03_facial_images_B_samples.png	488374
data_registry/D04_autism_prediction_audit.json	10518
data_registry/D04_autism_prediction_audit.md	3925
data_registry/D04_autism_prediction_licence_citation.txt	65
data_registry/D05_toddler_screening_audit.json	11945
data_registry/D05_toddler_screening_audit.md	4550
data_registry/D05_toddler_screening_licence_citation.txt	12067
data_registry/D06_early_autism_audit.json	5108
data_registry/D06_early_autism_audit.md	2026
data_registry/D06_early_autism_licence_citation.txt	65
data_registry/DATASET_DECISIONS.md	10472
data_registry/DATASET_REGISTRY.md	5347
data_registry/environment.txt	4701
data_registry/.gitkeep	0
data_registry/raw_snapshot_end.tsv	942155
data_registry/raw_snapshot_start.tsv	942155
.gitignore	72
logs/.gitkeep	0
logs/pip_constraints.txt	75
logs/pip_install.log	21511
logs/stage01_audit.log	1172
logs/stage01_facts.txt	5082
logs/stage01_pytest.txt	2383
logs/stage01_snapshot_compare.txt	183
.pytest_cache/CACHEDIR.TAG	191
.pytest_cache/.gitignore	37
.pytest_cache/README.md	302
.pytest_cache/v/cache/nodeids	1594
reports/.gitkeep	0
reports/_pdf_build/style.css	612
reports/STAGE_01_REPORT.md	19088
reports/STAGE_01_REPORT.pdf	292386
requirements.txt	212
results/.gitkeep	0
results/README.md	1765
scripts/00_snapshot_raw.py	896
scripts/01_audit_datasets.py	41773
scripts/01_build_registry.py	6620
scripts/01_check_cv_feasibility.py	1479
scripts/01_check_dd02_key.py	1993
scripts/01_check_environment.py	1658
scripts/01_git_commit.sh	767
scripts/01_make_pdf.sh	672
scripts/01_write_report.py	15252
scripts/.gitkeep	0
scripts/__pycache__/00_snapshot_raw.cpython-312.pyc	1995
splits/.gitkeep	0
src/data/.gitkeep	0
src/evaluation/.gitkeep	0
src/models/.gitkeep	0
src/training/.gitkeep	0
src/utils/.gitkeep	0
tests/.gitkeep	0
tests/__pycache__/test_stage01.cpython-312-pytest-9.1.1.pyc	18491
tests/test_stage01.py	2589
```
</details>

venv: moved `.venv` deleted; new one created and filled with
```
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt -c logs/pip_constraints.txt
```
(same recipe as Stage 1; matplotlib 3.6.3 has no Python 3.12 wheel so system packages are reused).
`torch==2.14.1` from PyPI resolved to `2.14.1+cu130` and `torch.cuda.is_available()` was True on the first try, so
the PyTorch CUDA wheel index was **not** needed. Full freeze: `logs/setup_pip_freeze.txt`.

New files: `CLAUDE.md`, `configs/paths.yaml`, `src/__init__.py`, `src/utils/__init__.py`, `src/utils/paths.py`,
`scripts/check_gpu.py`, `scripts/setup_verify_stage1_facts.py`, `tests/test_setup.py`, `pytest.ini`,
`reports/SETUP_REPORT.md`, `logs/setup_*`.

Changed files:
- `scripts/00_snapshot_raw.py`, `01_audit_datasets.py`, `01_check_dd02_key.py`, `01_check_cv_feasibility.py`,
  `01_build_registry.py`, `01_write_report.py` — hard-coded paths replaced by `src/utils/paths.py`
  (→ `configs/paths.yaml`). Re-running `01_build_registry.py`, `01_check_cv_feasibility.py` and
  `01_check_dd02_key.py` reproduced `DATASET_REGISTRY.md`, `D01_cv_feasibility.json` and
  `D01_dedupe_key_check.json` **byte-identically** (`cmp`). `01_audit_datasets.py` was not re-run (heavy); its
  dataset table resolves to the same 6 folder names.
- `data_registry/DATASET_DECISIONS.md` — DD-12 extended with the D02 part (report the effect of the near-duplicate
  grouping in Stage 8), which the brief requires and Stage 1 lacked. Effect size: UNKNOWN until Stage 8.
- `results/README.md` — `run_registry.csv` columns aligned with the brief (added `val_auc, gpu_mem_gb, seconds`),
  `blocked` status, metrics list, run_info GPU fields, validation-set rule.
- `.gitignore` — added `logs/*.log`.
- `reports/STAGE_01_REPORT.md` and `logs/stage01_pytest.txt` regenerated at the new location (only paths, tree,
  DD-12 title and pytest header changed). `reports/STAGE_01_REPORT.pdf` was **not** rebuilt and is stale.

Absolute paths now appear only in `configs/paths.yaml` among scripts/, src/, tests/, configs/ (checked by
`test_no_absolute_paths_outside_paths_yaml`). Documentation (`CLAUDE.md`, reports, data_registry, logs) still names
paths for humans.

## 3. Stage 1 fact checks

Produced by `scripts/setup_verify_stage1_facts.py` (loads the audit JSONs; ENV rows are live). Note on #8: 329,034
= total rows − distinct (RecordingTime, Time of Day, Participant, Trial) keys over all files pooled, so it also counts
repeats inside a single file; the brief's wording "duplicated across files" is slightly loose.

| # | dataset | claim | value from file | verdict | source |
|---|---|---|---|---|---|
| 1 | D01 | 2,252,943 gaze rows | `2252943` | CONFIRMED | D01_eye_tracking_audit.json |
| 2 | D01 | 25 gaze CSVs | `25` | CONFIRMED | D01_eye_tracking_audit.json |
| 3 | D01 | 9 column layouts | `9` | CONFIRMED | D01_eye_tracking_audit.json |
| 4 | D01 | metadata 59 participants (29 ASD / 30 TD) | `(59, {'TD': 30, 'ASD': 29})` | CONFIRMED | D01_eye_tracking_audit.json |
| 5 | D01 | 57 with gaze rows | `57` | CONFIRMED | D01_eye_tracking_audit.json |
| 6 | D01 | IDs 12 and 16 have no gaze rows | `[12, 16]` | CONFIRMED | D01_eye_tracking_audit.json |
| 7 | D01 | ~40% of rows are 'Unidentified' | `901786 rows = 0.4003` | CONFIRMED | D01_eye_tracking_audit.json |
| 8 | D01 | 329,034 duplicated rows | `329034 (= rows_total - distinct keys (RecordingTime [ms], Time of Day, Participant, Trial), all files pooled)` | CONFIRMED | D01_eye_tracking_audit.json |
| 9 | D01 | files 8 & 11 share all rows | `{'files': ['8.csv', '11.csv'], 'shared_keys': 114285, 'rows_a': 114285, 'rows_b': 114285}` | CONFIRMED | D01_eye_tracking_audit.json |
| 10 | D01 | file 1 is inside file 16 | `{'files': ['1.csv', '16.csv'], 'shared_keys': 34630, 'rows_a': 34630, 'rows_b': 110967}` | CONFIRMED | D01_eye_tracking_audit.json |
| 11 | D01 | CARS only for ASD | `{'ASD': {'cars_missing': 0, 'n': 29}, 'TD': {'cars_missing': 30, 'n': 30}}` | CONFIRMED | D01_eye_tracking_audit.json |
| 12 | D01 | ~60 Hz | `59.88` | CONFIRMED | D01_eye_tracking_audit.json |
| 13 | D02 | 2,940 images | `2940` | CONFIRMED | D02_facial_images_A_audit.json |
| 14 | D02 | train 2,386 / valid 254 / test 300 | `{'train': 2386, 'valid': 254, 'test': 300}` | CONFIRMED | D02_facial_images_A_audit.json |
| 15 | D02 | balanced classes | `{'autistic': 1470, 'nonautistic': 1470}` | CONFIRMED | D02_facial_images_A_audit.json |
| 16 | D02 | 0 unreadable | `0` | CONFIRMED | D02_facial_images_A_audit.json |
| 17 | D02 | 0 exact duplicates | `0` | CONFIRMED | D02_facial_images_A_audit.json |
| 18 | D02 | 11 near-duplicate pairs, 5 across splits | `(11, 5)` | CONFIRMED | D02_facial_images_A_audit.json |
| 19 | D02 | no subject IDs | `NO subject/participant ID exists: filena` | CONFIRMED | D02_facial_images_A_audit.json |
| 20 | D03 | same 2,940 images as D02 | `(2940, 2940)` | CONFIRMED | D02_facial_images_A_audit.json |
| 21 | D03 | 2,541 byte-identical | `2541` | CONFIRMED | D02_facial_images_A_audit.json |
| 22 | D03 | 399 re-saved copies of D02 test/valid | `(399, {'valid': 99, 'test': 300})` | CONFIRMED | D02_facial_images_A_audit.json |
| 23 | D04 | train 800 rows (639/161) | `(800, {'0': 639, '1': 161})` | CONFIRMED | D04_autism_prediction_audit.json |
| 24 | D04 | test.csv unlabelled | `[]` | CONFIRMED | D04_autism_prediction_audit.json |
| 25 | D05 | Toddler label == (A1..A10 sum >= 4), accuracy 100% | `1.0` | CONFIRMED | D05_toddler_screening_audit.json |
| 26 | D05 | Combined holds Toddler rows 5021-6074 | `[5021, 6074]` | CONFIRMED | D05_toddler_screening_audit.json |
| 27 | D05 | all 1,054 embedded labels = NO | `{'no': 1054}` | CONFIRMED | D05_toddler_screening_audit.json |
| 28 | D05 | 728 wrong labels | `728` | CONFIRMED | D05_toddler_screening_audit.json |
| 29 | D05 | 847 duplicate rows in Combined | `847` | CONFIRMED | D05_toddler_screening_audit.json |
| 30 | D06 | 1,000 rows (589/411) | `(1000, {'No': 589, 'Yes': 411})` | CONFIRMED | D06_early_autism_audit.json |
| 31 | D06 | one fully empty column | `['Unnamed: 10']` | CONFIRMED | D06_early_autism_audit.json |
| 32 | D06 | no leakage: best rule 0.601 vs majority 0.589 | `('ASD if Age >= 35', 0.601, 0.589)` | CONFIRMED | D06_early_autism_audit.json |
| 33 | ENV | NVIDIA RTX A2000 12GB, driver 580 | `NVIDIA RTX A2000 12GB, 580.178.04` | CONFIRMED | live nvidia-smi |
| 34 | ENV | driver CUDA 13.0 | `13.0` | CONFIRMED | live nvidia-smi |
| 35 | ENV | Python 3.12.3 | `3.12.3` | CONFIRMED | live |
| 36 | ENV | torch 2.14.1+cu130, CUDA available | `('2.14.1+cu130', True)` | CONFIRMED | live |
| 37 | ENV | .venv inside project dir | `/home/score/Documents/AI_project/asd_hybrid_10fold/.venv` | CONFIRMED | live |

TOTAL: 37 facts, 37 CONFIRMED, 0 CONTRADICTED

## 4. GPU check (`logs/setup_gpu_check.txt`)

```
$ nvidia-smi
Tue Oct 13 16:43:04 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.178.04             Driver Version: 580.178.04     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA RTX A2000 12GB          Off |   00000000:01:00.0  On |                  Off |
| 30%   37C    P8              9W /   70W |     370MiB /  12282MiB |      3%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A           48242      G   /usr/lib/xorg/Xorg                      127MiB |
|    0   N/A  N/A           48481      G   /usr/bin/gnome-shell                    132MiB |
|    0   N/A  N/A           49937      G   ...rack-uuid=3190708988185955192         73MiB |
+-----------------------------------------------------------------------------------------+

$ .venv/bin/python scripts/check_gpu.py
python 3.12.3 | /home/score/Documents/AI_project/asd_hybrid_10fold/.venv/bin/python
torch.__version__ = 2.14.1+cu130
torch.version.cuda = 13.0
torch.backends.cudnn.version() = 92400
torch.cuda.is_available() = True
nvidia driver = 580.178.04
device name = NVIDIA RTX A2000 12GB
total memory = 11.618 GiB
4096x4096 fp32 matmul on cuda: 29.47 ms (4.66 TFLOP/s)
max_memory_allocated = 264.1 MiB
GPU CHECK: OK
exit code: 0
```

## 5. Tests (`.venv/bin/python -m pytest -v`, `logs/setup_pytest.txt`)

```
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0 -- /home/score/Documents/AI_project/asd_hybrid_10fold/.venv/bin/python
rootdir: /home/score/Documents/AI_project/asd_hybrid_10fold
configfile: pytest.ini
testpaths: tests
plugins: anyio-4.15.1
collecting ... collected 37 items

tests/test_setup.py::test_project_dir_and_symlink PASSED                 [  2%]
tests/test_setup.py::test_paths_yaml_every_path_exists PASSED            [  5%]
tests/test_setup.py::test_datasets_dir_outside_project PASSED            [  8%]
tests/test_setup.py::test_no_absolute_paths_outside_paths_yaml PASSED    [ 10%]
tests/test_setup.py::test_venv_imports PASSED                            [ 13%]
tests/test_setup.py::test_cuda_available PASSED                          [ 16%]
tests/test_setup.py::test_check_gpu_script_exits_zero PASSED             [ 18%]
tests/test_setup.py::test_claude_md_sections PASSED                      [ 21%]
tests/test_setup.py::test_raw_data_snapshot_matches PASSED               [ 24%]
tests/test_setup.py::test_gitignore[.venv/] PASSED                       [ 27%]
tests/test_setup.py::test_gitignore[cache/] PASSED                       [ 29%]
tests/test_setup.py::test_gitignore[*.pt] PASSED                         [ 32%]
tests/test_setup.py::test_gitignore[logs/*.log] PASSED                   [ 35%]
tests/test_stage01.py::test_skeleton_dir_exists[configs] PASSED          [ 37%]
tests/test_stage01.py::test_skeleton_dir_exists[src/data] PASSED         [ 40%]
tests/test_stage01.py::test_skeleton_dir_exists[src/models] PASSED       [ 43%]
tests/test_stage01.py::test_skeleton_dir_exists[src/training] PASSED     [ 45%]
tests/test_stage01.py::test_skeleton_dir_exists[src/evaluation] PASSED   [ 48%]
tests/test_stage01.py::test_skeleton_dir_exists[src/utils] PASSED        [ 51%]
tests/test_stage01.py::test_skeleton_dir_exists[scripts] PASSED          [ 54%]
tests/test_stage01.py::test_skeleton_dir_exists[tests] PASSED            [ 56%]
tests/test_stage01.py::test_skeleton_dir_exists[data_registry] PASSED    [ 59%]
tests/test_stage01.py::test_skeleton_dir_exists[splits] PASSED           [ 62%]
tests/test_stage01.py::test_skeleton_dir_exists[cache] PASSED            [ 64%]
tests/test_stage01.py::test_skeleton_dir_exists[results] PASSED          [ 67%]
tests/test_stage01.py::test_skeleton_dir_exists[reports] PASSED          [ 70%]
tests/test_stage01.py::test_skeleton_dir_exists[logs] PASSED             [ 72%]
tests/test_stage01.py::test_results_readme_and_gitignore PASSED          [ 75%]
tests/test_stage01.py::test_audit_json_loads_and_nonzero[D01_eye_tracking] PASSED [ 78%]
tests/test_stage01.py::test_audit_json_loads_and_nonzero[D02_facial_images_A] PASSED [ 81%]
tests/test_stage01.py::test_audit_json_loads_and_nonzero[D03_facial_images_B] PASSED [ 83%]
tests/test_stage01.py::test_audit_json_loads_and_nonzero[D04_autism_prediction] PASSED [ 86%]
tests/test_stage01.py::test_audit_json_loads_and_nonzero[D05_toddler_screening] PASSED [ 89%]
tests/test_stage01.py::test_audit_json_loads_and_nonzero[D06_early_autism] PASSED [ 91%]
tests/test_stage01.py::test_registry_exists PASSED                       [ 94%]
tests/test_stage01.py::test_raw_data_unmodified PASSED                   [ 97%]
tests/test_stage01.py::test_cuda_available PASSED                        [100%]

============================== 37 passed in 5.32s ==============================
```

## 6. Stage 1 status

| check | result |
|---|---|
| (a) `tests/test_stage01.py` passes | yes — 24/24 |
| (b) raw snapshot == fresh snapshot of DATASETS_DIR | yes (`test_raw_data_unmodified`, `test_raw_data_snapshot_matches`) |
| (c) `DATASET_DECISIONS.md` exists | yes (DD-12 completed) |
| (d) `STAGE_01_REPORT.md` exists, ends with decision line | yes: `STAGE 1 DECISION: FAIL - git commits not made: git is not installed (`git --version` -> command not found)` |
| (e) git has commits | **NO — git is not installed** |

Stage 1 stays FAIL until git is installed and the commits are made. Required action (needs sudo, so by you):
`sudo apt install -y git`. Then the next session: `git init`, user.name "Samith Kolur", commit
"Stage 1: workspace, environment and dataset audit", regenerate the Stage 1 report (it will then see the commit),
commit "Setup: move into AI_project, paths.yaml, CLAUDE.md", and update CLAUDE.md §6.

## 7. Update 2026-10-13 — git requirement withdrawn by the user

The user decided that the project does not use git ("IMPORTANT OVERRIDE - NO GIT"). Every git instruction is replaced
by a local snapshot: `snapshots/stage_NN/` = code copy + `CHECKSUMS.sha256` + `SNAPSHOT_INFO.json`
(`scripts/make_snapshot.py`). git was never installed and no `.git` folder exists, so nothing was committed and
there is no git history. The only FAIL reason above (no commits) no longer applies:
- `scripts/01_write_report.py` now checks the local-snapshot rule instead of git; `STAGE_01_REPORT.md` regenerated.
- `CLAUDE.md` golden rule 1 and every other git mention updated.
- Stage 1 + Setup are recorded together in `snapshots/stage_01/` (made right after this update).
All other setup checks were already passing (section 5).

SETUP DECISION: PASS - all setup items done and verified (37 passed, 37/37 facts CONFIRMED); the git commit requirement was withdrawn by the user on 2026-10-13 and replaced by the local snapshot snapshots/stage_01/
