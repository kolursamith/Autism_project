# ASD hybrid 10-fold - Reliability-Weighted Multimodal AI for Early Autism Risk Screening

BITE308L Artificial Intelligence (VIT). Team: Samith Kolur (24BIT0117), Iniya K, Chandru Vel Kumar.

> Screening-support research only. Nothing in this repository is a diagnosis or a clinical accuracy estimate.

## Summary

We trained and compared ten hybrid deep-learning architectures (a pretrained or custom-CNN spatial encoder, a recurrent or Transformer temporal encoder and an attention head) for autism risk screening on two public datasets: eye-tracking recordings of 57 children (D01) and 2,940 web-collected facial images (D02). All models used one fixed training protocol and grouped 10-fold cross-validation; tabular baselines were run on three questionnaire datasets. On D01, child-level pooled ROC-AUC ranged from 0.574 to 0.832, but a logistic regression on three recording-quality features alone reached 0.823. No model was significantly better than this baseline, and only one model (M07) had a matched-subset AUC whose 95% bootstrap CI excluded 0.5 (0.724 [0.551, 0.883], 14 matched pairs); the other models' matched AUCs had CIs including 0.5. On D02, models reached mean ROC-AUCs of 0.867-0.931, yet simple non-facial image statistics alone reached 0.796, so dataset artefacts separate the classes. Removing near-duplicate test images changed AUCs by at most 0.002. Pretrained encoders scored higher on average than custom CNNs on both datasets. The results do not show that the models learned autism-related gaze or facial patterns and are not clinical accuracy estimates; they motivate fusion that weights each modality by its estimated reliability.

## Main findings (copied verbatim from `reports/FINAL_RESULTS_REPORT.md`, section 8)

- Can be claimed: per-child tracking quality alone predicts the label about as well as or better than every deep model (quality_LR3 pooled child AUC 0.823; models 0.574-0.832); no model is significantly better than this baseline, and 4 models (M02, M05, M09, M10) are significantly worse (CI of the difference below 0). Model child scores fall as tracking quality rises, also within the ASD group.
- By the pre-specified rule, M07 adds information beyond tracking quality, via rule 2 only: matched-subset AUC 0.724 [0.551, 0.883] on 14 pairs. This is weak evidence: rule 1 fails (difference vs quality_LR3 +0.009 [-0.098, +0.120]); 10 models were tested without multiplicity adjustment (with no signal beyond quality, the chance that at least one of 10 independent one-sided 97.5% bounds lies above 0.5 is 1 - 0.975^10 = 22%; the models are correlated, so this is only a guide); the matched set is small and keeps some residual imbalance (quality-only AUC on it 0.571, SMD -0.16); the exploratory incremental analysis gives +0.012 pooled AUC.
- Can be claimed: simple non-facial image statistics alone reach AUC 0.782-0.796, so the two classes differ systematically in image properties (colour, brightness, size/compression, sharpness); the deep models (0.867-0.931) score higher than this baseline, but how much of their performance comes from facial phenotype rather than such artefacts is not identified by these analyses.
- Near-duplicate leakage at pHash <= 8 does not explain the results (AUC change -0.002 to -0.001); different photos of the same child cannot be detected without subject IDs, so subject-level leakage is neither shown nor excluded.
- Models differ (Friedman p 3.7e-10); M04 has the highest mean fold AUC and is significantly better than 7 of 9 models after Holm (M01, M03, M05, M06, M08, M09, M10); pretrained backbones average higher than the custom CNNs.

## Limitations (copied from the report, section 9)

- Sample size: D01 has 57 children (5-6 test children per fold); child-level CIs are wide and the matched analysis rests on 14 pairs.
- D02 has no subject identifiers: different photos of the same child can fall into training and test folds; only near-identical copies were grouped. Results are image-level, not child-level.
- D02 images were collected from the web with unknown diagnosis verification, age range, consent and acquisition conditions; image artefacts alone separate the classes.
- The custom CNNs were capped at 40 epochs by the fixed protocol and on D02 more than half of their runs reached the 40-epoch limit without meeting the early-stopping criterion (Stage 6 report), so their comparison with pretrained encoders is conservative for the CNNs.
- cuDNN benchmark mode was on: re-running a fold reproduces results only to about 1e-3 in the losses (Stage 4); the stored results are the reference.
- D01 clip-level and child-level results answer different questions; clips of one child are not independent.
- The tabular datasets are small public questionnaires; D05 set_all is label-definitional and D06 shows no signal.

## Where things are

| path | content |
|---|---|
| `reports/FINAL_RESULTS_REPORT.md` / `.html` | final results report (the HTML is self-contained) |
| `reports/STAGE_NN_REPORT.md` | one report per stage (S1-S10), each ending in `STAGE N DECISION` |
| `results/final/` | final tables (`master_table`, `shortcut_table`, `stats_table`, `fold_auc_matrix`, ...; .csv + .md; every number has a source reference) |
| `reports/figures/final/` | final figures F01-F12 (300 dpi PNG + SVG) |
| `results/<dataset>/<model>/fold_XX/` | per-run outputs: predictions.csv, metrics.json, history.csv, run_info.json, config.yaml (best_model.pt not in the submission) |
| `results/stage8/` | shortcut-control and statistics outputs |
| `results/FINAL_MANIFEST.json` | sha256 of every file in results/, splits/, reports/, configs/, src/, tabular/, scripts/ |
| `splits/` | frozen 10-fold split files + SPLITS_MANIFEST.json |
| `src/`, `scripts/`, `tabular/`, `configs/`, `tests/` | code, configuration, tests |
| `snapshots/stage_NN/` | local snapshots (no git): code copy + CHECKSUMS.sha256 |

## How to reproduce

Raw datasets are not included; set their location in `configs/paths.yaml` (the only file with absolute paths).

```bash
python3 -m venv --system-site-packages .venv && .venv/bin/python -m pip install -r requirements.txt
source .venv/bin/activate
python scripts/check_gpu.py                      # must print GPU CHECK: OK
python -m pytest -v                              # full test suite
# S1 audit:   python scripts/01_audit_datasets.py; python scripts/01_build_registry.py
# S2 data:    python scripts/02a_prepare_d01.py; python scripts/02b_prepare_d02.py; python scripts/02c_prepare_tabular.py
#             python scripts/02d_make_splits.py   (deterministic; splits are frozen in splits/)
# S3 models:  python scripts/03_profile_models.py; python scripts/03b_build_feature_cache.py
# S4 engine:  python scripts/train.py --dataset D01 --model M01 --folds 1-2 --max-epochs 2 --out results_smoke
# S5 D01:     bash scripts/stage5_launch.sh     (nohup; status: python scripts/stage5_status.py)
# S6 D02:     bash scripts/stage6_launch.sh     (nohup; status: python scripts/stage6_status.py); python scripts/06b_verify.py
# S7 tabular: CUDA_VISIBLE_DEVICES= python tabular/run_stage7.py
# S8:         python scripts/08a_preflight.py ... scripts/08f_write_report.py
# S9:         python scripts/09a_preflight.py; 09b_tables.py; 09c_figures.py; 09d_write_report.py; 09e_write_stage_report.py
# S10 audit:  CUDA_VISIBLE_DEVICES= python scripts/10_audit.py; python scripts/10b_finalize.py report|freeze
```

Training is deterministic up to cuDNN benchmark effects (~1e-3 in the losses); the stored results are the reference. Training resumes from checkpoints; DONE folds are never re-run unless `--force`.

## Frozen state

After Stage 10, `results/`, `splits/` and `reports/figures/final/` are read-only (`chmod -R a-w`). To undo: `chmod -R u+w results splits reports/figures/final`.
