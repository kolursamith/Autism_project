State found on resume (2026-10-14 08:52). Previous session stopped 2026-10-13 ~17:40 during scripts/03_profile_models.py
(log ends at "M03 sequence bs=64: peak 6.00 GB", no traceback, no process left). Verification run: `pytest -v tests/test_stage03_models.py`
-> 99 passed, 1 failed (logs/stage03_pytest_resume_check.txt); weight files re-hashed against reports/stage03_weights.csv.

Pre-checks: STAGE_02_REPORT ends "STAGE 2 DECISION: PASS"; verify_manifest() OK (split hashes match); nvidia-smi: no compute
process on the GPU (only Xorg / gnome-shell / browser graphics contexts); no stale python job.

| item | requirement | found | evidence / action |
|---|---|---|---|
| B | sequence (B,8,3,224,224) + tokens (B,3,224,224) modes, normalisation inside the model | DONE | test_forward 20/20 pass; SpatialEncoder.preprocess |
| C1 | encoders.py temporal.py heads.py hybrid.py factory.py | DONE | all import; 100 tests collect |
| C2 | exact M01-M10 mapping | DONE | test_config_mapping 10/10 |
| C3 | timm pretrained=True weights + reports/stage03_weights.csv (name, tag, sha256) | DONE | 6 files, size + sha256 re-verified, no partial downloads; csv `file` column held absolute paths -> made relative to the HF hub cache on re-run |
| C4 | custom CNN spec documented | PARTIAL | in encoders.py docstring + report writer; report not yet written (-> G) |
| D1 | backbone_trainable policy (D01 frozen, D02 last_stage, custom CNN full, frozen modules in eval) | DONE | test_backward_and_step 20/20, test_last_stage_trains_part_of_backbone |
| D2 | feature_cache.py + live-vs-cached equivalence test (1e-3) | DONE | test_feature_cache_equivalence 7/7 (one harmless non-writable-numpy UserWarning -> fixed) |
| D3 | cache/features/ built for the frozen D01 backbones | MISSING | directory absent -> build with scripts/03b_build_feature_cache.py |
| E | tests/test_stage03_models.py: forward, attention, backward/grad, optimizer step, determinism, tiny overfit | DONE | 99/100 pass; the 1 failure is test_model_configs_have_batch_sizes (batch_size: null, depends on F) |
| F | scripts/03_profile_models.py -> reports/stage03_model_profile.csv + batch sizes in configs | PARTIAL | script exists; run interrupted after 2.5 of 10 models; no profile csv; all batch_size null -> re-run in full |
| G1 | reports/STAGE_03_REPORT.md | MISSING | scripts/03c_write_report.py exists but never ran (inputs missing); some summary sentences were hard-coded -> now computed |
| G2 | CLAUDE.md update | PARTIAL | "Model contracts" section present; stage table still "not started"; no Stage 3 session-log entry |
| G3 | snapshots/stage_03/ | MISSING | to be created with scripts/make_snapshot.py |
