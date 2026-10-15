# STAGE 3 REPORT - the 10 hybrid architectures + unit tests

Screening-support research, not diagnosis. No model was trained on real labels in this stage (tests use random inputs and random 0/1 labels only). No git: recorded by the local snapshot `snapshots/stage_03/`.

## 1. Summary

- `src/models/` implements one `HybridModel` for both input modes: spatial encoder -> projection (LayerNorm, Dropout 0.3, Linear -> 256, GELU) -> temporal encoder -> pooling -> classifier -> 1 logit (P(ASD) = sigmoid).
- All 10 models x 2 modes build, run forward/backward on cuda with fp16 autocast, and pass the tests (269 passed in 109.12s (0:01:49)).
- Pretrained weights: 6 timm ImageNet checkpoints downloaded and hashed (`reports/stage03_weights.csv`).
- Batch sizes (largest of 64/32/16/8/4 with peak < 10 GB): 64 for 20 of 20; highest steady-state peak 4.98 GB, highest first-step probe peak 7.53 GB; written to `configs/models/Mxx.yaml`.
- D01 feature cache (frozen backbones) implemented, equivalence-tested and built for 6 of 6 pretrained backbones (`cache/features/`).
- This stage was resumed after an interruption; see section 9.

## 2. Architectures (exact mapping)

| id | name | encoder | temporal | pooling / head | sequence: params total / trainable (M) | sequence: batch / peak GB | tokens: params total / trainable (M) | tokens: batch / peak GB |
|---|---|---|---|---|---|---|---|---|
| M01 | EfficientNetB3 + BiLSTM + Attention | efficientnet_b3.ra2_in1k | bilstm | additive attention -> Dropout -> Linear | 11.52 / 0.825 | 64 / 4.98 | 11.53 / 4.715 | 64 / 0.72 |
| M02 | CNN + BiLSTM + Attention | custom CNN 32-64-128-256 | bilstm | additive attention -> Dropout -> Linear | 0.88 / 0.884 | 64 / 2.73 | 0.93 / 0.934 | 64 / 1.53 |
| M03 | ResNet50 + LSTM + Attention | resnet50.a1_in1k | lstm | additive attention -> Dropout -> Linear | 24.25 / 0.743 | 64 / 3.38 | 24.26 / 15.720 | 64 / 0.66 |
| M04 | ConvNeXt + BiLSTM + Attention | convnext_tiny.in12k_ft_in1k | bilstm | additive attention -> Dropout -> Linear | 28.45 / 0.627 | 64 / 4.84 | 28.46 / 16.110 | 64 / 0.85 |
| M05 | CNN + Transformer + MLP | custom CNN 32-64-128-256 | transformer | mean pooling -> MLP 256-128-1 | 2.07 / 2.070 | 64 / 2.74 | 2.17 / 2.169 | 64 / 1.69 |
| M06 | DenseNet121 + LSTM + Attention | densenet121.ra_in1k | lstm | additive attention -> Dropout -> Linear | 7.43 / 0.479 | 64 / 3.01 | 7.45 / 2.652 | 64 / 0.44 |
| M07 | ViT + LSTM + Attention | vit_base_patch16_224.augreg2_in21k_ft_in1k | lstm | additive attention -> Dropout -> Linear | 86.21 / 0.413 | 64 / 3.23 | 86.26 / 7.552 | 64 / 1.04 |
| M08 | EfficientNet + Transformer + Attention | efficientnet_b0.ra_in1k | transformer | additive attention -> Dropout -> Linear | 5.95 / 1.946 | 64 / 3.52 | 5.98 / 3.098 | 64 / 0.50 |
| M09 | CNN + GRU + Attention | custom CNN 32-64-128-256 | gru | additive attention -> Dropout -> Linear | 0.62 / 0.620 | 64 / 2.73 | 0.67 / 0.670 | 64 / 1.47 |
| M10 | ResNet50 + BiLSTM + Transformer | resnet50.a1_in1k | bilstm_transformer | mean pooling -> Dropout -> Linear | 26.01 / 2.506 | 64 / 3.41 | 26.04 / 17.494 | 64 / 0.67 |

Sequence lengths: sequence mode L = 8 (time); tokens mode L = 7 x 7 = 49 (EfficientNet/ResNet/ConvNeXt/DenseNet) or 14 x 14 = 196 (ViT-B/16 patch tokens without CLS; custom CNN at 224 px). Tokens mode adds a learned positional embedding (L x 256) after the projection; the Transformer temporal encoders add their own learned positional embedding. M05 pools by the mean before its MLP head; M10 uses mean pooling (no attention).

Custom CNN (M02, M05, M09): 4 x [Conv3x3 -> BatchNorm -> ReLU -> MaxPool2], channels 32-64-128-256, Kaiming init, trained from scratch; input 112 x 112 in sequence mode (frames are resized inside the model with antialiasing) -> 7 x 7 x 256 map, 224 x 224 in tokens mode -> 14 x 14 x 256 map.

Preprocessing inside the model: uint8 -> float/255 -> (resize if needed) -> normalise with the mean/std the weights were trained with (ImageNet mean/std for all CNNs and the custom CNN; 0.5/0.5 for the ViT augreg weights, as given by timm's pretrained_cfg).

## 3. Pretrained weights (`reports/stage03_weights.csv`)

| models | timm_model | pretrained_tag | hf_hub_id | size_bytes | sha256 |
|---|---|---|---|---|---|
| M01 | efficientnet_b3 | ra2_in1k | timm/efficientnet_b3.ra2_in1k | 49335454 | 279d2a53898aa89dab43fd6bd7df9f706aea4cb9cf916223988bbcb8e5850469 |
| M02 M05 M09 | (none: custom CNN, random init, trained from scratch) |  |  |  |  |
| M03 M10 | resnet50 | a1_in1k | timm/resnet50.a1_in1k | 102469840 | 773525d5821de224f8f30c33377b7a795d7863e08522698200d3217d3f2a41bb |
| M04 | convnext_tiny | in12k_ft_in1k | timm/convnext_tiny.in12k_ft_in1k | 114374272 | a1aefa409b513cf209b085424eb3efffe4e4a9f511491bc2c12ea35209e6bb95 |
| M06 | densenet121 | ra_in1k | timm/densenet121.ra_in1k | 32334434 | eae900d8ae5b3716efa7f443fcc692aeb5a7f01f72581835c03c8c495f9eda10 |
| M07 | vit_base_patch16_224 | augreg2_in21k_ft_in1k | timm/vit_base_patch16_224.augreg2_in21k_ft_in1k | 346284714 | 32aa17d6e17b43500f531d5f6dc9bc93e56ed8841b8a75682e1bb295d722405b |
| M08 | efficientnet_b0 | ra_in1k | timm/efficientnet_b0.ra_in1k | 21355344 | d569899762ea9b1384ee07f4af64805cf8caa1c55f9253ebb1080dc40e87a2cd |

Note: the default timm tags were kept and pinned. Two are not pure ImageNet-1k: ConvNeXt-T `in12k_ft_in1k` (ImageNet-12k pre-training, ImageNet-1k fine-tune) and ViT-B/16 `augreg2_in21k_ft_in1k` (ImageNet-21k pre-training, ImageNet-1k fine-tune). EfficientNet-B3 `ra2_in1k` was trained at 288 px; it is used at 224 px (data contract).

## 4. Fine-tuning policy (config options; final choice confirmed in Stage 4)

| dataset / mode | pretrained backbones | custom CNN | learning rates |
|---|---|---|---|
| D01 / sequence | `frozen` (57 children; requires_grad=False, backbone in eval mode, run under no_grad; features can come from the cache) | `full` | head_lr 1e-3 |
| D02 / tokens | `last_stage` (EfficientNet: blocks.6 + conv_head + bn2; ResNet: layer4; ConvNeXt: stages.3; DenseNet: denseblock4 + norm5; ViT: blocks.11 + norm) | `full` | backbone_lr = head_lr / 10 = 1e-4 (custom CNN: 1e-3) |

Modules whose parameters are frozen stay in eval mode even when `model.train()` is called, so BatchNorm statistics in frozen parts are never updated (tested: frozen weights and running statistics are bit-identical after an optimizer step).

### Frozen-feature diversity on real D01 clips (no labels; `reports/stage03_feature_diversity.csv`)

| timm_name | real_D01_cos_between_clips | real_D01_cos_between_frames | random_noise_cos_between_clips | random_noise_cos_between_frames |
|---|---|---|---|---|
| efficientnet_b3.ra2_in1k | 0.6947 | 0.3479 | 0.9956 | 0.9645 |
| resnet50.a1_in1k | 0.8774 | 0.6427 | 0.9956 | 0.9662 |
| convnext_tiny.in12k_ft_in1k | 0.875 | 0.6792 | 0.9994 | 0.9959 |
| densenet121.ra_in1k | 0.8323 | 0.5632 | 0.9998 | 0.9987 |
| vit_base_patch16_224.augreg2_in21k_ft_in1k | 0.8645 | 0.6348 | 0.9946 | 0.9679 |
| efficientnet_b0.ra_in1k | 0.8491 | 0.6272 | 0.9984 | 0.9882 |

Lower mean cosine = more distinguishable inputs. Real gaze renderings give clearly more diverse frozen features than random noise, so a frozen backbone does carry clip-to-clip information. (This is also why the tiny-overfit test feeds random features to the head for frozen models: random images are nearly identical to a pretrained network.)

### Feature cache build (`logs/stage03_feature_cache.log`)

```
M01: efficientnet_b3.ra2_in1k -> /home/score/Documents/AI_project/asd_hybrid_10fold/cache/features/efficientnet_b3.ra2_in1k_D01.npy shape [1241, 8, 1536] float16 in 21.7 s
M03: resnet50.a1_in1k -> /home/score/Documents/AI_project/asd_hybrid_10fold/cache/features/resnet50.a1_in1k_D01.npy shape [1241, 8, 2048] float16 in 13.7 s
M04: convnext_tiny.in12k_ft_in1k -> /home/score/Documents/AI_project/asd_hybrid_10fold/cache/features/convnext_tiny.in12k_ft_in1k_D01.npy shape [1241, 8, 768] float16 in 24.1 s
M06: densenet121.ra_in1k -> /home/score/Documents/AI_project/asd_hybrid_10fold/cache/features/densenet121.ra_in1k_D01.npy shape [1241, 8, 1024] float16 in 17.0 s
M07: vit_base_patch16_224.augreg2_in21k_ft_in1k -> /home/score/Documents/AI_project/asd_hybrid_10fold/cache/features/vit_base_patch16_224.augreg2_in21k_ft_in1k_D01.npy shape [1241, 8, 768] float16 in 22.3 s
M08: efficientnet_b0.ra_in1k -> /home/score/Documents/AI_project/asd_hybrid_10fold/cache/features/efficientnet_b0.ra_in1k_D01.npy shape [1241, 8, 1280] float16 in 12.2 s
M10: shares the resnet50.a1_in1k cache built for M03
```

| cache file | shape | dtype | size_bytes | sha256 |
|---|---|---|---|---|
| cache/features/convnext_tiny.in12k_ft_in1k_D01.npy | (1241, 8, 768) | float16 | 15249536 | fb7f9ffcffb7bcdd22922eaae6474cf730f7d5137638727fe2832ad7387b2028 |
| cache/features/densenet121.ra_in1k_D01.npy | (1241, 8, 1024) | float16 | 20332672 | f62f547c20722b82da96e1baf76edbe0729e37ab8412f314df684a6a26082085 |
| cache/features/efficientnet_b0.ra_in1k_D01.npy | (1241, 8, 1280) | float16 | 25415808 | 541291c2ee4a61e32c3a94ca900e83acdfe40a0bab24febde9b8af004867498c |
| cache/features/efficientnet_b3.ra2_in1k_D01.npy | (1241, 8, 1536) | float16 | 30498944 | d47b18fce2946580a41f2511a53004fa9ebdc6e1165cc1a24a75041f4a7e707e |
| cache/features/resnet50.a1_in1k_D01.npy | (1241, 8, 2048) | float16 | 40665216 | ebfae0d69d7a1d70906c9f6c31c702daaa109706ec6818db09f3ec33702dc038 |
| cache/features/vit_base_patch16_224.augreg2_in21k_ft_in1k_D01.npy | (1241, 8, 768) | float16 | 15249536 | 113756f676ef1bec7b46ec431c93fbf11c55aafc5f198e31fa376876b17115ff |

## 5. Tests (`.venv/bin/python -m pytest -v`, full output, `logs/stage03_pytest.txt`)

Pre-flight (`logs/stage03_preflight.txt`):

```
# Stage 3 pre-flight 2026-10-13T17:24:54
A1 report last line: STAGE 2 DECISION: PASS
A1 snapshot stage_02 vs current (splits/, configs/): 69 recorded, 69 current, identical = True
A1 snapshot stage_02 vs current (all hashed dirs): identical = True
A1 SPLITS_MANIFEST hashes match: True | code_hash_at_split_time fa421378addc1eed72622c9223476192f929117a1decb1a6fc3c235676acf2a5
A3 code_hash(src,scripts,configs,tests) before Stage 3 = fa421378addc1eed72622c9223476192f929117a1decb1a6fc3c235676acf2a5
4096x4096 fp32 matmul on cuda: 31.47 ms (4.37 TFLOP/s)
max_memory_allocated = 264.1 MiB
GPU CHECK: OK
check_gpu exit=0
169 passed in 33.15s
```

Test design notes: forward/backward/determinism use random uint8 inputs; backward checks that every trainable tensor gets a gradient and changes after one AdamW step, frozen tensors keep grad None and stay identical; determinism builds each model twice with seed 42 (cudnn deterministic) and compares eval logits for exact equality; tiny overfit = 16 random samples, random labels, 60 steps at the config learning rates with dropout on (frozen backbones: random N(0,1) features into the head; others: full model on random coarse images); feature-cache test: live model vs head(cached fp16 features) on 4 real clips, |diff| <= 1e-3.

```
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0 -- /home/score/Documents/AI_project/asd_hybrid_10fold/.venv/bin/python
rootdir: /home/score/Documents/AI_project/asd_hybrid_10fold
configfile: pytest.ini
testpaths: tests
plugins: anyio-4.15.1
collecting ... collected 269 items

tests/test_setup.py::test_project_dir_and_symlink PASSED                 [  0%]
tests/test_setup.py::test_paths_yaml_every_path_exists PASSED            [  0%]
tests/test_setup.py::test_datasets_dir_outside_project PASSED            [  1%]
tests/test_setup.py::test_no_absolute_paths_outside_paths_yaml PASSED    [  1%]
tests/test_setup.py::test_venv_imports PASSED                            [  1%]
tests/test_setup.py::test_cuda_available PASSED                          [  2%]
tests/test_setup.py::test_check_gpu_script_exits_zero PASSED             [  2%]
tests/test_setup.py::test_claude_md_sections PASSED                      [  2%]
tests/test_setup.py::test_raw_data_snapshot_matches PASSED               [  3%]
tests/test_setup.py::test_gitignore[.venv/] PASSED                       [  3%]
tests/test_setup.py::test_gitignore[cache/] PASSED                       [  4%]
tests/test_setup.py::test_gitignore[*.pt] PASSED                         [  4%]
tests/test_setup.py::test_gitignore[logs/*.log] PASSED                   [  4%]
tests/test_stage01.py::test_skeleton_dir_exists[configs] PASSED          [  5%]
tests/test_stage01.py::test_skeleton_dir_exists[src/data] PASSED         [  5%]
tests/test_stage01.py::test_skeleton_dir_exists[src/models] PASSED       [  5%]
tests/test_stage01.py::test_skeleton_dir_exists[src/training] PASSED     [  6%]
tests/test_stage01.py::test_skeleton_dir_exists[src/evaluation] PASSED   [  6%]
tests/test_stage01.py::test_skeleton_dir_exists[src/utils] PASSED        [  7%]
tests/test_stage01.py::test_skeleton_dir_exists[scripts] PASSED          [  7%]
tests/test_stage01.py::test_skeleton_dir_exists[tests] PASSED            [  7%]
tests/test_stage01.py::test_skeleton_dir_exists[data_registry] PASSED    [  8%]
tests/test_stage01.py::test_skeleton_dir_exists[splits] PASSED           [  8%]
tests/test_stage01.py::test_skeleton_dir_exists[cache] PASSED            [  8%]
tests/test_stage01.py::test_skeleton_dir_exists[results] PASSED          [  9%]
tests/test_stage01.py::test_skeleton_dir_exists[reports] PASSED          [  9%]
tests/test_stage01.py::test_skeleton_dir_exists[logs] PASSED             [ 10%]
tests/test_stage01.py::test_results_readme_and_gitignore PASSED          [ 10%]
tests/test_stage01.py::test_audit_json_loads_and_nonzero[D01_eye_tracking] PASSED [ 10%]
tests/test_stage01.py::test_audit_json_loads_and_nonzero[D02_facial_images_A] PASSED [ 11%]
tests/test_stage01.py::test_audit_json_loads_and_nonzero[D03_facial_images_B] PASSED [ 11%]
tests/test_stage01.py::test_audit_json_loads_and_nonzero[D04_autism_prediction] PASSED [ 11%]
tests/test_stage01.py::test_audit_json_loads_and_nonzero[D05_toddler_screening] PASSED [ 12%]
tests/test_stage01.py::test_audit_json_loads_and_nonzero[D06_early_autism] PASSED [ 12%]
tests/test_stage01.py::test_registry_exists PASSED                       [ 13%]
tests/test_stage01.py::test_raw_data_unmodified PASSED                   [ 13%]
tests/test_stage01.py::test_cuda_available PASSED                        [ 13%]
tests/test_stage02_d01.py::test_no_unidentified_rows PASSED              [ 14%]
tests/test_stage02_d01.py::test_no_duplicate_participant_time_trial PASSED [ 14%]
tests/test_stage02_d01.py::test_no_cars_anywhere_in_cache_or_splits PASSED [ 14%]
tests/test_stage02_d01.py::test_npy_shape_matches_index PASSED           [ 15%]
tests/test_stage02_d01.py::test_clip_rules PASSED                        [ 15%]
tests/test_stage02_d01.py::test_no_clip_spans_two_trials PASSED          [ 15%]
tests/test_stage02_d01.py::test_20_random_clips_rerendered_from_raw_are_identical PASSED [ 16%]
tests/test_stage02_d01.py::test_rebuilt_gaze_equals_cached PASSED        [ 16%]
tests/test_stage02_d01.py::test_quality_table_not_in_model_inputs PASSED [ 17%]
tests/test_stage02_d02.py::test_manifest_pooled_and_complete PASSED      [ 17%]
tests/test_stage02_d02.py::test_no_d03_path_anywhere PASSED              [ 17%]
tests/test_stage02_d02.py::test_near_duplicate_pairs_recomputed_lie_in_same_outer_fold PASSED [ 18%]
tests/test_stage02_d02.py::test_no_group_has_mixed_labels PASSED         [ 18%]
tests/test_stage02_snapshot.py::test_latest_snapshot_checksums_match_current_files PASSED [ 18%]
tests/test_stage02_snapshot.py::test_latest_snapshot_info_and_code_copy PASSED [ 19%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[1-D01_eye_tracking] PASSED [ 19%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[1-D02_facial_images_A] PASSED [ 20%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[1-D04_autism_prediction] PASSED [ 20%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[1-D05_toddler_screening] PASSED [ 20%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[1-D06_early_autism] PASSED [ 21%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[2-D01_eye_tracking] PASSED [ 21%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[2-D02_facial_images_A] PASSED [ 21%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[2-D04_autism_prediction] PASSED [ 22%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[2-D05_toddler_screening] PASSED [ 22%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[2-D06_early_autism] PASSED [ 23%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[3-D01_eye_tracking] PASSED [ 23%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[3-D02_facial_images_A] PASSED [ 23%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[3-D04_autism_prediction] PASSED [ 24%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[3-D05_toddler_screening] PASSED [ 24%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[3-D06_early_autism] PASSED [ 24%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[4-D01_eye_tracking] PASSED [ 25%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[4-D02_facial_images_A] PASSED [ 25%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[4-D04_autism_prediction] PASSED [ 26%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[4-D05_toddler_screening] PASSED [ 26%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[4-D06_early_autism] PASSED [ 26%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[5-D01_eye_tracking] PASSED [ 27%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[5-D02_facial_images_A] PASSED [ 27%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[5-D04_autism_prediction] PASSED [ 27%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[5-D05_toddler_screening] PASSED [ 28%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[5-D06_early_autism] PASSED [ 28%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[6-D01_eye_tracking] PASSED [ 28%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[6-D02_facial_images_A] PASSED [ 29%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[6-D04_autism_prediction] PASSED [ 29%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[6-D05_toddler_screening] PASSED [ 30%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[6-D06_early_autism] PASSED [ 30%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[7-D01_eye_tracking] PASSED [ 30%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[7-D02_facial_images_A] PASSED [ 31%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[7-D04_autism_prediction] PASSED [ 31%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[7-D05_toddler_screening] PASSED [ 31%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[7-D06_early_autism] PASSED [ 32%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[8-D01_eye_tracking] PASSED [ 32%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[8-D02_facial_images_A] PASSED [ 33%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[8-D04_autism_prediction] PASSED [ 33%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[8-D05_toddler_screening] PASSED [ 33%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[8-D06_early_autism] PASSED [ 34%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[9-D01_eye_tracking] PASSED [ 34%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[9-D02_facial_images_A] PASSED [ 34%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[9-D04_autism_prediction] PASSED [ 35%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[9-D05_toddler_screening] PASSED [ 35%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[9-D06_early_autism] PASSED [ 36%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[10-D01_eye_tracking] PASSED [ 36%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[10-D02_facial_images_A] PASSED [ 36%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[10-D04_autism_prediction] PASSED [ 37%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[10-D05_toddler_screening] PASSED [ 37%]
tests/test_stage02_splits.py::test_parts_disjoint_in_samples_and_groups[10-D06_early_autism] PASSED [ 37%]
tests/test_stage02_splits.py::test_every_sample_in_exactly_one_test_fold[D01_eye_tracking] PASSED [ 38%]
tests/test_stage02_splits.py::test_every_sample_in_exactly_one_test_fold[D02_facial_images_A] PASSED [ 38%]
tests/test_stage02_splits.py::test_every_sample_in_exactly_one_test_fold[D04_autism_prediction] PASSED [ 39%]
tests/test_stage02_splits.py::test_every_sample_in_exactly_one_test_fold[D05_toddler_screening] PASSED [ 39%]
tests/test_stage02_splits.py::test_every_sample_in_exactly_one_test_fold[D06_early_autism] PASSED [ 39%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[1-D01_eye_tracking] PASSED [ 40%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[1-D02_facial_images_A] PASSED [ 40%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[1-D04_autism_prediction] PASSED [ 40%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[1-D05_toddler_screening] PASSED [ 41%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[1-D06_early_autism] PASSED [ 41%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[2-D01_eye_tracking] PASSED [ 42%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[2-D02_facial_images_A] PASSED [ 42%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[2-D04_autism_prediction] PASSED [ 42%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[2-D05_toddler_screening] PASSED [ 43%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[2-D06_early_autism] PASSED [ 43%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[3-D01_eye_tracking] PASSED [ 43%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[3-D02_facial_images_A] PASSED [ 44%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[3-D04_autism_prediction] PASSED [ 44%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[3-D05_toddler_screening] PASSED [ 44%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[3-D06_early_autism] PASSED [ 45%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[4-D01_eye_tracking] PASSED [ 45%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[4-D02_facial_images_A] PASSED [ 46%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[4-D04_autism_prediction] PASSED [ 46%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[4-D05_toddler_screening] PASSED [ 46%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[4-D06_early_autism] PASSED [ 47%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[5-D01_eye_tracking] PASSED [ 47%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[5-D02_facial_images_A] PASSED [ 47%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[5-D04_autism_prediction] PASSED [ 48%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[5-D05_toddler_screening] PASSED [ 48%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[5-D06_early_autism] PASSED [ 49%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[6-D01_eye_tracking] PASSED [ 49%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[6-D02_facial_images_A] PASSED [ 49%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[6-D04_autism_prediction] PASSED [ 50%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[6-D05_toddler_screening] PASSED [ 50%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[6-D06_early_autism] PASSED [ 50%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[7-D01_eye_tracking] PASSED [ 51%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[7-D02_facial_images_A] PASSED [ 51%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[7-D04_autism_prediction] PASSED [ 52%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[7-D05_toddler_screening] PASSED [ 52%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[7-D06_early_autism] PASSED [ 52%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[8-D01_eye_tracking] PASSED [ 53%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[8-D02_facial_images_A] PASSED [ 53%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[8-D04_autism_prediction] PASSED [ 53%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[8-D05_toddler_screening] PASSED [ 54%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[8-D06_early_autism] PASSED [ 54%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[9-D01_eye_tracking] PASSED [ 55%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[9-D02_facial_images_A] PASSED [ 55%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[9-D04_autism_prediction] PASSED [ 55%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[9-D05_toddler_screening] PASSED [ 56%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[9-D06_early_autism] PASSED [ 56%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[10-D01_eye_tracking] PASSED [ 56%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[10-D02_facial_images_A] PASSED [ 57%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[10-D04_autism_prediction] PASSED [ 57%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[10-D05_toddler_screening] PASSED [ 57%]
tests/test_stage02_splits.py::test_test_and_val_contain_both_classes[10-D06_early_autism] PASSED [ 58%]
tests/test_stage02_splits.py::test_d01_test_folds_have_asd_and_td_children PASSED [ 58%]
tests/test_stage02_splits.py::test_splits_manifest_hashes_match PASSED   [ 59%]
tests/test_stage02_splits.py::test_rerun_make_splits_is_byte_identical PASSED [ 59%]
tests/test_stage02_tabular.py::test_d04_only_labelled_train_rows PASSED  [ 59%]
tests/test_stage02_tabular.py::test_d04_result_decision_recorded PASSED  [ 60%]
tests/test_stage02_tabular.py::test_d05_honest_set_excludes_items_and_score PASSED [ 60%]
tests/test_stage02_tabular.py::test_d05_label_rule_on_clean_data PASSED  [ 60%]
tests/test_stage02_tabular.py::test_d05_other_csv_never_read_by_stage2_code PASSED [ 61%]
tests/test_stage02_tabular.py::test_d06_empty_column_absent PASSED       [ 61%]
tests/test_stage02_tabular.py::test_identical_feature_rows_share_group PASSED [ 62%]
tests/test_stage02_tabular.py::test_every_column_has_a_role PASSED       [ 62%]
tests/test_stage02_tabular.py::test_raw_data_snapshot_unchanged PASSED   [ 62%]
tests/test_stage03_models.py::test_config_mapping[M01] PASSED            [ 63%]
tests/test_stage03_models.py::test_config_mapping[M02] PASSED            [ 63%]
tests/test_stage03_models.py::test_config_mapping[M03] PASSED            [ 63%]
tests/test_stage03_models.py::test_config_mapping[M04] PASSED            [ 64%]
tests/test_stage03_models.py::test_config_mapping[M05] PASSED            [ 64%]
tests/test_stage03_models.py::test_config_mapping[M06] PASSED            [ 65%]
tests/test_stage03_models.py::test_config_mapping[M07] PASSED            [ 65%]
tests/test_stage03_models.py::test_config_mapping[M08] PASSED            [ 65%]
tests/test_stage03_models.py::test_config_mapping[M09] PASSED            [ 66%]
tests/test_stage03_models.py::test_config_mapping[M10] PASSED            [ 66%]
tests/test_stage03_models.py::test_factory_rejects_unknown PASSED        [ 66%]
tests/test_stage03_models.py::test_forward[M01-sequence] PASSED          [ 67%]
tests/test_stage03_models.py::test_forward[M01-tokens] PASSED            [ 67%]
tests/test_stage03_models.py::test_forward[M02-sequence] PASSED          [ 68%]
tests/test_stage03_models.py::test_forward[M02-tokens] PASSED            [ 68%]
tests/test_stage03_models.py::test_forward[M03-sequence] PASSED          [ 68%]
tests/test_stage03_models.py::test_forward[M03-tokens] PASSED            [ 69%]
tests/test_stage03_models.py::test_forward[M04-sequence] PASSED          [ 69%]
tests/test_stage03_models.py::test_forward[M04-tokens] PASSED            [ 69%]
tests/test_stage03_models.py::test_forward[M05-sequence] PASSED          [ 70%]
tests/test_stage03_models.py::test_forward[M05-tokens] PASSED            [ 70%]
tests/test_stage03_models.py::test_forward[M06-sequence] PASSED          [ 71%]
tests/test_stage03_models.py::test_forward[M06-tokens] PASSED            [ 71%]
tests/test_stage03_models.py::test_forward[M07-sequence] PASSED          [ 71%]
tests/test_stage03_models.py::test_forward[M07-tokens] PASSED            [ 72%]
tests/test_stage03_models.py::test_forward[M08-sequence] PASSED          [ 72%]
tests/test_stage03_models.py::test_forward[M08-tokens] PASSED            [ 72%]
tests/test_stage03_models.py::test_forward[M09-sequence] PASSED          [ 73%]
tests/test_stage03_models.py::test_forward[M09-tokens] PASSED            [ 73%]
tests/test_stage03_models.py::test_forward[M10-sequence] PASSED          [ 73%]
tests/test_stage03_models.py::test_forward[M10-tokens] PASSED            [ 74%]
tests/test_stage03_models.py::test_backward_and_step[M01-sequence] PASSED [ 74%]
tests/test_stage03_models.py::test_backward_and_step[M01-tokens] PASSED  [ 75%]
tests/test_stage03_models.py::test_backward_and_step[M02-sequence] PASSED [ 75%]
tests/test_stage03_models.py::test_backward_and_step[M02-tokens] PASSED  [ 75%]
tests/test_stage03_models.py::test_backward_and_step[M03-sequence] PASSED [ 76%]
tests/test_stage03_models.py::test_backward_and_step[M03-tokens] PASSED  [ 76%]
tests/test_stage03_models.py::test_backward_and_step[M04-sequence] PASSED [ 76%]
tests/test_stage03_models.py::test_backward_and_step[M04-tokens] PASSED  [ 77%]
tests/test_stage03_models.py::test_backward_and_step[M05-sequence] PASSED [ 77%]
tests/test_stage03_models.py::test_backward_and_step[M05-tokens] PASSED  [ 78%]
tests/test_stage03_models.py::test_backward_and_step[M06-sequence] PASSED [ 78%]
tests/test_stage03_models.py::test_backward_and_step[M06-tokens] PASSED  [ 78%]
tests/test_stage03_models.py::test_backward_and_step[M07-sequence] PASSED [ 79%]
tests/test_stage03_models.py::test_backward_and_step[M07-tokens] PASSED  [ 79%]
tests/test_stage03_models.py::test_backward_and_step[M08-sequence] PASSED [ 79%]
tests/test_stage03_models.py::test_backward_and_step[M08-tokens] PASSED  [ 80%]
tests/test_stage03_models.py::test_backward_and_step[M09-sequence] PASSED [ 80%]
tests/test_stage03_models.py::test_backward_and_step[M09-tokens] PASSED  [ 81%]
tests/test_stage03_models.py::test_backward_and_step[M10-sequence] PASSED [ 81%]
tests/test_stage03_models.py::test_backward_and_step[M10-tokens] PASSED  [ 81%]
tests/test_stage03_models.py::test_determinism_eval[M01-sequence] PASSED [ 82%]
tests/test_stage03_models.py::test_determinism_eval[M01-tokens] PASSED   [ 82%]
tests/test_stage03_models.py::test_determinism_eval[M02-sequence] PASSED [ 82%]
tests/test_stage03_models.py::test_determinism_eval[M02-tokens] PASSED   [ 83%]
tests/test_stage03_models.py::test_determinism_eval[M03-sequence] PASSED [ 83%]
tests/test_stage03_models.py::test_determinism_eval[M03-tokens] PASSED   [ 84%]
tests/test_stage03_models.py::test_determinism_eval[M04-sequence] PASSED [ 84%]
tests/test_stage03_models.py::test_determinism_eval[M04-tokens] PASSED   [ 84%]
tests/test_stage03_models.py::test_determinism_eval[M05-sequence] PASSED [ 85%]
tests/test_stage03_models.py::test_determinism_eval[M05-tokens] PASSED   [ 85%]
tests/test_stage03_models.py::test_determinism_eval[M06-sequence] PASSED [ 85%]
tests/test_stage03_models.py::test_determinism_eval[M06-tokens] PASSED   [ 86%]
tests/test_stage03_models.py::test_determinism_eval[M07-sequence] PASSED [ 86%]
tests/test_stage03_models.py::test_determinism_eval[M07-tokens] PASSED   [ 86%]
tests/test_stage03_models.py::test_determinism_eval[M08-sequence] PASSED [ 87%]
tests/test_stage03_models.py::test_determinism_eval[M08-tokens] PASSED   [ 87%]
tests/test_stage03_models.py::test_determinism_eval[M09-sequence] PASSED [ 88%]
tests/test_stage03_models.py::test_determinism_eval[M09-tokens] PASSED   [ 88%]
tests/test_stage03_models.py::test_determinism_eval[M10-sequence] PASSED [ 88%]
tests/test_stage03_models.py::test_determinism_eval[M10-tokens] PASSED   [ 89%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M01-sequence] PASSED [ 89%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M01-tokens] PASSED [ 89%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M02-sequence] PASSED [ 90%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M02-tokens] PASSED [ 90%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M03-sequence] PASSED [ 91%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M03-tokens] PASSED [ 91%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M04-sequence] PASSED [ 91%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M04-tokens] PASSED [ 92%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M05-sequence] PASSED [ 92%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M05-tokens] PASSED [ 92%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M06-sequence] PASSED [ 93%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M06-tokens] PASSED [ 93%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M07-sequence] PASSED [ 94%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M07-tokens] PASSED [ 94%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M08-sequence] PASSED [ 94%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M08-tokens] PASSED [ 95%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M09-sequence] PASSED [ 95%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M09-tokens] PASSED [ 95%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M10-sequence] PASSED [ 96%]
tests/test_stage03_models.py::test_tiny_overfit_random_data[M10-tokens] PASSED [ 96%]
tests/test_stage03_models.py::test_feature_cache_equivalence[M01] PASSED [ 97%]
tests/test_stage03_models.py::test_feature_cache_equivalence[M03] PASSED [ 97%]
tests/test_stage03_models.py::test_feature_cache_equivalence[M04] PASSED [ 97%]
tests/test_stage03_models.py::test_feature_cache_equivalence[M06] PASSED [ 98%]
tests/test_stage03_models.py::test_feature_cache_equivalence[M07] PASSED [ 98%]
tests/test_stage03_models.py::test_feature_cache_equivalence[M08] PASSED [ 98%]
tests/test_stage03_models.py::test_feature_cache_equivalence[M10] PASSED [ 99%]
tests/test_stage03_models.py::test_last_stage_trains_part_of_backbone PASSED [ 99%]
tests/test_stage03_models.py::test_model_configs_have_batch_sizes PASSED [100%]

======================= 269 passed in 109.12s (0:01:49) ========================
```

## 6. Profiling (`reports/stage03_model_profile.csv`)

One train step = AdamW + fp16 autocast + GradScaler on random inputs; peak = torch.cuda.max_memory_allocated during 25 train steps at the chosen batch size; `probe_peak_gb` = peak of the first train step at that batch size (includes the cudnn.benchmark workspace search; this is the value compared with the 10 GB limit when choosing the largest of 64/32/16/8/4, from `logs/stage03_profile.log`); times = median of 20 after 5 warm-ups. `_cached` = head-only step on cached features (frozen D01 path). cudnn.benchmark = True.

| model_id | mode | policy | seq_len | params_total | params_trainable | batch_size | probe_peak_gb | peak_mem_gb | s_per_train_step | s_per_infer_batch | steps_per_epoch | est_fold_min_live | s_per_train_step_cached | est_fold_min_cached |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M01 | sequence | frozen | 8 | 11521321 | 825089 | 64 | 7.53 | 4.978 | 0.6321 | 0.6355 | 15 | 6.32 | 0.00288 | 0.029 |
| M01 | tokens | last_stage | 49 | 11533865 | 4714747 | 64 | 1.0 | 0.719 | 0.0972 | 0.0828 | 36 | 2.33 |  |  |
| M02 | sequence | full | 8 | 883745 | 883745 | 64 | 3.17 | 2.728 | 0.1579 | 0.0508 | 15 | 1.58 |  |  |
| M02 | tokens | full | 196 | 933921 | 933921 | 64 | 1.57 | 1.529 | 0.081 | 0.0237 | 36 | 1.94 |  |  |
| M03 | sequence | frozen | 8 | 24251073 | 743041 | 64 | 6.0 | 3.385 | 0.5878 | 0.5849 | 15 | 5.88 | 0.00234 | 0.023 |
| M03 | tokens | last_stage | 49 | 24263617 | 15720321 | 64 | 0.85 | 0.655 | 0.0983 | 0.0765 | 36 | 2.36 |  |  |
| M04 | sequence | frozen | 8 | 28447073 | 626945 | 64 | 4.83 | 4.838 | 0.746 | 0.7414 | 15 | 7.46 | 0.00294 | 0.029 |
| M04 | tokens | last_stage | 49 | 28459617 | 16110081 | 64 | 0.72 | 0.852 | 0.1194 | 0.0967 | 36 | 2.87 |  |  |
| M05 | sequence | full | 8 | 2070305 | 2070305 | 64 | 2.73 | 2.743 | 0.1607 | 0.0517 | 15 | 1.61 |  |  |
| M05 | tokens | full | 196 | 2168609 | 2168609 | 64 | 1.67 | 1.689 | 0.0982 | 0.0285 | 36 | 2.36 |  |  |
| M06 | sequence | frozen | 8 | 7432705 | 478849 | 64 | 4.13 | 3.006 | 0.6687 | 0.6639 | 15 | 6.69 | 0.00247 | 0.025 |
| M06 | tokens | last_stage | 49 | 7445249 | 2651521 | 64 | 0.56 | 0.44 | 0.0994 | 0.087 | 36 | 2.38 |  |  |
| M07 | sequence | frozen | 8 | 86211457 | 412801 | 64 | 3.23 | 3.234 | 1.1905 | 1.1866 | 15 | 11.91 | 0.0025 | 0.025 |
| M07 | tokens | last_stage | 196 | 86261633 | 7552385 | 64 | 0.98 | 1.042 | 0.1837 | 0.1484 | 36 | 4.41 |  |  |
| M08 | sequence | frozen | 8 | 5953405 | 1945857 | 64 | 5.75 | 3.525 | 0.326 | 0.3246 | 15 | 3.26 | 0.00437 | 0.044 |
| M08 | tokens | last_stage | 49 | 5976445 | 3098289 | 64 | 0.76 | 0.503 | 0.0547 | 0.0437 | 36 | 1.31 |  |  |
| M09 | sequence | full | 8 | 620193 | 620193 | 64 | 2.72 | 2.725 | 0.1601 | 0.0513 | 15 | 1.6 |  |  |
| M09 | tokens | full | 196 | 670369 | 670369 | 64 | 1.47 | 1.471 | 0.081 | 0.0245 | 36 | 1.94 |  |  |
| M10 | sequence | frozen | 8 | 26014273 | 2506241 | 64 | 3.39 | 3.406 | 0.5962 | 0.5912 | 15 | 5.96 | 0.00509 | 0.051 |
| M10 | tokens | last_stage | 49 | 26037313 | 17494017 | 64 | 0.53 | 0.674 | 0.1063 | 0.0791 | 36 | 2.55 |  |  |

## 7. Estimated training time

ESTIMATE = ceil(mean train samples per fold / batch size) x measured s/step x 40 epochs (an ASSUMED epoch budget; Stage 4 sets the real one, with early stopping) x 10 folds. Excludes validation passes, data loading (D02 decodes JPEGs from disk) and early stopping, so it is a GPU-compute lower bound per epoch budget.

- **S5 (D01, 10 models x 10 folds)** with the feature cache for the 7 frozen-backbone models (custom CNNs live): **0.84 h** (+ one-time cache build, section 4); all backbones live instead: 8.69 h.
- **S6 (D02, 10 models x 10 folds, last_stage / full)**: **4.07 h**.

| model_id | mode | policy | batch_size | steps_per_epoch | est_min_per_fold |
|---|---|---|---|---|---|
| M01 | sequence | frozen | 64 | 15 | 0.029 |
| M01 | tokens | last_stage | 64 | 36 | 2.33 |
| M02 | sequence | full | 64 | 15 | 1.58 |
| M02 | tokens | full | 64 | 36 | 1.94 |
| M03 | sequence | frozen | 64 | 15 | 0.023 |
| M03 | tokens | last_stage | 64 | 36 | 2.36 |
| M04 | sequence | frozen | 64 | 15 | 0.029 |
| M04 | tokens | last_stage | 64 | 36 | 2.87 |
| M05 | sequence | full | 64 | 15 | 1.61 |
| M05 | tokens | full | 64 | 36 | 2.36 |
| M06 | sequence | frozen | 64 | 15 | 0.025 |
| M06 | tokens | last_stage | 64 | 36 | 2.38 |
| M07 | sequence | frozen | 64 | 15 | 0.025 |
| M07 | tokens | last_stage | 64 | 36 | 4.41 |
| M08 | sequence | frozen | 64 | 15 | 0.044 |
| M08 | tokens | last_stage | 64 | 36 | 1.31 |
| M09 | sequence | full | 64 | 15 | 1.6 |
| M09 | tokens | full | 64 | 36 | 1.94 |
| M10 | sequence | frozen | 64 | 15 | 0.051 |
| M10 | tokens | last_stage | 64 | 36 | 2.55 |

## 8. Open items for Stage 4

1. Confirm the fine-tuning policy (defaults above) and learning rates; D01 with frozen backbones should use the feature cache (call `load_cache`, which refuses a stale cache).
2. D01 class imbalance (437 ASD vs 804 TD clips): class-weighted loss or balanced sampling; early stopping on a validation metric that is robust with 7-8 validation children.
3. D02 augmentation (train only) is applied to uint8 images before the model; resizing to 224 in the loader.
4. Data-loader rules: pin_memory, num_workers = min(8, cpu_count), seeds; checkpoint/resume; run_info.json.
5. Call `src.data.splits.verify_manifest()` before every run; log max_memory_allocated per run.
6. Decide the epoch budget / early-stopping patience and re-check the section 7 estimate against the Stage 4 smoke test (which includes data loading and validation).

## 9. Interruption recovery

The previous session stopped during the profiling run. On resume nothing was deleted or regenerated blindly; every Stage 3 file was read and checked by running it (`logs/stage03_recovery_checklist.md`):

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

What was fixed / built in the resumed session:

1. `src/models/feature_cache.py` (and the cache test): copy memmap slices with `np.array` (removes the non-writable-tensor UserWarning; values unchanged).
2. `scripts/03_profile_models.py`: weight `file` paths stored relative to the Hugging Face hub cache (no absolute home-directory path in `reports/stage03_weights.csv`); `est_fold_min_*` columns added; the 40-epoch figure marked as an assumption.
3. `scripts/03b_build_feature_cache.py` run -> `cache/features/` (was missing).
4. `scripts/03_profile_models.py` re-run in full -> `reports/stage03_model_profile.csv`, batch sizes written to all 10 configs (fixes the one failing test). Weight sha256 values were re-verified; no weight file was re-downloaded (none was corrupt).
5. This report writer: summary numbers computed from the outputs instead of fixed text; cache table, minutes per fold and this section added.

STAGE 3 DECISION: PASS
