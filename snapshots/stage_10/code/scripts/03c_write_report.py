"""Stage 3 - write reports/STAGE_03_REPORT.md from the Stage 3 outputs (no hand-typed numbers).
Usage: .venv/bin/python scripts/03c_write_report.py"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.common import CACHE_DIR, LOG_DIR, PROJECT_DIR, md_table, sha256_file  # noqa: E402
from src.models.factory import MODEL_IDS, load_model_cfg  # noqa: E402

REP = os.path.join(PROJECT_DIR, "reports")


def R(p):
    with open(p) as fh:
        return fh.read().rstrip("\n")


def main():
    prof = pd.read_csv(os.path.join(REP, "stage03_model_profile.csv"))
    w = pd.read_csv(os.path.join(REP, "stage03_weights.csv"), dtype=str, keep_default_na=False)
    div = pd.read_csv(os.path.join(REP, "stage03_feature_diversity.csv"))
    pyt = R(os.path.join(LOG_DIR, "stage03_pytest.txt"))
    pre = R(os.path.join(LOG_DIR, "stage03_preflight.txt"))
    cache_log = R(os.path.join(LOG_DIR, "stage03_feature_cache.log"))
    last = pyt.splitlines()[-1]
    failed = re.search(r"(\d+) (failed|error)", last)

    arch = []
    for mid in MODEL_IDS:
        c = load_model_cfg(mid)
        enc = c["encoder"].get("timm_name", "custom CNN 32-64-128-256")
        t = c["temporal"]["type"]
        head = {"attention": "additive attention", "mean": "mean pooling"}[c["pooling"]] + \
            (" -> MLP 256-128-1" if c["head"]["type"] == "mlp" else " -> Dropout -> Linear")
        r = {"id": mid, "name": c["name"], "encoder": enc, "temporal": t, "pooling / head": head}
        for mode in ["sequence", "tokens"]:
            p = prof[(prof.model_id == mid) & (prof["mode"] == mode)].iloc[0]
            r[f"{mode}: params total / trainable (M)"] = f"{p.params_total / 1e6:.2f} / {p.params_trainable / 1e6:.3f}"
            r[f"{mode}: batch / peak GB"] = f"{int(p.batch_size)} / {p.peak_mem_gb:.2f}"
        arch.append(r)
    arch = pd.DataFrame(arch)

    seq = prof[prof["mode"] == "sequence"]
    tok = prof[prof["mode"] == "tokens"]
    s5_live = (seq.est_fold_hours_live * 10).sum()
    s5_mixed = sum((r.est_fold_hours_cached if r.policy == "frozen" else r.est_fold_hours_live) * 10
                   for r in seq.itertuples())
    s6 = (tok.est_fold_hours_live * 10).sum()
    bs_counts = prof.batch_size.value_counts().sort_index(ascending=False)
    bs_text = ", ".join(f"{int(b)} for {n} of {len(prof)}" for b, n in bs_counts.items())
    max_peak = prof.peak_mem_gb.max()

    caches = []
    for name in sorted(set(w.hf_hub_id[w.hf_hub_id != ""].str.replace("timm/", "", regex=False))):
        p = os.path.join(CACHE_DIR, "features", f"{name}_D01.npy")
        if os.path.isfile(p):
            a = np.load(p, mmap_mode="r")
            caches.append({"cache file": os.path.relpath(p, PROJECT_DIR), "shape": str(tuple(a.shape)),
                           "dtype": str(a.dtype), "size_bytes": os.path.getsize(p), "sha256": sha256_file(p)})
    caches = pd.DataFrame(caches)
    n_frozen = len(set(load_model_cfg(m)["encoder"]["timm_name"] for m in MODEL_IDS
                       if load_model_cfg(m)["encoder"]["type"] == "timm"))
    recovery = R(os.path.join(LOG_DIR, "stage03_recovery_checklist.md"))
    probe = {(m, mo, int(bs)): float(gb) for m, mo, bs, gb in
             re.findall(r"^\s+(M\d\d) (sequence|tokens) bs=(\d+): peak ([0-9.]+) GB", R(os.path.join(LOG_DIR, "stage03_profile.log")), re.M)}
    prof["probe_peak_gb"] = [probe.get((r.model_id, r.mode, int(r.batch_size))) for r in prof.itertuples()]

    cols = ["model_id", "mode", "policy", "seq_len", "params_total", "params_trainable", "batch_size", "probe_peak_gb",
            "peak_mem_gb",
            "s_per_train_step", "s_per_infer_batch", "steps_per_epoch", "est_fold_min_live",
            "s_per_train_step_cached", "est_fold_min_cached"]
    L = ["# STAGE 3 REPORT - the 10 hybrid architectures + unit tests", "",
         "Screening-support research, not diagnosis. No model was trained on real labels in this stage (tests use random "
         "inputs and random 0/1 labels only). No git: recorded by the local snapshot `snapshots/stage_03/`.", "",
         "## 1. Summary", "",
         "- `src/models/` implements one `HybridModel` for both input modes: spatial encoder -> projection (LayerNorm, "
         "Dropout 0.3, Linear -> 256, GELU) -> temporal encoder -> pooling -> classifier -> 1 logit (P(ASD) = sigmoid).",
         f"- All 10 models x 2 modes build, run forward/backward on cuda with fp16 autocast, and pass the tests ({last.strip('= ')}).",
         f"- Pretrained weights: {len(w) - 1} timm ImageNet checkpoints downloaded and hashed (`reports/stage03_weights.csv`).",
         f"- Batch sizes (largest of 64/32/16/8/4 with peak < 10 GB): {bs_text}; highest steady-state peak {max_peak:.2f} GB, highest first-step "
         f"probe peak {prof.probe_peak_gb.max():.2f} GB; "
         "written to `configs/models/Mxx.yaml`.",
         f"- D01 feature cache (frozen backbones) implemented, equivalence-tested and built for {len(caches)} of "
         f"{n_frozen} pretrained backbones (`cache/features/`).",
         "- This stage was resumed after an interruption; see section 9.", "",
         "## 2. Architectures (exact mapping)", "", md_table(arch), "",
         "Sequence lengths: sequence mode L = 8 (time); tokens mode L = 7 x 7 = 49 (EfficientNet/ResNet/ConvNeXt/DenseNet) "
         "or 14 x 14 = 196 (ViT-B/16 patch tokens without CLS; custom CNN at 224 px). Tokens mode adds a learned "
         "positional embedding (L x 256) after the projection; the Transformer temporal encoders add their own learned "
         "positional embedding. M05 pools by the mean before its MLP head; M10 uses mean pooling (no attention).", "",
         "Custom CNN (M02, M05, M09): 4 x [Conv3x3 -> BatchNorm -> ReLU -> MaxPool2], channels 32-64-128-256, Kaiming "
         "init, trained from scratch; input 112 x 112 in sequence mode (frames are resized inside the model with "
         "antialiasing) -> 7 x 7 x 256 map, 224 x 224 in tokens mode -> 14 x 14 x 256 map.", "",
         "Preprocessing inside the model: uint8 -> float/255 -> (resize if needed) -> normalise with the mean/std the "
         "weights were trained with (ImageNet mean/std for all CNNs and the custom CNN; 0.5/0.5 for the ViT augreg "
         "weights, as given by timm's pretrained_cfg).", "",
         "## 3. Pretrained weights (`reports/stage03_weights.csv`)", "", md_table(w.drop(columns="file")), "",
         "Note: the default timm tags were kept and pinned. Two are not pure ImageNet-1k: ConvNeXt-T `in12k_ft_in1k` "
         "(ImageNet-12k pre-training, ImageNet-1k fine-tune) and ViT-B/16 `augreg2_in21k_ft_in1k` (ImageNet-21k "
         "pre-training, ImageNet-1k fine-tune). EfficientNet-B3 `ra2_in1k` was trained at 288 px; it is used at 224 px "
         "(data contract).", "",
         "## 4. Fine-tuning policy (config options; final choice confirmed in Stage 4)", "",
         "| dataset / mode | pretrained backbones | custom CNN | learning rates |", "|---|---|---|---|",
         "| D01 / sequence | `frozen` (57 children; requires_grad=False, backbone in eval mode, run under no_grad; "
         "features can come from the cache) | `full` | head_lr 1e-3 |",
         "| D02 / tokens | `last_stage` (EfficientNet: blocks.6 + conv_head + bn2; ResNet: layer4; ConvNeXt: stages.3; "
         "DenseNet: denseblock4 + norm5; ViT: blocks.11 + norm) | `full` | backbone_lr = head_lr / 10 = 1e-4 "
         "(custom CNN: 1e-3) |", "",
         "Modules whose parameters are frozen stay in eval mode even when `model.train()` is called, so BatchNorm "
         "statistics in frozen parts are never updated (tested: frozen weights and running statistics are "
         "bit-identical after an optimizer step).", "",
         "### Frozen-feature diversity on real D01 clips (no labels; `reports/stage03_feature_diversity.csv`)", "",
         md_table(div), "",
         "Lower mean cosine = more distinguishable inputs. Real gaze renderings give clearly more diverse frozen "
         "features than random noise, so a frozen backbone does carry clip-to-clip information. (This is also why the "
         "tiny-overfit test feeds random features to the head for frozen models: random images are nearly identical "
         "to a pretrained network.)", "",
         "### Feature cache build (`logs/stage03_feature_cache.log`)", "", "```", cache_log, "```", "",
         (md_table(caches) if len(caches) else "NO CACHE FILES FOUND"), "",
         "## 5. Tests (`.venv/bin/python -m pytest -v`, full output, `logs/stage03_pytest.txt`)", "",
         "Pre-flight (`logs/stage03_preflight.txt`):", "", "```", pre, "```", "",
         "Test design notes: forward/backward/determinism use random uint8 inputs; backward checks that every trainable "
         "tensor gets a gradient and changes after one AdamW step, frozen tensors keep grad None and stay identical; "
         "determinism builds each model twice with seed 42 (cudnn deterministic) and compares eval logits for exact "
         "equality; tiny overfit = 16 random samples, random labels, 60 steps at the config learning rates with dropout "
         "on (frozen backbones: random N(0,1) features into the head; others: full model on random coarse images); "
         "feature-cache test: live model vs head(cached fp16 features) on 4 real clips, |diff| <= 1e-3.", "",
         "```", pyt, "```", "",
         "## 6. Profiling (`reports/stage03_model_profile.csv`)", "",
         "One train step = AdamW + fp16 autocast + GradScaler on random inputs; peak = torch.cuda.max_memory_allocated "
         "during 25 train steps at the chosen batch size; `probe_peak_gb` = peak of the first train step at that batch "
         "size (includes the cudnn.benchmark workspace search; this is the value compared with the 10 GB limit when "
         "choosing the largest of 64/32/16/8/4, from `logs/stage03_profile.log`); times = median of 20 "
         "after 5 warm-ups. `_cached` = head-only step on cached features (frozen D01 path). cudnn.benchmark = True.", "",
         md_table(prof[[c for c in cols if c in prof.columns]].fillna("")), "",
         "## 7. Estimated training time", "",
         "ESTIMATE = ceil(mean train samples per fold / batch size) x measured s/step x 40 epochs (an ASSUMED epoch "
         "budget; Stage 4 sets the real one, with early stopping) x 10 folds. Excludes validation passes, data loading "
         "(D02 decodes JPEGs from disk) and early stopping, so it is a GPU-compute lower bound per epoch budget.", "",
         f"- **S5 (D01, 10 models x 10 folds)** with the feature cache for the {int((seq.policy == 'frozen').sum())} "
         f"frozen-backbone models (custom CNNs "
         f"live): **{s5_mixed:.2f} h** (+ one-time cache build, section 4); all backbones live instead: {s5_live:.2f} h.",
         f"- **S6 (D02, 10 models x 10 folds, last_stage / full)**: **{s6:.2f} h**.", "",
         md_table(prof.assign(est_min_per_fold=[r.est_fold_min_cached if r.policy == "frozen" else r.est_fold_min_live
                                                for r in prof.itertuples()])
                  [["model_id", "mode", "policy", "batch_size", "steps_per_epoch", "est_min_per_fold"]]), "",
         "## 8. Open items for Stage 4", "",
         "1. Confirm the fine-tuning policy (defaults above) and learning rates; D01 with frozen backbones should use the "
         "feature cache (call `load_cache`, which refuses a stale cache).",
         "2. D01 class imbalance (437 ASD vs 804 TD clips): class-weighted loss or balanced sampling; early stopping on "
         "a validation metric that is robust with 7-8 validation children.",
         "3. D02 augmentation (train only) is applied to uint8 images before the model; resizing to 224 in the loader.",
         "4. Data-loader rules: pin_memory, num_workers = min(8, cpu_count), seeds; checkpoint/resume; run_info.json.",
         "5. Call `src.data.splits.verify_manifest()` before every run; log max_memory_allocated per run.",
         "6. Decide the epoch budget / early-stopping patience and re-check the section 7 estimate against the "
         "Stage 4 smoke test (which includes data loading and validation).", "",
         "## 9. Interruption recovery", "",
         "The previous session stopped during the profiling run. On resume nothing was deleted or regenerated blindly; "
         "every Stage 3 file was read and checked by running it (`logs/stage03_recovery_checklist.md`):", "",
         recovery, "",
         "What was fixed / built in the resumed session:", "",
         "1. `src/models/feature_cache.py` (and the cache test): copy memmap slices with `np.array` (removes the "
         "non-writable-tensor UserWarning; values unchanged).",
         "2. `scripts/03_profile_models.py`: weight `file` paths stored relative to the Hugging Face hub cache (no "
         "absolute home-directory path in `reports/stage03_weights.csv`); `est_fold_min_*` columns added; the 40-epoch figure marked as an assumption.",
         "3. `scripts/03b_build_feature_cache.py` run -> `cache/features/` (was missing).",
         "4. `scripts/03_profile_models.py` re-run in full -> `reports/stage03_model_profile.csv`, batch sizes written "
         "to all 10 configs (fixes the one failing test). Weight sha256 values were re-verified; no weight file was "
         "re-downloaded (none was corrupt).",
         "5. This report writer: summary numbers computed from the outputs instead of fixed text; cache table, "
         "minutes per fold and this section added.", ""]
    ok = failed is None and "passed" in last and (prof.status == "ok").all() and len(caches) == n_frozen
    L.append("STAGE 3 DECISION: PASS" if ok else "STAGE 3 DECISION: FAIL - "
             + "; ".join(([f"pytest: {last.strip('= ')}"] if not ("passed" in last and failed is None) else [])
                         + ([f"profiling: {list(prof[prof.status != 'ok'].model_id)}"] if not (prof.status == 'ok').all()
                            else [])
                         + ([f"feature cache: {len(caches)} of {n_frozen} built"] if len(caches) != n_frozen else [])))
    with open(os.path.join(REP, "STAGE_03_REPORT.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(L[-1])


if __name__ == "__main__":
    main()
