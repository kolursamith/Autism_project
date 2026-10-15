"""Stage 4 unit tests (no training): protocol config, leakage asserts on corrupted temp splits, metrics vs sklearn,
validation-only Youden threshold, registry skip/force/locking, summary aggregation on fake fold results."""
import json
import math
import os
import shutil
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from sklearn import metrics as skm

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
from src.data.splits import load_fold, verify_manifest  # noqa: E402
from src.training import registry  # noqa: E402
from src.training.data import D01, D02, d02_transforms, sample_table  # noqa: E402
from src.training.engine import (LeakageError, assert_no_test_ids, check_fold, load_training_cfg,  # noqa: E402
                                 make_scheduler, merge_config)
from src.training.metrics import binary_metrics, child_level, youden_threshold  # noqa: E402
from src.training.summarize import leaderboard, summarize_model  # noqa: E402

SPLITS = os.path.join(PROJECT_DIR, "splits")


# ------------------------------------------------------------------------------------------- protocol / config
def test_training_protocol_values_fixed():
    t = load_training_cfg()
    assert t["optimizer"] == {"name": "AdamW", "weight_decay": 0.01}
    assert t["lr"]["head_lr_pretrained"] == 5e-4 and t["lr"]["head_lr_custom_cnn"] == 1e-3
    assert t["lr"]["backbone_lr_factor_last_stage"] == 0.1
    assert t["schedule"] == {"name": "cosine", "warmup_epochs": 2} and t["max_epochs"] == 40
    assert t["early_stopping"] == {"monitor": "val_roc_auc", "mode": "max", "patience": 8, "min_delta": 1e-4}
    assert t["loss"]["pos_weight"] == "n_neg_over_n_pos_train" and t["grad_clip_norm"] == 1.0 and t["amp"] is True
    assert t["seed_base"] == 42 and t["deterministic"] is False and t["cudnn_benchmark"] is True
    a = t["data"][D02]["train_transform"]
    assert a == {"random_resized_crop": {"size": 224, "scale": [0.8, 1.0]}, "horizontal_flip_p": 0.5,
                 "color_jitter": {"brightness": 0.1, "contrast": 0.1, "saturation": 0.1}}
    assert t["data"][D01]["augmentation"] == "none"


@pytest.mark.parametrize("mid,ds,fold,head,bb,inp", [
    ("M01", D01, 1, 5e-4, 0.0, "cached_features"), ("M01", D02, 3, 5e-4, 5e-5, "images"),
    ("M07", D02, 10, 5e-4, 5e-5, "images"), ("M09", D01, 2, 1e-3, 1e-3, "clips"), ("M05", D02, 4, 1e-3, 1e-3, "images")])
def test_merge_config(mid, ds, fold, head, bb, inp):
    c = merge_config(ds, mid, fold)
    assert c["seed"] == 42 + fold and c["run"]["input"] == inp and c["run"]["batch_size"] == 64
    assert c["lr"]["head_lr"] == pytest.approx(head) and c["lr"]["backbone_lr"] == pytest.approx(bb)
    assert merge_config(ds, mid, fold, {"training.max_epochs": 2})["training"]["max_epochs"] == 2


def test_model_config_lr_mirrors_training_yaml():
    for i in range(1, 11):
        with open(os.path.join(PROJECT_DIR, "configs", "models", f"M{i:02d}.yaml")) as fh:
            m = yaml.safe_load(fh)
        c = merge_config(D02, f"M{i:02d}", 1)
        assert m["lr"] == pytest.approx(c["lr"]), i


def test_scheduler_warmup_and_cosine():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=1.0)
    s = make_scheduler(opt, steps_per_epoch=10, max_epochs=40, warmup_epochs=2)
    lrs = []
    for _ in range(400):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        s.step()
    assert lrs[0] == pytest.approx(1 / 20) and lrs[19] == pytest.approx(1.0)
    assert all(a <= b + 1e-12 for a, b in zip(lrs[:20], lrs[1:20]))          # increasing warm-up
    assert all(a >= b - 1e-12 for a, b in zip(lrs[20:], lrs[21:]))          # monotone cosine decay
    assert lrs[-1] < 1e-3


def test_d02_transforms_uint8():
    from PIL import Image
    t = load_training_cfg()["data"][D02]
    im = Image.fromarray(np.random.default_rng(0).integers(0, 256, (281, 180, 3), dtype=np.uint8))
    for train in [True, False]:
        x = d02_transforms(t, train)(im)
        assert x.dtype == torch.uint8 and tuple(x.shape) == (3, 224, 224)
    ev = d02_transforms(t, False)
    assert torch.equal(ev(im), ev(im))  # eval transform is deterministic


# ------------------------------------------------------------------------------------------- leakage
@pytest.mark.parametrize("ds", [D01, D02])
def test_real_splits_pass_check_fold(ds):
    table = sample_table(ds)
    for k in range(1, 11):
        check_fold(ds, k, load_fold(ds, k), table)


def _corrupt(ds, how):
    s = load_fold(ds, 1)
    t = sample_table(ds)
    if how == "id_in_train_and_test":
        s["train"] = s["train"] + [s["test"][0]]
    elif how == "duplicate":
        s["val"] = s["val"] + [s["val"][0]]
    elif how == "group_overlap":  # move one test sample to train: its group/child is now in train and test
        x = s["test"].pop(0)
        if ds == D02:  # D02 groups are mostly singletons: take a val sample and give it a test sample's group
            grp = t.loc[s["test"][0], "group_id"]
            t.loc[s["val"][0], "group_id"] = grp
            s["test"].insert(0, x)
        else:
            s["train"].append(x)
    elif how == "unknown_id":
        s["train"] = s["train"] + ["not_a_sample"]
    return s, t


@pytest.mark.parametrize("ds", [D01, D02])
@pytest.mark.parametrize("how", ["id_in_train_and_test", "duplicate", "group_overlap", "unknown_id"])
def test_leakage_asserts_trigger_on_corrupted_split(ds, how):
    s, t = _corrupt(ds, how)
    with pytest.raises(LeakageError):
        check_fold(ds, 1, s, t)


def test_test_set_must_match_folds_csv():
    s, t = load_fold(D01, 1), sample_table(D01)
    s["test"] = s["test"][1:]
    with pytest.raises(LeakageError):
        check_fold(D01, 1, s, t)


def test_verify_manifest_detects_modified_temp_copy(tmp_path):
    root = tmp_path / "splits"
    shutil.copytree(SPLITS, root)
    for dp, _, fns in os.walk(root):  # splits/ is read-only after the Stage 10 freeze; copytree copies the mode bits
        os.chmod(dp, 0o755)
        for f in fns:
            os.chmod(os.path.join(dp, f), 0o644)
    verify_manifest(str(root))  # intact copy passes
    p = root / D01 / "fold_03.json"
    d = json.loads(p.read_text())
    d["train"], d["test"] = d["train"] + d["test"][:1], d["test"][1:]
    p.write_text(json.dumps(d))
    with pytest.raises(RuntimeError):
        verify_manifest(str(root))
    verify_manifest()  # the real files were not touched


def test_assert_no_test_ids():
    assert_no_test_ids({"train": {"a", "b"}, "val": {"c"}}, {"d"})
    with pytest.raises(LeakageError):
        assert_no_test_ids({"train": {"a", "b"}, "val": {"d"}}, {"d"})


# ------------------------------------------------------------------------------------------- metrics
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("thr", [0.5, 0.3, 0.72])
def test_binary_metrics_match_sklearn(seed, thr):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, 200)
    p = np.clip(rng.normal(0.35 + 0.3 * y, 0.2), 0, 1)
    m = binary_metrics(y, p, thr)
    yp = (p >= thr).astype(int)
    assert m["accuracy"] == pytest.approx(skm.accuracy_score(y, yp))
    assert m["balanced_accuracy"] == pytest.approx(skm.balanced_accuracy_score(y, yp))
    assert m["precision"] == pytest.approx(skm.precision_score(y, yp, zero_division=0))
    assert m["recall"] == pytest.approx(skm.recall_score(y, yp)) == pytest.approx(m["sensitivity"])
    assert m["specificity"] == pytest.approx(skm.recall_score(y, yp, pos_label=0))
    assert m["f1"] == pytest.approx(skm.f1_score(y, yp))
    assert m["roc_auc"] == pytest.approx(skm.roc_auc_score(y, p))
    assert m["pr_auc"] == pytest.approx(skm.average_precision_score(y, p))
    assert m["brier"] == pytest.approx(skm.brier_score_loss(y, p))
    assert m["confusion_matrix"] == skm.confusion_matrix(y, yp, labels=[0, 1]).tolist()


def test_binary_metrics_known_values():
    m = binary_metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9], 0.5)
    assert m["confusion_matrix"] == [[1, 1], [1, 1]] and m["accuracy"] == 0.5 and m["roc_auc"] == 0.75
    assert m["brier"] == pytest.approx((0.01 + 0.36 + 0.36 + 0.01) / 4)
    assert math.isnan(binary_metrics([1, 1], [0.2, 0.9])["roc_auc"])  # single class -> undefined, not a crash


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_youden_threshold_brute_force_on_validation_only(seed):
    rng = np.random.default_rng(seed)
    yv = rng.integers(0, 2, 150)
    pv = np.clip(rng.normal(0.4 + 0.2 * yv, 0.2), 0, 1)
    t = youden_threshold(yv, pv)
    best = max(((pv[yv == 1] >= c).mean() - (pv[yv == 0] >= c).mean(), c) for c in np.unique(pv))
    pred = pv >= t
    assert pred[yv == 1].mean() - pred[yv == 0].mean() == pytest.approx(best[0])
    # the function only ever sees validation arrays: any test data cannot change it
    import inspect
    assert list(inspect.signature(youden_threshold).parameters) == ["y_val", "p_val"]
    assert youden_threshold(yv, pv) == t
    assert youden_threshold(np.ones(5), np.linspace(0, 1, 5)) == 0.5


def test_child_level_mean_probability():
    g, y, p = child_level(["p1", "p1", "p2", "p2", "p2"], [1, 1, 0, 0, 0], [0.2, 0.6, 0.1, 0.2, 0.3])
    assert list(g) == ["p1", "p2"] and list(y) == [1, 0] and p == pytest.approx([0.4, 0.2])
    with pytest.raises(ValueError):
        child_level(["p1", "p1"], [1, 0], [0.2, 0.3])


# ------------------------------------------------------------------------------------------- registry
def test_registry_skip_force_resume(tmp_path):
    r = str(tmp_path)
    assert registry.should_run(r, D01, "M01", 1) == (True, "new")
    registry.update(r, D01, "M01", 1, status="RUNNING", start="t0")
    assert registry.should_run(r, D01, "M01", 1)[0] and registry.get_status(r, D01, "M01", 1) == "RUNNING"
    registry.update(r, D01, "M01", 1, status="DONE", val_auc=0.7, test_auc=0.6)
    run, why = registry.should_run(r, D01, "M01", 1)
    assert run is False and "DONE" in why
    assert registry.should_run(r, D01, "M01", 1, force=True) == (True, "forced re-run")
    registry.update(r, D01, "M01", 2, status="FAILED", error="boom, with comma")
    df = registry.read_registry(r)
    assert list(df.columns) == registry.COLUMNS and len(df) == 2
    assert df.set_index("fold").loc[2, "error"] == "boom, with comma"
    with pytest.raises(ValueError):
        registry.update(r, D01, "M01", 3, status="OK")
    with pytest.raises(KeyError):
        registry.update(r, D01, "M01", 3, nonsense=1)


WRITER = """
import sys
sys.path.insert(0, sys.argv[1])
from src.training import registry
root = sys.argv[2]
for fold in map(int, sys.argv[3:]):
    for _ in range(5):
        registry.update(root, "D02_facial_images_A", "M03", fold, status="RUNNING")
    registry.update(root, "D02_facial_images_A", "M03", fold, status="DONE", seconds=fold)
"""


def test_registry_file_lock_no_lost_rows(tmp_path):
    """4 independent processes write 20 rows concurrently (read-modify-write each time): no row may be lost."""
    import subprocess
    procs = [subprocess.Popen([sys.executable, "-c", WRITER, PROJECT_DIR, str(tmp_path)]
                              + [str(k) for k in range(w, 21, 4)]) for w in range(1, 5)]
    assert all(p.wait(timeout=120) == 0 for p in procs)
    df = registry.read_registry(str(tmp_path))
    assert sorted(df.fold.astype(int)) == list(range(1, 21)) and (df.status == "DONE").all()


# ------------------------------------------------------------------------------------------- summary aggregation
def _fake_fold(root, ds, model, k, y, p, thr):
    d = os.path.join(root, ds, model, f"fold_{k:02d}")
    os.makedirs(d)
    ids = [f"s{k}_{i}" for i in range(len(y))]
    pd.DataFrame({"sample_id": ids, "group_id": [f"g{k}_{i // 2}" for i in range(len(y))], "y_true": y,
                  "y_prob": p, "fold": k}).to_csv(os.path.join(d, "predictions.csv"), index=False)
    m = {"dataset": ds, "model_id": model, "fold": k, "best_epoch": k, "val_threshold_youden": thr,
         "test": {"threshold_0.5": binary_metrics(y, p, 0.5), "threshold_val_youden": binary_metrics(y, p, thr)}}
    with open(os.path.join(d, "metrics.json"), "w") as fh:
        json.dump(m, fh)
    return m


def test_summary_aggregation_on_fake_folds(tmp_path):
    rng = np.random.default_rng(7)
    ms, ys, ps, preds = [], [], [], []
    for k, thr in zip([1, 2, 3], [0.4, 0.5, 0.6]):
        y = np.repeat(rng.integers(0, 2, 20), 2)  # pairs share a group and label
        p = np.clip(rng.normal(0.4 + 0.2 * y, 0.2), 0, 1)
        ms.append(_fake_fold(str(tmp_path), D02, "M03", k, y, p, thr))
        ys.append(y), ps.append(p), preds.append((p >= thr).astype(int))
    s = summarize_model(str(tmp_path), D02, "M03")
    auc = [m["test"]["threshold_0.5"]["roc_auc"] for m in ms]
    e = s["per_fold_metrics"]["test/threshold_0.5/roc_auc"]
    assert e["mean"] == pytest.approx(np.mean(auc)) and e["std"] == pytest.approx(np.std(auc, ddof=1))
    assert e["min"] == pytest.approx(min(auc)) and e["max"] == pytest.approx(max(auc)) and e["n_folds"] == 3
    assert s["per_fold_metrics"]["val_threshold_youden"]["mean"] == pytest.approx(0.5)
    Y, P = np.concatenate(ys), np.concatenate(ps)
    assert s["pooled_out_of_fold"]["threshold_0.5"]["roc_auc"] == pytest.approx(skm.roc_auc_score(Y, P))
    assert s["pooled_out_of_fold"]["threshold_val_youden_per_fold"]["accuracy"] == pytest.approx(
        skm.accuracy_score(Y, np.concatenate(preds)))
    assert os.path.isfile(tmp_path / D02 / "M03" / "summary.json")
    lb = leaderboard(str(tmp_path), D02)
    assert list(lb.model_id) == ["M03"] and lb.roc_auc_mean[0] == pytest.approx(np.mean(auc))
