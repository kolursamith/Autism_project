"""Stage 4 engine tests on the GPU with tiny subsets of the real fold-1 splits (random labels are NOT needed: these
only check mechanics; outputs go to pytest temp dirs, never to results/ or results_smoke/).
Resume: 3 epochs straight vs 1 epoch + simulated kill + resume to 3 epochs, same seed, deterministic cuDNN settings
-> identical history and test predictions within 1e-5."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
from src.data.splits import load_fold  # noqa: E402
from src.training import engine, registry  # noqa: E402
from src.training.checks import validate_model_results  # noqa: E402
from src.training.data import D01, D02  # noqa: E402
from src.training.metrics import youden_threshold  # noqa: E402

COLS = ["train_loss", "val_loss", "val_auc", "lr"]
CASES = [(D01, "M01", "cached_features"), (D02, "M09", "images (augmentation, 8 workers)")]


def setup_module():
    assert torch.cuda.is_available()


def _cfg(ds, mid, epochs=3):
    return engine.merge_config(ds, mid, 1, {"training.max_epochs": epochs, "training.deterministic": True})


def _run(ds, mid, root, **kw):
    return engine.train_one_fold(ds, mid, 1, cfg=_cfg(ds, mid), out_root=str(root), subset=12,
                                 log_path=str(root / "fold.log"), echo=False, **kw)


@pytest.fixture(scope="module", params=CASES, ids=[c[1] for c in CASES])
def resume_pair(request, tmp_path_factory):
    ds, mid, _ = request.param
    a = tmp_path_factory.mktemp(f"straight_{mid}")
    b = tmp_path_factory.mktemp(f"resumed_{mid}")
    _run(ds, mid, a)
    with pytest.raises(KeyboardInterrupt):
        _run(ds, mid, b, stop_after_epoch=1)
    fold_b = b / ds / mid / "fold_01"
    interrupted = {"status": registry.get_status(str(b), ds, mid, 1), "last_pt": (fold_b / "last.pt").is_file(),
                   "history_rows": len(pd.read_csv(fold_b / "history.csv")),
                   "best": (fold_b / "best_model.pt").is_file(), "metrics": (fold_b / "metrics.json").is_file()}
    _run(ds, mid, b)
    return ds, mid, a / ds / mid / "fold_01", fold_b, interrupted, a, b


def test_interrupt_leaves_running_and_checkpoint(resume_pair):
    *_, interrupted, _, _ = resume_pair
    assert interrupted == {"status": "RUNNING", "last_pt": True, "history_rows": 1, "best": True, "metrics": False}


def test_resume_identical_history_and_predictions(resume_pair):
    ds, mid, fa, fb, *_ = resume_pair
    ha, hb = pd.read_csv(fa / "history.csv"), pd.read_csv(fb / "history.csv")
    assert list(ha.epoch) == list(hb.epoch) == [1, 2, 3]
    diff = np.abs(ha[COLS].to_numpy() - hb[COLS].to_numpy()).max()
    assert diff <= 1e-5, (diff, ha[COLS], hb[COLS])
    pa, pb = pd.read_csv(fa / "predictions.csv"), pd.read_csv(fb / "predictions.csv")
    assert list(pa.sample_id) == list(pb.sample_id)
    assert np.abs(pa.y_prob - pb.y_prob).max() <= 1e-5


def test_resumed_run_info_and_cleanup(resume_pair):
    ds, mid, fa, fb, _, a, b = resume_pair
    ia, ib = (json.loads((f / "run_info.json").read_text()) for f in (fa, fb))
    assert ia["status"] == ib["status"] == "DONE" and ia["resumed"] is False and ib["resumed"] is True
    assert [s["start_epoch"] for s in ib["sessions"]] == [1, 2] and ib["sessions"][0]["interrupted"] is True
    assert ia["batch_counts"] == ib["batch_counts"]  # same number of batches in total, split across 2 sessions
    assert ib["test_ids_in_train_or_val_batches"] == 0 and ib["cudnn_deterministic"] is True
    assert not (fa / "last.pt").exists() and not (fb / "last.pt").exists()  # deleted when DONE
    assert registry.get_status(str(b), ds, mid, 1) == "DONE"


def test_youden_threshold_comes_from_validation_predictions(resume_pair):
    ds, mid, fa, *_ = resume_pair
    m = json.loads((fa / "metrics.json").read_text())
    v, t = pd.read_csv(fa / "val_predictions.csv"), pd.read_csv(fa / "predictions.csv")
    assert m["val_threshold_youden"] == pytest.approx(youden_threshold(v.y_true, v.y_prob))
    assert m["test"]["threshold_val_youden"]["threshold"] == pytest.approx(m["val_threshold_youden"])
    assert set(v.sample_id).isdisjoint(t.sample_id)
    if ds == D01:
        from src.training.metrics import child_level
        _, cy, cp = child_level(v.group_id, v.y_true, v.y_prob)
        assert m["val_threshold_youden_child"] == pytest.approx(youden_threshold(cy, cp))


def test_done_run_is_skipped_unless_forced(resume_pair):
    ds, mid, fa, _, _, a, _ = resume_pair
    mtime = (fa / "metrics.json").stat().st_mtime_ns
    r = _run(ds, mid, a)
    assert r["skipped"] is True and (fa / "metrics.json").stat().st_mtime_ns == mtime


def test_checks_detect_incomplete_results(resume_pair):
    ds, mid, fa, _, _, a, _ = resume_pair
    verdict, reasons = validate_model_results(ds, mid, str(a), folds=[1])
    assert verdict == "FAIL" and any("exactly" in r for r in reasons)  # subset run does not cover the test fold
    os.remove(fa / "history.csv")
    verdict, reasons = validate_model_results(ds, mid, str(a), folds=[1, 2])
    assert verdict == "FAIL" and any("missing" in r for r in reasons) and any("fold 2" in r for r in reasons)


def test_engine_refuses_corrupted_split(tmp_path, monkeypatch):
    def bad_fold(ds, k, root=None):
        s = load_fold(ds, k)
        s["train"] = s["train"] + s["test"][:1]
        return s
    monkeypatch.setattr(engine, "load_fold", bad_fold)
    with pytest.raises(engine.LeakageError):
        engine.train_one_fold(D01, "M01", 1, cfg=_cfg(D01, "M01"), out_root=str(tmp_path), subset=4,
                              log_path=str(tmp_path / "x.log"), echo=False)
    assert registry.get_status(str(tmp_path), D01, "M01", 1) is None  # nothing started


def test_resume_refuses_changed_config(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        _run(D01, "M01", tmp_path, stop_after_epoch=1)
    cfg = _cfg(D01, "M01")
    cfg["training"]["grad_clip_norm"] = 2.0
    with pytest.raises(RuntimeError, match="different config"):
        engine.train_one_fold(D01, "M01", 1, cfg=cfg, out_root=str(tmp_path), subset=12,
                              log_path=str(tmp_path / "fold.log"), echo=False)
    assert registry.get_status(str(tmp_path), D01, "M01", 1) == "FAILED"
