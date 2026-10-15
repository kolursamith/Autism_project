"""Stage 7 tests (CPU only, no torch): split integrity, feature-set leakage, files never read, train-only
preprocessing, validation-only early stopping / thresholds, completeness of the saved results.
Run: CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 .venv/bin/python -m pytest -v tabular/tests"""
import os

for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_v, "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import json  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import yaml  # noqa: E402

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_DIR)
from src.data.splits import load_fold, verify_manifest  # noqa: E402
from src.training.metrics import youden_threshold  # noqa: E402
from src.utils.paths import DATASETS_DIR  # noqa: E402
from tabular.models import MODEL_IDS, TAB_DIR, feature_columns, load_stage7_cfg, load_table  # noqa: E402
from tabular.run_stage7 import RESULTS, LeakageError, check_fold, check_results, run_fold  # noqa: E402

CFG = load_stage7_cfg()
TAB_DS = ["D04_autism_prediction", "D05_toddler_screening", "D06_early_autism"]
FORBIDDEN_D05 = [f"A{i}" for i in range(1, 11)] + ["Qchat-10-Score"]
EXPS = [(e["dataset"], e["experiment"]) for e in CFG["experiments"]]


def _exp(ds, exp):
    return next(e for e in CFG["experiments"] if e["dataset"] == ds and e["experiment"] == exp)


# ------------------------------------------------------------------------------------------- splits
def test_split_hashes_match_manifest():
    verify_manifest()


@pytest.mark.parametrize("ds", TAB_DS)
def test_train_val_test_disjoint_every_fold(ds):
    table, _ = load_table(ds)
    folds_csv = pd.read_csv(os.path.join(PROJECT_DIR, "splits", ds, "folds.csv"))
    all_test = []
    for k in range(1, 11):
        s = load_fold(ds, k)
        tr, va, te = set(s["train"]), set(s["val"]), set(s["test"])
        assert not (tr & va) and not (tr & te) and not (va & te), (ds, k)
        g = {p: set(table.loc[list(x), "group_id"]) for p, x in [("tr", tr), ("va", va), ("te", te)]}
        assert not (g["tr"] & g["va"]) and not (g["tr"] & g["te"]) and not (g["va"] & g["te"]), (ds, k)
        assert len(tr | va | te) == len(table)
        check_fold(ds, k, s, table)
        all_test += s["test"]
    assert sorted(all_test) == sorted(folds_csv.sample_id)  # every sample is tested exactly once


def test_check_fold_detects_corrupted_split():
    table, _ = load_table("D06_early_autism")
    s = load_fold("D06_early_autism", 1)
    s["train"] = s["train"] + s["test"][:1]
    with pytest.raises(LeakageError):
        check_fold("D06_early_autism", 1, s, table)


# ------------------------------------------------------------------------------------------- feature sets
def test_set_honest_has_no_items_or_score():
    _, cols = load_table("D05_toddler_screening")
    num, cat = feature_columns(cols, "set_honest")
    assert not set(num + cat) & set(FORBIDDEN_D05)
    table, _ = load_table("D05_toddler_screening")
    r = run_fold(CFG, _exp("D05_toddler_screening", "set_honest"), "logreg", 1, table, cols, write=False)
    seen = set(r["pipe"].named_steps["prep"].feature_names_in_)
    assert not seen & set(FORBIDDEN_D05), seen
    for m in MODEL_IDS:  # every saved set_honest run
        for k in range(1, 11):
            c = yaml.safe_load(open(os.path.join(RESULTS, "D05_toddler_screening", "set_honest", m, f"fold_{k:02d}",
                                                 "config.yaml")))
            assert not set(c["numeric_features"] + c["categorical_features"]) & set(FORBIDDEN_D05), (m, k)


def test_set_all_d05_label_is_the_item_rule():
    table, _ = load_table("D05_toddler_screening")
    rule = (table[[f"A{i}" for i in range(1, 11)]].sum(axis=1) >= 4).astype(int)
    assert (rule == table.label).all()
    rc = pd.read_csv(os.path.join(RESULTS, "D05_toddler_screening", "set_all", "rule_comparison.csv"))
    assert set(MODEL_IDS) <= set(rc.model)


# ------------------------------------------------------------------------------------------- files never read
OPENED = []


def _hook(event, args):
    if event == "open" and OPENED is not None and isinstance(args[0], (str, bytes, os.PathLike)):
        OPENED.append(os.fsdecode(args[0]))


sys.addaudithook(_hook)


def test_raw_files_never_read():
    """Audit every file opened while loading the tables and running a fold of every experiment (all 5 models): the
    raw dataset folder is never opened, so D04 test.csv and the D05 Combined file cannot be read."""
    OPENED.clear()
    for ds, exp in EXPS:
        table, cols = load_table(ds)
        for m in MODEL_IDS:
            run_fold(CFG, _exp(ds, exp), m, 2, table, cols, write=False)
    paths = [os.path.realpath(p) for p in OPENED]
    assert any(p.endswith(os.path.join("D04_autism_prediction", "clean.csv")) for p in paths)  # the hook works
    raw = os.path.realpath(DATASETS_DIR)
    assert not [p for p in paths if p.startswith(raw + os.sep)]
    assert not [p for p in paths if os.path.basename(p) == "test.csv" or "Combined" in os.path.basename(p)]
    for fn in os.listdir(TAB_DIR):  # the code does not even name the raw folder
        if fn.endswith(".py"):
            src = open(os.path.join(TAB_DIR, fn)).read()
            assert "DATASETS_DIR" not in src and "dataset_dir(" not in src, fn


# ------------------------------------------------------------------------------------------- train-only preprocessing
@pytest.mark.parametrize("ds,exp", EXPS)
def test_preprocessing_fitted_on_train_part_only(ds, exp):
    table, cols = load_table(ds)
    r = run_fold(CFG, _exp(ds, exp), "logreg", 3, table, cols, write=False)
    prep = r["pipe"].named_steps["prep"]
    tr = table.loc[r["parts"]["train"]]
    everything = table
    if r["numeric"]:
        num = prep.named_transformers_["num"]
        med = tr[r["numeric"]].median().to_numpy()
        assert np.allclose(num.named_steps["impute"].statistics_, med)
        expected = tr[r["numeric"]].fillna(tr[r["numeric"]].median()).mean().to_numpy()
        assert np.allclose(num.named_steps["scale"].mean_, expected, atol=1e-9)        # == train-part mean
        assert not np.allclose(num.named_steps["scale"].mean_, everything[r["numeric"]].mean().to_numpy(),
                               atol=1e-9)                                             # != all-data mean
    if r["categorical"]:
        oh = prep.named_transformers_["cat"].named_steps["onehot"]
        imp = prep.named_transformers_["cat"].named_steps["impute"]
        for c, cats in zip(r["categorical"], oh.categories_):
            train_vals = set(tr[c].fillna(imp.statistics_[r["categorical"].index(c)]))
            assert set(cats) == train_vals, c                                          # categories from train only


# ------------------------------------------------------------------------------------------- validation-only choices
def test_mlp_early_stopping_uses_validation_part_only():
    for ds, exp in EXPS:
        table, cols = load_table(ds)
        r = run_fold(CFG, _exp(ds, exp), "mlp", 4, table, cols, write=False)
        clf = r["pipe"].named_steps["clf"]
        assert clf.n_val_seen_ == len(r["parts"]["val"]) and clf.early_stopping is False
        assert len(clf.val_history_) == clf.epochs_run_ and clf.best_val_auc_ == pytest.approx(max(clf.val_history_))
        assert clf.val_history_[clf.best_epoch_ - 1] == pytest.approx(clf.best_val_auc_)


def test_saved_thresholds_come_from_validation_predictions():
    for ds, exp in EXPS:
        for m in MODEL_IDS:
            for k in range(1, 11):
                d = os.path.join(RESULTS, ds, exp, m, f"fold_{k:02d}")
                mt = json.load(open(os.path.join(d, "metrics.json")))
                v = pd.read_csv(os.path.join(d, "val_predictions.csv"))
                t = pd.read_csv(os.path.join(d, "predictions.csv"))
                assert mt["val_threshold_youden"] == pytest.approx(youden_threshold(v.y_true, v.y_prob))
                assert set(v.sample_id).isdisjoint(t.sample_id)
                assert set(v.sample_id) == set(load_fold(ds, k)["val"])


# ------------------------------------------------------------------------------------------- outputs
def test_all_results_complete_and_valid():
    chk = check_results(CFG)
    assert len(chk) == len(EXPS) * len(MODEL_IDS)
    bad = {k: v["reasons"][:3] for k, v in chk.items() if v["verdict"] != "PASS"}
    assert not bad, bad
    for ds, exp in EXPS:
        lb = pd.read_csv(os.path.join(RESULTS, ds, exp, "leaderboard.csv"))
        assert sorted(lb.model) == sorted(MODEL_IDS) and (lb.n_folds == 10).all()


def test_torch_never_imported():
    assert "torch" not in sys.modules
