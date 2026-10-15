"""Stage 8: statistics helpers (bootstrap, DeLong, Friedman/Nemenyi, Nadeau-Bengio, Holm, ECE, matching), the D01
quality baseline's train-only fitting, and that results/D0*/ is unchanged since the Stage 8 pre-flight."""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
from src.analysis import stats as S  # noqa: E402
from src.analysis.quality import QUALITY_FEATURES, fit_predict_lr, fold_children  # noqa: E402
from src.data.splits import load_fold  # noqa: E402

OUT = os.path.join(PROJECT_DIR, "results", "stage8")


def _preflight_module():
    spec = importlib.util.spec_from_file_location("preflight08", os.path.join(PROJECT_DIR, "scripts", "08a_preflight.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------------------------- bootstrap
def test_bootstrap_reproducible_with_seed():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(30), 3)
    y = (np.arange(30) % 2).repeat(3)
    p = rng.random(90) + 0.3 * y
    a = S.bootstrap_indices(groups, y, 200, seed=42)
    b = S.bootstrap_indices(groups, y, 200, seed=42)
    c = S.bootstrap_indices(groups, y, 200, seed=43)
    assert all(np.array_equal(x, z) for x, z in zip(a, b))
    assert not all(np.array_equal(x, z) for x, z in zip(a, c))
    ba, bb = S.boot_stat(S.auc, a, y, p), S.boot_stat(S.auc, b, y, p)
    assert np.array_equal(ba, bb) and S.ci(ba) == S.ci(bb)


def test_cluster_bootstrap_keeps_groups_whole_and_both_classes():
    groups = np.repeat(np.arange(10), [1, 2, 3, 1, 2, 3, 1, 2, 3, 2])
    y = np.isin(groups, [0, 2, 4, 6, 8]).astype(int)
    for ix in S.bootstrap_indices(groups, y, 50, seed=42):
        g, cnt = np.unique(groups[ix], return_counts=True)
        sizes = pd.Series(groups).value_counts()
        assert all(c % sizes[gg] == 0 for gg, c in zip(g, cnt))   # whole groups only
        assert set(y[ix]) == {0, 1}                                  # stratified: both classes
        assert len(np.unique(groups[ix][y[ix] == 1])) <= 5


# ---------------------------------------------------------------------------------------------- DeLong
def _delong_bruteforce(y, preds):
    """Textbook DeLong: psi(X, Y) = 1 / 0.5 / 0 placements, explicit loops."""
    y = np.asarray(y)
    X = [np.asarray(p)[y == 1] for p in preds]
    Y = [np.asarray(p)[y == 0] for p in preds]
    m, n = len(X[0]), len(Y[0])
    psi = lambda a, b: 1.0 if a > b else (0.5 if a == b else 0.0)  # noqa: E731
    V10 = np.array([[np.mean([psi(X[r][i], Y[r][j]) for j in range(n)]) for i in range(m)] for r in range(len(preds))])
    V01 = np.array([[np.mean([psi(X[r][i], Y[r][j]) for i in range(m)]) for j in range(n)] for r in range(len(preds))])
    aucs = V10.mean(axis=1)
    Sm = np.atleast_2d(np.cov(V10)) / m + np.atleast_2d(np.cov(V01)) / n
    return aucs, Sm


def test_delong_known_example():
    y = [1, 1, 1, 0, 0, 0]
    p1 = [0.9, 0.8, 0.4, 0.7, 0.3, 0.2]   # AUC = 8/9 exactly (one discordant pair: 0.4 < 0.7)
    p2 = [0.9, 0.3, 0.4, 0.7, 0.8, 0.2]
    r = S.delong(y, p1, p2)
    assert r["auc1"] == pytest.approx(8 / 9)
    aucs, Sm = _delong_bruteforce(y, [p1, p2])
    assert r["auc2"] == pytest.approx(aucs[1])
    var = Sm[0, 0] + Sm[1, 1] - 2 * Sm[0, 1]
    assert r["se_diff"] == pytest.approx(np.sqrt(var))
    from scipy.stats import norm
    assert r["p"] == pytest.approx(2 * norm.sf(abs((aucs[0] - aucs[1]) / np.sqrt(var))))
    assert S.delong(y, p1)["var1"] == pytest.approx(Sm[0, 0])


def test_delong_with_ties_matches_bruteforce_and_identical_scores_give_p1():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 40)
    p1 = np.round(rng.random(40) + 0.4 * y, 1)    # rounding creates ties
    p2 = np.round(rng.random(40) + 0.2 * y, 1)
    r = S.delong(y, p1, p2)
    aucs, Sm = _delong_bruteforce(y, [p1, p2])
    assert [r["auc1"], r["auc2"]] == pytest.approx(list(aucs))
    assert r["se_diff"] == pytest.approx(np.sqrt(Sm[0, 0] + Sm[1, 1] - 2 * Sm[0, 1]))
    assert S.delong(y, p1, p1)["p"] == 1.0
    assert S.auc(y, p1) == pytest.approx(aucs[0])


# ---------------------------------------------------------------------------------------------- Friedman / Nemenyi
def test_friedman_toy_matrix():
    mat = np.array([[0.9, 0.8, 0.7]] * 4)    # model A always best, C always worst: ranks 1, 2, 3 in all 4 blocks
    fr = S.friedman(mat)
    assert fr["chi2"] == pytest.approx(8.0)                 # 12N/(k(k+1)) sum R^2 - 3N(k+1) = 56 - 48
    assert fr["p"] == pytest.approx(np.exp(-4.0))           # chi2 with 2 df: sf(8) = exp(-4)
    assert list(fr["avg_ranks"]) == [1.0, 2.0, 3.0]
    ne = S.nemenyi(fr["avg_ranks"], 4)
    assert ne["q_alpha"] == pytest.approx(2.343, abs=1e-3)  # Demšar (2006) table, k = 3
    assert ne["cd"] == pytest.approx(2.343 * np.sqrt(3 * 4 / (6 * 4)), abs=1e-3)
    assert S.nemenyi(np.arange(1, 11), 10)["q_alpha"] == pytest.approx(3.164, abs=1e-3)  # k = 10
    assert ne["p"][0, 2] < ne["p"][0, 1] and np.allclose(ne["p"], ne["p"].T)


def test_nadeau_bengio_and_holm():
    a = np.array([0.80, 0.82, 0.79, 0.81, 0.83])
    b = a - np.array([0.01, 0.02, 0.0, 0.01, 0.03])
    r = S.nadeau_bengio(a, b, n_train=900, n_test=100)
    d = a - b
    t = d.mean() / np.sqrt((1 / 5 + 100 / 900) * d.var(ddof=1))
    assert r["t"] == pytest.approx(t)
    from scipy.stats import t as tdist
    assert r["p"] == pytest.approx(2 * tdist.sf(abs(t), 4))
    assert list(S.holm([0.01, 0.04, 0.03])) == pytest.approx([0.03, 0.06, 0.06])


def test_ece_and_matching():
    assert S.ece([0, 1, 0, 1], [0.0, 1.0, 0.0, 1.0]) == 0.0
    assert S.ece([1, 1], [0.25, 0.25]) == pytest.approx(0.75)
    pairs = S.greedy_caliper_match(["a1", "a2"], [0.50, 0.90], ["t1", "t2", "t3"], [0.52, 0.51, 0.70], 0.05)
    assert pairs == [("a1", "t2", pytest.approx(0.01))]      # a2 has no control within 0.05


# ---------------------------------------------------------------------------------------------- D01 quality baseline
def test_quality_baseline_uses_only_train_fold_children():
    from src.data.common import CACHE_DIR
    ds = "D01_eye_tracking"
    folds = pd.read_csv(os.path.join(PROJECT_DIR, "splits", ds, "folds.csv"))
    q = pd.read_csv(os.path.join(CACHE_DIR, ds, "quality_per_child.csv"))
    q["group_id"] = q.participant_id.map(lambda i: f"p{int(i):02d}")
    for k in [1, 5, 10]:
        tr_ids, te_ids = fold_children(folds, load_fold, ds, k)
        assert set(tr_ids).isdisjoint(te_ids)
        tr, te = q[q.group_id.isin(tr_ids)], q[q.group_id.isin(te_ids)].copy()
        p = fit_predict_lr(tr, te, QUALITY_FEATURES)
        te["label"] = 1 - te["label"]                               # test labels must not matter
        assert np.array_equal(p, fit_predict_lr(tr, te, QUALITY_FEATURES))
        with pytest.raises(ValueError):
            fit_predict_lr(pd.concat([tr, te.iloc[:1]]), te, QUALITY_FEATURES)
    log = pd.read_csv(os.path.join(OUT, "d01_quality_baseline_folds.csv"))
    for r in log.itertuples():
        tr_ids, te_ids = fold_children(folds, load_fold, ds, r.fold)
        assert r.train_children.split() == tr_ids and r.test_children.split() == te_ids


# ---------------------------------------------------------------------------------------------- read-only results
def test_results_d0x_unchanged_since_preflight():
    mod = _preflight_module()
    with open(os.path.join(OUT, "preflight_results_manifest.sha256")) as fh:
        recorded = fh.read().splitlines()
    current = mod.results_manifest_lines()
    assert len(recorded) > 2000
    assert recorded == current, sorted(set(recorded) ^ set(current))[:10]


def test_stage8_outputs_present():
    for f in ["d01_model_vs_quality.csv", "d01_matched_auc.csv", "d01_quality_baselines.csv", "d02_near_dup_sensitivity.csv",
              "d02_image_stats_baseline.csv", "d02_original_split.csv", "stats_bootstrap_ci.csv",
              "stats_friedman_nemenyi.json", "stats_nadeau_bengio.csv", "stats_calibration.csv"]:
        assert os.path.isfile(os.path.join(OUT, f)), f
    ci = pd.read_csv(os.path.join(OUT, "stats_bootstrap_ci.csv"))
    ok = ci.dropna(subset=["pooled_auc"])
    assert ((ok.ci_lo <= ok.pooled_auc + 1e-12) & (ok.pooled_auc <= ok.ci_hi + 1e-12)).all()
