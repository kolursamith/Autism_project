"""Stage 8 statistics (no model is trained here): rank AUC, (cluster, stratified) bootstrap, DeLong test for two
correlated AUCs, Friedman + Nemenyi, Nadeau-Bengio corrected resampled t-test, Holm, ECE, greedy caliper matching."""
import numpy as np
from scipy import stats

SEED = 42
N_BOOT = 2000


def auc(y, p):
    """ROC-AUC = Mann-Whitney U / (n_pos n_neg) with mid-ranks for ties; NaN if one class only."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = stats.rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ---------------------------------------------------------------------------------------------- bootstrap
def bootstrap_indices(groups, labels, n_boot=N_BOOT, seed=SEED, stratify=True):
    """Cluster bootstrap: resample whole groups with replacement (within each group-label stratum when stratify=True and
    every group has one label, so every resample keeps both classes). Returns a list of sample-index arrays; the same
    list is reused for every model of a dataset (paired resampling)."""
    groups = np.asarray(groups)
    labels = np.asarray(labels).astype(int)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    g_lab = np.array([labels[m].max() if labels[m].min() == labels[m].max() else -1 for m in members])
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero(g_lab == c) for c in (0, 1)] if stratify and (g_lab >= 0).all() \
        else [np.arange(len(uniq))]
    out = []
    for _ in range(n_boot):
        pick = np.concatenate([rng.choice(s, size=len(s), replace=True) for s in strata])
        out.append(np.concatenate([members[g] for g in pick]))
    return out


def boot_stat(fn, idx_list, *arrays):
    """fn(*[a[idx] for a in arrays]) for every resample (NaN-safe)."""
    return np.array([fn(*[np.asarray(a)[ix] for a in arrays]) for ix in idx_list], dtype=float)


def ci(values, level=0.95):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    a = (1 - level) / 2
    return float(np.quantile(v, a)), float(np.quantile(v, 1 - a))


def boot_p_two_sided(diffs, null=0.0):
    """Bootstrap two-sided p for H0: statistic = null (share of resamples on the other side, doubled, capped at 1)."""
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    return float(min(1.0, 2 * min((d <= null).mean(), (d >= null).mean())))


# ---------------------------------------------------------------------------------------------- DeLong
def _midrank(x):
    return stats.rankdata(x)


def delong(y, p1, p2=None):
    """DeLong (1988) via Sun & Xu (2014) placements. Returns dict with auc1 [, auc2, diff, se_diff, z, p] for the
    two-sided test of AUC1 == AUC2 on the same samples (paired)."""
    y = np.asarray(y).astype(int)
    preds = [np.asarray(p1, float)] + ([np.asarray(p2, float)] if p2 is not None else [])
    pos, neg = y == 1, y == 0
    m, n = int(pos.sum()), int(neg.sum())
    aucs, v10, v01 = [], [], []
    for p in preds:
        x, z = p[pos], p[neg]
        tx, tz, tall = _midrank(x), _midrank(z), _midrank(np.concatenate([x, z]))
        a = (tall[:m].sum() - m * (m + 1) / 2) / (m * n)
        aucs.append(a)
        v10.append((tall[:m] - tx) / n)          # structural components of positives
        v01.append(1.0 - (tall[m:] - tz) / m)    # of negatives
    s10, s01 = np.atleast_2d(np.cov(np.array(v10))), np.atleast_2d(np.cov(np.array(v01)))
    S = s10 / m + s01 / n
    out = {"auc1": float(aucs[0]), "var1": float(S[0, 0])}
    if p2 is None:
        return out
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    diff = aucs[0] - aucs[1]
    if var <= 0:
        zstat, pval = (0.0, 1.0) if diff == 0 else (np.inf * np.sign(diff), 0.0)
    else:
        zstat = diff / np.sqrt(var)
        pval = 2 * stats.norm.sf(abs(zstat))
    out.update({"auc2": float(aucs[1]), "diff": float(diff), "se_diff": float(np.sqrt(max(var, 0))),
                "z": float(zstat), "p": float(pval)})
    return out


# ---------------------------------------------------------------------------------------------- model comparison
def friedman(mat):
    """mat: (n_blocks=folds, k=models) scores, higher = better. Returns chi2, p, Iman-Davenport F and p, average ranks
    (rank 1 = best within each fold, ties averaged)."""
    mat = np.asarray(mat, float)
    N, k = mat.shape
    ranks = np.vstack([stats.rankdata(-row) for row in mat])
    R = ranks.mean(axis=0)
    chi2, p = stats.friedmanchisquare(*[mat[:, j] for j in range(k)])
    ff = (N - 1) * chi2 / (N * (k - 1) - chi2) if N * (k - 1) != chi2 else np.inf
    p_ff = stats.f.sf(ff, k - 1, (k - 1) * (N - 1))
    return {"chi2": float(chi2), "p": float(p), "iman_davenport_F": float(ff), "iman_davenport_p": float(p_ff),
            "avg_ranks": R, "n_blocks": N, "k": k}


def nemenyi(avg_ranks, n_blocks, alpha=0.05):
    """Critical difference and the pairwise p-value matrix (two-sided, studentized range with infinite df)."""
    R = np.asarray(avg_ranks, float)
    k = len(R)
    se = np.sqrt(k * (k + 1) / (6.0 * n_blocks))
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    P = np.ones((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            P[i, j] = P[j, i] = float(stats.studentized_range.sf(abs(R[i] - R[j]) / se * np.sqrt(2), k, np.inf))
    return {"cd": float(q * se), "q_alpha": float(q), "p": P}


def nadeau_bengio(a, b, n_train, n_test):
    """Corrected resampled t-test (Nadeau & Bengio 2003) for per-fold scores a, b of the same J folds;
    n_train / n_test = mean training / test size per fold. Returns mean diff, t, two-sided p (df J-1)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    J = len(d)
    var = d.var(ddof=1)
    denom = np.sqrt((1.0 / J + n_test / n_train) * var)
    if denom == 0:
        return {"mean_diff": float(d.mean()), "t": float("nan"), "p": 1.0 if d.mean() == 0 else 0.0}
    t = d.mean() / denom
    return {"mean_diff": float(d.mean()), "t": float(t), "p": float(2 * stats.t.sf(abs(t), J - 1))}


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for r, i in enumerate(order):
        running = max(running, (len(p) - r) * p[i])
        adj[i] = min(1.0, running)
    return adj


# ---------------------------------------------------------------------------------------------- calibration
def ece(y, p, n_bins=10):
    """Expected calibration error, equal-width bins on [0, 1] (p = 1 goes to the last bin)."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    b = np.minimum((p * n_bins).astype(int), n_bins - 1)
    e = 0.0
    for k in range(n_bins):
        s = b == k
        if s.any():
            e += s.mean() * abs(y[s].mean() - p[s].mean())
    return float(e)


def reliability_bins(y, p, n_bins=10):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    b = np.minimum((p * n_bins).astype(int), n_bins - 1)
    rows = []
    for k in range(n_bins):
        s = b == k
        rows.append({"bin": k, "n": int(s.sum()), "mean_p": float(p[s].mean()) if s.any() else np.nan,
                     "frac_pos": float(y[s].mean()) if s.any() else np.nan})
    return rows


# ---------------------------------------------------------------------------------------------- matching
def greedy_caliper_match(ids_t, x_t, ids_c, x_c, caliper):
    """Greedy 1:1 nearest-neighbour matching without replacement: all (treated, control) pairs sorted by |x_t - x_c|
    (ties by ids), accepted while both are free and the distance <= caliper. Returns [(id_t, id_c, dist)]."""
    pairs = sorted(((abs(a - b), str(i), str(j), i, j) for i, a in zip(ids_t, x_t) for j, b in zip(ids_c, x_c)))
    used_t, used_c, out = set(), set(), []
    for dist, _, _, i, j in pairs:
        if dist > caliper:
            break
        if i in used_t or j in used_c:
            continue
        used_t.add(i)
        used_c.add(j)
        out.append((i, j, float(dist)))
    return out
