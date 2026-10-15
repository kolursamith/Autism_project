"""Stage 8 C - D02 facial images: dataset shortcuts. Reads saved out-of-fold predictions, the manifest and (read-only)
the raw images; no deep model is (re)trained. Outputs results/stage8/d02_*.
Usage: CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/08c_d02_shortcuts.py | tee logs/stage08_d02.txt"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.analysis import stats as S  # noqa: E402
from src.analysis.data import D02, MODELS, OUT_DIR, load_oof  # noqa: E402
from src.analysis.image_stats import FEATURES, image_features  # noqa: E402
from src.data.common import CACHE_DIR, PROJECT_DIR, write_csv, write_json  # noqa: E402
from src.data.d02_faces import hamming_matrix  # noqa: E402
from src.data.splits import load_fold  # noqa: E402
from src.utils.paths import DATASETS_DIR  # noqa: E402

NEAR_DUP_MAX = 8
RF_PARAMS = {"n_estimators": 500, "min_samples_leaf": 2, "max_features": "sqrt", "random_state": 42, "n_jobs": 2}
LR_PARAMS = {"C": 1.0, "max_iter": 5000}
ARTEFACT_THRESHOLD = 0.75


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    man = pd.read_csv(os.path.join(CACHE_DIR, D02, "manifest.csv"))
    folds = pd.read_csv(os.path.join(PROJECT_DIR, "splits", D02, "folds.csv")).set_index("sample_id")
    man["fold"] = man.image_id.map(folds.outer_fold)
    assert man.fold.notna().all()
    pos = {s: i for i, s in enumerate(man.image_id)}
    P = {m: load_oof(D02, m) for m in MODELS}

    # ------------------------------------------------------------------ C1 near-duplicate sensitivity
    H = hamming_matrix(man.phash.tolist())
    flag_rows, count_rows = [], []
    for k in range(1, 11):
        f = load_fold(D02, k)
        tr = np.array([pos[s] for s in list(f["train"]) + list(f["val"])])
        te = np.array([pos[s] for s in f["test"]])
        dmin = H[np.ix_(te, tr)].min(axis=1)
        nn = tr[H[np.ix_(te, tr)].argmin(axis=1)]
        for i, d, j in zip(te, dmin, nn):
            flag_rows.append({"image_id": man.image_id[i], "fold": k, "label": int(man.label[i]),
                              "min_hamming_to_train": int(d), "nearest_train_image": man.image_id[j],
                              "nearest_train_label": int(man.label[j]), "near_dup": bool(d <= NEAR_DUP_MAX)})
        count_rows.append({"fold": k, "n_test": len(te), "n_near_dup_le8": int((dmin <= NEAR_DUP_MAX).sum()),
                           "n_le4": int((dmin <= 4).sum())})
    F = pd.DataFrame(flag_rows)
    write_csv(F, os.path.join(OUT_DIR, "d02_near_dup_test_images.csv"))
    cnt = pd.DataFrame(count_rows)
    write_csv(cnt, os.path.join(OUT_DIR, "d02_near_dup_counts.csv"))
    nd = set(F.image_id[F.near_dup])
    same_lab = F[F.near_dup]
    print(f"C1 test images with a pHash <= {NEAR_DUP_MAX} neighbour among the fold's training (train+val) images: "
          f"{len(nd)} of {len(F)} (per fold {cnt.n_near_dup_le8.tolist()}); <= 4: {int(cnt.n_le4.sum())}; nearest "
          f"neighbour has the same label: {int((same_lab.label == same_lab.nearest_train_label).sum())} of {len(same_lab)}")
    rows = []
    for m in MODELS:
        p = P[m]
        keep = ~p.sample_id.isin(nd)
        fa = [S.auc(g.y_true, g.y_prob) for _, g in p.groupby("fold")]
        fr = [S.auc(g.y_true, g.y_prob) for _, g in p[keep].groupby("fold")]
        rows.append({"model": m, "n_all": len(p), "n_removed": int((~keep).sum()), "pooled_auc_all": S.auc(p.y_true, p.y_prob),
                     "pooled_auc_removed": S.auc(p.y_true[keep], p.y_prob[keep]),
                     "fold_mean_auc_all": float(np.mean(fa)), "fold_mean_auc_removed": float(np.mean(fr)),
                     "fold_std_auc_removed": float(np.std(fr, ddof=1)),
                     "auc_on_near_dup_images_only": S.auc(p.y_true[~keep], p.y_prob[~keep])})
    C1 = pd.DataFrame(rows)
    C1["pooled_diff"] = C1.pooled_auc_removed - C1.pooled_auc_all
    C1["fold_mean_diff"] = C1.fold_mean_auc_removed - C1.fold_mean_auc_all
    write_csv(C1, os.path.join(OUT_DIR, "d02_near_dup_sensitivity.csv"))
    for r in C1.itertuples():
        print(f"C1 {r.model} fold-mean AUC all {r.fold_mean_auc_all:.3f} -> without near-dups {r.fold_mean_auc_removed:.3f} "
              f"({r.fold_mean_diff:+.3f}); pooled {r.pooled_auc_all:.3f} -> {r.pooled_auc_removed:.3f}")

    # ------------------------------------------------------------------ C2 image-statistics baseline
    feat_path = os.path.join(OUT_DIR, "d02_image_stats.csv")
    if os.path.isfile(feat_path):
        X = pd.read_csv(feat_path)
    else:
        X = pd.DataFrame([{"image_id": r.image_id, **image_features(os.path.join(DATASETS_DIR, r.rel_path))}
                          for r in man.itertuples()])
        write_csv(X, feat_path)
    X = man[["image_id", "label", "fold", "group_id", "original_split"]].merge(X, on="image_id")
    assert len(X) == len(man) and X[FEATURES].notna().all().all(), X[FEATURES].isna().sum()
    uni = [{"feature": c, "auc_signed": S.auc(X.label, X[c]), "mean_autistic": X.loc[X.label == 1, c].mean(),
            "mean_non_autistic": X.loc[X.label == 0, c].mean()} for c in FEATURES]
    U = pd.DataFrame(uni)
    U["auc_direction_free"] = np.maximum(U.auc_signed, 1 - U.auc_signed)
    U = U.sort_values("auc_direction_free", ascending=False)
    write_csv(U, os.path.join(OUT_DIR, "d02_image_stats_univariate.csv"))
    X["p_lr"], X["p_rf"] = np.nan, np.nan
    for k in range(1, 11):
        f = load_fold(D02, k)
        trn = X.image_id.isin(set(f["train"]) | set(f["val"]))
        tst = X.image_id.isin(set(f["test"]))
        assert not (trn & tst).any() and tst.sum() == len(f["test"])
        lr = make_pipeline(StandardScaler(), LogisticRegression(random_state=42, **LR_PARAMS))
        lr.fit(X.loc[trn, FEATURES], X.label[trn])
        X.loc[tst, "p_lr"] = lr.predict_proba(X.loc[tst, FEATURES])[:, 1]
        rf = RandomForestClassifier(**RF_PARAMS).fit(X.loc[trn, FEATURES], X.label[trn])
        X.loc[tst, "p_rf"] = rf.predict_proba(X.loc[tst, FEATURES])[:, 1]
    rows = []
    for name, col in [("image_stats_logreg", "p_lr"), ("image_stats_random_forest", "p_rf")]:
        fa = [S.auc(g.label, g[col]) for _, g in X.groupby("fold")]
        rows.append({"baseline": name, "fold_mean_auc": float(np.mean(fa)), "fold_std_auc": float(np.std(fa, ddof=1)),
                     "pooled_auc": S.auc(X.label, X[col]), "fold_aucs": " ".join(f"{a:.3f}" for a in fa),
                     "params": str(LR_PARAMS if col == "p_lr" else RF_PARAMS)})
    C2 = pd.DataFrame(rows)
    C2["artefacts_alone_separate_classes"] = C2.fold_mean_auc >= ARTEFACT_THRESHOLD
    write_csv(C2, os.path.join(OUT_DIR, "d02_image_stats_baseline.csv"))
    write_csv(X[["image_id", "label", "fold", "group_id", "p_lr", "p_rf"]], os.path.join(OUT_DIR, "d02_image_stats_oof.csv"))
    for r in C2.itertuples():
        print(f"C2 {r.baseline}: AUC {r.fold_mean_auc:.3f} ± {r.fold_std_auc:.3f} (pooled {r.pooled_auc:.3f}); "
              f">= {ARTEFACT_THRESHOLD}: {r.artefacts_alone_separate_classes}")
    print("C2 strongest single features (direction-free AUC): " +
          ", ".join(f"{r.feature} {r.auc_direction_free:.3f}" for r in U.head(5).itertuples()))

    # ------------------------------------------------------------------ C3 original-split analysis with bootstrap CIs
    orig = man.set_index("image_id").original_split
    base = P[MODELS[0]].copy()
    idx = S.bootstrap_indices(base.group_id, base.y_true, S.N_BOOT, S.SEED, stratify=True)
    rows = []
    for m in MODELS:
        p = P[m].set_index("sample_id").loc[base.sample_id].reset_index()
        is_t = (p.sample_id.map(orig) == "test").to_numpy()
        y, s = p.y_true.to_numpy(), p.y_prob.to_numpy()

        def sub(ix, which):
            sel = ix[is_t[ix]] if which else ix[~is_t[ix]]
            return S.auc(y[sel], s[sel])
        bt = np.array([sub(ix, True) for ix in idx])
        br = np.array([sub(ix, False) for ix in idx])
        at, ar = S.auc(y[is_t], s[is_t]), S.auc(y[~is_t], s[~is_t])
        rows.append({"model": m, "n_orig_test": int(is_t.sum()), "n_rest": int((~is_t).sum()),
                     "auc_orig_test": at, "ci_lo_orig_test": S.ci(bt)[0], "ci_hi_orig_test": S.ci(bt)[1],
                     "auc_rest": ar, "ci_lo_rest": S.ci(br)[0], "ci_hi_rest": S.ci(br)[1],
                     "diff": at - ar, "diff_ci_lo": S.ci(bt - br)[0], "diff_ci_hi": S.ci(bt - br)[1]})
    C3 = pd.DataFrame(rows)
    write_csv(C3, os.path.join(OUT_DIR, "d02_original_split.csv"))
    for r in C3.itertuples():
        print(f"C3 {r.model} orig-test {r.auc_orig_test:.3f} [{r.ci_lo_orig_test:.3f}, {r.ci_hi_orig_test:.3f}] vs rest "
              f"{r.auc_rest:.3f} [{r.ci_lo_rest:.3f}, {r.ci_hi_rest:.3f}]; diff {r.diff:+.3f} [{r.diff_ci_lo:+.3f}, {r.diff_ci_hi:+.3f}]")
    write_json({"near_dup_max_hamming": NEAR_DUP_MAX, "training_images": "train + val of the fold (all non-test images)",
                "rf_params": RF_PARAMS, "lr_params": LR_PARAMS, "artefact_threshold": ARTEFACT_THRESHOLD,
                "bootstrap": f"{S.N_BOOT} stratified resamples of near-duplicate groups, seed {S.SEED}"},
               os.path.join(OUT_DIR, "d02_settings.json"))


if __name__ == "__main__":
    main()
