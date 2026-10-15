"""Stage 8 D - statistics for all datasets from saved out-of-fold predictions / metrics.json (read-only):
bootstrap CIs of pooled OOF AUC, Friedman + Nemenyi + Nadeau-Bengio (deep models), calibration, pretrained vs scratch.
Outputs results/stage8/stats_*. Usage: CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/08d_statistics.py | tee logs/stage08_stats.txt"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.analysis import stats as S  # noqa: E402
from src.analysis.data import (CUSTOM_CNN, D01, D02, MODELS, OUT_DIR, PRETRAINED, TAB_MODELS, TABULAR,  # noqa: E402
                               child_scores, load_oof, model_dir)
from src.data.common import write_csv, write_json  # noqa: E402
from src.data.splits import load_fold  # noqa: E402


def fold_metric(ds, model, block, key="roc_auc", experiment=None):
    out = []
    for k in range(1, 11):
        with open(os.path.join(model_dir(ds, model, experiment), f"fold_{k:02d}", "metrics.json")) as fh:
            out.append(json.load(fh)[block]["threshold_0.5"][key])
    return np.array(out, float)


def boot_rows(name, level, unit, y, p, groups, label, extra=None):
    idx = S.bootstrap_indices(groups, y, S.N_BOOT, S.SEED, stratify=True)
    b = S.boot_stat(S.auc, idx, y, p)
    lo, hi = S.ci(b)
    g = pd.Series(np.asarray(y)).groupby(np.asarray(groups)).nunique()
    return {"dataset": name, "level": level, "model": label, "pooled_auc": S.auc(y, p), "ci_lo": lo, "ci_hi": hi,
            "n": len(y), "n_groups": int(len(g)), "resampling_unit": unit,
            "stratified": bool((g == 1).all()), **(extra or {})}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    # ------------------------------------------------------------------ D1 bootstrap CIs of pooled OOF AUC
    rows, calib, rel = [], [], []
    deep = {D01: {m: load_oof(D01, m) for m in MODELS}, D02: {m: load_oof(D02, m) for m in MODELS}}
    for m in MODELS:
        P = deep[D01][m]
        rows.append(boot_rows(D01, "clip", "child", P.y_true, P.y_prob, P.group_id, m))
        C = child_scores(P)
        rows.append(boot_rows(D01, "child", "child", C.y, C.p, C.group_id, m))
        P2 = deep[D02][m]
        rows.append(boot_rows(D02, "image", "near-duplicate group", P2.y_true, P2.y_prob, P2.group_id, m))
        for ds, level, y, p in [(D01, "clip", P.y_true, P.y_prob), (D01, "child", C.y, C.p),
                                (D02, "image", P2.y_true, P2.y_prob)]:
            calib.append({"dataset": ds, "level": level, "model": m, "n": len(y),
                          "brier": float(np.mean((np.asarray(p) - np.asarray(y)) ** 2)), "ece_10": S.ece(y, p),
                          "mean_p": float(np.mean(p)), "frac_pos": float(np.mean(y))})
            if level != "child":
                rel += [{"dataset": ds, "model": m, **r} for r in S.reliability_bins(y, p)]
    q = pd.read_csv(os.path.join(OUT_DIR, "d01_quality_child_scores.csv"))
    rows.append(boot_rows(D01, "child", "child", q.y, q.score_lr3, q.group_id, "baseline: quality_LR3"))
    rows.append(boot_rows(D01, "child", "child", q.y, q.score_1mvr, q.group_id, "baseline: 1 - valid_ratio"))
    X = pd.read_csv(os.path.join(OUT_DIR, "d02_image_stats_oof.csv"))
    rows.append(boot_rows(D02, "image", "near-duplicate group", X.label, X.p_lr, X.group_id, "baseline: image_stats_logreg"))
    rows.append(boot_rows(D02, "image", "near-duplicate group", X.label, X.p_rf, X.group_id, "baseline: image_stats_random_forest"))
    for ds, exps in TABULAR.items():
        for e in exps:
            for m in TAB_MODELS:
                P = load_oof(ds, m, e)
                rows.append(boot_rows(f"{ds}/{e}", "row", "duplicate-feature group", P.y_true, P.y_prob, P.group_id, m))
                calib.append({"dataset": f"{ds}/{e}", "level": "row", "model": m, "n": len(P),
                              "brier": float(np.mean((P.y_prob - P.y_true) ** 2)), "ece_10": S.ece(P.y_true, P.y_prob),
                              "mean_p": float(P.y_prob.mean()), "frac_pos": float(P.y_true.mean())})
    CI = pd.DataFrame(rows)
    write_csv(CI, os.path.join(OUT_DIR, "stats_bootstrap_ci.csv"))
    write_csv(pd.DataFrame(calib), os.path.join(OUT_DIR, "stats_calibration.csv"))
    write_csv(pd.DataFrame(rel), os.path.join(OUT_DIR, "stats_reliability_bins.csv"))
    print(f"D1 bootstrap CIs: {len(CI)} rows ({S.N_BOOT} resamples, seed {S.SEED}) -> stats_bootstrap_ci.csv")
    for (ds, lev), g in CI[~CI.model.str.startswith("baseline") & (CI.model != "dummy_majority")].groupby(["dataset", "level"], sort=False):
        b = g.sort_values("pooled_auc", ascending=False).iloc[0]
        print(f"D1 top {ds} [{lev}]: {b.model} pooled AUC {b.pooled_auc:.3f} [{b.ci_lo:.3f}, {b.ci_hi:.3f}]")

    # ------------------------------------------------------------------ D2 Friedman / Nemenyi / Nadeau-Bengio
    fr_out, nb_rows, mats = {}, [], []
    for ds, block, tag in [(D01, "test", "D01_clip"), (D01, "test_child", "D01_child"), (D02, "test", "D02_image")]:
        mat = np.column_stack([fold_metric(ds, m, block) for m in MODELS])
        mats += [{"analysis": tag, "fold": k + 1, **{m: mat[k, j] for j, m in enumerate(MODELS)}} for k in range(10)]
        valid = np.isfinite(mat).all(axis=1)
        fr = S.friedman(mat[valid])
        ne = S.nemenyi(fr["avg_ranks"], fr["n_blocks"])
        sizes = [(len(load_fold(ds, k)["train"]) + len(load_fold(ds, k)["val"]), len(load_fold(ds, k)["test"]))
                 for k in range(1, 11)]
        n_train, n_test = float(np.mean([s[0] for s in sizes])), float(np.mean([s[1] for s in sizes]))
        means = np.nanmean(mat, axis=0)
        best = int(np.argmax(means))
        res = []
        for j, m in enumerate(MODELS):
            if j == best:
                continue
            r = S.nadeau_bengio(mat[valid, best], mat[valid, j], n_train, n_test)
            res.append({"analysis": tag, "best": MODELS[best], "other": m, "mean_auc_best": means[best],
                        "mean_auc_other": means[j], **r})
        adj = S.holm([r["p"] for r in res])
        for r, a in zip(res, adj):
            r["p_holm"] = float(a)
        nb_rows += res
        fr_out[tag] = {"metric": f"{block}/threshold_0.5/roc_auc per fold", "n_folds_used": int(valid.sum()),
                       "chi2": fr["chi2"], "p": fr["p"], "iman_davenport_F": fr["iman_davenport_F"],
                       "iman_davenport_p": fr["iman_davenport_p"],
                       "avg_ranks": dict(zip(MODELS, map(float, fr["avg_ranks"]))), "nemenyi_cd": ne["cd"],
                       "nemenyi_q_alpha": ne["q_alpha"],
                       "nemenyi_p": {MODELS[i]: dict(zip(MODELS, map(float, ne["p"][i]))) for i in range(10)},
                       "n_sig_pairs_nemenyi": int((ne["p"][np.triu_indices(10, 1)] < 0.05).sum()),
                       "nadeau_bengio_n_train": n_train, "nadeau_bengio_n_test": n_test}
        print(f"D2 Friedman {tag}: chi2 {fr['chi2']:.2f}, p {fr['p']:.2g} (folds {int(valid.sum())}); Nemenyi CD "
              f"{ne['cd']:.2f}; significant Nemenyi pairs {fr_out[tag]['n_sig_pairs_nemenyi']}; best {MODELS[best]}; "
              f"Nadeau-Bengio Holm p < 0.05 vs: {[r['other'] for r in res if r['p_holm'] < 0.05]}")
    write_json(fr_out, os.path.join(OUT_DIR, "stats_friedman_nemenyi.json"))
    write_csv(pd.DataFrame(nb_rows), os.path.join(OUT_DIR, "stats_nadeau_bengio.csv"))
    write_csv(pd.DataFrame(mats), os.path.join(OUT_DIR, "stats_fold_auc_matrix.csv"))

    # ------------------------------------------------------------------ D4 pretrained vs from scratch (descriptive)
    rows = []
    for tag, ds, block, lev in [("D01 clip", D01, "test", "clip"), ("D01 child", D01, "test_child", "child"),
                                ("D02 image", D02, "test", "image")]:
        fm = {m: float(np.nanmean(fold_metric(ds, m, block))) for m in MODELS}
        pooled = CI[(CI.dataset == ds) & (CI.level == lev)].set_index("model").pooled_auc
        for grp, ms in [("pretrained (M01,M03,M04,M06,M07,M08,M10)", PRETRAINED), ("from scratch (M02,M05,M09)", CUSTOM_CNN)]:
            rows.append({"analysis": tag, "group": grp, "n_models": len(ms),
                         "mean_of_fold_mean_auc": float(np.mean([fm[m] for m in ms])),
                         "min_fold_mean_auc": float(np.min([fm[m] for m in ms])), "max_fold_mean_auc": float(np.max([fm[m] for m in ms])),
                         "mean_pooled_auc": float(np.mean([pooled[m] for m in ms]))})
    PS = pd.DataFrame(rows)
    write_csv(PS, os.path.join(OUT_DIR, "stats_pretrained_vs_scratch.csv"))
    for r in PS.itertuples():
        print(f"D4 {r.analysis:9s} {r.group[:12]:12s}: mean fold-mean AUC {r.mean_of_fold_mean_auc:.3f} "
              f"({r.min_fold_mean_auc:.3f}-{r.max_fold_mean_auc:.3f})")


if __name__ == "__main__":
    main()
