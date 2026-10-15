"""Stage 8 B - D01 eye-tracking: tracking-quality shortcut (DD-12). Reads saved out-of-fold predictions and
cache/D01_eye_tracking/{quality_per_child,clips_index}.csv; no deep model is (re)trained. Outputs results/stage8/d01_*.
Usage: CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/08b_d01_quality.py | tee logs/stage08_d01.txt"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sps  # noqa: E402

from src.analysis import stats as S  # noqa: E402
from src.analysis.data import D01, MODELS, OUT_DIR, child_scores, load_oof  # noqa: E402
from src.analysis.quality import QUALITY_FEATURES, fit_predict_lr, fold_children, tertile_labels  # noqa: E402
from src.data.common import CACHE_DIR, PROJECT_DIR, write_csv, write_json  # noqa: E402
from src.data.splits import load_fold  # noqa: E402

CALIPER = 0.05


def per_fold_auc(df, score, y="y", fold="fold"):
    v = [S.auc(g[y], g[score]) for _, g in df.groupby(fold)]
    return float(np.nanmean(v)), float(np.nanstd(v, ddof=1)), v


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    folds = pd.read_csv(os.path.join(PROJECT_DIR, "splits", D01, "folds.csv"))
    q = pd.read_csv(os.path.join(CACHE_DIR, D01, "quality_per_child.csv"))
    q["group_id"] = q.participant_id.map(lambda i: f"p{int(i):02d}")
    child_fold = folds.groupby("group_id").outer_fold.first()
    q = q[q.group_id.isin(child_fold.index)].copy()
    q["fold"] = q.group_id.map(child_fold)
    clips = pd.read_csv(os.path.join(CACHE_DIR, D01, "clips_index.csv"))
    clips = clips.merge(folds[["sample_id", "outer_fold"]], left_on="clip_id", right_on="sample_id")
    print(f"B0 children with clips: {len(q)} ({int(q.label.sum())} ASD / {int((1 - q.label).sum())} TD); clips "
          f"{len(clips)}; quality features {QUALITY_FEATURES}")

    # ------------------------------------------------------------------ B1 quality-only baselines (OOF child scores)
    q["score_1mvr"] = 1 - q.mean_valid_ratio            # fixed score, no fitting (direction from Stage 5)
    q["score_lr3"] = np.nan
    q["score_lr1"] = np.nan
    fold_log = []
    for k in range(1, 11):
        tr_ids, te_ids = fold_children(folds, load_fold, D01, k)
        tr, te = q[q.group_id.isin(tr_ids)], q[q.group_id.isin(te_ids)]
        assert set(te.group_id) == set(q.group_id[q.fold == k])
        q.loc[te.index, "score_lr3"] = fit_predict_lr(tr, te, QUALITY_FEATURES)
        q.loc[te.index, "score_lr1"] = fit_predict_lr(tr, te, ["mean_valid_ratio"])
        fold_log.append({"fold": k, "n_train_children": len(tr), "n_test_children": len(te),
                         "train_children": " ".join(tr.group_id), "test_children": " ".join(te.group_id)})
    write_csv(pd.DataFrame(fold_log), os.path.join(OUT_DIR, "d01_quality_baseline_folds.csv"))
    q = q.rename(columns={"label": "y"})
    base_rows = []
    for name, col, desc in [("quality_LR3", "score_lr3", "LR on " + " + ".join(QUALITY_FEATURES) + " (fit per fold)"),
                            ("quality_LR_valid_ratio", "score_lr1", "LR on mean_valid_ratio (fit per fold)"),
                            ("one_minus_valid_ratio", "score_1mvr", "1 - mean valid_ratio (no fitting)")]:
        m, s, _ = per_fold_auc(q, col)
        base_rows.append({"baseline": name, "level": "child", "description": desc, "n": len(q),
                          "pooled_auc": S.auc(q.y, q[col]), "fold_mean_auc": m, "fold_std_auc": s})
    clips["score_1mvr"] = 1 - clips.valid_ratio
    clips["score_lr_clip"] = np.nan
    for k in range(1, 11):
        tr, te = clips[clips.outer_fold != k], clips[clips.outer_fold == k]
        clips.loc[te.index, "score_lr_clip"] = fit_predict_lr(tr.drop(columns="participant_id"), te.drop(columns="participant_id"),
                                                              ["valid_ratio", "blink_ratio"])
    for name, col, desc in [("clip_one_minus_valid_ratio", "score_1mvr", "each clip: 1 - its own valid_ratio (no fitting)"),
                            ("clip_LR_valid_blink", "score_lr_clip", "LR on clip valid_ratio + blink_ratio (fit per fold on training clips)")]:
        m, s, _ = per_fold_auc(clips, col, y="label", fold="outer_fold")
        base_rows.append({"baseline": name, "level": "clip", "description": desc, "n": len(clips),
                          "pooled_auc": S.auc(clips.label, clips[col]), "fold_mean_auc": m, "fold_std_auc": s})
    B1 = pd.DataFrame(base_rows)
    write_csv(B1, os.path.join(OUT_DIR, "d01_quality_baselines.csv"))
    write_csv(q[["group_id", "y", "fold", "n_clips"] + QUALITY_FEATURES + ["score_lr3", "score_lr1", "score_1mvr"]],
              os.path.join(OUT_DIR, "d01_quality_child_scores.csv"))
    for r in base_rows:
        print(f"B1 {r['baseline']:28s} {r['level']:5s} pooled AUC {r['pooled_auc']:.3f}; per-fold {r['fold_mean_auc']:.3f} "
              f"± {r['fold_std_auc']:.3f}")

    # ------------------------------------------------------------------ child table with every model's OOF score
    C = q[["group_id", "y", "fold", "mean_valid_ratio", "score_lr3", "score_1mvr"]].copy()
    clipP = {}
    for m in MODELS:
        P = load_oof(D01, m)
        clipP[m] = P
        cs = child_scores(P).set_index("group_id")
        assert set(cs.index) == set(C.group_id) and (cs.y.loc[C.group_id].to_numpy() == C.y.to_numpy()).all()
        C[m] = cs.p.loc[C.group_id].to_numpy()
    C = C.reset_index(drop=True)
    write_csv(C, os.path.join(OUT_DIR, "d01_child_oof_scores.csv"))

    # ------------------------------------------------------------------ B2 paired comparison model vs quality-only
    idx = S.bootstrap_indices(C.group_id, C.y, S.N_BOOT, S.SEED, stratify=True)
    rows = []
    for base, bcol in [("quality_LR3", "score_lr3"), ("one_minus_valid_ratio", "score_1mvr")]:
        b_boot = S.boot_stat(S.auc, idx, C.y, C[bcol])
        for m in MODELS:
            m_boot = S.boot_stat(S.auc, idx, C.y, C[m])
            d = m_boot - b_boot
            dl = S.delong(C.y, C[m], C[bcol])
            lo, hi = S.ci(d)
            rows.append({"model": m, "baseline": base, "model_auc": S.auc(C.y, C[m]), "baseline_auc": S.auc(C.y, C[bcol]),
                         "diff": S.auc(C.y, C[m]) - S.auc(C.y, C[bcol]), "diff_ci_lo": lo, "diff_ci_hi": hi,
                         "model_auc_ci_lo": S.ci(m_boot)[0], "model_auc_ci_hi": S.ci(m_boot)[1],
                         "delong_z": dl["z"], "delong_p": dl["p"], "n_children": len(C), "n_boot": S.N_BOOT})
    B2 = pd.DataFrame(rows)
    write_csv(B2, os.path.join(OUT_DIR, "d01_model_vs_quality.csv"))
    for r in B2[B2.baseline == "quality_LR3"].itertuples():
        print(f"B2 {r.model} child AUC {r.model_auc:.3f} vs quality_LR3 {r.baseline_auc:.3f}: diff {r.diff:+.3f} "
              f"[{r.diff_ci_lo:+.3f}, {r.diff_ci_hi:+.3f}], DeLong p {r.delong_p:.3f}")

    # ------------------------------------------------------------------ B3 Spearman dependence on valid_ratio
    rows = []
    for m in MODELS:
        for grp, sel in [("all", C.y >= 0), ("ASD", C.y == 1), ("TD", C.y == 0)]:
            rho, p = sps.spearmanr(C.loc[sel, m], C.loc[sel, "mean_valid_ratio"])
            rows.append({"model": m, "children": grp, "n": int(sel.sum()), "spearman_rho": float(rho), "p": float(p)})
    B3 = pd.DataFrame(rows)
    write_csv(B3, os.path.join(OUT_DIR, "d01_spearman_valid_ratio.csv"))

    # ------------------------------------------------------------------ B4a quality tertiles
    rows = []
    clips_t, (c1, c2) = tertile_labels(clips.valid_ratio.to_numpy())
    vr = clips.set_index("clip_id").valid_ratio
    tert = pd.Series(clips_t, index=clips.clip_id)
    for m in MODELS + ["quality: 1 - valid_ratio"]:
        P = clipP["M01"] if m.startswith("quality") else clipP[m]
        sc = (1 - P.sample_id.map(vr)) if m.startswith("quality") else P.y_prob
        t = P.sample_id.map(tert).to_numpy()
        for j in range(3):
            s = t == j
            rows.append({"level": "clip", "model": m, "tertile": ["low", "mid", "high"][j], "n": int(s.sum()),
                         "n_pos": int(P.y_true[s].sum()), "n_neg": int((1 - P.y_true[s]).sum()),
                         "auc": S.auc(P.y_true[s], sc[s])})
    child_t, (h1, h2) = tertile_labels(C.mean_valid_ratio.to_numpy())
    for m in MODELS + ["score_1mvr", "score_lr3"]:
        for j in range(3):
            s = child_t == j
            n1, n0 = int(C.y[s].sum()), int((1 - C.y[s]).sum())
            rows.append({"level": "child", "model": {"score_1mvr": "quality: 1 - valid_ratio",
                                                     "score_lr3": "quality_LR3"}.get(m, m),
                         "tertile": ["low", "mid", "high"][j], "n": int(s.sum()), "n_pos": n1, "n_neg": n0,
                         "auc": S.auc(C.y[s], C[m][s]) if n1 and n0 else float("nan")})
    B4a = pd.DataFrame(rows)
    write_csv(B4a, os.path.join(OUT_DIR, "d01_quality_tertiles.csv"))
    write_json({"clip_valid_ratio_cuts": [c1, c2], "child_mean_valid_ratio_cuts": [h1, h2]},
               os.path.join(OUT_DIR, "d01_quality_tertile_cuts.json"))

    # ------------------------------------------------------------------ B4b quality-matched subset
    asd, td = C[C.y == 1], C[C.y == 0]
    pairs = S.greedy_caliper_match(list(asd.group_id), list(asd.mean_valid_ratio), list(td.group_id),
                                   list(td.mean_valid_ratio), CALIPER)
    pr = pd.DataFrame(pairs, columns=["asd_child", "td_child", "abs_diff_valid_ratio"])
    write_csv(pr, os.path.join(OUT_DIR, "d01_matched_pairs.csv"))
    Cm = C.set_index("group_id")
    a_ids, t_ids = list(pr.asd_child), list(pr.td_child)
    M = pd.concat([Cm.loc[a_ids], Cm.loc[t_ids]]).reset_index()
    npair = len(pr)
    rng = np.random.default_rng(S.SEED)
    pidx = [rng.integers(0, npair, npair) for _ in range(S.N_BOOT)]
    boot_idx = [np.concatenate([pi, pi + npair]) for pi in pidx]  # resample pairs (ASD row i, TD row npair + i)

    def smd(a, b):
        return float((a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2))
    rows = []
    for m in MODELS + ["score_1mvr", "score_lr3"]:
        a = S.auc(M.y, M[m])
        bs = S.boot_stat(S.auc, boot_idx, M.y, M[m])
        lo, hi = S.ci(bs)
        rows.append({"model": {"score_1mvr": "quality: 1 - valid_ratio", "score_lr3": "quality_LR3"}.get(m, m),
                     "n_pairs": npair, "auc_matched": a, "ci_lo": lo, "ci_hi": hi, "ci_excludes_0.5_above": lo > 0.5})
    B4b = pd.DataFrame(rows)
    write_csv(B4b, os.path.join(OUT_DIR, "d01_matched_auc.csv"))
    bal = {"caliper": CALIPER, "n_asd": int(len(asd)), "n_td": int(len(td)), "n_pairs": npair,
           "mean_valid_ratio_asd_all": float(asd.mean_valid_ratio.mean()), "mean_valid_ratio_td_all": float(td.mean_valid_ratio.mean()),
           "smd_all": smd(asd.mean_valid_ratio, td.mean_valid_ratio),
           "mean_valid_ratio_asd_matched": float(Cm.loc[a_ids].mean_valid_ratio.mean()),
           "mean_valid_ratio_td_matched": float(Cm.loc[t_ids].mean_valid_ratio.mean()),
           "smd_matched": smd(Cm.loc[a_ids].mean_valid_ratio, Cm.loc[t_ids].mean_valid_ratio),
           "max_abs_diff_in_pair": float(pr.abs_diff_valid_ratio.max()) if npair else None,
           "bootstrap": f"{S.N_BOOT} resamples of matched pairs, seed {S.SEED}"}
    write_json(bal, os.path.join(OUT_DIR, "d01_matching_balance.json"))
    print(f"B4b matching (caliper {CALIPER}): {npair} pairs of {len(asd)} ASD / {len(td)} TD; SMD valid_ratio "
          f"{bal['smd_all']:.2f} -> {bal['smd_matched']:.2f}")
    for r in B4b.itertuples():
        print(f"B4b {r.model:26s} matched AUC {r.auc_matched:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]")

    # ------------------------------------------------------------------ B5 incremental value (exploratory)
    rows, per_fold = [], []
    for m in MODELS:
        inc_q, inc_qm = pd.Series(np.nan, index=C.index), pd.Series(np.nan, index=C.index)
        for k in range(1, 11):
            tr_ids, te_ids = fold_children(folds, load_fold, D01, k)
            tr = q[q.group_id.isin(tr_ids)].merge(C[["group_id", m]], on="group_id")
            te = q[q.group_id.isin(te_ids)].merge(C[["group_id", m]], on="group_id")
            ti = C.index[C.group_id.isin(te_ids)]
            order = C.loc[ti, "group_id"].to_numpy()
            te = te.set_index("group_id").loc[order].reset_index()
            inc_q.loc[ti] = fit_predict_lr(tr, te, QUALITY_FEATURES, label="y")
            inc_qm.loc[ti] = fit_predict_lr(tr, te, QUALITY_FEATURES + [m], label="y")
            a_q, a_qm = S.auc(te.y, inc_q.loc[ti]), S.auc(te.y, inc_qm.loc[ti])
            per_fold.append({"model": m, "fold": k, "auc_quality": a_q, "auc_quality_plus_score": a_qm, "diff": a_qm - a_q})
        pf = pd.DataFrame([r for r in per_fold if r["model"] == m])
        rows.append({"model": m, "pooled_auc_quality": S.auc(C.y, inc_q), "pooled_auc_quality_plus_score": S.auc(C.y, inc_qm),
                     "pooled_diff": S.auc(C.y, inc_qm) - S.auc(C.y, inc_q),
                     "fold_mean_auc_quality": pf.auc_quality.mean(), "fold_mean_auc_quality_plus_score": pf.auc_quality_plus_score.mean(),
                     "fold_mean_diff": pf["diff"].mean(), "fold_std_diff": pf["diff"].std(ddof=1),
                     "folds_improved": int((pf["diff"] > 0).sum()), "folds_worse": int((pf["diff"] < 0).sum())})
    B5 = pd.DataFrame(rows)
    write_csv(B5, os.path.join(OUT_DIR, "d01_incremental_value.csv"))
    write_csv(pd.DataFrame(per_fold), os.path.join(OUT_DIR, "d01_incremental_value_per_fold.csv"))
    for r in B5.itertuples():
        print(f"B5 {r.model} pooled AUC quality {r.pooled_auc_quality:.3f} -> +score {r.pooled_auc_quality_plus_score:.3f} "
              f"({r.pooled_diff:+.3f}); per-fold mean diff {r.fold_mean_diff:+.3f}")

    # ------------------------------------------------------------------ B6 interpretation rule (applied)
    rows = []
    for m in MODELS:
        b2 = B2[(B2.model == m) & (B2.baseline == "quality_LR3")].iloc[0]
        b4 = B4b[B4b.model == m].iloc[0]
        r1 = b2.diff_ci_lo > 0
        r2 = bool(b4.ci_lo > 0.5)
        rows.append({"model": m, "rule1_diff_ci_excludes_0_for_model": bool(r1), "rule2_matched_ci_excludes_0.5": r2,
                     "adds_information_beyond_tracking_quality": bool(r1 or r2),
                     "secondary_vs_1mvr_ci_lo": float(B2[(B2.model == m) & (B2.baseline == "one_minus_valid_ratio")].diff_ci_lo.iloc[0])})
    B6 = pd.DataFrame(rows)
    write_csv(B6, os.path.join(OUT_DIR, "d01_interpretation.csv"))
    print(f"B6 models that add information beyond tracking quality (rule): "
          f"{list(B6.model[B6.adds_information_beyond_tracking_quality]) or 'none'}")


if __name__ == "__main__":
    main()
