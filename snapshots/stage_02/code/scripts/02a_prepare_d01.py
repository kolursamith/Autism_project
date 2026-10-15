"""Stage 2 - D01 eye-tracking: load, clean, window, render (C1-C5). Outputs -> cache/D01_eye_tracking/.
Re-runnable: overwrites its outputs. Log: logs/stage02_d01.log
Usage: .venv/bin/python scripts/02a_prepare_d01.py"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data import d01_eye as d  # noqa: E402
from src.data.common import cache_dir, get_logger, load_cfg, md_table, set_seed, write_csv, write_json  # noqa: E402

DS = "D01_eye_tracking"


def main():
    t0 = time.time()
    cfg = load_cfg("D01")
    set_seed(cfg["seed"])
    log = get_logger("stage02_d01")
    out = cache_dir(DS)

    # C1 + C2 a-d
    df, mapping, steps, pairs, excl, n_raw = d.build_clean_gaze(cfg, log.info)
    write_csv(mapping, os.path.join(out, "column_mapping.csv"))
    log.info(f"C1: {mapping.source_file.nunique()} files, {mapping.n_source_columns.nunique()} distinct column counts, "
             f"{n_raw} raw rows")
    absent = mapping[mapping.source_column == ""].groupby("canonical")["source_file"].apply(lambda s: ", ".join(s))
    log.info("C1: canonical fields absent in some files (filled with NaN / right eye used):\n" + absent.to_string())

    # C2 e: screen resolution evidence + invalid gaze (rows kept)
    ev = d.resolution_evidence(df, cfg)
    log.info(f"C2e: resolution evidence {ev}")
    if ev["nearest_standard"] != [cfg["screen"]["width"], cfg["screen"]["height"]]:
        raise SystemExit(f"screen resolution in D01.yaml {cfg['screen']} disagrees with the data {ev}")
    corner = d.corner_evidence(df, cfg)
    log.info(f"C2e: origin-artefact evidence {corner}")
    overlap = d.overlapping_trial_pairs(df)
    log.info(f"C3: trials overlapping in time within a (file, child): {overlap}")
    n_invalid = int((~df["valid"]).sum())
    steps = pd.concat([steps, pd.DataFrame([{"step": f"e. mark invalid gaze (rows kept): {n_invalid} rows invalid",
                                             "rows_before": len(df), "rows_after": len(df), "removed": 0}])],
                      ignore_index=True)
    write_csv(steps, os.path.join(out, "cleaning_steps.csv"))
    write_csv(pairs, os.path.join(out, "dedupe_file_pairs.csv"))
    write_csv(excl, os.path.join(out, "excluded_participants.csv"))
    log.info("C2 cleaning steps:\n" + md_table(steps))
    log.info("C2b duplicates removed per file pair:\n" + md_table(pairs))
    log.info("C2c excluded participants:\n" + md_table(excl))

    dup = int(df.duplicated(["participant_id", "t_ms", "trial"]).sum())
    log.info(f"C2b check: duplicate (participant_id, t_ms, trial) after dedupe = {dup}")
    assert dup == 0
    log.info(f"eye used: {df['eye'].value_counts().to_dict()}; events: {df['event'].value_counts().to_dict()}")

    # canonical clean rows, sorted per recording
    df = d.recordings(df)
    keep_cols = ["source_file", "row_in_file", "participant_id", "label", "trial", "stimulus", "t_ms", "event", "eye",
                 "x_px", "y_px", "x_right", "y_right", "pupil_mm", "tracking_ratio", "valid"]
    df[keep_cols].to_pickle(os.path.join(out, "gaze_clean.pkl"))

    # C3 clips
    clips, recs = d.make_clips(df, cfg)
    clips = d.apply_cap(clips, cfg)
    write_csv(recs, os.path.join(out, "recordings.csv"))
    write_csv(clips, os.path.join(out, "clips_index_uncapped.csv"))
    log.info(f"C3: {len(recs)} recordings, {int(recs.n_segments.sum())} continuous segments, "
             f"{len(clips)} candidate 8-s clips; status counts {clips.status.value_counts().to_dict()}")
    kept = clips[clips.status == "kept"].sort_values(["participant_id", "source_file", "trial", "start_ms"],
                                                     kind="mergesort").reset_index(drop=True)
    kept["row_in_npy"] = np.arange(len(kept))
    idx_cols = ["clip_id", "participant_id", "label", "source_file", "trial", "stimulus", "start_ms", "end_ms",
                "n_samples", "valid_ratio", "blink_ratio", "fixation_ratio", "saccade_ratio", "row_in_npy"]
    write_csv(kept[idx_cols], os.path.join(out, "clips_index.csv"))

    # participants table (report only: age, gender) + children without clips
    meta = d.read_metadata()
    per = df.groupby("participant_id").agg(n_rows=("t_ms", "size"), valid_fraction_all_rows=("valid", "mean"),
                                           tracking_ratio_mean=("tracking_ratio", "mean")).reset_index()
    cnt = clips.pivot_table(index="participant_id", columns="status", values="clip_id", aggfunc="count",
                            fill_value=0).reset_index()
    parts = meta.merge(per, on="participant_id", how="left").merge(cnt, on="participant_id", how="left").fillna(
        {c: 0 for c in cnt.columns if c != "participant_id"})
    for c in ["kept", "dropped_by_cap", "rejected_gap", "rejected_valid_ratio", "rejected_other_trial_in_span"]:
        if c not in parts:
            parts[c] = 0
    parts["n_clips_uncapped"] = parts["kept"] + parts["dropped_by_cap"]
    parts["n_clips"] = parts["kept"]

    def reason(r):
        if r["n_rows"] != r["n_rows"] or r["n_rows"] == 0:
            return "excluded: no identified gaze rows"
        if r["kept"] == 0:
            return ("excluded: no 8-s clip with valid_ratio >= 0.5" if r["rejected_valid_ratio"] > 0
                    else "excluded: no continuous 8-s segment")
        return "included"
    parts["status"] = parts.apply(reason, axis=1)
    pcols = ["participant_id", "label", "class", "age", "gender", "n_rows", "valid_fraction_all_rows",
             "tracking_ratio_mean", "rejected_gap", "rejected_other_trial_in_span", "rejected_valid_ratio",
             "n_clips_uncapped", "n_clips", "status"]
    parts = parts[pcols].sort_values("participant_id")
    write_csv(parts, os.path.join(out, "participants.csv"))
    inc = parts[parts.status == "included"]
    log.info("C3 children per class after filtering: " + str(inc.groupby("class").size().to_dict()))
    log.info("C3 children excluded:\n" + md_table(parts[parts.status != "included"][["participant_id", "class", "status"]]))
    s = inc["n_clips"]
    log.info(f"C3 clips per child (capped): min {s.min()}, median {s.median()}, max {s.max()}; "
             f"uncapped: min {inc.n_clips_uncapped.min()}, median {inc.n_clips_uncapped.median()}, "
             f"max {inc.n_clips_uncapped.max()}")
    log.info(f"C3 clips per class (capped): {kept.groupby('label').size().to_dict()} (1=ASD, 0=TD); "
             f"uncapped: {clips[clips.status.isin(['kept', 'dropped_by_cap'])].groupby('label').size().to_dict()}")
    stim = kept.groupby(["stimulus", "label"]).size().unstack(fill_value=0)
    write_csv(stim.reset_index(), os.path.join(out, "clips_per_stimulus_and_label.csv"))

    # C4 render into a memmap
    cfgr = cfg["render"]
    shape = (len(kept), cfg["clips"]["windows_per_clip"], 3, cfgr["size"], cfgr["size"])
    npy = os.path.join(out, "clips_uint8.npy")
    arr = np.lib.format.open_memmap(npy, mode="w+", dtype=np.uint8, shape=shape)
    groups = df.groupby(["source_file", "participant_id", "trial"], sort=False).indices
    t1 = time.time()
    for r in kept.itertuples():
        rec = df.iloc[groups[(r.source_file, r.participant_id, r.trial)]]
        arr[r.row_in_npy] = d.render_clip(rec, r.start_ms, cfg)
    arr.flush()
    nonblank = int((arr.reshape(len(kept) * shape[1], -1).max(axis=1) > 0).sum())
    log.info(f"C4: rendered {shape} in {time.time() - t1:.1f} s -> {npy}; non-black windows {nonblank} of "
             f"{len(kept) * shape[1]}")
    del arr

    # C5 quality per child (shortcut control DD-12; NOT model inputs)
    q = kept.groupby("participant_id").agg(n_clips=("clip_id", "size"), mean_valid_ratio=("valid_ratio", "mean"),
                                           mean_blink_ratio=("blink_ratio", "mean")).reset_index()
    q = parts[["participant_id", "label", "tracking_ratio_mean", "valid_fraction_all_rows"]].merge(
        q, on="participant_id", how="left")
    q = q.rename(columns={"tracking_ratio_mean": "tracking_ratio_before_filtering",
                          "valid_fraction_all_rows": "valid_fraction_before_filtering"})
    q = q[q.participant_id.isin(inc.participant_id)]
    q = q[["participant_id", "label", "n_clips", "mean_valid_ratio", "mean_blink_ratio",
           "tracking_ratio_before_filtering", "valid_fraction_before_filtering"]]
    write_csv(q, os.path.join(out, "quality_per_child.csv"))
    means = q.drop(columns="participant_id").groupby("label").mean().T.rename(columns={1: "ASD_mean", 0: "TD_mean"})
    means = means[["ASD_mean", "TD_mean"]].round(4)
    log.info("C5 quality features, ASD vs TD means over children:\n" + md_table(means.reset_index()))

    summary = {
        "n_raw_rows": n_raw, "cleaning_steps": steps.to_dict("records"), "resolution_evidence": ev,
        "origin_artifact_evidence": corner, "trial_overlap": overlap,
        "n_recordings": len(recs), "n_segments": int(recs.n_segments.sum()), "n_candidate_clips": len(clips),
        "clip_status_counts": clips.status.value_counts().to_dict(), "npy_shape": list(shape),
        "children_included_per_class": inc.groupby("class").size().to_dict(),
        "children_excluded": parts[parts.status != "included"][["participant_id", "status"]].to_dict("records"),
        "clips_per_child_capped": {"min": int(s.min()), "median": float(s.median()), "max": int(s.max())},
        "clips_per_child_uncapped": {"min": int(inc.n_clips_uncapped.min()),
                                     "median": float(inc.n_clips_uncapped.median()),
                                     "max": int(inc.n_clips_uncapped.max())},
        "clips_per_label_capped": kept.groupby("label").size().to_dict(),
        "quality_means": means.to_dict(), "nonblank_windows": nonblank, "seconds": round(time.time() - t0, 1),
    }
    write_json(summary, os.path.join(out, "d01_summary.json"))
    log.info(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
