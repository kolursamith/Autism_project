"""D01 eye-tracking: load 25 CSVs into one canonical schema, clean (DD-01..DD-03), window into clips, render.

Canonical row schema: source_file, row_in_file, participant_id, trial, stimulus, t_ms, event, eye, x_px, y_px,
pupil_mm, tracking_ratio (+ label after the metadata join, + valid after invalid-gaze marking).
CARS Score is never read (DD-03): the metadata file is read with usecols=[id, class, age, gender]."""
import os

import cv2
import numpy as np
import pandas as pd

from src.data.common import dataset_dir

GAZE_SUBDIR = "Eye-tracking Output"
META_FILE = "Metadata_Participants.csv"

# canonical field -> list of source columns in order of preference (first present one is used)
SOURCE_COLUMNS = {
    "participant_id": ["Participant"],
    "trial": ["Trial"],
    "stimulus": ["Stimulus"],
    "t_ms": ["RecordingTime [ms]"],
    "event_right": ["Category Right"],
    "event_left": ["Category Left"],
    "x_right": ["Point of Regard Right X [px]"],
    "y_right": ["Point of Regard Right Y [px]"],
    "x_left": ["Point of Regard Left X [px]"],
    "y_left": ["Point of Regard Left Y [px]"],
    "pupil_right": ["Pupil Diameter Right [mm]"],
    "pupil_left": ["Pupil Diameter Left [mm]"],
    "tracking_ratio": ["Tracking Ratio [%]"],
}
REQUIRED = ["participant_id", "trial", "stimulus", "t_ms", "event_right", "x_right", "y_right", "tracking_ratio"]
EVENTS = {"Fixation": "Fixation", "Saccade": "Saccade", "Blink": "Blink"}  # everything else -> "other"


def gaze_dir():
    return os.path.join(dataset_dir("D01_eye_tracking"), GAZE_SUBDIR)


def gaze_files():
    return sorted(os.listdir(gaze_dir()), key=lambda s: int(os.path.splitext(s)[0]))


# ---------------------------------------------------------------- C1 load
def column_mapping():
    """One row per (file, canonical field): which source column feeds it ('' = absent)."""
    rows = []
    for f in gaze_files():
        header = pd.read_csv(os.path.join(gaze_dir(), f), nrows=0).columns.tolist()
        for field, cands in SOURCE_COLUMNS.items():
            src = next((c for c in cands if c in header), "")
            rows.append({"source_file": f, "n_source_columns": len(header), "canonical": field, "source_column": src})
    return pd.DataFrame(rows)


def _num(s):
    return pd.to_numeric(s, errors="coerce")  # '-' and blanks -> NaN


def load_file(fname, mapping):
    """Read one gaze CSV and return it in the canonical schema (before any cleaning)."""
    m = mapping[(mapping.source_file == fname) & (mapping.source_column != "")]
    src_to_field = dict(zip(m.source_column, m.canonical))
    missing = [r for r in REQUIRED if r not in src_to_field.values()]
    if missing:
        raise ValueError(f"{fname}: required fields missing {missing}")
    raw = pd.read_csv(os.path.join(gaze_dir(), fname), usecols=list(src_to_field), dtype=str)
    raw = raw.rename(columns=src_to_field)
    n = len(raw)
    out = pd.DataFrame({"source_file": fname, "row_in_file": np.arange(n, dtype=np.int64)})
    out["participant_raw"] = raw["participant_id"].fillna("").str.strip()
    out["trial"] = raw["trial"].fillna("").str.strip()
    out["stimulus"] = raw["stimulus"].fillna("").str.strip()
    out["t_ms"] = _num(raw["t_ms"]).round(3)
    xr, yr = _num(raw["x_right"]).round(4), _num(raw["y_right"]).round(4)
    xl = _num(raw["x_left"]).round(4) if "x_left" in raw else pd.Series(np.nan, index=raw.index)
    yl = _num(raw["y_left"]).round(4) if "y_left" in raw else pd.Series(np.nan, index=raw.index)
    # right-eye coordinates are kept separately: they form the DD-02 dedupe key (present in all 9 layouts)
    out["x_right"], out["y_right"] = xr, yr
    right_ok = xr.notna() & yr.notna() & ~((xr == 0) & (yr == 0))
    left_ok = xl.notna() & yl.notna() & ~((xl == 0) & (yl == 0))
    use_left = ~right_ok & left_ok
    out["eye"] = np.where(right_ok, "R", np.where(use_left, "L", "none"))
    out["x_px"] = np.where(use_left, xl, xr)
    out["y_px"] = np.where(use_left, yl, yr)
    ev = raw["event_right"].fillna("").str.strip()
    if "event_left" in raw:
        evl = raw["event_left"].fillna("").str.strip()
        ev = ev.where(~use_left.to_numpy(), evl)
    out["event"] = ev.map(EVENTS).fillna("other")
    pr = _num(raw["pupil_right"]) if "pupil_right" in raw else pd.Series(np.nan, index=raw.index)
    pl = _num(raw["pupil_left"]) if "pupil_left" in raw else pd.Series(np.nan, index=raw.index)
    out["pupil_mm"] = pd.concat([pr, pl], axis=1).mean(axis=1, skipna=True)
    out["tracking_ratio"] = _num(raw["tracking_ratio"])
    return out


def load_all():
    mapping = column_mapping()
    frames = [load_file(f, mapping) for f in gaze_files()]
    return pd.concat(frames, ignore_index=True), mapping


def read_metadata():
    """Participant table WITHOUT the CARS column (DD-03)."""
    meta = pd.read_csv(os.path.join(dataset_dir("D01_eye_tracking"), META_FILE),
                       usecols=["ParticipantID", "Gender", "Age", "Class"])
    meta = meta.rename(columns={"ParticipantID": "participant_id", "Gender": "gender", "Age": "age", "Class": "class"})
    meta["label"] = (meta["class"] == "ASD").astype(int)
    return meta


# ---------------------------------------------------------------- C2 clean
def clean(df, cfg, log=print):
    """Steps a-d. Returns (clean_df, steps_table, dedupe_pairs_table, excluded_participants_table)."""
    steps = []

    def step(name, before, after):
        steps.append({"step": name, "rows_before": before, "rows_after": after, "removed": before - after})

    # a. unidentified / missing participant
    n0 = len(df)
    pid = pd.to_numeric(df["participant_raw"], errors="coerce")
    keep = pid.notna()
    df = df[keep].copy()
    df["participant_id"] = pid[keep].astype(int)
    df = df.drop(columns="participant_raw")
    step("a. drop 'Unidentified' / missing participant", n0, len(df))

    # b. cross-file duplicates (DD-02): numeric key, keep first occurrence in file order 1..25, row order
    n1 = len(df)
    df["_file_no"] = df["source_file"].str.replace(".csv", "", regex=False).astype(int)
    df = df.sort_values(["_file_no", "row_in_file"], kind="mergesort")
    key = ["participant_id", "t_ms", "trial", "x_right", "y_right"]
    dup = df.duplicated(key, keep="first")
    first_file = df.groupby(key, dropna=False, sort=False)["source_file"].transform("first")
    pairs = (pd.DataFrame({"kept_in": first_file[dup], "removed_from": df.loc[dup, "source_file"]})
             .value_counts().rename("rows_removed").reset_index()
             .sort_values(["rows_removed", "kept_in", "removed_from"], ascending=[False, True, True]))
    df = df[~dup].copy()
    step("b. remove duplicate rows (DD-02 key, keep first)", n1, len(df))

    # b2. same (participant, t_ms, trial) exported again at a coarser coordinate precision (e.g. 0.1 px vs 0.0001 px):
    # a later row is a duplicate if its right-eye gaze differs from the first row by <= tol in x and y.
    n1b = len(df)
    tol = cfg["dedupe"]["coordinate_tolerance_px"] + 1e-6  # + float epsilon (0.05 is stored as 0.05000000000001)
    k3 = ["participant_id", "t_ms", "trial"]
    later = df.duplicated(k3, keep="first")
    firsts = df[~later].set_index(k3)[["x_right", "y_right", "source_file"]]
    cand = df[later].join(firsts, on=k3, rsuffix="_first")
    close = ((cand["x_right"] - cand["x_right_first"]).abs() <= tol) & ((cand["y_right"] - cand["y_right_first"]).abs() <= tol)
    conflicts = cand[~close]
    if len(conflicts):
        raise ValueError(f"{len(conflicts)} rows share (participant, t_ms, trial) with different gaze: inspect before continuing")
    pairs2 = (pd.DataFrame({"kept_in": cand["source_file_first"], "removed_from": cand["source_file"]})
              .value_counts().rename("rows_removed").reset_index())
    pairs2["step"] = "b2"
    pairs["step"] = "b"
    pairs = pd.concat([pairs, pairs2], ignore_index=True)
    df = df.drop(index=cand.index)
    step(f"b2. remove same (participant, t_ms, trial) with gaze within {cfg['dedupe']['coordinate_tolerance_px']} px "
         "(export precision)", n1b, len(df))

    # c. participants in the metadata
    meta = read_metadata()
    n2 = len(df)
    in_meta = df["participant_id"].isin(meta["participant_id"])
    gaze_ids = set(df["participant_id"].unique())
    excl = [{"participant_id": int(p), "reason": "gaze rows but not in Metadata_Participants.csv"}
            for p in sorted(gaze_ids - set(meta["participant_id"]))]
    excl += [{"participant_id": int(p), "reason": "in metadata but no identified gaze rows"}
             for p in sorted(set(meta["participant_id"]) - gaze_ids)]
    df = df[in_meta].copy()
    step("c. keep participants present in metadata", n2, len(df))

    # d. label join (ASD=1, TD=0); CARS is never read
    n3 = len(df)
    df = df.merge(meta[["participant_id", "label"]], on="participant_id", how="inner", validate="many_to_one")
    df = df.sort_values(["_file_no", "row_in_file"], kind="mergesort").drop(columns="_file_no").reset_index(drop=True)
    step("d. join label (ASD=1, TD=0); only id + class read from metadata (DD-03)", n3, len(df))
    return df, pd.DataFrame(steps), pairs, pd.DataFrame(excl, columns=["participant_id", "reason"])


def estimate_resolution(df, cfg):
    """99.9th percentile of x and y over non-missing, non-(0,0) gaze, and the nearest standard resolution."""
    ok = df["x_px"].notna() & df["y_px"].notna() & ~((df["x_px"] == 0) & (df["y_px"] == 0))
    q = cfg["screen"]["percentile"]
    px, py = float(np.percentile(df.loc[ok, "x_px"], q)), float(np.percentile(df.loc[ok, "y_px"], q))
    std = [tuple(r) for r in cfg["screen"]["standard_resolutions"]]
    best = min(std, key=lambda wh: abs(wh[0] - px) / wh[0] + abs(wh[1] - py) / wh[1])
    return {"p_x": round(px, 1), "p_y": round(py, 1), "n_points": int(ok.sum()), "nearest_standard": list(best)}


def mark_invalid(df, cfg):
    """e. valid = gaze present, not a (0,0) artefact (within origin_artifact_radius_px of (0,0)), and within the
    screen +- margin. Rows are kept so the time base stays intact."""
    W, H = cfg["screen"]["width"], cfg["screen"]["height"]
    m, r0 = cfg["invalid_gaze"]["margin_px"], cfg["invalid_gaze"]["origin_artifact_radius_px"]
    x, y = df["x_px"], df["y_px"]
    df["valid"] = (x.notna() & y.notna() & (np.hypot(x, y) > r0)
                   & (x >= -m) & (x <= W + m) & (y >= -m) & (y <= H + m))
    return df


def build_clean_gaze(cfg, log=print):
    """Full C1 + C2 pipeline from the raw CSVs (used by 02a and by the re-render test)."""
    raw, mapping = load_all()
    df, steps, pairs, excl = clean(raw, cfg, log)
    df = mark_invalid(df, cfg)
    return df, mapping, steps, pairs, excl, len(raw)


def resolution_evidence(df, cfg):
    """Evidence for the screen resolution: high percentile -> nearest standard, and the tracker clamp limits."""
    ok = df["x_px"].notna() & df["y_px"].notna() & ~((df["x_px"] == 0) & (df["y_px"] == 0))
    x, y = df.loc[ok, "x_px"], df.loc[ok, "y_px"]
    ev = estimate_resolution(df, cfg)
    ev.update({"x_min": round(float(x.min()), 1), "x_max": round(float(x.max()), 1),
               "y_min": round(float(y.min()), 1), "y_max": round(float(y.max()), 1),
               "x_range_centre": round(float(x.min() + x.max()) / 2, 1),
               "y_range_centre": round(float(y.min() + y.max()) / 2, 1),
               "x_median": round(float(x.median()), 1), "y_median": round(float(y.median()), 1)})
    return ev


# ---------------------------------------------------------------- C3 clips
def recordings(df):
    """Rows grouped by (source_file, participant_id, trial), each sorted by t_ms (stable)."""
    df = df.sort_values(["source_file", "participant_id", "trial", "t_ms"], kind="mergesort").reset_index(drop=True)
    return df


def clip_rows(t, start, end):
    """Boolean mask of samples belonging to the clip [start, end)."""
    return (t >= start) & (t < end)


def make_clips(df, cfg):
    """Candidate clips for every continuous segment of every recording. Returns (clips_df, recordings_df).
    A segment ends where two consecutive samples are more than gap_max_ms apart, so no clip contains such a gap."""
    c = cfg["clips"]
    win_ms, n_win, gap = c["window_ms"], c["windows_per_clip"], c["gap_max_ms"]
    clip_ms = win_ms * n_win
    rows, recs = [], []
    # all samples of each (file, child), time-sorted, to reject clips whose span contains another trial's samples
    span = {}
    for key, r in df.groupby(["source_file", "participant_id"], sort=False):
        o = np.argsort(r["t_ms"].to_numpy(), kind="mergesort")
        span[key] = (r["t_ms"].to_numpy()[o], r["trial"].to_numpy()[o])
    for (f, pid, trial), r in df.groupby(["source_file", "participant_id", "trial"], sort=True):
        t = r["t_ms"].to_numpy()
        if np.any(np.diff(t) <= 0):
            raise ValueError(f"non-increasing t_ms in {f} p{pid} {trial}")
        breaks = np.flatnonzero(np.diff(t) > gap) + 1
        seg_starts = np.r_[0, breaks]
        seg_ends = np.r_[breaks, len(t)]
        recs.append({"source_file": f, "participant_id": pid, "trial": trial, "n_rows": len(t),
                     "t_min": t[0], "t_max": t[-1], "n_segments": len(seg_starts)})
        valid = r["valid"].to_numpy()
        ev = r["event"].to_numpy()
        stim = r["stimulus"].to_numpy()
        label = int(r["label"].iloc[0])
        fno = int(f.replace(".csv", ""))
        for si, (a, b) in enumerate(zip(seg_starts, seg_ends)):
            s0, sl = t[a], t[b - 1]
            k = 0
            ts, trs = span[(f, pid)]
            while round(s0 + clip_ms * (k + 1), 3) <= sl:  # clip ends at or before the segment's last sample
                start = round(s0 + clip_ms * k, 3)
                end = round(start + clip_ms, 3)
                m = clip_rows(t, start, end)
                tt = t[m]
                gaps = np.diff(np.r_[start, tt, end]) if len(tt) else np.array([clip_ms])
                lo, hi = np.searchsorted(ts, start, "left"), np.searchsorted(ts, end, "left")
                other_trial = bool((trs[lo:hi] != trial).any())
                st = pd.Series(stim[m]).value_counts()
                rows.append({
                    "clip_id": f"p{pid:02d}_f{fno:02d}_{trial}_s{si:03d}_c{k:03d}",
                    "participant_id": pid, "label": label, "source_file": f, "trial": trial,
                    "stimulus": st.index[0] if len(st) else "", "segment": si, "start_ms": start, "end_ms": end,
                    "n_samples": int(m.sum()), "max_gap_ms": round(float(gaps.max()), 3),
                    "other_trial_in_span": other_trial,
                    "valid_ratio": float(valid[m].mean()) if m.any() else 0.0,
                    "blink_ratio": float((ev[m] == "Blink").mean()) if m.any() else 0.0,
                    "fixation_ratio": float((ev[m] == "Fixation").mean()) if m.any() else 0.0,
                    "saccade_ratio": float((ev[m] == "Saccade").mean()) if m.any() else 0.0,
                })
                k += 1
    clips = pd.DataFrame(rows)
    clips["status"] = np.select(
        [clips["other_trial_in_span"], clips["max_gap_ms"] > gap, clips["valid_ratio"] < c["min_valid_ratio"]],
        ["rejected_other_trial_in_span", "rejected_gap", "rejected_valid_ratio"], default="candidate")
    return clips, pd.DataFrame(recs)


def apply_cap(clips, cfg):
    """Random cap of clips per child among the candidates (numpy default_rng(seed), children in id order)."""
    cap, rng = cfg["clips"]["max_clips_per_child"], np.random.default_rng(cfg["seed"])
    clips = clips.copy()
    for pid in sorted(clips["participant_id"].unique()):
        idx = clips.index[(clips["participant_id"] == pid) & (clips["status"] == "candidate")].to_numpy()
        if len(idx) > cap:
            drop = np.setdiff1d(idx, rng.choice(idx, size=cap, replace=False))
            clips.loc[drop, "status"] = "dropped_by_cap"
    clips.loc[clips["status"] == "candidate", "status"] = "kept"
    return clips


# ---------------------------------------------------------------- C4 render
def to_pixels(x, y, cfg):
    s = cfg["render"]["size"]
    px = np.clip(np.floor(x / cfg["screen"]["width"] * s), 0, s - 1).astype(np.int32)
    py = np.clip(np.floor(y / cfg["screen"]["height"] * s), 0, s - 1).astype(np.int32)
    return px, py


def render_window(x, y, valid, is_fix, t, wstart, cfg):
    """One 1-s window -> (3, S, S) uint8: scan path, fixation density, time encoding. Invalid samples are not drawn."""
    rc = cfg["render"]
    S, win = rc["size"], cfg["clips"]["window_ms"]
    img = np.zeros((3, S, S), np.uint8)
    v = valid.astype(bool)
    if not v.any():
        return img
    px, py = to_pixels(x[v], y[v], cfg)
    # ch0 scan path: lines joining consecutive valid points
    ch0 = np.zeros((S, S), np.uint8)
    pts = np.stack([px, py], 1).reshape(-1, 1, 2)
    if len(pts) >= 2:
        cv2.polylines(ch0, [pts], isClosed=False, color=255, thickness=rc["line_thickness"], lineType=cv2.LINE_8)
    else:
        ch0[py, px] = 255
    img[0] = ch0
    # ch1 fixation density
    f = is_fix[v].astype(bool)
    if f.any():
        h = np.zeros((S, S), np.float32)
        np.add.at(h, (py[f], px[f]), 1.0)
        h = cv2.GaussianBlur(h, (0, 0), sigmaX=rc["fixation_sigma_px"], sigmaY=rc["fixation_sigma_px"],
                             borderType=cv2.BORDER_CONSTANT)
        if h.max() > 0:
            img[1] = np.round(h / h.max() * 255).astype(np.uint8)
    # ch2 time encoding: brightness rises linearly over the second
    lo, hi = rc["time_brightness_min"], rc["time_brightness_max"]
    b = np.clip(np.round(lo + (hi - lo) * (t[v] - wstart) / win), lo, hi).astype(np.uint8)
    ch2 = np.zeros((S, S), np.uint8)
    np.maximum.at(ch2, (py, px), b)
    img[2] = ch2
    return img


def render_clip(rec, start, cfg):
    """rec = rows of ONE (source_file, participant_id, trial) recording (columns t_ms, x_px, y_px, valid, event).
    Returns (n_windows, 3, S, S) uint8 for the clip starting at `start`."""
    c = cfg["clips"]
    win, n_win = c["window_ms"], c["windows_per_clip"]
    t = rec["t_ms"].to_numpy()
    out = np.zeros((n_win, 3, cfg["render"]["size"], cfg["render"]["size"]), np.uint8)
    for w in range(n_win):
        ws = round(start + w * win, 3)
        m = clip_rows(t, ws, round(ws + win, 3))
        r = rec[m]
        out[w] = render_window(r["x_px"].to_numpy(), r["y_px"].to_numpy(), r["valid"].to_numpy(),
                               (r["event"] == "Fixation").to_numpy(), r["t_ms"].to_numpy(), ws, cfg)
    return out


def corner_evidence(df, cfg, radius=30):
    """Evidence for the origin-artefact rule: samples valid WITHOUT that rule, near each screen corner."""
    W, H, m = cfg["screen"]["width"], cfg["screen"]["height"], cfg["invalid_gaze"]["margin_px"]
    x, y = df["x_px"], df["y_px"]
    base = x.notna() & y.notna() & ~((x == 0) & (y == 0)) & (x >= -m) & (x <= W + m) & (y >= -m) & (y <= H + m)
    out = {}
    for name, (cx, cy) in {"top_left": (0, 0), "top_right": (W, 0), "bottom_left": (0, H), "bottom_right": (W, H)}.items():
        out[name] = int((base & ((x - cx).abs() < radius) & ((y - cy).abs() < radius)).sum())
    one_zero = base & ((x == 0) ^ (y == 0))
    other = np.where(x[one_zero] == 0, y[one_zero], x[one_zero])
    return {"radius_px": radius, "valid_without_origin_rule_near_corner": out,
            "n_exactly_one_coordinate_zero": int(one_zero.sum()),
            "other_coordinate_range_when_one_is_zero": [round(float(other.min()), 1), round(float(other.max()), 1)]}


def overlapping_trial_pairs(df):
    """Number of (file, child) trial pairs whose [first, last] sample-time intervals overlap, and the files involved."""
    rec = df.groupby(["source_file", "participant_id", "trial"]).t_ms.agg(["min", "max"]).reset_index()
    n, files = 0, set()
    for (f, _), x in rec.groupby(["source_file", "participant_id"]):
        x = x.sort_values("min").to_numpy()
        for i in range(len(x)):
            k = int(np.searchsorted(x[:, 3].astype(float), x[i][4], "left")) - (i + 1)
            if k > 0:
                n += k
                files.add(f)
    return {"overlapping_trial_pairs": n, "files": sorted(files, key=lambda s: int(s[:-4]))}
