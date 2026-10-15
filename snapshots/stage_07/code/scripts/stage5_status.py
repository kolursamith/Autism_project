"""Stage 5 progress (D01, M01 -> M10 x 10 folds): queue process alive?, folds DONE / RUNNING / FAILED / PENDING per
model, current model / fold / epoch, mean test ROC-AUC of DONE folds (clip and child level), elapsed time, estimated
time remaining, last 5 queue-log lines, GPU memory / utilisation. Read-only.
Remaining time: per model = (10 - DONE folds) x that model's mean DONE-fold seconds; models without a DONE fold use
the Stage 4 estimate (reports/stage04_time_estimate.csv, ~20 epochs) x the measured/estimated ratio of the folds
finished so far. Usage: .venv/bin/python scripts/stage5_status.py"""
import datetime
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.data.common import LOG_DIR, PROJECT_DIR  # noqa: E402

DS = "D01_eye_tracking"
OUT = os.path.join(PROJECT_DIR, "results")
MODELS = [f"M{i:02d}" for i in range(1, 11)]


def fmt(sec):
    sec = int(max(0, sec))
    return f"{sec // 3600}h {sec % 3600 // 60:02d}m {sec % 60:02d}s"


def pid_status():
    p = os.path.join(LOG_DIR, "stage5.pid")
    if not os.path.isfile(p):
        return None, "no logs/stage5.pid (queue not launched, or stopped cleanly)"
    pid = int(open(p).read().strip())
    try:
        cmd = open(f"/proc/{pid}/cmdline").read().replace("\0", " ")
    except FileNotFoundError:
        return pid, f"PID {pid}: NOT running (stale pid file - crashed, finished or machine rebooted)"
    if "run_stage5.sh" not in cmd:
        return pid, f"PID {pid}: alive but it is not run_stage5.sh ({cmd.strip()[:60]}) - stale pid file"
    return pid, f"PID {pid}: RUNNING (run_stage5.sh)"


def gpu():
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
                       capture_output=True, text=True)
    return r.stdout.strip() or "UNKNOWN"


def main():
    now = datetime.datetime.now()
    print(f"STAGE 5 STATUS  {now.isoformat(timespec='seconds')}  ({DS})")
    print("queue process :", pid_status()[1])
    reg_p = os.path.join(OUT, "run_registry.csv")
    reg = pd.read_csv(reg_p, keep_default_na=False) if os.path.isfile(reg_p) else pd.DataFrame(
        columns=["dataset", "model", "fold", "status", "start", "seconds"])
    reg = reg[reg.dataset == DS]

    est = pd.read_csv(os.path.join(PROJECT_DIR, "reports", "stage04_time_estimate.csv"))
    est = est[est.dataset == DS].set_index("model_id")["est_fold_min_E20"] * 60
    rows, done_sec, done_est = [], [], []
    for m in MODELS:
        r = reg[reg.model == m]
        n = {s: int((r.status == s).sum()) for s in ["DONE", "RUNNING", "FAILED"]}
        n["PENDING"] = 10 - sum(n.values())
        clip, child = [], []
        d = r[r.status == "DONE"]
        for k in d.fold.astype(int):
            mp = os.path.join(OUT, DS, m, f"fold_{k:02d}", "metrics.json")
            if os.path.isfile(mp):
                mt = json.load(open(mp))
                clip.append(mt["test"]["threshold_0.5"]["roc_auc"])
                child.append(mt.get("test_child", {}).get("threshold_0.5", {}).get("roc_auc", np.nan))
        secs = pd.to_numeric(d.seconds, errors="coerce").dropna()
        done_sec += list(secs)
        done_est += [est[m]] * len(secs)
        rows.append({"model": m, **n, "mean_s_per_fold": secs.mean() if len(secs) else np.nan,
                     "test_auc_clip": np.nanmean(clip) if clip else np.nan,
                     "test_auc_child": np.nanmean(child) if child and not np.all(np.isnan(child)) else np.nan,
                     "n_auc": len(clip)})
    t = pd.DataFrame(rows)
    show = t.copy()
    for c in ["test_auc_clip", "test_auc_child"]:
        show[c] = show[c].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
    show["mean_s_per_fold"] = show["mean_s_per_fold"].map(lambda v: "" if pd.isna(v) else f"{v:.0f}")
    print("\nper model (folds; mean test ROC-AUC over DONE folds, threshold-free; child = mean clip prob per child):")
    print(show.to_string(index=False))

    # current fold
    cur = reg[reg.status == "RUNNING"]
    print()
    if cur.empty:
        print("current       : no RUNNING fold")
    for _, c in cur.iterrows():
        d = os.path.join(OUT, DS, c.model, f"fold_{int(c.fold):02d}")
        h = pd.read_csv(os.path.join(d, "history.csv")) if os.path.isfile(os.path.join(d, "history.csv")) else None
        mx = yaml.safe_load(open(os.path.join(d, "config.yaml")))["training"]["max_epochs"] \
            if os.path.isfile(os.path.join(d, "config.yaml")) else "?"
        ep = 0 if h is None else len(h)
        extra = "" if h is None or h.empty else (f" | last epoch {h.seconds.iloc[-1]:.1f}s, val_auc {h.val_auc.iloc[-1]:.4f},"
                                                 f" best val_auc {h.val_auc.max():.4f} @ {int(h.val_auc.idxmax()) + 1}")
        print(f"current       : {c.model} fold {int(c.fold)} - {ep}/{mx} epochs completed (started {c.start}){extra}")

    # time
    starts = []
    hist = os.path.join(LOG_DIR, "stage5_launch_history.jsonl")
    if os.path.isfile(hist):
        starts = [json.loads(l)["start_time"] for l in open(hist) if l.strip()]
    if starts:
        first = datetime.datetime.fromisoformat(starts[0]).replace(tzinfo=None)
        print(f"elapsed       : {fmt((now - first).total_seconds())} since the first launch ({starts[0]}; "
              f"{len(starts)} launch(es)); compute in DONE folds {fmt(sum(done_sec))}")
    ratio = (sum(done_sec) / sum(done_est)) if done_est else 1.0
    remaining = 0.0
    for r in t.itertuples():
        left = 10 - r.DONE
        per = r.mean_s_per_fold if not pd.isna(r.mean_s_per_fold) else est[r.model] * ratio
        remaining += left * per
    print(f"remaining     : ~{fmt(remaining)} (estimate; {int(t.DONE.sum())}/100 folds DONE; measured/Stage-4-estimate "
          f"ratio so far {ratio:.2f})")
    if t.FAILED.sum():
        print(f"FAILED folds  : {int(t.FAILED.sum())} - the queue stops at a failed model; see logs/stage5_queue.log")

    # log + gpu
    qlog = os.path.join(LOG_DIR, "stage5_queue.log")
    print("\nlast 5 queue log lines:")
    if os.path.isfile(qlog):
        lines = [l.rstrip("\n") for l in open(qlog, errors="replace") if l.strip()]
        for l in lines[-5:]:
            print("  " + l[:220])
    else:
        print("  (no logs/stage5_queue.log yet)")
    print(f"\nGPU (memory used, total, utilisation): {gpu()}")


if __name__ == "__main__":
    main()
