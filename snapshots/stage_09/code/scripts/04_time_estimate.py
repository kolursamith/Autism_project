"""Stage 4 - training-time estimate for S5 (D01) and S6 (D02) from the MEASURED smoke epochs (results_smoke/) and the
Stage 3 GPU profile (reports/stage03_model_profile.csv). Output: reports/stage04_time_estimate.csv + printed totals.

Per input type (cached features / clips / images) the smoke runs give
  overhead_epoch = measured steady epoch seconds - steps/epoch x profiled s/step of the smoked model
                   (data loading, worker start, validation pass, logging, checkpoint)
  overhead_fold  = mean(first epoch - steady epoch) + (run seconds - sum of epoch seconds)
                   (cuDNN autotune, cold page cache, model build, weight hash check, final val + test prediction)
and every model of that type is estimated as
  epoch_s = overhead_epoch(type) + steps/epoch x s/step(model)      fold_s = overhead_fold(type) + E x epoch_s
for E = 20 epochs (assumed average with early stopping) and E = 40 (max_epochs, worst case). x 10 folds x 10 models.
Usage: .venv/bin/python scripts/04_time_estimate.py"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.common import PROJECT_DIR, md_table, write_csv  # noqa: E402
from src.training.engine import merge_config  # noqa: E402

SMOKE = os.path.join(PROJECT_DIR, "results_smoke")
SCENARIOS = {"E20": 20, "E40": 40}


def step_seconds(prof, mid, mode, inp):
    r = prof[(prof.model_id == mid) & (prof["mode"] == mode)].iloc[0]
    return float(r.s_per_train_step_cached if inp == "cached_features" else r.s_per_train_step), int(r.steps_per_epoch)


def main():
    prof = pd.read_csv(os.path.join(PROJECT_DIR, "reports", "stage03_model_profile.csv"))
    meas = []
    for d in sorted(glob.glob(os.path.join(SMOKE, "**", "fold_[0-9][0-9]"), recursive=True)):
        info = json.load(open(os.path.join(d, "run_info.json")))
        if info["resumed"]:
            continue  # the interrupted run has two first epochs; its straight reference is used instead
        h = pd.read_csv(os.path.join(d, "history.csv"))
        cfg = merge_config(info["dataset"], info["model_id"], info["fold"])
        inp, mode = cfg["run"]["input"], cfg["run"]["mode"]
        s_step, steps = step_seconds(prof, info["model_id"], mode, inp)
        steady = float(h.seconds[1:].mean())
        meas.append({"run": os.path.relpath(d, SMOKE), "input": inp, "steady_epoch_s": steady,
                     "first_epoch_extra_s": max(0.0, float(h.seconds.iloc[0]) - steady),
                     "fixed_s": float(info["seconds"] - h.seconds.sum()),
                     "overhead_epoch_s": steady - steps * s_step})
    M = pd.DataFrame(meas)
    ov = M.groupby("input").agg(overhead_epoch_s=("overhead_epoch_s", "mean"),
                                overhead_fold_s=("first_epoch_extra_s", "mean"), fixed_s=("fixed_s", "mean"),
                                n_runs=("run", "count"))
    ov["overhead_fold_s"] += ov.pop("fixed_s")
    rows = []
    for ds, mode_ds in [("D01_eye_tracking", "S5"), ("D02_facial_images_A", "S6")]:
        for i in range(1, 11):
            mid = f"M{i:02d}"
            cfg = merge_config(ds, mid, 1)
            inp = cfg["run"]["input"]
            s_step, steps = step_seconds(prof, mid, cfg["run"]["mode"], inp)
            o = ov.loc[inp]
            epoch_s = o.overhead_epoch_s + steps * s_step
            r = {"stage": mode_ds, "dataset": ds, "model_id": mid, "input": inp, "steps_per_epoch": steps,
                 "s_per_step_profile": s_step, "est_epoch_s": round(epoch_s, 2),
                 "measured_in_smoke": bool((M.run.str.startswith(f"{ds}/{mid}/")
                                            | M.run.str.contains(f"/{ds}/{mid}/")).any())}
            for name, E in SCENARIOS.items():
                r[f"est_fold_min_{name}"] = round((o.overhead_fold_s + E * epoch_s) / 60, 2)
                r[f"est_model_h_{name}"] = round(10 * (o.overhead_fold_s + E * epoch_s) / 3600, 3)
            rows.append(r)
    est = pd.DataFrame(rows)
    write_csv(est, os.path.join(PROJECT_DIR, "reports", "stage04_time_estimate.csv"))
    write_csv(M.round(3), os.path.join(PROJECT_DIR, "reports", "stage04_smoke_timing.csv"))
    write_csv(ov.reset_index().round(3), os.path.join(PROJECT_DIR, "reports", "stage04_overheads.csv"))
    print("measured smoke timing:\n" + md_table(M.round(3)))
    print("\noverheads per input type:\n" + md_table(ov.reset_index().round(3)))
    print("\nper model:\n" + md_table(est))
    for st in ["S5", "S6"]:
        e = est[est.stage == st]
        print(f"{st}: {e.est_model_h_E20.sum():.2f} h (20 epochs avg) | {e.est_model_h_E40.sum():.2f} h (40 epochs)")


if __name__ == "__main__":
    main()
