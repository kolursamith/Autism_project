"""Stage 1 - feasibility check for DD-04 only (no split manifest is saved; that is Stage 2).
Can StratifiedGroupKFold(10, shuffle=True, random_state=42) over D01 participants with gaze give every
test fold >=1 ASD and >=1 TD? Writes data_registry/D01_cv_feasibility.json."""
import json, os, sys
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

REG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_registry")
e = json.load(open(os.path.join(REG, "D01_eye_tracking_audit.json")))["eye_tracking"]
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.paths import DATASETS_DIR, dataset_dir  # noqa: E402  (configs/paths.yaml)
meta = pd.read_csv(os.path.join(dataset_dir("D01_eye_tracking"), "Metadata_Participants.csv"))
excluded = set(e["participants_metadata_but_no_gaze"])
meta = meta[~meta["ParticipantID"].isin(excluded)].reset_index(drop=True)
y = (meta["Class"] == "ASD").astype(int).to_numpy()
g = meta["ParticipantID"].to_numpy()
folds = []
for k, (_, te) in enumerate(StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42).split(np.zeros(len(y)), y, g), 1):
    folds.append({"fold": k, "n_test": int(len(te)), "asd": int(y[te].sum()), "td": int((1 - y[te]).sum())})
res = {"n_participants": int(len(y)), "asd": int(y.sum()), "td": int((1 - y).sum()), "excluded_no_gaze": sorted(excluded),
       "folds": folds, "every_fold_has_asd_and_td": all(f["asd"] >= 1 and f["td"] >= 1 for f in folds)}
json.dump(res, open(os.path.join(REG, "D01_cv_feasibility.json"), "w"), indent=1)
print(json.dumps(res))
