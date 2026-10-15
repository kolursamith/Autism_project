"""Stage 1 - read-only check of the DD-02 dedupe key for D01 (string vs numeric parsing).
Writes data_registry/D01_dedupe_key_check.json."""
import json, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.paths import DATASETS_DIR, dataset_dir  # noqa: E402  (configs/paths.yaml)
GDIR = os.path.join(dataset_dir("D01_eye_tracking"), "Eye-tracking Output")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_registry", "D01_dedupe_key_check.json")
COLS = ["Participant", "RecordingTime [ms]", "Trial", "Point of Regard Right X [px]", "Point of Regard Right Y [px]"]
frames = []
for f in sorted(os.listdir(GDIR), key=lambda s: int(s[:-4])):
    d = pd.read_csv(os.path.join(GDIR, f), usecols=COLS, dtype=str)
    d["file"] = f
    frames.append(d)
df = pd.concat(frames, ignore_index=True)
ident = ~df["Participant"].str.startswith("Unidentified")
num = df.copy()
num["RecordingTime [ms]"] = pd.to_numeric(num["RecordingTime [ms]"], errors="coerce").round(3)
for c in COLS[3:]:
    num[c] = pd.to_numeric(num[c], errors="coerce").round(4)  # '-' -> NaN
def stats(d, mask):
    k = pd.util.hash_pandas_object(d.loc[mask, COLS], index=False)
    dup = k.duplicated()
    within = d.loc[mask].assign(k=k.values).duplicated(["file", "k"]).sum()
    return {"rows": int(mask.sum()), "distinct": int((~dup).sum()), "duplicate_rows_removed": int(dup.sum()),
            "of_which_duplicate_within_same_file": int(within)}
res = {
    "key": COLS,
    "string_key_all_rows": stats(df, pd.Series(True, index=df.index)),
    "string_key_identified_rows": stats(df, ident),
    "numeric_key_all_rows(time round 3dp, coords round 4dp, '-'->NaN)": stats(num, pd.Series(True, index=df.index)),
    "numeric_key_identified_rows": stats(num, ident),
    "example_formats_PoR_X": df["Point of Regard Right X [px]"].drop_duplicates().head(8).tolist(),
    "example_formats_RecordingTime_first_row_per_file": df.groupby("file")["RecordingTime [ms]"].first().head(6).to_dict(),
}
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
