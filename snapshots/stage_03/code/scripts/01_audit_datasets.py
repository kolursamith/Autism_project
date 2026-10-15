"""Stage 1 - read-only audit of the 6 raw datasets.

Nothing under DATASETS_DIR is written, moved or extracted. Outputs go to data_registry/.
Usage: .venv/bin/python scripts/01_audit_datasets.py
"""
import collections
import datetime
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import zipfile
from xml.etree import ElementTree

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.paths import DATASETS_DIR, dataset_dir  # noqa: E402  (configs/paths.yaml)
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REG = os.path.join(PROJECT_DIR, "data_registry")

# dataset_id -> (folder name on disk, kind); folder names come from configs/paths.yaml
KINDS = {
    "D01_eye_tracking": "eye_tracking",
    "D02_facial_images_A": "images",
    "D03_facial_images_B": "images",
    "D04_autism_prediction": "tabular",
    "D05_toddler_screening": "tabular",
    "D06_early_autism": "tabular",
}
DATASETS = {d: (os.path.basename(dataset_dir(d)), kind) for d, kind in KINDS.items()}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
TAB_EXT = {".csv", ".xlsx", ".xls", ".arff", ".tsv"}
ARCHIVE_EXT = {".zip", ".rar", ".7z", ".tar", ".gz", ".tgz"}
SPLIT_NAMES = {"train", "valid", "val", "validation", "test"}
PHASH_MAX_DIST = 4
POSITIVE = {"yes", "asd", "1", "autistic", "true"}


def jdefault(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def log(*a):
    print(f"[{datetime.datetime.now():%H:%M:%S}]", *a, flush=True)


# --------------------------------------------------------------------------- a) tree + b) type
def tree_summary(root):
    by_ext, size, max_depth, n_files = collections.Counter(), 0, 0, 0
    levels = []
    for dp, dns, fns in os.walk(root):
        dns.sort()
        rel = os.path.relpath(dp, root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        max_depth = max(max_depth, depth + (1 if fns else 0))
        for f in fns:
            ext = os.path.splitext(f)[1].lower() or "<none>"
            by_ext[ext] += 1
            size += os.path.getsize(os.path.join(dp, f))
            n_files += 1
        if depth <= 3:
            levels.append({"dir": rel, "depth": depth, "n_files": len(fns), "n_subdirs": len(dns)})
    return {"n_files": n_files, "files_by_extension": dict(by_ext.most_common()),
            "total_size_bytes": size, "total_size_mb": round(size / 2**20, 2),
            "max_depth": max_depth, "tree_first_3_levels": levels}


def list_archives(root):
    out = []
    for dp, _, fns in os.walk(root):
        for f in fns:
            p = os.path.join(dp, f)
            ext = os.path.splitext(f)[1].lower()
            if ext not in ARCHIVE_EXT:
                continue
            entry = {"path": os.path.relpath(p, root)}
            if ext == ".zip":
                with zipfile.ZipFile(p) as z:
                    entry["contents"] = [i.filename for i in z.infolist()]
            elif ext == ".rar":
                try:
                    r = subprocess.run(["unrar", "lb", p], capture_output=True, text=True)
                    entry["contents"] = r.stdout.splitlines()
                except FileNotFoundError:
                    entry["contents"] = "UNKNOWN (unrar not installed; not extracted)"
            else:
                entry["contents"] = "UNKNOWN (format not listed)"
            out.append(entry)
    return out


def data_type(tree):
    exts = tree["files_by_extension"]
    n_img = sum(v for k, v in exts.items() if k in IMG_EXT)
    n_tab = sum(v for k, v in exts.items() if k in TAB_EXT)
    if n_img and n_img >= n_tab:
        return "images"
    if n_tab:
        return "tabular"
    return "other"


def licence_files(root, did):
    """Copy text of licence/citation/readme/description files into the registry (read-only)."""
    found = []
    for dp, _, fns in os.walk(root):
        for f in sorted(fns):
            low = f.lower()
            ext = os.path.splitext(low)[1]
            if not (re.search(r"licen|citation|cite|readme|description|copying", low)
                    or ext in {".txt", ".md", ".docx", ".pdf"}):
                continue
            p = os.path.join(dp, f)
            if ext == ".docx":
                with zipfile.ZipFile(p) as z:
                    xml = z.read("word/document.xml")
                ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
                paras = ["".join(t.text or "" for t in para.iter(ns + "t"))
                         for para in ElementTree.fromstring(xml).iter(ns + "p")]
                text = "\n".join(paras)
            elif ext in {".txt", ".md", ""}:
                with open(p, encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
            else:
                text = "UNKNOWN (binary format, text not extracted)"
            found.append({"path": os.path.relpath(p, root), "text": text})
    out = os.path.join(REG, f"{did}_licence_citation.txt")
    with open(out, "w") as fh:
        if not found:
            fh.write("No licence, citation or readme file found in the dataset folder.\n")
        for e in found:
            fh.write(f"===== {e['path']} =====\n{e['text']}\n\n")
    return {"files": [e["path"] for e in found], "copied_to": os.path.relpath(out, PROJECT_DIR),
            "licence_file_present": any(re.search(r"licen", e["path"].lower()) for e in found)}


# --------------------------------------------------------------------------- c) images
def norm_class(name):
    return re.sub(r"[^a-z]", "", name.lower())  # Non-Autistic / Non_Autistic -> nonautistic


def collect_images(root):
    recs = []
    for dp, _, fns in os.walk(root):
        for f in sorted(fns):
            if os.path.splitext(f)[1].lower() not in IMG_EXT:
                continue
            rel = os.path.relpath(os.path.join(dp, f), root)
            parts = rel.split(os.sep)
            if len(parts) == 1:
                recs.append({"rel": rel, "split": None, "class_raw": None, "class": None})
                continue
            cls = parts[-2]
            split = parts[-3] if len(parts) >= 3 and parts[-3].lower() in SPLIT_NAMES else "none"
            recs.append({"rel": rel, "split": split, "class_raw": cls, "class": norm_class(cls)})
    return recs


def hash_images(root, recs):
    for r in recs:
        p = os.path.join(root, r["rel"])
        with open(p, "rb") as fh:
            r["md5"] = hashlib.md5(fh.read()).hexdigest()
        try:
            with Image.open(p) as im:
                im.load()
                r["w"], r["h"], r["mode"] = im.width, im.height, im.mode
                r["phash"] = str(imagehash.phash(im.convert("RGB")))
            r["readable"] = True
        except Exception as e:  # noqa: BLE001 - we want to record every failure
            r["readable"], r["error"] = False, repr(e)
            r["w"] = r["h"] = r["mode"] = r["phash"] = None


POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def phash_pairs(hashes, max_dist):
    """All index pairs (i<j) with Hamming distance <= max_dist between 64-bit hex phashes."""
    b = np.array([np.frombuffer(bytes.fromhex(h), dtype=np.uint8) for h in hashes])
    pairs = []
    for i in range(len(b) - 1):
        d = POP8[np.bitwise_xor(b[i + 1:], b[i])].sum(axis=1)
        for j in np.nonzero(d <= max_dist)[0]:
            pairs.append((i, i + 1 + int(j), int(d[j])))
    return pairs


def components(n, pairs):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, j, *_ in pairs:
        parent[find(i)] = find(j)
    groups = collections.defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return [g for g in groups.values() if len(g) > 1]


def describe_pairs(recs, pairs):
    out = {"pairs_total": len(pairs), "within_same_split": 0, "across_splits": 0,
           "same_class": 0, "different_class(label conflict)": 0, "across_split_matrix": {},
           "pairs_not_exact_md5": 0}
    m = collections.Counter()
    for i, j, *_ in pairs:
        a, b = recs[i], recs[j]
        if a["md5"] != b["md5"]:
            out["pairs_not_exact_md5"] += 1
        if a["split"] == b["split"]:
            out["within_same_split"] += 1
        else:
            out["across_splits"] += 1
            m["|".join(sorted([a["split"], b["split"]]))] += 1
        if a["class"] == b["class"]:
            out["same_class"] += 1
        else:
            out["different_class(label conflict)"] += 1
    out["across_split_matrix"] = dict(m)
    return out


def exact_dup_groups(recs):
    by = collections.defaultdict(list)
    for i, r in enumerate(recs):
        by[r["md5"]].append(i)
    groups = [g for g in by.values() if len(g) > 1]
    within = sum(1 for g in groups if len({recs[i]["split"] for i in g}) == 1)
    across = len(groups) - within
    conflict = sum(1 for g in groups if len({recs[i]["class"] for i in g}) > 1)
    split_combo = collections.Counter("|".join(sorted({recs[i]["split"] for i in g})) for g in groups)
    return {
        "n_groups": len(groups),
        "n_images_in_groups": sum(len(g) for g in groups),
        "n_redundant_copies": sum(len(g) - 1 for g in groups),
        "groups_within_one_split": within,
        "groups_across_splits": across,
        "groups_split_combinations": dict(split_combo),
        "groups_with_label_conflict": conflict,
        "examples": [[recs[i]["rel"] for i in g] for g in groups[:15]],
    }


def contact_sheet(root, recs, did, per_class=16, thumb=96):
    rng = random.Random(SEED)
    by_cls = collections.defaultdict(list)
    for r in recs:
        if r["readable"] and r["class"]:
            by_cls[r["class_raw"] if False else r["class"]].append(r)
    classes = sorted(by_cls)
    cols, pad, title_h = 4, 4, 22
    block_w = cols * (thumb + pad) + pad
    rows = (per_class + cols - 1) // cols
    sheet = Image.new("RGB", (block_w * len(classes), title_h + rows * (thumb + pad) + pad), "white")
    draw = ImageDraw.Draw(sheet)
    for ci, c in enumerate(classes):
        picks = rng.sample(by_cls[c], min(per_class, len(by_cls[c])))
        draw.text((ci * block_w + pad, 4), f"class={c} (n={len(by_cls[c])}, 16 random, seed 42)", fill="black")
        for k, r in enumerate(picks):
            with Image.open(os.path.join(root, r["rel"])) as im:
                t = im.convert("RGB")
                t.thumbnail((thumb, thumb))
            x = ci * block_w + pad + (k % cols) * (thumb + pad)
            y = title_h + (k // cols) * (thumb + pad)
            sheet.paste(t, (x, y))
    out = os.path.join(REG, f"{did}_samples.png")
    sheet.save(out)
    return os.path.relpath(out, PROJECT_DIR)


def stats(vals):
    a = np.array(vals)
    return {"min": int(a.min()), "median": float(np.median(a)), "max": int(a.max())}


def audit_images(root, did):
    all_recs = collect_images(root)
    loose = [r["rel"] for r in all_recs if r["class"] is None]
    recs = [r for r in all_recs if r["class"] is not None]
    log(did, f"hashing {len(recs)} images")
    hash_images(root, recs)
    ok = [r for r in recs if r["readable"]]
    splits = sorted({r["split"] for r in recs})
    counts = collections.defaultdict(dict)
    raw_names = collections.defaultdict(set)
    for r in recs:
        counts[r["split"]][r["class"]] = counts[r["split"]].get(r["class"], 0) + 1
        raw_names[r["split"]].add(r["class_raw"])
    class_totals = collections.Counter(r["class"] for r in recs)
    pairs = phash_pairs([r["phash"] for r in ok], PHASH_MAX_DIST)
    near = describe_pairs(ok, pairs)
    near["n_groups(connected components)"] = len(components(len(ok), pairs))
    near["examples"] = [[ok[i]["rel"], ok[j]["rel"], d] for i, j, d in pairs[:15]]
    # subject IDs: filenames are <class>.<n>.jpg or <n>.jpg; no CSV / metadata in folder
    name_patterns = collections.Counter(re.sub(r"\d+", "<n>", os.path.basename(r["rel"])) for r in recs)
    side = [f for f in tree_summary(root)["files_by_extension"] if f in TAB_EXT]
    res = {
        "image_files_used_as_samples": len(recs),
        "image_files_outside_class_folders(not samples)": loose,
        "splits": splits,
        "class_folder_names_per_split(raw)": {s: sorted(v) for s, v in raw_names.items()},
        "class_label_normalisation": "lowercase, non-letters removed (e.g. 'Non-Autistic' and 'Non_Autistic' -> 'nonautistic')",
        "count_per_split_x_class": counts,
        "class_totals": dict(class_totals),
        "width": stats([r["w"] for r in ok]),
        "height": stats([r["h"] for r in ok]),
        "n_distinct_sizes": len({(r["w"], r["h"]) for r in ok}),
        "colour_modes": dict(collections.Counter(r["mode"] for r in ok)),
        "n_unreadable": len(recs) - len(ok),
        "unreadable_files": [[r["rel"], r["error"]] for r in recs if not r["readable"]],
        "exact_duplicates_md5": exact_dup_groups(recs),
        "near_duplicates_phash": {"threshold_hamming_le": PHASH_MAX_DIST, **near},
        "subject_ids": {
            "filename_patterns": dict(name_patterns.most_common(10)),
            "metadata_or_csv_files_in_folder": side,
            "finding": "NO subject/participant ID exists: filenames are only <class>.<index>.jpg or <index>.jpg "
                       "and the folder contains no CSV/metadata. The same child may appear in several images "
                       "and cannot be grouped." if not side else "UNKNOWN - tabular files present, inspect.",
        },
        "contact_sheet": contact_sheet(root, ok, did),
    }
    return res, recs


def cross_image_datasets(recs_a, recs_b, id_a, id_b):
    md5_b = collections.defaultdict(list)
    for r in recs_b:
        md5_b[r["md5"]].append(r)
    matched_a = [r for r in recs_a if r["md5"] in md5_b]
    b_set = {r["md5"] for r in recs_b}
    a_set = {r["md5"] for r in recs_a}
    split_of_match = collections.Counter(r["split"] for r in matched_a)
    label_agree = sum(1 for r in matched_a if any(x["class"] == r["class"] for x in md5_b[r["md5"]]))
    ok_a = [r for r in recs_a if r["readable"]]
    ok_b = [r for r in recs_b if r["readable"]]
    pairs = phash_pairs([r["phash"] for r in ok_a] + [r["phash"] for r in ok_b], PHASH_MAX_DIST)
    na = len(ok_a)
    cross = [(i, j, d) for i, j, d in pairs if (i < na) != (j < na)]
    # nearest B image (min Hamming distance) for every A image
    hb = np.array([np.frombuffer(bytes.fromhex(r["phash"]), dtype=np.uint8) for r in ok_b])
    nearest = []
    for r in ok_a:
        d = POP8[np.bitwise_xor(hb, np.frombuffer(bytes.fromhex(r["phash"]), dtype=np.uint8))].sum(axis=1)
        k = int(d.argmin())
        nearest.append((r, ok_b[k], int(d[k])))
    within = [(r, b, d) for r, b, d in nearest if d <= PHASH_MAX_DIST]
    dist_hist = collections.Counter(min(d, 99) for _, _, d in nearest)
    non_exact = [(r, b, d) for r, b, d in nearest if r["md5"] not in md5_b]
    return {
        "datasets": [id_a, id_b],
        "exact_md5": {
            f"{id_a}_images_with_exact_copy_in_{id_b}": len(matched_a),
            f"{id_a}_images_total": len(recs_a),
            f"{id_b}_unique_md5_total": len(b_set),
            "unique_md5_shared": len(a_set & b_set),
            f"unique_md5_only_in_{id_a}": len(a_set - b_set),
            f"unique_md5_only_in_{id_b}": len(b_set - a_set),
            f"matched_{id_a}_images_by_split": dict(split_of_match),
            "matched_images_with_same_class_label": label_agree,
        },
        "near_phash_cross_dataset_pairs(hamming<=4)": len(cross),
        f"{id_a}_images_with_near_or_exact_match_in_{id_b}(nearest hamming<=4)": len(within),
        f"{id_a}_nearest_match_split_in_{id_b}": dict(collections.Counter(b["split"] for _, b, _ in within)),
        "nearest_match_same_class": sum(1 for r, b, _ in within if r["class"] == b["class"]),
        "nearest_distance_histogram": {str(k): v for k, v in sorted(dist_hist.items())},
        f"{id_a}_non_byte_identical_images": {
            "n": len(non_exact),
            "nearest_distance_histogram": {str(k): v for k, v in sorted(collections.Counter(d for *_, d in non_exact).items())},
            "nearest_match_split": dict(collections.Counter(b["split"] for _, b, d in non_exact if d <= PHASH_MAX_DIST)),
            "same_pixel_size_as_nearest": sum(1 for r, b, _ in non_exact if (r["w"], r["h"]) == (b["w"], b["h"])),
            "examples(a, nearest_b, dist)": [[r["rel"], b["rel"], d] for r, b, d in non_exact[:10]],
        },
    }


# --------------------------------------------------------------------------- d) tabular
ID_RE = re.compile(r"^(id|case_no|child_id|participantid|subject_?id)$", re.I)
LABEL_RE = re.compile(r"class|diagnos", re.I)
ITEM_RE = re.compile(r"^A\d+(_Score)?$")


def read_table(p):
    ext = os.path.splitext(p)[1].lower()
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    return pd.read_csv(p)  # C parser handles \n, \r\n and bare \r line endings


def best_threshold_rule(x, y, name):
    """Best single rule 'x >= t' or 'x <= t' predicting y (1 = ASD). Returns dict with confusion matrix."""
    x = pd.to_numeric(x, errors="coerce")
    m = x.notna() & y.notna()
    x, y = x[m].to_numpy(float), y[m].to_numpy(int)
    best = None
    for t in np.unique(x):
        for op in (">=", "<="):
            pred = (x >= t) if op == ">=" else (x <= t)
            acc = float((pred == y).mean())
            if best is None or acc > best["accuracy"]:
                tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
                fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
                best = {"feature": name, "rule": f"ASD if {name} {op} {t:g}", "accuracy": round(acc, 6),
                        "n": int(len(y)), "confusion_matrix[[TN,FP],[FN,TP]]": [[tn, fp], [fn, tp]]}
    return best


def audit_table(p, root):
    df = read_table(p)
    rel = os.path.relpath(p, root)
    out = {"file": rel, "shape": list(df.shape), "columns": list(df.columns),
           "dtypes": {c: str(t) for c, t in df.dtypes.items()},
           "missing_per_column": {c: int(v) for c, v in df.isna().sum().items()},
           "question_mark_values_per_column": {c: int((df[c].astype(str).str.strip() == "?").sum())
                                               for c in df.columns if (df[c].astype(str).str.strip() == "?").any()},
           "fully_empty_columns": [c for c in df.columns if df[c].isna().all()]}
    id_cols = [c for c in df.columns if ID_RE.match(str(c).strip())]
    out["id_columns"] = id_cols
    out["id_columns_unique"] = {c: bool(df[c].is_unique) for c in id_cols}
    feat = df.drop(columns=id_cols)
    out["duplicate_rows_full"] = int(df.duplicated().sum())
    out["duplicate_rows_excluding_id"] = int(feat.duplicated().sum())
    cats = {}
    for c in df.columns:
        if c in id_cols:
            continue
        if df[c].dtype == object or df[c].nunique(dropna=True) <= 20:
            u = df[c].dropna().unique().tolist()
            cats[c] = {"n_unique": len(u), "values(first 20)": sorted(map(str, u))[:20]}
    out["categorical_unique_values"] = cats
    labels = [c for c in df.columns if LABEL_RE.search(str(c))]
    out["label_columns"] = labels
    if not labels:
        out["label_note"] = "NO label column in this file (unlabelled)."
        return out, df, None
    lab = labels[0]
    out["class_counts"] = {str(k): int(v) for k, v in df[lab].value_counts(dropna=False).items()}
    y = df[lab].astype(str).str.strip().str.lower().map(lambda v: 1 if v in POSITIVE else 0)
    out["positive_class_mapping"] = f"{lab} in {sorted(POSITIVE)} (case-insensitive) -> 1 (ASD)"
    vc = y.value_counts()
    out["class_balance_pos_fraction"] = round(float(vc.get(1, 0) / len(y)), 4)
    # ---- leakage
    items = [c for c in df.columns if ITEM_RE.match(str(c))]
    leak = {"item_columns": items, "rules": []}
    cand = {}
    if items:
        cand["sum(" + ",".join(items[:1]) + ".." + items[-1] + ")"] = df[items].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    for c in df.columns:
        if c in id_cols or c == lab or c in items:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any():
            cand[c] = df[c]
    for name, x in cand.items():
        leak["rules"].append(best_threshold_rule(x, y, name))
    leak["rules"].sort(key=lambda r: -r["accuracy"])
    if items:
        s = df[items].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        for c in df.columns:
            if re.search(r"score|result", str(c), re.I) and not ITEM_RE.match(str(c)) and pd.api.types.is_numeric_dtype(df[c]):
                leak[f"item_sum_equals_{c}"] = {"rows_equal": int((s == df[c]).sum()), "n": int(len(df)),
                                                "max_abs_diff": float((s - df[c]).abs().max())}
    leak["majority_class_baseline_accuracy"] = round(float(vc.max() / len(y)), 6)
    if items and "Age" in df.columns:  # item-sum rule within age bands (years)
        s = df[items].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        bands = {}
        for lo, hi in [(0, 3), (4, 11), (12, 17), (18, 200)]:
            m = (df["Age"] >= lo) & (df["Age"] <= hi)
            if m.sum():
                r = best_threshold_rule(s[m], y[m], "item_sum")
                r["n_pos"] = int(y[m].sum())
                bands[f"Age {lo}-{hi}"] = r
        leak["item_sum_rule_by_age_band"] = bands
        exact_rows = sum(b["n"] for b in bands.values() if b["accuracy"] == 1.0)
        leak["rows_in_age_bands_with_exact_rule"] = exact_rows
    top = leak["rules"][0] if leak["rules"] else None
    leak["exact_leakage_found"] = bool(top and top["accuracy"] == 1.0)
    leak["summary"] = (f"LEAKAGE: label reproduced exactly by '{top['rule']}' (accuracy 1.0)" if leak["exact_leakage_found"]
                       else f"No exact single-feature rule over the whole file; best is '{top['rule']}' accuracy {top['accuracy']} "
                            f"(majority baseline {leak['majority_class_baseline_accuracy']})" if top
                       else "No numeric features to test.")
    if not leak["exact_leakage_found"] and leak.get("rows_in_age_bands_with_exact_rule"):
        leak["summary"] += (f"; PARTIAL LEAKAGE: within age bands the item-sum rule is exact for "
                            f"{leak['rows_in_age_bands_with_exact_rule']} of {len(df)} rows")
    out["leakage_check"] = leak
    return out, df, lab


def embedded_blocks(tables):
    """Is one file's rows contained, in order, as a contiguous block of another file?
    Compared on item columns A1..A10 + Sex (both present in each file); labels then compared."""
    out = []
    for (fa, (da, la)), (fb, (db, lb)) in [(x, y) for x in tables.items() for y in tables.items() if x[0] != y[0]]:
        items = [c for c in da.columns if ITEM_RE.match(str(c)) and c in db.columns]
        if not items or "Sex" not in da.columns or "Sex" not in db.columns or len(db) >= len(da):
            continue
        cols = items + ["Sex"]
        A = da[cols].astype(str).to_numpy()
        B = db[cols].astype(str).to_numpy()
        for start in range(len(da) - len(db) + 1):
            if (A[start] == B[0]).all() and (A[start:start + len(db)] == B).all():
                ya = da[la].astype(str).str.strip().str.lower().iloc[start:start + len(db)].to_numpy()
                yb = db[lb].astype(str).str.strip().str.lower().to_numpy()
                out.append({"container": fa, "contained": fb, "compared_on": cols,
                            "container_rows": [start, start + len(db) - 1],
                            "label_container_counts": dict(collections.Counter(ya)),
                            "label_contained_counts": dict(collections.Counter(yb)),
                            "rows_with_different_label": int((ya != yb).sum())})
                break
    return out


def audit_tabular(root):
    files, tables = [], {}
    for dp, _, fns in os.walk(root):
        for f in sorted(fns):
            if os.path.splitext(f)[1].lower() in TAB_EXT:
                o, df, lab = audit_table(os.path.join(dp, f), root)
                files.append(o)
                if lab:
                    tables[o["file"]] = (df, lab)
    return {"files": files, "cross_file_embedded_blocks": embedded_blocks(tables)}


# --------------------------------------------------------------------------- e) eye tracking
def audit_eye(root):
    gdir = os.path.join(root, "Eye-tracking Output")
    meta = pd.read_csv(os.path.join(root, "Metadata_Participants.csv"))
    meta_ids = set(meta["ParticipantID"].astype(int))
    files = sorted([f for f in os.listdir(gdir) if f.endswith(".csv")], key=lambda s: int(s[:-4]))
    per_file, all_cols, total_rows, pid_rows = [], {}, 0, collections.Counter()
    key_sets, md5s, dt_all = {}, {}, []
    usecols = ["RecordingTime [ms]", "Time of Day [h:m:s:ms]", "Trial", "Stimulus", "Participant", "Category Group",
               "Point of Regard Right X [px]", "Point of Regard Right Y [px]"]
    dd02_cols = ["Participant", "RecordingTime [ms]", "Trial", "Point of Regard Right X [px]", "Point of Regard Right Y [px]"]
    dd02_hash, dd02_ident = {}, {}
    for f in files:
        p = os.path.join(gdir, f)
        log("eye", f)
        with open(p, "rb") as fh:
            md5s[f] = hashlib.md5(fh.read()).hexdigest()
        cols = list(pd.read_csv(p, nrows=0).columns)
        all_cols[f] = cols
        df = pd.read_csv(p, usecols=usecols, dtype=str)
        n = len(df)
        total_rows += n
        part = df["Participant"]
        pc = part.value_counts()
        pid_rows.update(pc.to_dict())
        t = pd.to_numeric(df["RecordingTime [ms]"], errors="coerce")
        g = (part != part.shift()) | (df["Trial"] != df["Trial"].shift())
        d = t.diff()[~g & t.notna()]
        d = d[d > 0]
        dt_all.append(d.to_numpy())
        key_sets[f] = set(zip(df["RecordingTime [ms]"], df["Time of Day [h:m:s:ms]"], part, df["Trial"]))
        h = pd.util.hash_pandas_object(df[dd02_cols], index=False).to_numpy()
        dd02_hash[f] = h
        dd02_ident[f] = ~part.str.startswith("Unidentified").to_numpy()
        per_file.append({"file": f, "rows": n, "n_columns": len(cols),
                         "participants": sorted(pc.index, key=lambda v: (not v.isdigit(), int(v) if v.isdigit() else 0, v)),
                         "rows_unidentified": int(part.str.startswith("Unidentified").sum()),
                         "n_trials": int(df["Trial"].nunique()),
                         "stimuli": sorted(df["Stimulus"].dropna().unique().tolist())[:20],
                         "median_dt_ms": float(np.median(d)) if len(d) else None})
    # column-set variants
    variants = collections.defaultdict(list)
    for f, c in all_cols.items():
        variants[tuple(c)].append(f)
    numeric_ids = {int(k) for k in pid_rows if str(k).isdigit()}
    unident = {k: int(v) for k, v in pid_rows.items() if not str(k).isdigit()}
    # duplicated content across files (row key = RecordingTime, TimeOfDay, Participant, Trial)
    seen = collections.Counter()
    for s in key_sets.values():
        seen.update(s)
    overlap = {f: int(sum(1 for k in s if seen[k] > 1)) for f, s in key_sets.items()}
    md5_groups = collections.defaultdict(list)
    for f, h in md5s.items():
        md5_groups[h].append(f)
    distinct_keys = len(seen)
    # DD-02 key (participant + recording timestamp + trial + gaze coordinates), hashed to uint64
    allh = np.concatenate(list(dd02_hash.values()))
    alli = np.concatenate(list(dd02_ident.values()))
    uniq = {f: np.unique(h) for f, h in dd02_hash.items()}
    pair_overlap = []
    fl = list(uniq)
    for i in range(len(fl)):
        for j in range(i + 1, len(fl)):
            k = len(np.intersect1d(uniq[fl[i]], uniq[fl[j]], assume_unique=True))
            if k:
                pair_overlap.append({"files": [fl[i], fl[j]], "shared_keys": int(k),
                                     "rows_a": int(len(dd02_hash[fl[i]])), "rows_b": int(len(dd02_hash[fl[j]]))})
    pair_overlap.sort(key=lambda d: -d["shared_keys"])
    dd02 = {
        "key": "(Participant, RecordingTime [ms], Trial, Point of Regard Right X [px], Point of Regard Right Y [px]) hashed with pandas.util.hash_pandas_object (64-bit)",
        "rows_total": int(len(allh)),
        "distinct_keys_all_rows": int(len(np.unique(allh))),
        "rows_identified_participant": int(alli.sum()),
        "distinct_keys_identified_rows(=rows after drop Unidentified + dedupe)": int(len(np.unique(allh[alli]))),
        "within_file_duplicate_rows": {f: int(len(h) - len(uniq[f])) for f, h in dd02_hash.items() if len(h) != len(uniq[f])},
        "file_pairs_sharing_keys": pair_overlap,
    }
    dt = np.concatenate(dt_all)
    cls = meta["Class"].value_counts().to_dict()
    gaze_with_meta = sorted(numeric_ids & meta_ids)
    rows_by_class = collections.Counter()
    m = dict(zip(meta["ParticipantID"].astype(int), meta["Class"]))
    for k, v in pid_rows.items():
        rows_by_class[m.get(int(k), "no_metadata") if str(k).isdigit() else "unidentified"] += v
    return {
        "n_csv_files_gaze": len(files),
        "total_gaze_rows": total_rows,
        "column_set_variants": [{"n_columns": len(k), "files": v, "columns": list(k)} for k, v in variants.items()],
        "per_file": per_file,
        "metadata_file": "Metadata_Participants.csv",
        "metadata_columns": list(meta.columns),
        "metadata_n_participants": int(len(meta)),
        "metadata_class_counts": {str(k): int(v) for k, v in cls.items()},
        "metadata_missing": {c: int(v) for c, v in meta.isna().sum().items()},
        "cars_missing_by_class": {str(k): {"cars_missing": int(g["CARS Score"].isna().sum()), "n": int(len(g))}
                                  for k, g in meta.groupby("Class")},
        "metadata_gender_counts": {str(k): int(v) for k, v in meta["Gender"].value_counts().items()},
        "metadata_age_years": {"min": float(meta["Age"].min()), "median": float(meta["Age"].median()), "max": float(meta["Age"].max())},
        "unique_participant_values_in_gaze": int(len(pid_rows)),
        "unique_numeric_participant_ids_in_gaze": int(len(numeric_ids)),
        "non_numeric_participant_values_rows": unident,
        "participants_with_metadata_and_gaze": int(len(gaze_with_meta)),
        "participants_with_metadata_and_gaze_by_class": dict(collections.Counter(m[i] for i in gaze_with_meta)),
        "participants_metadata_but_no_gaze": sorted(meta_ids - numeric_ids),
        "participant_ids_gaze_but_no_metadata": sorted(numeric_ids - meta_ids),
        "gaze_rows_by_class(all files, incl. duplicates)": dict(rows_by_class),
        "file_md5_identical_groups": [v for v in md5_groups.values() if len(v) > 1],
        "duplicate_content_check": {
            "row_key": "(RecordingTime [ms], Time of Day, Participant, Trial)",
            "distinct_row_keys_over_all_files": distinct_keys,
            "rows_total": total_rows,
            "rows_whose_key_also_occurs_in_another_file_per_file": overlap,
        },
        "duplicate_check_dd02_key": dd02,
        "tracking_ratio_column_in_files": sum("Tracking Ratio [%]" in c for c in all_cols.values()),
        "sampling": {
            "method": "median of positive RecordingTime differences between consecutive rows with the same Participant and Trial, in file order",
            "median_dt_ms": float(np.median(dt)),
            "implied_rate_hz": round(1000.0 / float(np.median(dt)), 2),
            "p05_dt_ms": float(np.percentile(dt, 5)), "p95_dt_ms": float(np.percentile(dt, 95)),
            "n_diffs": int(len(dt)),
        },
    }


# --------------------------------------------------------------------------- markdown
def md_dataset(did, a):
    L = [f"# {did} audit", "", f"- Folder: `{a['folder']}`", f"- Detected data type: **{a['data_type']}**",
         f"- Files: {a['tree']['n_files']} | size: {a['tree']['total_size_mb']} MB | max depth: {a['tree']['max_depth']}",
         f"- Files by extension: {a['tree']['files_by_extension']}",
         f"- Archives (zip/rar): {a['archives'] or 'none'}",
         f"- Licence/citation files: {a['licence']['files'] or 'none'} (licence file present: {a['licence']['licence_file_present']}) -> `{a['licence']['copied_to']}`",
         "", "## Folder tree (first 3 levels)", "", "| dir | depth | files | subdirs |", "|---|---|---|---|"]
    for t in a["tree"]["tree_first_3_levels"]:
        L.append(f"| `{t['dir']}` | {t['depth']} | {t['n_files']} | {t['n_subdirs']} |")
    if "images" in a:
        im = a["images"]
        L += ["", "## Images", "", f"- Samples (images inside class folders): {im['image_files_used_as_samples']}",
              f"- Image files outside class folders (not samples): {im['image_files_outside_class_folders(not samples)'] or 'none'}",
              f"- Class folder names per split (raw): {im['class_folder_names_per_split(raw)']}",
              "", "| split | " + " | ".join(sorted(im["class_totals"])) + " | total |", "|---" * (len(im["class_totals"]) + 2) + "|"]
        for s, c in im["count_per_split_x_class"].items():
            L.append(f"| {s} | " + " | ".join(str(c.get(k, 0)) for k in sorted(im["class_totals"])) + f" | {sum(c.values())} |")
        e, n = im["exact_duplicates_md5"], im["near_duplicates_phash"]
        L += ["", f"- Width min/median/max: {im['width']} ; Height: {im['height']} ; distinct sizes: {im['n_distinct_sizes']}",
              f"- Colour modes: {im['colour_modes']} ; unreadable files: {im['n_unreadable']}",
              f"- **Exact duplicates (md5)**: {e['n_groups']} groups, {e['n_images_in_groups']} images, {e['n_redundant_copies']} redundant copies; "
              f"within one split: {e['groups_within_one_split']}, across splits: {e['groups_across_splits']} {e['groups_split_combinations']}; "
              f"groups with label conflict: {e['groups_with_label_conflict']}",
              f"- **Near duplicates (phash Hamming <= 4)**: {n['pairs_total']} pairs ({n['pairs_not_exact_md5']} not byte-identical); "
              f"within split {n['within_same_split']}, across splits {n['across_splits']} {n['across_split_matrix']}; "
              f"different class: {n['different_class(label conflict)']}; connected groups: {n['n_groups(connected components)']}",
              f"- Subject IDs: {im['subject_ids']['finding']}",
              f"- Contact sheet: `{im['contact_sheet']}`"]
    if "tabular" in a:
        for f in a["tabular"]["files"]:
            L += ["", f"## `{f['file']}`", "", f"- Shape: {f['shape']}", f"- ID columns: {f['id_columns'] or 'none'} (unique: {f['id_columns_unique']})",
                  f"- Label column(s): {f['label_columns'] or 'NONE'}", f"- Class counts: {f.get('class_counts', f.get('label_note'))}",
                  f"- Duplicate rows: full={f['duplicate_rows_full']}, excluding ID={f['duplicate_rows_excluding_id']}",
                  f"- Fully empty columns: {f['fully_empty_columns'] or 'none'}",
                  f"- '?' placeholder values: {f['question_mark_values_per_column'] or 'none'}",
                  "", "| column | dtype | missing | unique values (first 20) |", "|---|---|---|---|"]
            for c in f["columns"]:
                u = f["categorical_unique_values"].get(c)
                L.append(f"| `{c}` | {f['dtypes'][c]} | {f['missing_per_column'][c]} | {u['values(first 20)'] if u else '(numeric/continuous)'} |")
            if "leakage_check" in f:
                lk = f["leakage_check"]
                L += ["", f"**Leakage check:** {lk['summary']}", "", "| feature | best rule | accuracy | [[TN,FP],[FN,TP]] |", "|---|---|---|---|"]
                for r in lk["rules"]:
                    L.append(f"| {r['feature']} | {r['rule']} | {r['accuracy']} | {r['confusion_matrix[[TN,FP],[FN,TP]]']} |")
                for k, v in lk.items():
                    if k.startswith("item_sum_equals_"):
                        L.append(f"\n- {k}: {v}")
                L.append(f"\n- Majority-class baseline accuracy: {lk['majority_class_baseline_accuracy']}")
                for band, r in lk.get("item_sum_rule_by_age_band", {}).items():
                    L.append(f"- {band}: n={r['n']} (ASD={r['n_pos']}) best '{r['rule']}' accuracy {r['accuracy']} CM {r['confusion_matrix[[TN,FP],[FN,TP]]']}")
        for b in a["tabular"].get("cross_file_embedded_blocks", []):
            L += ["", f"## Cross-file finding: `{b['contained']}` is embedded in `{b['container']}`", "",
                  f"- All {b['container_rows'][1] - b['container_rows'][0] + 1} rows found, in order, at 0-based rows "
                  f"{b['container_rows'][0]}..{b['container_rows'][1]} of `{b['container']}` (compared on {b['compared_on']})",
                  f"- Labels there: {b['label_container_counts']} vs original {b['label_contained_counts']} -> "
                  f"**{b['rows_with_different_label']} rows carry a different label**"]
    if "eye_tracking" in a:
        e = a["eye_tracking"]
        L += ["", "## Eye-tracking", "", f"- Gaze CSV files: {e['n_csv_files_gaze']} ; total rows: {e['total_gaze_rows']}",
              f"- Column-set variants: " + "; ".join(f"{v['n_columns']} cols -> files {v['files']}" for v in e["column_set_variants"]),
              f"- Metadata: `{e['metadata_file']}` columns {e['metadata_columns']}; {e['metadata_n_participants']} participants; classes {e['metadata_class_counts']}; gender {e['metadata_gender_counts']}; age {e['metadata_age_years']}; missing {e['metadata_missing']}",
              f"- Unique Participant values in gaze: {e['unique_participant_values_in_gaze']} (numeric IDs: {e['unique_numeric_participant_ids_in_gaze']})",
              f"- Non-numeric Participant rows: {e['non_numeric_participant_values_rows']}",
              f"- Participants with metadata AND gaze: {e['participants_with_metadata_and_gaze']} {e['participants_with_metadata_and_gaze_by_class']}",
              f"- Metadata but no gaze rows: {e['participants_metadata_but_no_gaze'] or 'none'}",
              f"- Gaze IDs without metadata: {e['participant_ids_gaze_but_no_metadata'] or 'none'}",
              f"- Gaze rows by class: {e['gaze_rows_by_class(all files, incl. duplicates)']}",
              f"- Byte-identical gaze files (md5): {e['file_md5_identical_groups'] or 'none'}",
              f"- Distinct row keys {e['duplicate_content_check']['row_key']}: {e['duplicate_content_check']['distinct_row_keys_over_all_files']} of {e['duplicate_content_check']['rows_total']} rows",
              f"- Sampling: median dt {e['sampling']['median_dt_ms']} ms (~{e['sampling']['implied_rate_hz']} Hz), p05 {e['sampling']['p05_dt_ms']}, p95 {e['sampling']['p95_dt_ms']}",
              "", "| file | rows | cols | trials | unidentified rows | rows shared with other files | median dt ms | participants |", "|---|---|---|---|---|---|---|---|"]
        ov = e["duplicate_content_check"]["rows_whose_key_also_occurs_in_another_file_per_file"]
        for p in e["per_file"]:
            L.append(f"| {p['file']} | {p['rows']} | {p['n_columns']} | {p['n_trials']} | {p['rows_unidentified']} | {ov[p['file']]} | {p['median_dt_ms']} | {', '.join(p['participants'])} |")
    return "\n".join(L) + "\n"


def main():
    started = datetime.datetime.now().isoformat(timespec="seconds")
    audits, img_recs = {}, {}
    for did, (folder, kind) in DATASETS.items():
        root = os.path.join(DATASETS_DIR, folder)
        log("auditing", did, "->", folder)
        tree = tree_summary(root)
        a = {"dataset_id": did, "folder": folder, "path": root, "audited_at": started, "seed": SEED,
             "tree": tree, "data_type": "eye_tracking_timeseries_csv" if kind == "eye_tracking" else data_type(tree),
             "archives": list_archives(root), "licence": licence_files(root, did)}
        if kind == "images":
            a["images"], img_recs[did] = audit_images(root, did)
            a["n_samples"] = a["images"]["image_files_used_as_samples"]
        elif kind == "tabular":
            a["tabular"] = audit_tabular(root)
            a["n_samples"] = sum(f["shape"][0] for f in a["tabular"]["files"])
        else:
            a["eye_tracking"] = audit_eye(root)
            a["n_samples"] = a["eye_tracking"]["total_gaze_rows"]
        audits[did] = a
    cross = cross_image_datasets(img_recs["D03_facial_images_B"], img_recs["D02_facial_images_A"],
                                 "D03_facial_images_B", "D02_facial_images_A")
    for did in ("D02_facial_images_A", "D03_facial_images_B"):
        audits[did]["cross_dataset_duplicates"] = cross
    for did, a in audits.items():
        with open(os.path.join(REG, f"{did}_audit.json"), "w") as fh:
            json.dump(a, fh, indent=1, default=jdefault)
        md = md_dataset(did, a)
        if "cross_dataset_duplicates" in a:
            md += "\n## Cross-dataset duplicates (D02 vs D03)\n\n```json\n" + json.dumps(cross, indent=1, default=jdefault) + "\n```\n"
        with open(os.path.join(REG, f"{did}_audit.md"), "w") as fh:
            fh.write(md)
    log("done; wrote", len(audits), "audits")


if __name__ == "__main__":
    sys.exit(main())
