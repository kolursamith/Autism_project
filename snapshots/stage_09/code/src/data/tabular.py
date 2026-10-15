"""D04 / D05 / D06 deterministic cleaning (DD-08, DD-09, DD-10). Nothing is fitted here: no scaling, encoding or
imputation (that happens per fold in Stage 7). Every change is recorded in a change log."""
import os

import numpy as np
import pandas as pd

from src.data.common import dataset_dir


def read_raw(cfg):
    """Reads ONLY cfg['file'] (one labelled file per dataset) as strings."""
    return pd.read_csv(os.path.join(dataset_dir(cfg["dataset_id"]), cfg["file"]), dtype=str, keep_default_na=False)


def clean(cfg):
    """Returns (clean_df, change_log_df, columns_spec dict, checks dict)."""
    raw = read_raw(cfg)
    df = raw.copy()
    changes = []

    def log(column, change, n):
        if n:
            changes.append({"column": column, "change": change, "rows_affected": int(n)})

    # excluded columns
    for col, reason in cfg.get("excluded", {}).items():
        if col not in df:
            raise KeyError(col)
        df = df.drop(columns=col)
        changes.append({"column": col, "change": f"column dropped: {reason}", "rows_affected": len(df)})

    # strings: trim, lower-case categorical values, missing tokens -> NaN
    miss = set(cfg.get("missing_tokens", []))
    for col in cfg["categorical"] + [cfg["label"]["column"]]:
        s0 = df[col]
        s = s0.str.strip()
        log(col, "trimmed surrounding whitespace", (s != s0).sum())
        is_miss = s.isin(miss) | (s == "")
        log(col, f"missing token {sorted(miss)} -> missing", is_miss.sum())
        if col in cfg["categorical"]:
            low = s.str.lower()
            merged = {v: low[s == v].iloc[0] for v in s.unique() if v != v.lower()}
            log(col, f"lower-cased values {merged}", (low != s).sum())
            s = low
        df[col] = s.mask(is_miss)

    # numeric
    for col in cfg["numeric"]:
        v = pd.to_numeric(df[col].str.strip(), errors="coerce")
        bad = v.isna() & (df[col].str.strip() != "")
        log(col, "non-numeric -> missing", bad.sum())
        df[col] = v

    # label -> 0/1
    lab = cfg["label"]
    pos = {str(x).lower() for x in lab["positive"]}
    neg = {str(x).lower() for x in lab["negative"]}
    lv = df[lab["column"]].str.lower()
    if not lv.isin(pos | neg).all():
        raise ValueError(f"unexpected label values {sorted(set(lv) - pos - neg)}")
    df["label"] = lv.isin(pos).astype(int)
    changes.append({"column": lab["column"], "change": f"label {sorted(pos)} -> 1, {sorted(neg)} -> 0, stored as 'label'",
                    "rows_affected": len(df)})

    # sample id
    idc = cfg["id_column"]
    if not df[idc].is_unique:
        raise ValueError("id column not unique")
    df["sample_id"] = cfg["sample_prefix"] + df[idc].astype(str)

    # duplicate groups: identical feature rows (all features, id and label ignored) share a group_id
    feats = cfg["numeric"] + cfg["categorical"]
    key = df[feats].apply(lambda col: col.map(lambda v: "<NA>" if pd.isna(v) else repr(v))).agg("|".join, axis=1)
    first = {}
    for k in key:
        first.setdefault(k, len(first))
    df["group_id"] = [f"g{first[k]:04d}" for k in key]
    gl = df.groupby("group_id")["label"].nunique()
    gs = df.groupby("group_id").size()
    checks = {
        "n_rows_raw": len(raw), "n_rows_clean": len(df),
        "duplicate_rows_full_excluding_id": int(raw.drop(columns=idc).duplicated().sum()),
        "rows_in_duplicate_feature_groups": int(gs[gs > 1].sum()), "duplicate_feature_groups": int((gs > 1).sum()),
        "duplicate_groups_with_mixed_labels": int((gl > 1).sum()), "n_groups": int(len(gs)),
        "label_counts": df["label"].value_counts().sort_index().to_dict(),
        "missing_per_column": df[feats].isna().sum().to_dict(),
    }
    cols = ["sample_id", "group_id"] + [idc] + feats + ["label"]
    spec = {"dataset_id": cfg["dataset_id"], "source_file": cfg["file"], "columns": {}}
    spec["columns"]["sample_id"] = {"role": "id", "note": f"'{cfg['sample_prefix']}' + {idc}"}
    spec["columns"]["group_id"] = {"role": "id", "note": "duplicate group: identical feature rows share it (same fold)"}
    spec["columns"][idc] = {"role": "id", "note": "original row id, not a feature"}
    for c in cfg["numeric"]:
        spec["columns"][c] = {"role": "feature_numeric"}
    for c in cfg["categorical"]:
        spec["columns"][c] = {"role": "feature_categorical"}
    spec["columns"]["label"] = {"role": "label", "note": f"from '{lab['column']}': 1 = {sorted(pos)}"}
    for c, reason in cfg.get("excluded", {}).items():
        spec["columns"][c] = {"role": "excluded", "reason": reason}
    spec["columns"][lab["column"]] = {"role": "excluded", "reason": "raw label column (replaced by 'label')"}
    spec["feature_sets"] = {name: list(v) for name, v in cfg.get("feature_sets", {"set_all": feats}).items()}
    for name, fs in spec["feature_sets"].items():
        unknown = set(fs) - set(feats)
        if unknown:
            raise ValueError(f"{name}: {unknown}")
    return df[cols], pd.DataFrame(changes), spec, checks


def item_sum_rule(df, items, threshold):
    """Accuracy of 'label == (sum(items) >= threshold)' and the confusion matrix [[TN, FP], [FN, TP]]."""
    pred = (df[items].sum(axis=1) >= threshold).astype(int)
    y = df["label"]
    cm = [[int(((y == 0) & (pred == 0)).sum()), int(((y == 0) & (pred == 1)).sum())],
          [int(((y == 1) & (pred == 0)).sum()), int(((y == 1) & (pred == 1)).sum())]]
    return float((pred == y).mean()), cm


def result_function_test(df, items, col):
    """Is `col` a direct (deterministic) function of the item columns?"""
    key = df[items].astype(int).astype(str).agg("".join, axis=1)
    per = df.groupby(key)[col].nunique()
    X = np.c_[np.ones(len(df)), df[items].to_numpy(float)]
    y = df[col].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return {"equals_item_sum_fraction": float((df[items].sum(axis=1) == df[col]).mean()),
            "n_distinct_item_vectors": int(per.size), "max_distinct_values_per_item_vector": int(per.max()),
            "item_vectors_with_more_than_one_value": int((per > 1).sum()), "linear_r2_on_items": round(float(r2), 4),
            "is_function_of_items": bool(per.max() == 1)}
