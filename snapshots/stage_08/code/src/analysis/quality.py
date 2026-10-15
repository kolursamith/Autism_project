"""D01 tracking-quality shortcut baselines (DD-12). Quality features are per-child summaries of the recording itself
(never model inputs); every fitted baseline is fit on the TRAINING children of an outer fold only."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

QUALITY_FEATURES = ["mean_valid_ratio", "mean_blink_ratio", "tracking_ratio_before_filtering"]
LR_PARAMS = {"C": 1.0, "max_iter": 2000}


def fit_predict_lr(train, test, features, label="label", seed=42):
    """Standardise + logistic regression fit on `train` rows only, P(ASD) for `test` rows. Refuses overlapping ids."""
    if "group_id" in train and "group_id" in test:
        overlap = set(train["group_id"]) & set(test["group_id"])
        if overlap:
            raise ValueError(f"train/test overlap: {sorted(overlap)[:5]}")
    model = make_pipeline(StandardScaler(), LogisticRegression(random_state=seed, **LR_PARAMS))
    model.fit(train[features].to_numpy(float), train[label].to_numpy(int))
    return model.predict_proba(test[features].to_numpy(float))[:, 1]


def fold_children(folds_df, load_fold, ds, k):
    """(train+val child ids, test child ids) of outer fold k from the frozen split files."""
    f = load_fold(ds, k)
    g = folds_df.set_index("sample_id").group_id
    train = sorted(set(g.loc[f["train"]]) | set(g.loc[f["val"]]))
    test = sorted(set(g.loc[f["test"]]))
    assert not set(train) & set(test)
    return train, test


def tertile_labels(x):
    """0/1/2 by the 1/3 and 2/3 quantiles of x (low / mid / high)."""
    q1, q2 = np.quantile(x, [1 / 3, 2 / 3])
    return np.where(x <= q1, 0, np.where(x <= q2, 1, 2)), (float(q1), float(q2))
