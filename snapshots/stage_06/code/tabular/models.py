"""Stage 7 tabular baselines: data access, train-only preprocessing pipeline, the 5 fixed models.
CPU only, no torch. Data are read ONLY from cache/<ds>/clean.csv + columns.yaml (Stage 2 outputs); the raw dataset
folder is never opened (so D04 test.csv and the D05 Combined file cannot be read)."""
import copy
import os

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

TAB_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TAB_DIR)
CACHE_DIR = os.path.join(PROJECT_DIR, "cache")
MODEL_IDS = ["dummy_majority", "logreg", "random_forest", "hist_gb", "mlp"]


def load_stage7_cfg(path=os.path.join(TAB_DIR, "configs", "stage7.yaml")):
    with open(path) as fh:
        return yaml.safe_load(fh)


def load_table(ds):
    """clean.csv indexed by sample_id + columns.yaml (roles, feature_sets)."""
    with open(os.path.join(CACHE_DIR, ds, "columns.yaml")) as fh:
        cols = yaml.safe_load(fh)
    df = pd.read_csv(os.path.join(CACHE_DIR, ds, "clean.csv"), keep_default_na=True).set_index("sample_id", drop=False)
    return df, cols


def feature_columns(cols, feature_set):
    """(numeric, categorical) feature columns of a feature set, by role in columns.yaml."""
    feats = cols["feature_sets"][feature_set]
    roles = {c: v["role"] for c, v in cols["columns"].items()}
    bad = [c for c in feats if roles.get(c) not in {"feature_numeric", "feature_categorical"}]
    if bad:
        raise ValueError(f"not feature columns: {bad}")
    return ([c for c in feats if roles[c] == "feature_numeric"], [c for c in feats if roles[c] == "feature_categorical"])


def make_preprocessor(numeric, categorical):
    num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    parts = ([("num", num, numeric)] if numeric else []) + ([("cat", cat, categorical)] if categorical else [])
    return ColumnTransformer(parts, remainder="drop")


class ValEarlyStoppingMLP(MLPClassifier):
    """MLPClassifier trained with partial_fit (one pass over the TRAIN part per epoch); early stopping on the ROC-AUC
    of the fold's VALIDATION part (passed explicitly as X_val, y_val); the weights of the best epoch are kept.
    sklearn's own early_stopping (an internal split of the training data) is NOT used."""

    def __init__(self, hidden_layer_sizes=(64, 32), random_state=None, patience=10, min_delta=1e-4, max_epochs=200):
        super().__init__(hidden_layer_sizes=hidden_layer_sizes, random_state=random_state)
        self.patience, self.min_delta, self.max_epochs = patience, min_delta, max_epochs

    def fit(self, X, y, X_val=None, y_val=None):
        if X_val is None:
            raise ValueError("ValEarlyStoppingMLP needs the fold's validation part")
        classes = np.array([0, 1])
        best, best_state, bad, self.val_history_ = -np.inf, None, 0, []
        for epoch in range(1, self.max_epochs + 1):
            super().partial_fit(X, y, classes=classes)
            auc = roc_auc_score(y_val, super().predict_proba(X_val)[:, 1])
            self.val_history_.append(auc)
            if auc > best + self.min_delta:
                best, bad = auc, 0
                best_state = (copy.deepcopy(self.coefs_), copy.deepcopy(self.intercepts_), epoch)
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.coefs_, self.intercepts_, self.best_epoch_ = best_state
        self.epochs_run_, self.best_val_auc_, self.n_val_seen_ = epoch, best, len(y_val)
        return self


def make_classifier(model_id, cfg, seed):
    m = cfg["models"][model_id]
    p = dict(m["params"])
    if model_id == "dummy_majority":
        return DummyClassifier(**p)
    if model_id == "logreg":
        return LogisticRegression(**p, random_state=seed)
    if model_id == "random_forest":
        return RandomForestClassifier(**p, n_jobs=cfg["n_jobs"], random_state=seed)
    if model_id == "hist_gb":
        return HistGradientBoostingClassifier(**p, random_state=seed)
    if model_id == "mlp":
        es = m["early_stopping"]
        return ValEarlyStoppingMLP(hidden_layer_sizes=tuple(p["hidden_layer_sizes"]), random_state=seed,
                                   patience=es["patience"], min_delta=es["min_delta"], max_epochs=es["max_epochs"])
    raise ValueError(model_id)


def fit_pipeline(model_id, cfg, seed, numeric, categorical, X_tr, y_tr, X_val, y_val):
    """Pipeline(preprocess -> classifier) fitted on the TRAIN part; the validation part is used only by the MLP
    (early stopping, after being transformed with the train-fitted preprocessor)."""
    pipe = Pipeline([("prep", make_preprocessor(numeric, categorical)), ("clf", make_classifier(model_id, cfg, seed))])
    if model_id == "hist_gb":
        pipe.fit(X_tr, y_tr, clf__sample_weight=compute_sample_weight("balanced", y_tr))
    elif model_id == "mlp":
        prep = pipe.named_steps["prep"].fit(X_tr, y_tr)
        pipe.named_steps["clf"].fit(prep.transform(X_tr), y_tr, X_val=prep.transform(X_val), y_val=y_val)
    else:
        pipe.fit(X_tr, y_tr)
    return pipe
