"""Read-only access to saved out-of-fold predictions (results/<ds>/... are never written by Stage 8)."""
import os

import pandas as pd

from src.data.common import PROJECT_DIR

RESULTS = os.path.join(PROJECT_DIR, "results")
OUT_DIR = os.path.join(RESULTS, "stage8")
FIG_DIR = os.path.join(PROJECT_DIR, "reports", "figures")
D01, D02 = "D01_eye_tracking", "D02_facial_images_A"
MODELS = [f"M{i:02d}" for i in range(1, 11)]
CUSTOM_CNN = ["M02", "M05", "M09"]
PRETRAINED = ["M01", "M03", "M04", "M06", "M07", "M08", "M10"]
NAMES = {"M01": "EfficientNetB3 + BiLSTM + Attn", "M02": "CNN + BiLSTM + Attn", "M03": "ResNet50 + LSTM + Attn",
         "M04": "ConvNeXt + BiLSTM + Attn", "M05": "CNN + Transformer + MLP", "M06": "DenseNet121 + LSTM + Attn",
         "M07": "ViT + LSTM + Attn", "M08": "EfficientNet + Transformer + Attn", "M09": "CNN + GRU + Attn",
         "M10": "ResNet50 + BiLSTM + Transformer"}
TABULAR = {"D04_autism_prediction": ["set_all"], "D05_toddler_screening": ["set_all", "set_honest"],
           "D06_early_autism": ["set_all"]}
TAB_MODELS = ["dummy_majority", "logreg", "random_forest", "hist_gb", "mlp"]


def model_dir(ds, model, experiment=None):
    return os.path.join(RESULTS, ds, *( [experiment] if experiment else []), model)


def load_oof(ds, model, experiment=None, folds=range(1, 11)):
    """All test predictions of one model (10 folds concatenated): sample_id, group_id, y_true, y_prob, fold."""
    d = model_dir(ds, model, experiment)
    P = pd.concat([pd.read_csv(os.path.join(d, f"fold_{k:02d}", "predictions.csv")) for k in folds],
                  ignore_index=True)
    assert not P.sample_id.duplicated().any()
    return P


def child_scores(P):
    """Child probability = mean clip probability per child (group_id); one row per child with its test fold."""
    C = P.groupby("group_id", sort=True).agg(y=("y_true", "first"), y_n=("y_true", "nunique"), p=("y_prob", "mean"),
                                              fold=("fold", "first"), n_clips=("y_prob", "size")).reset_index()
    assert (C.y_n == 1).all()
    return C.drop(columns="y_n")
