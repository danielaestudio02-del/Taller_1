"""Construcción del pipeline completo (preprocesamiento + estimador) desde un config."""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from common import CATEGORICAL

MODELS = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
}


def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def build_preprocessor(prep: dict, numeric: list[str]) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if prep.get("log1p"):
        num_steps.append(("log1p", FunctionTransformer(signed_log1p, feature_names_out="one-to-one")))
    if prep["kind"] == "linear":
        num_steps.append(("scale", StandardScaler()))
        cat_enc = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=prep.get("min_frequency", 20),
                                sparse_output=False)
    elif prep["kind"] == "tree":
        cat_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                 encoded_missing_value=-1)
    else:
        raise ValueError(f"preprocesamiento desconocido: {prep['kind']}")
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", cat_enc)])
    return ColumnTransformer([("num", Pipeline(num_steps), numeric), ("cat", cat, CATEGORICAL)])


def build_pipeline(cfg: dict, numeric: list[str]) -> Pipeline:
    model_cfg = cfg["model"]
    params = dict(model_cfg.get("params", {}))
    params["random_state"] = cfg["seed"]
    estimator = MODELS[model_cfg["type"]](**params)
    return Pipeline([("preprocess", build_preprocessor(cfg["preprocess"], numeric)), ("model", estimator)])
