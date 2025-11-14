# -*- coding: utf-8 -*-
"""
Shared model-training utilities for the 4th Down Decision Tool.

These helpers are imported by both the background data pipeline and the
standalone model retraining script so we have a single source of truth for
feature sets and evaluation logic.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier, XGBRegressor

CLASSIFIER_FEATURES = [
    "ydstogo",
    "qtr",
    "half_seconds_remaining",
    "yardline_100",
    "score_differential",
]

TOP_FEATURES = ["ydstogo", "qtr", "score_differential", "yardline_100"]


def evaluate_classifiers(X: pd.DataFrame, y: pd.Series) -> None:
    """Cross-validate several off-the-shelf classifiers for context."""
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    models: Dict[str, object] = {
        "LogReg": LogisticRegression(max_iter=10000),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(),
        "LinReg": LogisticRegression(
            fit_intercept=True, max_iter=10000, solver="liblinear"
        ),
        "XGBoost": XGBClassifier(eval_metric="logloss"),
        "NaiveBayes": GaussianNB(),
    }

    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=kf, scoring="accuracy")
        print(f"{name} CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
        model.fit(X, y)
        _display_feature_importance(model, name, X.columns)


def _display_feature_importance(
    model, model_name: str, feature_names: Iterable[str]
) -> None:
    print(f"\n--- Feature Importances for {model_name} ---")
    if hasattr(model, "coef_"):
        coefs = model.coef_[0]
        for feat, coef in sorted(
            zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True
        ):
            print(f"{feat}: {coef:.4f}")
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        for feat, imp in sorted(
            zip(feature_names, importances), key=lambda x: x[1], reverse=True
        ):
            print(f"{feat}: {imp:.4f}")
    else:
        print("Feature importances not supported for this model.")


def train_primary_go_model(
    go_data_cleaned: pd.DataFrame,
) -> Tuple[LogisticRegression, float, float]:
    """Train the logistic regression selected as the production go-for-it model."""
    X_top = go_data_cleaned[TOP_FEATURES]
    y = go_data_cleaned["fourth_down_converted"]
    kf = KFold(n_splits=25, shuffle=True, random_state=42)
    model = LogisticRegression(
        fit_intercept=True, max_iter=100000, solver="liblinear"
    )
    model.fit(X_top, y)
    scores = cross_val_score(model, X_top, y, cv=kf, scoring="accuracy")
    return model, scores.mean(), scores.std()


def train_regression_models(
    go_data_cleaned: pd.DataFrame,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    """Train EPA/WPA regressors for success and failure scenarios."""
    success_df = go_data_cleaned[go_data_cleaned["fourth_down_converted"] == 1]
    fail_df = go_data_cleaned[go_data_cleaned["fourth_down_converted"] == 0]

    epa_model_success = select_best_regressor(
        success_df[CLASSIFIER_FEATURES], success_df["epa"], label="EPA Success"
    )
    wpa_model_success = select_best_regressor(
        success_df[CLASSIFIER_FEATURES], success_df["wpa"], label="WPA Success"
    )
    epa_model_fail = select_best_regressor(
        fail_df[CLASSIFIER_FEATURES], fail_df["epa"], label="EPA Fail"
    )
    wpa_model_fail = select_best_regressor(
        fail_df[CLASSIFIER_FEATURES], fail_df["wpa"], label="WPA Fail"
    )

    fail_averages = (
        fail_df.groupby(["yardline_100", "ydstogo"])[["epa", "wpa"]]
        .mean()
        .reset_index()
    )

    models = {
        "epa_success": epa_model_success,
        "wpa_success": wpa_model_success,
        "epa_fail": epa_model_fail,
        "wpa_fail": wpa_model_fail,
    }

    return models, fail_averages


def select_best_regressor(
    X: pd.DataFrame, y: pd.Series, label: str = "epa/wpa"
) -> object:
    """Pick the best-performing regressor via 5-fold CV."""
    models: Dict[str, object] = {
        "Linear": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, eval_metric="rmse", random_state=42),
    }
    best_model = None
    best_score = -np.inf
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring="r2")
        mean_score = scores.mean()
        print(f"{label} - {name} R2 CV Score: {mean_score:.4f}")
        if mean_score > best_score:
            best_score = mean_score
            best_model = model
    best_model.fit(X, y)
    return best_model


def predict_conversion_prob(
    input_dict: Dict[str, float], model: LogisticRegression
) -> float:
    df = pd.DataFrame([input_dict])[TOP_FEATURES]
    return float(model.predict_proba(df)[0][1])


def predict_epa_wpa(
    input_dict: Dict[str, float],
    models: Dict[str, object],
) -> Tuple[float, float, float, float]:
    df = pd.DataFrame([input_dict])[CLASSIFIER_FEATURES]
    epa_success_pred = float(models["epa_success"].predict(df)[0])
    epa_fail_pred = float(models["epa_fail"].predict(df)[0])
    wpa_success_pred = float(models["wpa_success"].predict(df)[0])
    wpa_fail_pred = float(models["wpa_fail"].predict(df)[0])
    return epa_success_pred, epa_fail_pred, wpa_success_pred, wpa_fail_pred


def expected_gain(
    input_dict: Dict[str, float],
    conversion_model: LogisticRegression,
    regression_models: Dict[str, object],
) -> Dict[str, float]:
    success_prob = predict_conversion_prob(input_dict, conversion_model)
    epa_succ, epa_fail, wpa_succ, wpa_fail = predict_epa_wpa(
        input_dict, regression_models
    )
    expected_epa = (success_prob * epa_succ) + ((1 - success_prob) * epa_fail)
    expected_wpa = (success_prob * wpa_succ) + ((1 - success_prob) * wpa_fail)
    return {
        "conversion_prob": success_prob,
        "expected_epa": expected_epa,
        "expected_wpa": expected_wpa,
        "epa_success": epa_succ,
        "epa_fail": epa_fail,
        "wpa_success": wpa_succ,
        "wpa_fail": wpa_fail,
    }


__all__ = [
    "CLASSIFIER_FEATURES",
    "TOP_FEATURES",
    "evaluate_classifiers",
    "train_primary_go_model",
    "train_regression_models",
    "select_best_regressor",
    "predict_conversion_prob",
    "predict_epa_wpa",
    "expected_gain",
]

