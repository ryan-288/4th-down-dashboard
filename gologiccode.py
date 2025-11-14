# -*- coding: utf-8 -*-
"""
Model selection and serialization utilities for the 4th Down Decision Tool.

This module expects that `backgroundcalculation.py` has already generated the
`artifacts/go_attempts.csv` dataset. It retrains classifier/regressor models
that estimate 4th-down conversion probability along with EPA/WPA outcomes and
persists the resulting pickles back into the `artifacts/` directory.
"""

from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd

from model_training import (
    CLASSIFIER_FEATURES,
    evaluate_classifiers,
    expected_gain,
    train_primary_go_model,
    train_regression_models,
)

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
GO_ATTEMPTS_PATH = ARTIFACTS_DIR / "go_attempts.csv"
GO_MODEL_PATH = ARTIFACTS_DIR / "go_for_it_model.pkl"
EPA_SUCCESS_MODEL_PATH = ARTIFACTS_DIR / "epa_model_success.pkl"
WPA_SUCCESS_MODEL_PATH = ARTIFACTS_DIR / "wpa_model_success.pkl"
EPA_FAIL_MODEL_PATH = ARTIFACTS_DIR / "epa_model_fail.pkl"
WPA_FAIL_MODEL_PATH = ARTIFACTS_DIR / "wpa_model_fail.pkl"
FAIL_AVERAGES_PATH = ARTIFACTS_DIR / "fail_epa_wpa_averages.csv"


def load_go_attempts() -> pd.DataFrame:
    if not GO_ATTEMPTS_PATH.exists():
        raise FileNotFoundError(
            f"{GO_ATTEMPTS_PATH} not found. Run backgroundcalculation.py first."
        )
    return pd.read_csv(GO_ATTEMPTS_PATH)


def main():
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    go_attempts = load_go_attempts()
    go_data_cleaned = go_attempts.dropna(
        subset=CLASSIFIER_FEATURES + ["fourth_down_converted", "epa", "wpa"]
    )

    X = go_data_cleaned[CLASSIFIER_FEATURES]
    y = go_data_cleaned["fourth_down_converted"]
    evaluate_classifiers(X, y)

    go_model, mean_acc, std_acc = train_primary_go_model(go_data_cleaned)
    joblib.dump(go_model, GO_MODEL_PATH)
    print(
        f"Saved primary go-for-it model to {GO_MODEL_PATH} "
        f"(CV accuracy {mean_acc:.4f} ± {std_acc:.4f})"
    )

    regression_models, fail_averages = train_regression_models(go_data_cleaned)
    fail_averages.to_csv(FAIL_AVERAGES_PATH, index=False)
    joblib.dump(regression_models["epa_success"], EPA_SUCCESS_MODEL_PATH)
    joblib.dump(regression_models["wpa_success"], WPA_SUCCESS_MODEL_PATH)
    joblib.dump(regression_models["epa_fail"], EPA_FAIL_MODEL_PATH)
    joblib.dump(regression_models["wpa_fail"], WPA_FAIL_MODEL_PATH)
    print(
        f"Saved EPA/WPA regression models to {ARTIFACTS_DIR} "
        f"and fail averages to {FAIL_AVERAGES_PATH}"
    )

    example_input = {
        "ydstogo": 3,
        "qtr": 4,
        "half_seconds_remaining": 120,
        "yardline_100": 40,
        "score_differential": -3,
    }
    results = expected_gain(example_input, go_model, regression_models)
    print(f"Conversion Probability: {results['conversion_prob']:.2f}")
    print(
        f"EPA (Success): {results['epa_success']:.2f} | "
        f"EPA (Fail): {results['epa_fail']:.2f}"
    )
    print(
        f"WPA (Success): {results['wpa_success']:.2f} | "
        f"WPA (Fail): {results['wpa_fail']:.2f}"
    )
    print(f"Expected EPA: {results['expected_epa']:.2f}")
    print(f"Expected WPA: {results['expected_wpa']:.2f}")


if __name__ == "__main__":
    main()
