# -*- coding: utf-8 -*-
"""
End-to-end data pipeline for the 4th Down Decision Tool.

This script downloads play-by-play data, engineers punt/field-goal/go-for-it
features, trains predictive models, and persists all artifacts in the local
`artifacts/` directory. The API layer and Dash app can then load these
artifacts without recomputing heavy analytics on startup.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import json

import joblib
import nfl_data_py as nfl
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.linear_model import LogisticRegression

from model_training import (
    CLASSIFIER_FEATURES,
    evaluate_classifiers,
    expected_gain,
    train_primary_go_model,
    train_regression_models,
)

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

DEFAULT_SEASONS = [2021, 2022, 2023, 2024]

GO_ATTEMPTS_PATH = ARTIFACTS_DIR / "go_attempts.csv"
GO_MODEL_PATH = ARTIFACTS_DIR / "go_for_it_model.pkl"
EPA_SUCCESS_MODEL_PATH = ARTIFACTS_DIR / "epa_model_success.pkl"
WPA_SUCCESS_MODEL_PATH = ARTIFACTS_DIR / "wpa_model_success.pkl"
EPA_FAIL_MODEL_PATH = ARTIFACTS_DIR / "epa_model_fail.pkl"
WPA_FAIL_MODEL_PATH = ARTIFACTS_DIR / "wpa_model_fail.pkl"
PUNT_SUMMARY_PATH = ARTIFACTS_DIR / "punt_summary.csv"
SCORE_PROB_PATH = ARTIFACTS_DIR / "scoreprobability.csv"
OPP_SCORE_PROB_PATH = ARTIFACTS_DIR / "opponentscoreprobability.csv"
FAIL_AVERAGES_PATH = ARTIFACTS_DIR / "fail_epa_wpa_averages.csv"
FIELD_GOAL_SUMMARY_PATH = ARTIFACTS_DIR / "field_goal_summary.csv"
FIELD_GOAL_MODEL_PATH = ARTIFACTS_DIR / "field_goal_model.pkl"
FIELD_GOAL_OUTCOMES_PATH = ARTIFACTS_DIR / "field_goal_outcomes.json"


@dataclass
class DecisionDatasets:
    decisiondata: pd.DataFrame
    go_attempts: pd.DataFrame
    punt_attempts: pd.DataFrame
    field_goal_attempts: pd.DataFrame
    first_down_samples: pd.DataFrame


@dataclass
class PuntInterpolators:
    epa: interp1d
    wpa: interp1d
    touchback: interp1d
    opp_td: interp1d
    opp_fg: interp1d
    opp_no_score: interp1d


def ensure_artifact_dir() -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)


def fetch_play_by_play(seasons: Iterable[int]) -> pd.DataFrame:
    print(f"Downloading play-by-play data for seasons: {list(seasons)}")
    return nfl.import_pbp_data(list(seasons))


def prepare_datasets(data: pd.DataFrame) -> DecisionDatasets:
    firsts = data[
        ((data["down"] == 1.0) & (data["ydstogo"] == 10))
        | (
            (data["down"] == 1.0)
            & (data["goal_to_go"] == 1.0)
            & (data["ydstogo"] <= 10)
        )
    ]

    columns_to_keep = [
        "posteam",
        "posteam_type",
        "defteam",
        "side_of_field",
        "yardline_100",
        "half_seconds_remaining",
        "game_seconds_remaining",
        "qtr",
        "down",
        "ydstogo",
        "ydsnet",
        "yards_gained",
        "epa",
        "wp",
        "def_wp",
        "wpa",
        "vegas_wpa",
        "pass_attempt",
        "season",
        "cp",
        "cpoe",
        "goal_to_go",
        "air_yards",
        "field_goal_attempt",
        "field_goal_result",
        "kick_distance",
        "score_differential",
        "no_score_prob",
        "opp_fg_prob",
        "opp_td_prob",
        "fg_prob",
        "td_prob",
        "punt_blocked",
        "punt_inside_twenty",
        "touchback",
        "punt_attempt",
        "fourth_down_converted",
        "touchdown",
    ]

    filtered = data.loc[
        (data["down"] == 4.0) | (data["field_goal_attempt"] == 1.0), columns_to_keep
    ]
    decisiondata = filtered.copy()

    decisiondata_4th = decisiondata[decisiondata["down"] == 4]
    go_attempts = decisiondata_4th[
        (decisiondata_4th["field_goal_attempt"] != 1.0)
        & (decisiondata_4th["punt_attempt"] != 1.0)
    ]
    punt_attempts = decisiondata_4th[decisiondata_4th["punt_attempt"] == 1.0]
    field_goal_attempts = decisiondata[decisiondata["field_goal_attempt"] == 1.0]

    print(f"Decision dataset shape: {decisiondata.shape}")
    print(f"Go attempts shape: {go_attempts.shape}")
    print(f"Punt attempts shape: {punt_attempts.shape}")
    print(f"Field goal attempts shape: {field_goal_attempts.shape}")

    return DecisionDatasets(
        decisiondata=decisiondata,
        go_attempts=go_attempts,
        punt_attempts=punt_attempts,
        field_goal_attempts=field_goal_attempts,
        first_down_samples=firsts,
    )


def build_punt_summary(punt_attempts: pd.DataFrame) -> pd.DataFrame:
    summary = (
        punt_attempts.groupby("yardline_100")
        .agg(
            {
                "epa": "mean",
                "wpa": "mean",
                "opp_td_prob": "mean",
                "opp_fg_prob": "mean",
                "no_score_prob": "mean",
                "touchback": "mean",
                "punt_inside_twenty": "mean",
            }
        )
        .reset_index()
    )
    summary["weighted_points"] = summary["epa"]
    summary.columns = [
        "field_position",
        "punt_epa",
        "punt_wpa",
        "opp_td_prob",
        "opp_fg_prob",
        "opp_no_score_prob",
        "touchback_prob",
        "inside_twenty_prob",
        "punt_weighted_points",
    ]
    return summary


def create_punt_interpolators(punt_summary: pd.DataFrame) -> PuntInterpolators:
    return PuntInterpolators(
        epa=interp1d(
            punt_summary["field_position"],
            punt_summary["punt_epa"],
            kind="linear",
            fill_value="extrapolate",
        ),
        wpa=interp1d(
            punt_summary["field_position"],
            punt_summary["punt_wpa"],
            kind="linear",
            fill_value="extrapolate",
        ),
        touchback=interp1d(
            punt_summary["field_position"],
            punt_summary["touchback_prob"],
            kind="linear",
            fill_value="extrapolate",
        ),
        opp_td=interp1d(
            punt_summary["field_position"],
            punt_summary["opp_td_prob"],
            kind="linear",
            fill_value="extrapolate",
        ),
        opp_fg=interp1d(
            punt_summary["field_position"],
            punt_summary["opp_fg_prob"],
            kind="linear",
            fill_value="extrapolate",
        ),
        opp_no_score=interp1d(
            punt_summary["field_position"],
            punt_summary["opp_no_score_prob"],
            kind="linear",
            fill_value="extrapolate",
        ),
    )


def convert_coach_yardline_to_yardline_100(yardline: int, team_side: str) -> int:
    if not (1 <= yardline <= 50):
        raise ValueError("Yardline must be between 1 and 50")
    if team_side.lower() == "own":
        return 100 - yardline
    if team_side.lower() == "opponent":
        return yardline
    raise ValueError("team_side must be 'own' or 'opponent'")


def punt_decision_metrics(
    interpolators: PuntInterpolators,
    coach_yardline: int,
    team_side: str,
    gross_punt_yards: float,
) -> Dict[str, float]:
    yardline_100 = convert_coach_yardline_to_yardline_100(coach_yardline, team_side)
    tb_prob = float(interpolators.touchback(yardline_100))
    raw_landing_yl_100 = yardline_100 + gross_punt_yards
    
    # Clamp landing position to valid range (0-100)
    # If punt goes into end zone, treat as touchback
    if raw_landing_yl_100 >= 100:
        # Punt goes into end zone - treat as touchback
        pos_if_no_tb = 0  # End zone
        tb_prob = 1.0  # Force touchback
    else:
        pos_if_no_tb = max(0.0, 100 - raw_landing_yl_100)
    
    pos_if_tb = 80
    adjusted_fp = tb_prob * pos_if_tb + (1 - tb_prob) * pos_if_no_tb
    
    # Clamp adjusted field position to valid range (0-100)
    adjusted_fp = max(0.0, min(100.0, adjusted_fp))

    epa = float(interpolators.epa(adjusted_fp))
    wpa = float(interpolators.wpa(adjusted_fp))
    opp_td = float(interpolators.opp_td(adjusted_fp))
    opp_fg = float(interpolators.opp_fg(adjusted_fp))
    opp_no_score = float(interpolators.opp_no_score(adjusted_fp))

    # Clamp field positions for epa calculations
    landing_fp = max(0.0, min(100.0, 100 - raw_landing_yl_100))
    epa_no_tb = float(interpolators.epa(landing_fp))
    epa_tb = float(interpolators.epa(80))
    weighted_points = (epa_no_tb * (1 - tb_prob)) - (epa_tb * tb_prob)

    return {
        "epa": epa,
        "wpa": wpa,
        "opp_td_prob": opp_td,
        "opp_fg_prob": opp_fg,
        "opp_no_score_prob": opp_no_score,
        "touchback_prob": tb_prob,
        "weighted_points_added": weighted_points,
    }


def train_field_goal_model(
    field_goal_attempts: pd.DataFrame,
) -> Tuple[pd.DataFrame, LogisticRegression]:
    fg_attempts_filtered = field_goal_attempts.dropna(
        subset=["kick_distance", "field_goal_result"]
    ).copy()
    fg_attempts_filtered["made_binary"] = (
        fg_attempts_filtered["field_goal_result"] == "made"
    ).astype(int)

    X = fg_attempts_filtered["kick_distance"].values.reshape(-1, 1)
    y = fg_attempts_filtered["made_binary"].values

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    distances = np.arange(19, 71).reshape(-1, 1)
    predicted_probs = model.predict_proba(distances)[:, 1]

    attempts = (
        fg_attempts_filtered["kick_distance"]
        .value_counts()
        .reindex(range(19, 71), fill_value=0)
        .sort_index()
    )
    made = (
        fg_attempts_filtered[fg_attempts_filtered["field_goal_result"] == "made"][
            "kick_distance"
        ]
        .value_counts()
        .reindex(range(19, 71), fill_value=0)
        .sort_index()
    )

    summary_df = pd.DataFrame(
        {
            "Kick Distance (yards)": range(19, 71),
            "Attempts": attempts.values,
            "Made": made.values,
            "Make_Prob": np.divide(
                made.values,
                attempts.values,
                out=np.zeros_like(made.values, dtype=float),
                where=attempts.values != 0,
            ),
            "Model_Predicted_Prob": predicted_probs,
            "Model_Missed_Predicted_Prob": 1 - predicted_probs,
        }
    )
    return summary_df, model


def summarize_field_goal_outcomes(
    field_goal_attempts: pd.DataFrame,
) -> Dict[str, Dict[str, float]]:
    made_mask = field_goal_attempts["field_goal_result"] == "made"
    made = field_goal_attempts[made_mask]
    missed = field_goal_attempts[~made_mask]

    def stats(df: pd.DataFrame) -> Dict[str, float]:
        if df.empty:
            return {"epa": 0.0, "wpa": 0.0}
        return {
            "epa": float(df["epa"].mean()),
            "wpa": float(df["wpa"].mean()),
        }

    return {
        "made": stats(made),
        "missed": stats(missed),
    }


def compute_score_probability_tables(
    decisiondata: pd.DataFrame, first_down_samples: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    scoreprobability = (
        decisiondata.groupby(["yardline_100", "ydstogo"])
        .agg(
            {
                "td_prob": "mean",
                "fg_prob": "mean",
                "opp_td_prob": "mean",
                "opp_fg_prob": "mean",
                "no_score_prob": "mean",
            }
        )
        .reset_index()
    )

    opponentscoreprobability = (
        first_down_samples.groupby(["yardline_100", "ydstogo"])
        .agg(
            {
                "td_prob": "mean",
                "fg_prob": "mean",
                "opp_td_prob": "mean",
                "opp_fg_prob": "mean",
                "no_score_prob": "mean",
            }
        )
        .reset_index()
    )

    return scoreprobability, opponentscoreprobability


def train_go_models(go_attempts: pd.DataFrame):
    go_attempts.to_csv(GO_ATTEMPTS_PATH, index=False)
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
    joblib.dump(regression_models["epa_success"], EPA_SUCCESS_MODEL_PATH)
    joblib.dump(regression_models["wpa_success"], WPA_SUCCESS_MODEL_PATH)
    joblib.dump(regression_models["epa_fail"], EPA_FAIL_MODEL_PATH)
    joblib.dump(regression_models["wpa_fail"], WPA_FAIL_MODEL_PATH)
    fail_averages.to_csv(FAIL_AVERAGES_PATH, index=False)

    print(f"Saved regression models and fail averages to {ARTIFACTS_DIR}")
    return go_model, regression_models, fail_averages


def run_pipeline(seasons: Iterable[int] = DEFAULT_SEASONS):
    ensure_artifact_dir()
    data = fetch_play_by_play(seasons)
    datasets = prepare_datasets(data)

    punt_summary = build_punt_summary(datasets.punt_attempts)
    punt_summary.to_csv(PUNT_SUMMARY_PATH, index=False)
    interpolators = create_punt_interpolators(punt_summary)

    fg_summary, fg_model = train_field_goal_model(datasets.field_goal_attempts)
    fg_summary.to_csv(FIELD_GOAL_SUMMARY_PATH, index=False)
    joblib.dump(fg_model, FIELD_GOAL_MODEL_PATH)
    fg_outcomes = summarize_field_goal_outcomes(datasets.field_goal_attempts)
    FIELD_GOAL_OUTCOMES_PATH.write_text(json.dumps(fg_outcomes, indent=2))

    scoreprobability, opponentscoreprobability = compute_score_probability_tables(
        datasets.decisiondata, datasets.first_down_samples
    )
    scoreprobability.to_csv(SCORE_PROB_PATH, index=False)
    opponentscoreprobability.to_csv(OPP_SCORE_PROB_PATH, index=False)

    go_model, regression_models, _ = train_go_models(datasets.go_attempts)

    example_input = {
        "ydstogo": 3,
        "qtr": 4,
        "half_seconds_remaining": 120,
        "yardline_100": 40,
        "score_differential": -3,
    }
    results = expected_gain(example_input, go_model, regression_models)
    print("Example go-for-it recommendation snapshot:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")

    metrics_example = punt_decision_metrics(
        interpolators, coach_yardline=35, team_side="own", gross_punt_yards=45
    )
    print("Example punt metrics snapshot:")
    for key, value in metrics_example.items():
        print(f"  {key}: {value:.4f}")

    print(f"Artifacts written to {ARTIFACTS_DIR.resolve()}")


if __name__ == "__main__":
    run_pipeline()

