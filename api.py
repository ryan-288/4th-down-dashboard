# -*- coding: utf-8 -*-
"""
FastAPI service exposing the 4th Down Decision Tool predictions.

Run locally with:
    uvicorn api:app --reload
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backgroundcalculation import (
    ARTIFACTS_DIR,
    EPA_FAIL_MODEL_PATH,
    EPA_SUCCESS_MODEL_PATH,
    FIELD_GOAL_MODEL_PATH,
    FIELD_GOAL_OUTCOMES_PATH,
    GO_MODEL_PATH,
    PUNT_SUMMARY_PATH,
    WPA_FAIL_MODEL_PATH,
    WPA_SUCCESS_MODEL_PATH,
    create_punt_interpolators,
    punt_decision_metrics,
)
from model_training import expected_gain

app = FastAPI(title="4th Down Decision Tool API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DecisionRequest(BaseModel):
    ydstogo: float = Field(..., gt=0, description="Yards needed for a first down.")
    yardline_100: float = Field(
        ..., ge=0, le=100, description="Distance (in yards) to the opponent's end zone."
    )
    qtr: int = Field(..., ge=1, le=5, description="Quarter number (OT=5).")
    half_seconds_remaining: float = Field(
        ...,
        ge=0,
        description="Seconds remaining in the current half (0-1800).",
    )
    score_differential: float = Field(
        ..., description="Offense score minus defense score."
    )
    gross_punt_yards: float = Field(
        45.0, ge=0, description="Expected gross punt distance for the scenario."
    )
    kick_distance: Optional[float] = Field(
        None,
        ge=15,
        le=70,
        description="Field goal kick distance in yards (15-70). If not provided, calculated from yardline.",
    )


class Artifacts:
    def __init__(self):
        self.go_model = None
        self.regression_models: Dict[str, object] = {}
        self.field_goal_model = None
        self.field_goal_outcomes: Dict[str, Dict[str, float]] = {}
        self.punt_interpolators = None

    def loaded(self) -> bool:
        return (
            self.go_model is not None
            and self.regression_models
            and self.field_goal_model is not None
            and self.field_goal_outcomes
            and self.punt_interpolators is not None
        )


artifacts = Artifacts()


def _load_joblib(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact: {path}")
    return joblib.load(path)


def load_artifacts():
    punt_summary = pd.read_csv(PUNT_SUMMARY_PATH)
    artifacts.punt_interpolators = create_punt_interpolators(punt_summary)

    artifacts.go_model = _load_joblib(GO_MODEL_PATH)
    artifacts.regression_models = {
        "epa_success": _load_joblib(EPA_SUCCESS_MODEL_PATH),
        "wpa_success": _load_joblib(WPA_SUCCESS_MODEL_PATH),
        "epa_fail": _load_joblib(EPA_FAIL_MODEL_PATH),
        "wpa_fail": _load_joblib(WPA_FAIL_MODEL_PATH),
    }
    artifacts.field_goal_model = _load_joblib(FIELD_GOAL_MODEL_PATH)
    artifacts.field_goal_outcomes = json.loads(FIELD_GOAL_OUTCOMES_PATH.read_text())


def infer_coach_yardline(yardline_100: float) -> Tuple[float, str]:
    if yardline_100 > 50:
        coach_yardline = 100 - yardline_100
        team_side = "own"
    else:
        coach_yardline = yardline_100
        team_side = "opponent"
    coach_yardline = float(max(1, min(50, coach_yardline)))
    return coach_yardline, team_side


def compute_field_goal_metrics(
    payload: DecisionRequest,
) -> Dict[str, float]:
    # Use provided kick_distance or calculate from yardline
    if payload.kick_distance is not None:
        kick_distance = float(payload.kick_distance)
    else:
        kick_distance = float(payload.yardline_100 + 17)
    
    # Ensure kick_distance is within valid range
    kick_distance = max(15.0, min(70.0, kick_distance))
    
    make_prob = float(
        artifacts.field_goal_model.predict_proba(np.array([[kick_distance]]))[0][1]
    )
    miss_prob = 1 - make_prob
    made_stats = artifacts.field_goal_outcomes.get("made", {"epa": 0.0, "wpa": 0.0})
    missed_stats = artifacts.field_goal_outcomes.get("missed", {"epa": 0.0, "wpa": 0.0})
    expected_epa = make_prob * made_stats["epa"] + miss_prob * missed_stats["epa"]
    expected_wpa = make_prob * made_stats["wpa"] + miss_prob * missed_stats["wpa"]
    
    return {
        "kick_distance": kick_distance,
        "make_prob": make_prob,
        "miss_prob": miss_prob,
        "expected_epa": expected_epa,
        "expected_wpa": expected_wpa,
    }


def choose_recommendation(options: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    best_play, metrics = max(
        options.items(), key=lambda item: item[1].get("expected_wpa", float("-inf"))
    )
    return {"play": best_play, "expected_wpa": metrics.get("expected_wpa", 0.0)}


@app.on_event("startup")
def startup_event():
    if not ARTIFACTS_DIR.exists():
        raise RuntimeError(
            f"Artifacts directory not found at {ARTIFACTS_DIR}. "
            "Run backgroundcalculation.py to generate model artifacts."
        )
    load_artifacts()


@app.get("/")
def root():
    return {
        "message": "4th Down Decision Tool API",
        "endpoints": {
            "health": "/health",
            "decision": "/decision (POST)",
            "docs": "/docs"
        }
    }


@app.get("/health")
def health():
    status = "ok" if artifacts.loaded() else "not_ready"
    return {"status": status}


@app.post("/decision")
def make_decision(payload: DecisionRequest):
    if not artifacts.loaded():
        raise HTTPException(status_code=503, detail="Artifacts not loaded")

    go_input = {
        "ydstogo": payload.ydstogo,
        "qtr": payload.qtr,
        "half_seconds_remaining": payload.half_seconds_remaining,
        "yardline_100": payload.yardline_100,
        "score_differential": payload.score_differential,
    }
    go_metrics = expected_gain(go_input, artifacts.go_model, artifacts.regression_models)

    coach_yardline, team_side = infer_coach_yardline(payload.yardline_100)
    punt_metrics = punt_decision_metrics(
        artifacts.punt_interpolators,
        coach_yardline=coach_yardline,
        team_side=team_side,
        gross_punt_yards=payload.gross_punt_yards,
    )

    field_goal_metrics = compute_field_goal_metrics(payload)

    recommendation = choose_recommendation(
        {
            "go_for_it": go_metrics,
            "field_goal": field_goal_metrics,
            "punt": {
                **punt_metrics,
                "expected_wpa": punt_metrics["wpa"],
                "expected_epa": punt_metrics["epa"],
            },
        }
    )

    return {
        "inputs": payload.dict(),
        "go_for_it": go_metrics,
        "field_goal": field_goal_metrics,
        "punt": punt_metrics,
        "recommendation": recommendation,
    }

