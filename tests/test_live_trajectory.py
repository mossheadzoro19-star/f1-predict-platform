from __future__ import annotations

import pandas as pd

from f1_predict.modeling.live_trajectory import evaluate_trajectories


def _rows():
    return pd.DataFrame([
        {"session_key":"r1","lap_number":1,"driver_number":1,"winner":1,"win_probability":0.4},
        {"session_key":"r1","lap_number":1,"driver_number":2,"winner":0,"win_probability":0.6},
        {"session_key":"r1","lap_number":2,"driver_number":1,"winner":1,"win_probability":0.7},
        {"session_key":"r1","lap_number":2,"driver_number":2,"winner":0,"win_probability":0.3},
        {"session_key":"r1","lap_number":3,"driver_number":1,"winner":1,"win_probability":0.8},
        {"session_key":"r1","lap_number":3,"driver_number":2,"winner":0,"win_probability":0.2},
    ])


def test_trajectory_metrics():
    result = evaluate_trajectories(_rows())
    row = result.iloc[0]
    assert row["first_correct_leader_lap"] == 2
    assert row["persistent_correct_leader_lap"] == 2
    assert row["predicted_leader_changes"] == 1
    assert row["max_winner_probability"] == 0.8
    assert row["final_winner_probability"] == 0.8


def test_trajectory_rejects_missing_columns():
    bad = _rows().drop(columns=["win_probability"])
    try:
        evaluate_trajectories(bad)
    except ValueError as exc:
        assert "Missing trajectory columns" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
