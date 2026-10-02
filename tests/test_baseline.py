import pandas as pd
import pytest

from f1_predict.modeling.baseline import (
    evaluate_race_probabilities,
    fit_baseline,
    predict_race_probabilities,
)
from f1_predict.modeling.features import FeatureContract


def _frame():
    rows = []
    for season in (2022, 2023, 2024):
        for round_ in (1, 2):
            for driver in ("a", "b", "c"):
                row = {
                    "season": season,
                    "round": round_,
                    "driver_id": driver,
                    "prediction_time": pd.Timestamp(f"{season}-03-{round_:02d}T12:00:00Z"),
                    "winner": int(driver == "a"),
                }
                for feature in FeatureContract.pre_race().features:
                    row[feature] = 1.0
                rows.append(row)
    return pd.DataFrame(rows)


def test_baseline_outputs_one_probability_per_driver_race():
    frame = _frame()
    model = fit_baseline(frame[frame["season"] <= 2023])
    predictions = predict_race_probabilities(model, frame[frame["season"] == 2024])

    totals = predictions.groupby(["season", "round"])["win_probability"].sum()
    assert len(predictions) == 6
    assert totals.tolist() == pytest.approx([1.0, 1.0])


def test_baseline_handles_missing_pre_race_values():
    frame = _frame()
    frame.loc[0, "q2_seconds"] = float("nan")
    model = fit_baseline(frame[frame["season"] <= 2023])
    predictions = predict_race_probabilities(model, frame[frame["season"] == 2024])
    assert predictions["win_probability"].notna().all()


def test_race_probability_metrics():
    actuals = pd.DataFrame(
        [
            {"season": 2024, "round": 1, "driver_id": "a", "winner": 1},
            {"season": 2024, "round": 1, "driver_id": "b", "winner": 0},
            {"season": 2024, "round": 1, "driver_id": "c", "winner": 0},
        ]
    )
    predictions = pd.DataFrame(
        [
            {"season": 2024, "round": 1, "driver_id": "a", "win_probability": 0.8},
            {"season": 2024, "round": 1, "driver_id": "b", "win_probability": 0.1},
            {"season": 2024, "round": 1, "driver_id": "c", "win_probability": 0.1},
        ]
    )
    metrics = evaluate_race_probabilities(predictions, actuals)

    assert metrics.races == 1
    assert metrics.winner_accuracy == 1.0
    assert metrics.log_loss == pytest.approx(-__import__("math").log(0.8))
    assert metrics.brier_score == pytest.approx(((0.8 - 1) ** 2 + 0.1**2 + 0.1**2) / 3)
