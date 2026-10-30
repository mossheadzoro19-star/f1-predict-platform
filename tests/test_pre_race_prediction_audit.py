import numpy as np
import pandas as pd
import pytest

import f1_predict.modeling.pre_race_prediction_audit as audit
from f1_predict.modeling.pre_race_model import FEATURE_COLUMNS


class _FixedXGBoost:
    def fit(self, features, target, verbose=False):
        self.fitted_rows = len(features)
        return self

    def predict_proba(self, features):
        # Driver 0 receives the strongest score in each synthetic race.
        scores = np.tile(np.array([0.8, 0.15, 0.05]), (len(features) // 3, 1)).reshape(-1)
        return np.column_stack([1.0 - scores, scores])


def _frame(seasons=(2020, 2021)):
    rows = []
    for season in seasons:
        for round_number in (1, 2):
            for driver_number in range(3):
                row = {
                    "season": season,
                    "round": round_number,
                    "driver_id": f"driver-{driver_number}",
                    "winner": int(driver_number == 0),
                    "race_name": f"Race {round_number}",
                    "circuit_id": f"circuit-{round_number}",
                }
                row.update(
                    {feature: float(driver_number + 1) for feature in FEATURE_COLUMNS}
                )
                rows.append(row)
    return pd.DataFrame(rows)


def test_audit_rejects_reserved_2025_holdout():
    with pytest.raises(ValueError, match="2025 is reserved"):
        audit.validate_audit_frame(
            _frame(seasons=(2020, 2021, 2025)),
            validation_years=(2021,),
        )


def test_driver_probabilities_are_preserved_and_sum_to_one_per_race():
    frame = _frame().query("season == 2021").copy()
    probabilities = np.tile([0.7, 0.2, 0.1], 2)

    result = audit.build_driver_predictions(frame, probabilities, train_end_year=2020)

    assert len(result) == 6
    assert result.groupby(["season", "round"])["win_probability"].sum().to_numpy() == pytest.approx([1, 1])
    assert set(result["train_end_year"]) == {2020}


def test_error_audit_flags_low_probability_winner_and_confident_wrong_pick():
    frame = _frame().query("season == 2021").copy()
    probabilities = np.tile([0.70, 0.25, 0.05], 2)
    predictions = audit.build_driver_predictions(frame, probabilities, 2020)
    # Make driver 1 the actual winner in the first race.
    predictions.loc[
        (predictions["round"] == 1) & (predictions["driver_id"] == "driver-0"),
        "winner",
    ] = 0
    predictions.loc[
        (predictions["round"] == 1) & (predictions["driver_id"] == "driver-1"),
        "winner",
    ] = 1

    result = audit.build_race_error_audit(
        frame.assign(
            winner=np.where(
                (frame["round"] == 1) & (frame["driver_id"] == "driver-0"),
                0,
                np.where(
                    (frame["round"] == 1) & (frame["driver_id"] == "driver-1"),
                    1,
                    frame["winner"],
                ),
            )
        ),
        predictions,
        confidence_threshold=0.60,
        low_winner_probability=0.30,
    )
    first = result.loc[result["round"] == 1].iloc[0]
    assert first["predicted_winner_driver_id"] == "driver-0"
    assert first["actual_winner_driver_id"] == "driver-1"
    assert bool(first["high_confidence_wrong"])
    assert bool(first["low_probability_actual_winner"])
    assert first["actual_winner_qualifying_position"] == 2


def test_reliability_bins_report_observed_frequency():
    predictions = pd.DataFrame({
        "win_probability": [0.1, 0.2, 0.8, 0.9],
        "winner": [0, 0, 1, 0],
    })
    result = audit.build_candidate_reliability_bins(predictions, bins=2)

    assert len(result) == 2
    assert result.iloc[0]["candidate_rows"] == 2
    assert result.iloc[0]["observed_winner_frequency"] == 0.0
    assert result.iloc[1]["candidate_rows"] == 2
    assert result.iloc[1]["observed_winner_frequency"] == 0.5


def test_top_choice_reliability_uses_race_hit_rate():
    race_audit = pd.DataFrame({
        "predicted_winner_probability": [0.6, 0.7, 0.8, 0.9],
        "top_prediction_correct": [True, False, True, False],
    })
    result = audit.build_top_choice_reliability_bins(race_audit, bins=4)

    assert result["races"].sum() == 4
    assert result["top_pick_hit_rate"].tolist() == pytest.approx([0.5, 0.5])


def test_run_prediction_audit_trains_only_on_prior_seasons(monkeypatch):
    frame = _frame()
    monkeypatch.setattr(audit, "_make_xgboost", _FixedXGBoost)

    summary, predictions, race_audit, calibration = audit.run_prediction_audit(
        frame,
        validation_years=(2021,),
        bins=5,
    )

    assert summary["validation_years"] == [2021]
    assert summary["validation_races"] == 2
    assert summary["driver_race_candidate_rows"] == 6
    assert set(predictions["train_end_year"]) == {2020}
    assert len(race_audit) == 2
    assert summary["overall_race_metrics"]["log_loss"] > 0
    assert len(calibration) > 0
