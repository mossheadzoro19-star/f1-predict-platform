import numpy as np
import pandas as pd
import pytest

import f1_predict.modeling.pre_race_walk_forward as walk_forward
from f1_predict.modeling.pre_race_model import FEATURE_COLUMNS


class _FixedProbabilityModel:
    def predict_proba(self, features):
        # Use the first synthetic feature to create distinct driver scores.
        # A constant model would tie every driver, correctly giving all of
        # them rank 1 under the production rank(method="min") semantics.
        positive = features.iloc[:, 0].to_numpy(dtype=float) / 3.0
        return np.column_stack([1.0 - positive, positive])


def _walk_forward_frame(seasons=(2020, 2021, 2022)):
    rows = []
    for season in seasons:
        for round_number in (1, 2):
            for driver_number in range(2):
                row = {
                    "season": season,
                    "round": round_number,
                    "driver_id": f"driver-{driver_number}",
                    "winner": int(driver_number == 1),
                }
                row.update(
                    {feature: float(driver_number + 1) for feature in FEATURE_COLUMNS}
                )
                rows.append(row)
    return pd.DataFrame(rows)


def test_walk_forward_validation_rejects_reserved_holdout():
    frame = _walk_forward_frame(seasons=(2020, 2021, 2022, 2025))

    with pytest.raises(ValueError, match="2025 is reserved"):
        walk_forward.validate_walk_forward_frame(frame, validation_years=(2021, 2022))


def test_walk_forward_requires_each_validation_season():
    frame = _walk_forward_frame(seasons=(2020, 2022))

    with pytest.raises(ValueError, match="Validation season 2021 is missing"):
        walk_forward.validate_walk_forward_frame(frame, validation_years=(2021, 2022))


def test_walk_forward_uses_only_prior_seasons_and_reports_each_fold(monkeypatch):
    frame = _walk_forward_frame()

    def fake_run_experiment(
        supplied_frame,
        train_end_year,
        validation_start_year,
        validation_end_year,
    ):
        assert validation_start_year == validation_end_year
        assert supplied_frame.loc[
            supplied_frame["season"] > train_end_year, "season"
        ].min() == validation_start_year
        return {}, {
            "logistic_regression": _FixedProbabilityModel(),
            "xgboost": _FixedProbabilityModel(),
        }

    monkeypatch.setattr(walk_forward, "run_experiment", fake_run_experiment)

    report, audit, probabilities = walk_forward.run_walk_forward(
        frame,
        validation_years=(2021, 2022),
    )

    assert [fold["validation_season"] for fold in report["folds"]] == [2021, 2022]
    assert report["folds"][0]["training_seasons"] == [2020]
    assert report["folds"][1]["training_seasons"] == [2020, 2021]
    assert report["validation_races"] == 4
    assert set(audit["season"]) == {2021, 2022}
    assert set(audit["model"]) == {
        "uniform",
        "logistic_regression",
        "xgboost",
    }
    assert len(audit) == 4 * 3
    assert len(probabilities) == 4 * 2 * 3
    assert set(probabilities["model"]) == {
        "uniform", "logistic_regression", "xgboost"
    }
    assert probabilities.groupby(["season", "round", "model"])[
        "predicted_win_probability"
    ].sum().round(10).eq(1.0).all()
    assert set(probabilities["probability_rank"]) == {1, 2}
    assert probabilities.groupby(["season", "round", "model"])[
        "actual_winner"
    ].sum().eq(1).all()
    assert set(report["combined_out_of_fold_metrics"]) == {
        "uniform",
        "logistic_regression",
        "xgboost",
    }


def test_walk_forward_years_must_be_unique_and_ascending():
    frame = _walk_forward_frame()

    with pytest.raises(ValueError, match="unique and in ascending order"):
        walk_forward.validate_walk_forward_frame(
            frame,
            validation_years=(2022, 2021),
        )
