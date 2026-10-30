import numpy as np
import pandas as pd
import pytest

import f1_predict.modeling.race_level_experiment as experiment
from f1_predict.modeling.pre_race_model import FEATURE_COLUMNS
from f1_predict.modeling.race_level_model import ConditionalLogitRaceModel


def _frame(seasons=(2020, 2021, 2022), rounds=(1, 2)):
    rows = []
    for season in seasons:
        for round_number in rounds:
            for driver_number in range(3):
                row = {
                    "season": season,
                    "round": round_number,
                    "driver_id": f"driver-{driver_number}",
                    "winner": int(driver_number == (round_number - 1) % 3),
                }
                row.update({
                    feature: float(driver_number + 1) for feature in FEATURE_COLUMNS
                })
                rows.append(row)
    return pd.DataFrame(rows)


def test_conditional_logit_probabilities_sum_to_one_per_race():
    frame = _frame()
    model = ConditionalLogitRaceModel().fit(frame.query("season < 2022"))
    predictions = model.predict_race_probabilities(frame.query("season == 2022"))
    totals = predictions.groupby(["season", "round"])["win_probability"].sum()
    assert np.allclose(totals.to_numpy(), 1.0)
    assert predictions["win_probability"].between(0.0, 1.0).all()


def test_conditional_logit_reports_successful_optimization():
    model = ConditionalLogitRaceModel().fit(_frame())
    assert model.coef_ is not None
    assert len(model.coef_) == len(FEATURE_COLUMNS)
    assert model.optimization_["success"] is True


def test_conditional_logit_analytic_gradient_matches_finite_difference():
    model = ConditionalLogitRaceModel(l2_strength=0.2)
    matrix = np.array([
        [-1.0, 0.5],
        [0.2, -0.3],
        [0.8, 1.1],
        [-0.4, 0.7],
    ])
    groups = [np.array([0, 1]), np.array([2, 3])]
    targets = np.array([1.0, 0.0, 0.0, 1.0])
    beta = np.array([0.15, -0.25])
    _, analytic = model._objective(beta, matrix, groups, targets)

    epsilon = 1e-6
    numeric = np.zeros_like(beta)
    for index in range(len(beta)):
        step = np.zeros_like(beta)
        step[index] = epsilon
        plus = model._objective(beta + step, matrix, groups, targets)[0]
        minus = model._objective(beta - step, matrix, groups, targets)[0]
        numeric[index] = (plus - minus) / (2 * epsilon)

    assert np.allclose(analytic, numeric, rtol=1e-5, atol=1e-7)


def test_prediction_requires_fitted_model():
    with pytest.raises(RuntimeError, match="fitted"):
        ConditionalLogitRaceModel().predict_race_probabilities(_frame())


def test_conditional_logit_rejects_races_without_one_winner():
    frame = _frame()
    frame.loc[(frame["season"] == 2020) & (frame["round"] == 1), "winner"] = 0
    with pytest.raises(ValueError, match="exactly one winner"):
        ConditionalLogitRaceModel().fit(frame)


def test_conditional_logit_rejects_duplicate_driver_race_rows():
    frame = _frame()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        ConditionalLogitRaceModel().fit(frame)


def test_experiment_rejects_reserved_2025_holdout():
    frame = _frame(seasons=(2020, 2021, 2022, 2025))
    with pytest.raises(ValueError, match="2025 is reserved"):
        experiment.validate_experiment_frame(frame, validation_years=(2021, 2022))


def test_experiment_requires_prior_training_season():
    frame = _frame(seasons=(2021, 2022))
    with pytest.raises(ValueError, match="No earlier training seasons"):
        experiment.validate_experiment_frame(frame, validation_years=(2021, 2022))


def test_experiment_validation_years_must_be_sorted_unique():
    frame = _frame()
    with pytest.raises(ValueError, match="unique and in ascending order"):
        experiment.validate_experiment_frame(frame, validation_years=(2022, 2021))
