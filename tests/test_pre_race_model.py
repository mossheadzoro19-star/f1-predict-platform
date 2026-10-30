import numpy as np
import pandas as pd
import pytest

from f1_predict.modeling.pre_race_model import (
    FEATURE_COLUMNS,
    chronological_split,
    evaluate_predictions,
    uniform_predictions,
    validate_dataset,
)


def _frame():
    rows = []
    for season in (2020, 2021, 2022):
        for round_number in (1, 2):
            for driver_number in range(3):
                row = {
                    "season": season,
                    "round": round_number,
                    "driver_id": f"driver-{driver_number}",
                    "winner": int(driver_number == (round_number - 1)),
                }
                row.update({feature: float(driver_number + 1) for feature in FEATURE_COLUMNS})
                rows.append(row)
    return pd.DataFrame(rows)


def test_dataset_validation_and_chronological_split():
    frame = _frame()
    validate_dataset(frame)

    train, validation = chronological_split(
        frame,
        train_end_year=2021,
        validation_start_year=2022,
        validation_end_year=2022,
    )

    assert set(train["season"]) == {2020, 2021}
    assert set(validation["season"]) == {2022}
    assert train["season"].max() < validation["season"].min()


def test_uniform_predictions_sum_to_one_for_every_race():
    predictions = uniform_predictions(_frame())

    totals = predictions.groupby(["season", "round"])["win_probability"].sum()
    assert np.allclose(totals.to_numpy(), 1.0)
    assert predictions.groupby(["season", "round"])["win_probability"].nunique().eq(1).all()


def test_evaluation_scores_uniform_three_driver_distribution():
    predictions = uniform_predictions(_frame().query("season == 2022"))
    metrics = evaluate_predictions(predictions)

    assert metrics["races"] == 2
    assert metrics["log_loss"] == pytest.approx(np.log(3))
    assert metrics["multiclass_brier_score"] == pytest.approx(2 / 3)


def test_dataset_validation_rejects_duplicate_rows():
    frame = _frame()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate"):
        validate_dataset(frame)


def test_dataset_validation_rejects_races_without_exactly_one_winner():
    frame = _frame()
    frame.loc[(frame["season"] == 2020) & (frame["round"] == 1), "winner"] = 0

    with pytest.raises(ValueError, match="exactly one winner"):
        validate_dataset(frame)
