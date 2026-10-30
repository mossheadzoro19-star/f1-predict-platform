import numpy as np
import pandas as pd
import pytest

import f1_predict.modeling.pre_race_ablation as ablation
from f1_predict.modeling.pre_race_model import FEATURE_COLUMNS, _race_probabilities


def _ablation_frame(seasons=(2020, 2021, 2022)):
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


def test_ablation_rejects_reserved_2025_holdout():
    frame = _ablation_frame(seasons=(2020, 2021, 2022, 2025))

    with pytest.raises(ValueError, match="2025 is reserved"):
        ablation.validate_ablation_frame(frame, validation_years=(2021, 2022))


def test_ablation_requires_validation_seasons():
    frame = _ablation_frame(seasons=(2020, 2022))

    with pytest.raises(ValueError, match="Validation season 2021 is missing"):
        ablation.validate_ablation_frame(frame, validation_years=(2021, 2022))


def test_feature_sets_are_valid_subsets_of_baseline():
    baseline = set(FEATURE_COLUMNS)

    assert set(ablation.FEATURE_SETS["full_baseline"]) == baseline
    assert all(set(columns).issubset(baseline) for columns in ablation.FEATURE_SETS.values())
    assert set(ablation.FEATURE_SETS["qualifying_only"]) == set(
        ablation.QUALIFYING_FEATURES
    )
    assert not (
        set(ablation.FEATURE_SETS["recent_form_qualifying"])
        & set(ablation.CAREER_FEATURES)
    )


def test_ablation_runs_identical_chronological_folds(monkeypatch):
    frame = _ablation_frame()
    calls = []

    def fake_fit_predict(train, validation, feature_columns, model_name):
        train_end = int(train["season"].max())
        calls.append((train_end, int(validation["season"].iloc[0]), model_name))
        scores = validation["driver_id"].map(
            {"driver-0": 0.4, "driver-1": 0.6}
        ).to_numpy()
        return _race_probabilities(validation, scores)

    monkeypatch.setattr(ablation, "_fit_predict", fake_fit_predict)

    report, audit = ablation.run_ablation(frame, validation_years=(2021, 2022))

    assert [fold["validation_season"] for fold in report["folds"]] == [2021, 2022]
    assert report["folds"][0]["train_end_year"] == 2020
    assert report["folds"][1]["train_end_year"] == 2021
    assert report["validation_races"] == 4
    assert len(calls) == len(ablation.FEATURE_SETS) * 2 * 2
    assert all(train_end < validation_year for train_end, validation_year, _ in calls)
    assert set(audit["feature_set"]) == set(ablation.FEATURE_SETS)
    assert set(audit["model"]) == {"logistic_regression", "xgboost"}
    assert len(audit) == 4 * len(ablation.FEATURE_SETS) * 2
    assert len(report["combined_out_of_fold_metrics"]) == len(ablation.FEATURE_SETS) * 2


def test_ablation_years_must_be_unique_and_ascending():
    frame = _ablation_frame()

    with pytest.raises(ValueError, match="unique and in ascending order"):
        ablation.validate_ablation_frame(frame, validation_years=(2022, 2021))
