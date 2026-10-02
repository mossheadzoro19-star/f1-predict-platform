import pandas as pd
import pytest

from f1_predict.modeling.features import (
    POST_RACE_OR_TARGET_COLUMNS,
    FeatureContract,
    select_pre_race_dataset,
)
from f1_predict.modeling.split import chronological_race_split


def _frame():
    rows = []
    for season in (2022, 2023, 2024, 2025):
        for round_ in (1, 2):
            for driver in ("a", "b"):
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


def test_pre_race_contract_excludes_post_race_columns():
    contract = FeatureContract.pre_race()
    assert not set(contract.features).intersection(POST_RACE_OR_TARGET_COLUMNS)


def test_select_pre_race_dataset_uses_explicit_allow_list():
    frame = _frame()
    frame["unexpected_numeric_column"] = 999
    X, y = select_pre_race_dataset(frame)

    assert list(X.columns) == list(FeatureContract.pre_race().features)
    assert "unexpected_numeric_column" not in X.columns
    assert y.tolist().count(1) == 4


def test_feature_contract_rejects_missing_predictor():
    frame = _frame().drop(columns=["grid"])
    with pytest.raises(ValueError, match="Missing pre-race feature"):
        select_pre_race_dataset(frame)


def test_feature_contract_rejects_post_race_predictor_if_added_to_contract():
    contract = FeatureContract(
        features=("points",),
        target="winner",
        excluded_post_race=POST_RACE_OR_TARGET_COLUMNS,
    )
    with pytest.raises(ValueError, match="forbidden post-race"):
        select_pre_race_dataset(_frame(), contract)


def test_chronological_split_keeps_races_disjoint():
    result = chronological_race_split(_frame(), train_end_year=2023, validation_end_year=2024)

    assert result.train["season"].max() == 2023
    assert result.validation["season"].unique().tolist() == [2024]
    assert result.test["season"].unique().tolist() == [2025]

    train_races = set(zip(result.train["season"], result.train["round"]))
    validation_races = set(zip(result.validation["season"], result.validation["round"]))
    test_races = set(zip(result.test["season"], result.test["round"]))

    assert train_races.isdisjoint(validation_races)
    assert train_races.isdisjoint(test_races)
    assert validation_races.isdisjoint(test_races)
