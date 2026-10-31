import numpy as np
import pandas as pd
import pytest

import f1_predict.modeling.pre_race_holdout as holdout
from f1_predict.modeling.pre_race_model import FEATURE_COLUMNS


class _FakeXGBoost:
    def fit(self, features, target, verbose=False):
        self.fit_rows = len(features)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, features):
        positive = np.where(features["grid"].to_numpy() == 1, 0.8, 0.2)
        return np.column_stack([1 - positive, positive])


def _frame(seasons=(2023, 2024, 2025)):
    rows = []
    for season in seasons:
        for rnd in (1, 2):
            for driver, grid in (("driver-a", 1), ("driver-b", 2)):
                row = {
                    "season": season,
                    "round": rnd,
                    "driver_id": driver,
                    "winner": int(driver == ("driver-a" if rnd == 1 else "driver-b")),
                }
                row.update({feature: float(grid) for feature in FEATURE_COLUMNS})
                rows.append(row)
    return pd.DataFrame(rows)


def test_final_holdout_trains_only_through_2024_and_scores_2025(monkeypatch):
    fake_model = _FakeXGBoost()
    monkeypatch.setattr(holdout, "_make_xgboost", lambda: fake_model)

    report, predictions = holdout.run_final_holdout(_frame())

    assert fake_model.fit_rows == 8
    assert report["training_seasons"] == [2023, 2024]
    assert report["holdout_season"] == 2025
    assert report["holdout_races"] == 2
    assert set(predictions["season"]) == {2025}
    assert len(predictions) == 4
    assert predictions.groupby(["season", "round"])[
        "predicted_win_probability"
    ].sum().round(10).eq(1.0).all()


def test_final_holdout_rejects_missing_2025():
    with pytest.raises(ValueError, match="2025 final holdout season is missing"):
        holdout.validate_holdout_frame(_frame(seasons=(2023, 2024)))


def test_final_holdout_rejects_seasons_after_2025():
    with pytest.raises(ValueError, match="after 2025"):
        holdout.validate_holdout_frame(_frame(seasons=(2023, 2024, 2025, 2026)))


def test_final_holdout_requires_training_through_2024():
    with pytest.raises(ValueError, match="extend through 2024"):
        holdout.validate_holdout_frame(_frame(seasons=(2023, 2025)))
