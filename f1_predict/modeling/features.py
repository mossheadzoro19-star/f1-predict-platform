from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


TARGET_COLUMN = "winner"

# Explicit allow-list: post-race observations are deliberately excluded.
PRE_RACE_FEATURES = [
    "qualifying_position",
    "grid",
    "q1_seconds",
    "q2_seconds",
    "q3_seconds",
    "field_size",
    "driver_finish_position_last_3",
    "driver_finish_position_last_5",
    "driver_points_last_3",
    "driver_points_last_5",
    "driver_prior_win_rate",
    "driver_prior_starts",
    "constructor_points_last_3",
    "constructor_points_last_5",
    "constructor_prior_win_rate",
    "constructor_prior_starts",
    "driver_circuit_finish_last_3",
    "driver_circuit_finish_last_5",
    "driver_circuit_prior_starts",
]


POST_RACE_OR_TARGET_COLUMNS = [
    "winner",
    "position",
    "finish_position_numeric",
    "points",
    "laps",
    "status",
]


@dataclass(frozen=True)
class FeatureContract:
    """Formal contract for the pre-race winner-probability dataset."""

    features: tuple[str, ...]
    target: str
    excluded_post_race: tuple[str, ...]

    @classmethod
    def pre_race(cls) -> "FeatureContract":
        return cls(
            features=tuple(PRE_RACE_FEATURES),
            target=TARGET_COLUMN,
            excluded_post_race=tuple(POST_RACE_OR_TARGET_COLUMNS),
        )


def validate_feature_contract(frame: pd.DataFrame, contract: FeatureContract | None = None) -> None:
    """Reject missing predictors and any accidental post-race predictor columns."""
    contract = contract or FeatureContract.pre_race()

    missing = sorted(set(contract.features) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing pre-race feature columns: {missing}")

    forbidden = sorted(
        set(contract.excluded_post_race).intersection(contract.features)
    )
    if forbidden:
        raise ValueError(f"Feature contract contains forbidden post-race columns: {forbidden}")

    if contract.target not in frame.columns:
        raise ValueError(f"Missing target column: {contract.target}")

    if frame[list(contract.features)].columns.duplicated().any():
        raise ValueError("Duplicate feature columns detected")


def select_pre_race_dataset(
    frame: pd.DataFrame,
    contract: FeatureContract | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return X/y using only the explicit pre-race feature allow-list."""
    contract = contract or FeatureContract.pre_race()
    validate_feature_contract(frame, contract)

    X = frame.loc[:, list(contract.features)].copy()
    y = frame.loc[:, contract.target].astype("int8").copy()

    if y.isna().any():
        raise ValueError("Target contains missing values")

    if not y.isin([0, 1]).all():
        raise ValueError("Winner target must be binary")

    return X, y
