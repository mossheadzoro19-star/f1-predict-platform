from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


TARGET_COLUMN = "winner"

LIVE_NUMERIC_FEATURES = [
    "lap_number",
    "position_position",
    "interval_gap_to_leader",
    "interval_laps_behind_leader",
    "interval_interval",
    "interval_laps_behind_car_ahead",
    "stint_tyre_age_at_start",
    "position_change_1_lap",
    "position_change_3_laps",
    "position_change_5_laps",
    "gap_change_1_lap",
    "gap_change_3_laps",
    "gap_change_5_laps",
    "interval_change_1_lap",
    "interval_change_3_laps",
    "interval_change_5_laps",
    "previous_lap_duration",
    "previous_3_lap_mean",
    "pit_stops_completed",
    "laps_since_pit",
    "weather_air_temperature",
    "weather_track_temperature",
    "weather_humidity",
    "weather_rainfall",
    "weather_wind_speed",
]

LIVE_CATEGORICAL_FEATURES = [
    "stint_compound",
]


@dataclass(frozen=True)
class LiveFeatureContract:
    """Point-in-time feature contract for live race winner inference."""

    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    target: str

    @classmethod
    def default(cls) -> "LiveFeatureContract":
        return cls(
            numeric_features=tuple(LIVE_NUMERIC_FEATURES),
            categorical_features=tuple(LIVE_CATEGORICAL_FEATURES),
            target=TARGET_COLUMN,
        )

    @property
    def features(self) -> tuple[str, ...]:
        return self.numeric_features + self.categorical_features


def validate_live_feature_contract(
    frame: pd.DataFrame,
    contract: LiveFeatureContract | None = None,
) -> None:
    """Validate that all live predictors exist and are point-in-time safe."""
    contract = contract or LiveFeatureContract.default()
    required = set(contract.features) | {
        contract.target,
        "session_key",
        "driver_number",
        "prediction_time",
        "lap_number",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing live race-state columns: {missing}")

    if frame[list(contract.numeric_features)].columns.duplicated().any():
        raise ValueError("Duplicate numeric live feature columns detected")

    if frame[list(contract.categorical_features)].columns.duplicated().any():
        raise ValueError("Duplicate categorical live feature columns detected")

    target = pd.to_numeric(frame[contract.target], errors="coerce")
    if target.isna().any() or not target.isin([0, 1]).all():
        raise ValueError("Live winner target must be binary")

    if frame["prediction_time"].isna().any():
        raise ValueError("Live prediction_time contains missing values")


def select_live_dataset(
    frame: pd.DataFrame,
    contract: LiveFeatureContract | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return only the point-in-time live predictors and winner target."""
    contract = contract or LiveFeatureContract.default()
    validate_live_feature_contract(frame, contract)

    X = frame.loc[:, list(contract.features)].copy()
    y = frame.loc[:, contract.target].astype("int8").copy()

    # Treat tyre compound as a categorical value, not an ordinal number.
    # Use plain object dtype with np.nan for sklearn compatibility. Pandas\n    # StringDtype uses pd.NA, which can reach SimpleImputer and make comparisons\n    # such as ``value != value`` ambiguous in some sklearn/pandas combinations.\n    X["stint_compound"] = X["stint_compound"].astype("object")\n    X["stint_compound"] = X["stint_compound"].where(X["stint_compound"].notna(), None)

    # OpenF1 may represent missing numeric observations with nulls. The model
    # pipeline is responsible for imputation; we preserve missingness here.
    for column in contract.numeric_features:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    return X, y


def validate_race_snapshot_targets(frame: pd.DataFrame) -> None:
    """Ensure every race has exactly one eventual winning driver."""
    winners = (
        frame.loc[frame[TARGET_COLUMN] == 1]
        .groupby("session_key")["driver_number"]
        .nunique()
    )
    expected_races = frame["session_key"].nunique()
    if len(winners) != expected_races or not (winners == 1).all():
        invalid = winners[winners != 1]
        raise ValueError(
            "Each race must have exactly one unique winner driver. "
            f"Invalid races: {invalid.to_dict()}"
        )


def validate_temporal_order(frame: pd.DataFrame) -> None:
    """Reject backward prediction timestamps within the complete dataset."""
    ordered = frame.sort_values(
        ["prediction_time", "session_key", "driver_number"]
    )
    if (ordered["prediction_time"].diff().dropna() < pd.Timedelta(0)).any():
        raise ValueError("Prediction timestamps are not globally chronological.")


def summarize_live_dataset(frame: pd.DataFrame) -> dict[str, object]:
    """Return compact quality statistics for the OpenF1 race-state dataset."""
    validate_live_feature_contract(frame)
    validate_race_snapshot_targets(frame)
    validate_temporal_order(frame)

    return {
        "rows": int(len(frame)),
        "races": int(frame["session_key"].nunique()),
        "drivers": int(frame["driver_number"].nunique()),
        "seasons": sorted(frame["season"].unique().tolist()),
        "duplicate_snapshots": int(
            frame.duplicated(
                ["session_key", "driver_number", "lap_number", "prediction_time"]
            ).sum()
        ),
        "winner_drivers_per_race": int(
            frame.loc[frame[TARGET_COLUMN] == 1]
            .groupby("session_key")["driver_number"]
            .nunique()
            .max()
        ),
    }
