from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


PRIOR_FEATURES = [
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


def load_feature_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load the generated point-in-time feature dataset."""
    path = path or ROOT / "data" / "processed" / "features" / "race_driver_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Feature dataset not found: {path}")
    return pd.read_parquet(path)


def check_unique_driver_race_keys(frame: pd.DataFrame) -> None:
    """Every driver may appear at most once in a race."""
    keys = ["season", "round", "driver_id"]
    if frame.duplicated(keys).any():
        duplicates = frame.loc[frame.duplicated(keys, keep=False), keys]
        raise AssertionError(f"Duplicate driver-race keys found:\n{duplicates}")


def check_one_winner_per_race(frame: pd.DataFrame) -> None:
    """Each completed historical race must have exactly one recorded winner."""
    winners = frame.groupby(["season", "round"])["winner"].sum()
    invalid = winners[winners != 1]
    if not invalid.empty:
        raise AssertionError(f"Races without exactly one winner: {invalid.to_dict()}")


def check_first_start_features(frame: pd.DataFrame) -> None:
    """A driver's first observed race must have zero prior starts."""
    ordered = frame.sort_values(["prediction_time", "season", "round", "driver_id"])
    first_rows = ordered.groupby("driver_id", sort=False).head(1)

    if not (first_rows["driver_prior_starts"] == 0).all():
        bad = first_rows.loc[
            first_rows["driver_prior_starts"] != 0,
            ["season", "round", "driver_id", "driver_prior_starts"],
        ]
        raise AssertionError(f"First driver starts have non-zero prior starts:\n{bad}")


def check_prior_counts_monotonic(frame: pd.DataFrame) -> None:
    """Prior-start counters must equal the number of earlier rows for each driver."""
    ordered = frame.sort_values(
        ["prediction_time", "season", "round", "driver_id"]
    ).reset_index(drop=True)

    expected = ordered.groupby("driver_id", sort=False).cumcount()
    actual = ordered["driver_prior_starts"].astype("int64")

    mismatches = ordered.loc[
        actual.ne(expected),
        [
            "season",
            "round",
            "driver_id",
            "prediction_time",
            "driver_prior_starts",
        ],
    ].copy()

    if not mismatches.empty:
        mismatches["expected_prior_starts"] = expected.loc[mismatches.index].to_numpy()
        first = mismatches.iloc[0]
        raise AssertionError(
            "driver_prior_starts is inconsistent with the global chronological "
            "history. This usually means the feature Parquet was generated "
            "before the cross-season history fix. "
            f"First mismatch: driver={first['driver_id']}, "
            f"season={first['season']}, round={first['round']}, "
            f"actual={first['driver_prior_starts']}, "
            f"expected={first['expected_prior_starts']}"
        )


def check_prior_features_absent_before_history(frame: pd.DataFrame) -> None:
    """A first-ever driver race cannot contain historical rolling/rate values."""
    ordered = frame.sort_values(["prediction_time", "season", "round", "driver_id"])
    first_rows = ordered.groupby("driver_id", sort=False).head(1)

    historical_cols = [
        column
        for column in PRIOR_FEATURES
        if column in first_rows.columns and column != "driver_prior_starts"
    ]

    for column in historical_cols:
        if first_rows[column].notna().any():
            bad = first_rows.loc[
                first_rows[column].notna(),
                ["season", "round", "driver_id", column],
            ]
            raise AssertionError(
                f"First driver race contains historical value in {column}:\n{bad}"
            )


def check_prediction_timestamps(frame: pd.DataFrame) -> None:
    """Prediction timestamps must exist and be chronological."""
    if frame["prediction_time"].isna().any():
        raise AssertionError("One or more prediction timestamps are missing")

    if not frame["prediction_time"].is_monotonic_increasing:
        raise AssertionError("Feature dataset is not globally chronological")


def run_leakage_audit(frame: pd.DataFrame) -> dict[str, str]:
    """Run the complete point-in-time feature integrity audit."""
    required = {
        "season",
        "round",
        "driver_id",
        "constructor_id",
        "circuit_id",
        "prediction_time",
        "winner",
        "driver_prior_starts",
        "constructor_prior_starts",
        "driver_circuit_prior_starts",
    }
    missing = required - set(frame.columns)
    if missing:
        raise AssertionError(f"Missing required leakage-audit columns: {sorted(missing)}")

    check_unique_driver_race_keys(frame)
    check_one_winner_per_race(frame)
    check_first_start_features(frame)
    check_prior_counts_monotonic(frame)
    check_prior_features_absent_before_history(frame)
    check_prediction_timestamps(frame)

    return {
        "driver_race_keys": "PASS",
        "one_winner_per_race": "PASS",
        "first_driver_start": "PASS",
        "prior_counts_monotonic": "PASS",
        "first_race_history": "PASS",
        "prediction_timestamps": "PASS",
        "leakage_audit": "PASS",
    }


def main() -> None:
    frame = load_feature_dataset()
    results = run_leakage_audit(frame)

    print("=" * 60)
    print("Point-in-Time Feature Leakage Audit")
    print("=" * 60)
    print(f"Rows:  {len(frame)}")
    print(f"Races: {frame[['season', 'round']].drop_duplicates().shape[0]}")
    print()
    for name, status in results.items():
        print(f"{name.replace('_', ' ').title():28} {status}")
    print()
    print("LEAKAGE AUDIT: PASS")


if __name__ == "__main__":
    main()
