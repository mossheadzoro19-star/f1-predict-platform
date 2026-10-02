from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    """Chronological train/validation/test partitions."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def chronological_race_split(
    frame: pd.DataFrame,
    train_end_year: int,
    validation_end_year: int,
) -> TemporalSplit:
    """Split complete races chronologically by season without shuffling."""
    required = {"season", "round", "prediction_time"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing split columns: {sorted(missing)}")

    if train_end_year >= validation_end_year:
        raise ValueError("train_end_year must be less than validation_end_year")

    ordered = frame.sort_values(
        ["prediction_time", "season", "round", "driver_id"]
    ).reset_index(drop=True)

    train = ordered[ordered["season"] <= train_end_year].copy()
    validation = ordered[
        (ordered["season"] > train_end_year)
        & (ordered["season"] <= validation_end_year)
    ].copy()
    test = ordered[ordered["season"] > validation_end_year].copy()

    if train.empty or validation.empty or test.empty:
        raise ValueError(
            "Temporal split produced an empty partition. "
            f"Available seasons: {sorted(ordered['season'].unique().tolist())}"
        )

    train_races = set(zip(train["season"], train["round"]))
    validation_races = set(zip(validation["season"], validation["round"]))
    test_races = set(zip(test["season"], test["round"]))

    if train_races & validation_races or train_races & test_races or validation_races & test_races:
        raise AssertionError("A race appears in more than one temporal partition")

    return TemporalSplit(train=train, validation=validation, test=test)
