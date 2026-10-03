from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class LiveTemporalSplit:
    """Chronological train/validation/test partitions for race-state observations."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def chronological_live_race_split(
    frame: pd.DataFrame,
    train_end_year: int,
    validation_end_year: int,
) -> LiveTemporalSplit:
    """Split complete OpenF1 sessions by season without shuffling rows."""
    required = {"season", "session_key", "prediction_time"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing live split columns: {sorted(missing)}")

    if train_end_year >= validation_end_year:
        raise ValueError("train_end_year must be less than validation_end_year")

    ordered = frame.sort_values(
        ["prediction_time", "session_key", "driver_number"],
        ignore_index=True,
    )

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

    train_sessions = set(train["session_key"].unique())
    validation_sessions = set(validation["session_key"].unique())
    test_sessions = set(test["session_key"].unique())

    if (
        train_sessions & validation_sessions
        or train_sessions & test_sessions
        or validation_sessions & test_sessions
    ):
        raise AssertionError("A race session appears in more than one temporal partition")

    return LiveTemporalSplit(
        train=train,
        validation=validation,
        test=test,
    )
