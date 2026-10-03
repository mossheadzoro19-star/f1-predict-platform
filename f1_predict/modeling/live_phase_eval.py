from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from f1_predict.modeling.live_baseline import (
    LivePredictionMetrics,
    evaluate_live_probabilities,
    predict_live_probabilities,
)
from f1_predict.modeling.live_features import LiveFeatureContract
from f1_predict.modeling.live_split import chronological_live_race_split
from f1_predict.modeling.live_xgboost import fit_live_xgboost


DEFAULT_PHASES: tuple[tuple[str, int, int | None], ...] = (
    ("early", 1, 10),
    ("mid", 11, 30),
    ("late", 31, None),
)


@dataclass(frozen=True)
class RacePhaseMetrics:
    """Winner-probability performance for one fixed lap phase."""

    phase: str
    snapshots: int
    races: int
    winner_accuracy: float
    log_loss: float
    brier_score: float
    mean_winner_probability: float
    winner_probability_volatility: float


def assign_race_phase(
    lap_number: pd.Series,
    phases: tuple[tuple[str, int, int | None], ...] = DEFAULT_PHASES,
) -> pd.Series:
    """Assign fixed lap phases using only the current lap number."""
    if not phases:
        raise ValueError("At least one race phase is required")

    result = pd.Series(pd.NA, index=lap_number.index, dtype="string")
    numeric_laps = pd.to_numeric(lap_number, errors="coerce")

    for name, start, end in phases:
        if start < 1 or (end is not None and end < start):
            raise ValueError("Race phase bounds are invalid")
        mask = numeric_laps >= start
        if end is not None:
            mask &= numeric_laps <= end
        result.loc[mask] = name

    if result.isna().any():
        raise ValueError("Some lap numbers do not belong to a configured phase")

    return result


def _scorable_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Keep snapshots where the eventual winner is still in the observed field."""
    winner_count = predictions.groupby(
        ["session_key", "lap_number"]
    )["winner"].transform("sum")
    return predictions.loc[winner_count == 1].copy()


def evaluate_race_phases(
    predictions: pd.DataFrame,
    phases: tuple[tuple[str, int, int | None], ...] = DEFAULT_PHASES,
) -> pd.DataFrame:
    """Evaluate winner probabilities separately across fixed race phases."""
    required = {
        "session_key",
        "lap_number",
        "driver_number",
        "winner",
        "win_probability",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing phase-evaluation columns: {sorted(missing)}")

    frame = predictions.copy()
    frame["phase"] = assign_race_phase(frame["lap_number"], phases)
    rows: list[RacePhaseMetrics] = []

    for name, _, _ in phases:
        phase = _scorable_predictions(frame.loc[frame["phase"] == name])
        if phase.empty:
            continue

        metrics: LivePredictionMetrics = evaluate_live_probabilities(phase)

        winner_rows = phase.loc[phase["winner"] == 1].copy()
        winner_rows = winner_rows.sort_values(
            ["session_key", "lap_number", "prediction_time"]
            if "prediction_time" in winner_rows.columns
            else ["session_key", "lap_number"]
        )
        winner_probability = (
            winner_rows.groupby(["session_key", "lap_number"])["win_probability"]
            .first()
            .reset_index()
        )
        winner_probability = winner_probability.sort_values(
            ["session_key", "lap_number"]
        )
        volatility = (
            winner_probability.groupby("session_key")["win_probability"]
            .diff()
            .abs()
            .mean()
        )

        rows.append(
            RacePhaseMetrics(
                phase=name,
                snapshots=metrics.snapshots,
                races=metrics.races,
                winner_accuracy=metrics.winner_accuracy,
                log_loss=metrics.log_loss,
                brier_score=metrics.brier_score,
                mean_winner_probability=float(
                    winner_probability["win_probability"].mean()
                ),
                winner_probability_volatility=float(
                    volatility if pd.notna(volatility) else 0.0
                ),
            )
        )

    if not rows:
        raise ValueError("No scorable race phases available")

    return pd.DataFrame([row.__dict__ for row in rows])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate live XGBoost performance across race phases."
    )
    parser.add_argument(
        "--data",
        default="data/processed/features/openf1_race_state.parquet",
    )
    parser.add_argument("--train-end-year", type=int, default=2023)
    parser.add_argument("--validation-end-year", type=int, default=2024)
    args = parser.parse_args()

    frame = pd.read_parquet(args.data)
    split = chronological_live_race_split(
        frame,
        train_end_year=args.train_end_year,
        validation_end_year=args.validation_end_year,
    )
    contract = LiveFeatureContract.default()

    validation_model = fit_live_xgboost(split.train, contract)
    validation_predictions = predict_live_probabilities(
        validation_model,
        split.validation,
        contract,
    )
    validation_phases = evaluate_race_phases(validation_predictions)

    final_train = pd.concat([split.train, split.validation], ignore_index=True)
    final_model = fit_live_xgboost(final_train, contract)
    test_predictions = predict_live_probabilities(
        final_model,
        split.test,
        contract,
    )
    test_phases = evaluate_race_phases(test_predictions)

    print(
        "LIVE XGBOOST RACE-PHASE EVALUATION\n"
        f"TRAIN: {len(split.train)} rows, {split.train['session_key'].nunique()} races\n"
        f"VALIDATION: {len(split.validation)} rows, {split.validation['session_key'].nunique()} races\n"
        f"TEST: {len(split.test)} rows, {split.test['session_key'].nunique()} races\n"
    )
    print(
        f"{'Phase':<8} {'Snapshots':>9} {'Races':>6} "
        f"{'Val Acc':>9} {'Val LL':>9} {'Val Brier':>10} "
        f"{'Val Pwin':>9} {'Val Vol':>9}"
    )
    print("-" * 86)
    validation_by_phase = validation_phases.set_index("phase")
    for phase in validation_by_phase.index:
        row = validation_by_phase.loc[phase]
        print(
            f"{phase:<8} {int(row.snapshots):>9} {int(row.races):>6} "
            f"{row.winner_accuracy:>9.4f} {row.log_loss:>9.4f} "
            f"{row.brier_score:>10.4f} {row.mean_winner_probability:>9.4f} "
            f"{row.winner_probability_volatility:>9.4f}"
        )

    print("\nTEST")
    print(
        f"{'Phase':<8} {'Snapshots':>9} {'Races':>6} "
        f"{'Test Acc':>9} {'Test LL':>9} {'Test Brier':>10} "
        f"{'Test Pwin':>9} {'Test Vol':>9}"
    )
    print("-" * 86)
    test_by_phase = test_phases.set_index("phase")
    for phase in test_by_phase.index:
        row = test_by_phase.loc[phase]
        print(
            f"{phase:<8} {int(row.snapshots):>9} {int(row.races):>6} "
            f"{row.winner_accuracy:>9.4f} {row.log_loss:>9.4f} "
            f"{row.brier_score:>10.4f} {row.mean_winner_probability:>9.4f} "
            f"{row.winner_probability_volatility:>9.4f}"
        )


if __name__ == "__main__":
    main()
