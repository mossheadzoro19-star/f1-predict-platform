from __future__ import annotations

"""Evaluate race-level temperature scaling on validation data only."""

import argparse

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.model_selection import GroupKFold

from f1_predict.modeling.live_features import LiveFeatureContract
from f1_predict.modeling.live_xgboost import (
    LiveXGBoostConfig,
    fit_live_xgboost,
    predict_live_probabilities,
)
from f1_predict.modeling.live_split import chronological_live_race_split
from f1_predict.modeling.race_state_evaluation import build_race_state_report


SNAPSHOT_KEYS = ["session_key", "lap_number"]


def _temperature_scale_predictions(
    predictions: pd.DataFrame,
    temperature: float,
) -> pd.DataFrame:
    """Apply one positive temperature and renormalize each race/lap distribution."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a finite positive number")

    result = predictions.copy()
    output = pd.Series(index=result.index, dtype="float64")
    for _, group in result.groupby(SNAPSHOT_KEYS, sort=False):
        probabilities = np.clip(
            group["win_probability"].to_numpy(dtype=float), 1e-15, 1.0
        )
        logits = np.log(probabilities) / temperature
        logits -= logits.max()
        scaled = np.exp(logits)
        scaled /= scaled.sum()
        output.loc[group.index] = scaled

    result["win_probability"] = output.astype(float)
    return result


def _multiclass_log_loss(predictions: pd.DataFrame) -> float:
    """Average negative log probability assigned to the winner per race/lap."""
    losses: list[float] = []
    for _, group in predictions.groupby(SNAPSHOT_KEYS, sort=False):
        winner = group.loc[group["winner"] == 1, "driver_number"]
        if len(winner) != 1:
            continue
        p = group.loc[
            group["driver_number"] == winner.iloc[0], "win_probability"
        ]
        if len(p) != 1:
            raise ValueError("Each snapshot must contain exactly one row for its winner")
        losses.append(-float(np.log(np.clip(float(p.iloc[0]), 1e-15, 1.0))))

    if not losses:
        raise ValueError("No scorable snapshots available for temperature fitting")
    return float(np.mean(losses))


def _fit_temperature(oof_predictions: pd.DataFrame) -> float:
    """Fit temperature only on out-of-fold predictions from training races."""
    def objective(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        scaled = _temperature_scale_predictions(oof_predictions, temperature)
        return _multiclass_log_loss(scaled)

    result = minimize_scalar(
        objective,
        method="bounded",
        bounds=(-3.0, 3.0),
        options={"xatol": 1e-3},
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError("Could not fit temperature scaling on out-of-fold predictions")
    return float(np.exp(result.x))


def _make_oof_predictions(
    train: pd.DataFrame,
    contract: LiveFeatureContract,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Generate predictions for each training race from a model that did not see it."""
    sessions = train["session_key"].drop_duplicates().to_numpy()
    folds = min(n_splits, len(sessions))
    if folds < 2:
        raise ValueError("At least two distinct training races are required for OOF calibration")

    splitter = GroupKFold(n_splits=folds)
    predictions: list[pd.DataFrame] = []
    for train_indices, heldout_indices in splitter.split(sessions, groups=sessions):
        fold_train_sessions = set(sessions[train_indices])
        fold_heldout_sessions = set(sessions[heldout_indices])
        fold_train = train[train["session_key"].isin(fold_train_sessions)]
        fold_heldout = train[train["session_key"].isin(fold_heldout_sessions)]

        model = fit_live_xgboost(
            fold_train,
            contract=contract,
            config=LiveXGBoostConfig(balance_classes=False),
        )
        predictions.append(predict_live_probabilities(model, fold_heldout, contract))

    oof = pd.concat(predictions, ignore_index=True)
    if oof["session_key"].nunique() != len(sessions):
        raise AssertionError("OOF predictions do not cover every training race exactly")
    return oof


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare raw and temperature-scaled unweighted XGBoost on validation only. "
            "Temperature is fitted on race-grouped out-of-fold training predictions; "
            "the test season is not evaluated."
        )
    )
    parser.add_argument(
        "--data",
        default="data/processed/features/jolpica_race_state_enriched.parquet",
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

    print(
        "VALIDATION-ONLY XGBOOST TEMPERATURE CALIBRATION\n"
        f"TRAIN: {len(split.train):,} rows, "
        f"{split.train['session_key'].nunique()} races\n"
        f"VALIDATION: {len(split.validation):,} rows, "
        f"{split.validation['session_key'].nunique()} races\n"
        "The held-out test season is not evaluated."
    )

    print("\nGenerating race-grouped out-of-fold predictions from training races...")
    oof_predictions = _make_oof_predictions(split.train, contract)
    temperature = _fit_temperature(oof_predictions)

    validation_model = fit_live_xgboost(
        split.train,
        contract=contract,
        config=LiveXGBoostConfig(balance_classes=False),
    )
    raw_predictions = predict_live_probabilities(
        validation_model, split.validation, contract
    )
    calibrated_predictions = _temperature_scale_predictions(
        raw_predictions, temperature
    )

    reports = {
        "RAW UNWEIGHTED": build_race_state_report(raw_predictions),
        "TEMPERATURE CALIBRATED": build_race_state_report(calibrated_predictions),
    }
    print(f"Fitted temperature: {temperature:.4f}")
    print("\nVALIDATION METRICS")
    print(
        f"{'Metric':<34}{'RAW':>12}{'CALIBRATED':>14}"
    )
    print("-" * 60)
    for metric in (
        "winner_accuracy",
        "current_leader_accuracy",
        "model_vs_leader_accuracy_delta",
        "log_loss",
        "brier_score",
    ):
        raw = reports["RAW UNWEIGHTED"]["overall"][metric]
        calibrated = reports["TEMPERATURE CALIBRATED"]["overall"][metric]
        print(f"{metric:<34}{raw:>12.4f}{calibrated:>14.4f}")

    for name, report in reports.items():
        print(f"\n{name} CONFIDENCE DIAGNOSTICS")
        print(
            pd.DataFrame(report["confidence"]).to_string(
                index=False, float_format=lambda value: f"{value:.4f}"
            )
        )
        print(f"\n{name} RACE-PHASE METRICS")
        print(
            pd.DataFrame(report["phase"])[
                ["phase", "snapshots", "winner_accuracy", "log_loss", "brier_score"]
            ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
        )


if __name__ == "__main__":
    main()
