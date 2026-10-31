from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from f1_predict.modeling.pre_race_model import evaluate_predictions

RACE_KEYS = ["season", "round"]
REQUIRED_COLUMNS = {
    "season", "round", "model", "driver_id", "actual_winner",
    "predicted_win_probability",
}
MAX_ALLOWED_SEASON = 2024


def validate_probability_frame(frame: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Probability data is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Probability data is empty")
    if frame["season"].astype(int).max() > MAX_ALLOWED_SEASON:
        raise ValueError("2025 is reserved and cannot be used in calibration")
    if frame.duplicated(["season", "round", "model", "driver_id"]).any():
        raise ValueError("Duplicate driver/race/model probability rows")
    probabilities = pd.to_numeric(
        frame["predicted_win_probability"], errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or (probabilities <= 0).any():
        raise ValueError("Probabilities must be finite and strictly positive")
    for key, race in frame.groupby(["season", "round", "model"], sort=False):
        if int(race["actual_winner"].sum()) != 1:
            raise ValueError(f"Race/model {key} must contain exactly one winner")
        if not np.isclose(
            race["predicted_win_probability"].sum(), 1.0, atol=1e-8
        ):
            raise ValueError(f"Race/model {key} probabilities must sum to one")


def _race_frame(frame: pd.DataFrame, probability_column: str) -> pd.DataFrame:
    return frame.rename(
        columns={
            "actual_winner": "winner",
            probability_column: "win_probability",
        }
    )[["season", "round", "driver_id", "winner", "win_probability"]].copy()


def _temperature_probabilities(frame: pd.DataFrame, temperature: float) -> pd.Series:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and greater than zero")
    values = frame["predicted_win_probability"].to_numpy(dtype=float)
    log_scores = np.log(np.clip(values, 1e-15, 1.0)) / temperature
    adjusted = np.empty_like(log_scores)
    working = frame[["season", "round"]].copy()
    working["_position"] = np.arange(len(working))
    for _, group in working.groupby(RACE_KEYS, sort=False):
        positions = group["_position"].to_numpy(dtype=int)
        scores = log_scores[positions]
        scores = scores - scores.max()
        exp_scores = np.exp(scores)
        adjusted[positions] = exp_scores / exp_scores.sum()
    return pd.Series(adjusted, index=frame.index, dtype=float)


def fit_temperature(frame: pd.DataFrame) -> tuple[float, dict[str, float]]:
    """Fit one scalar temperature by race-level NLL on calibration seasons only."""
    validate_probability_frame(frame)
    if frame["model"].nunique() != 1:
        raise ValueError("Fit temperature on exactly one model at a time")
    log_temperatures = np.linspace(-3.0, 3.0, 1201)
    race_data = _race_frame(frame, "predicted_win_probability")
    baseline_loss = evaluate_predictions(race_data)["log_loss"]
    best_loss = float("inf")
    best_temperature = 1.0
    for log_temperature in log_temperatures:
        temperature = float(np.exp(log_temperature))
        candidate = frame.copy()
        candidate["calibrated_probability"] = _temperature_probabilities(
            candidate, temperature
        )
        metrics = evaluate_predictions(
            _race_frame(candidate, "calibrated_probability")
        )
        loss = metrics["log_loss"]
        if loss < best_loss - 1e-12 or (
            abs(loss - best_loss) <= 1e-12
            and abs(log_temperature) < abs(np.log(best_temperature))
        ):
            best_loss = loss
            best_temperature = temperature
    return best_temperature, {
        "fit_log_loss_before": float(baseline_loss),
        "fit_log_loss_after": float(best_loss),
    }


def run_calibration_experiment(
    probabilities: pd.DataFrame,
    fit_seasons: tuple[int, ...] = (2021, 2022),
    evaluation_seasons: tuple[int, ...] = (2023, 2024),
) -> tuple[dict, pd.DataFrame]:
    validate_probability_frame(probabilities)
    if set(fit_seasons) & set(evaluation_seasons):
        raise ValueError("Calibration and evaluation seasons must not overlap")
    if max((*fit_seasons, *evaluation_seasons)) > MAX_ALLOWED_SEASON:
        raise ValueError("2025 is reserved and cannot be used in calibration")
    if not fit_seasons or not evaluation_seasons:
        raise ValueError("Fit and evaluation season lists must both be non-empty")
    xgb = probabilities.loc[probabilities["model"] == "xgboost"].copy()
    if xgb.empty:
        raise ValueError("No XGBoost probabilities were found")
    available = set(xgb["season"].astype(int).unique())
    missing = (set(fit_seasons) | set(evaluation_seasons)) - available
    if missing:
        raise ValueError(f"Required seasons are missing: {sorted(missing)}")
    fit = xgb[xgb["season"].isin(fit_seasons)].copy()
    evaluation = xgb[xgb["season"].isin(evaluation_seasons)].copy()
    temperature, fit_metrics = fit_temperature(fit)
    evaluation["calibrated_win_probability"] = _temperature_probabilities(
        evaluation, temperature
    )
    results = {}
    for label, subset in [
        ("combined", evaluation),
        *[(str(year), evaluation[evaluation["season"] == year])
          for year in evaluation_seasons],
    ]:
        baseline = evaluate_predictions(
            _race_frame(subset, "predicted_win_probability")
        )
        calibrated = evaluate_predictions(
            _race_frame(subset, "calibrated_win_probability")
        )
        results[label] = {
            "races": baseline["races"],
            "baseline": baseline,
            "temperature_scaled": calibrated,
        }
    report = {
        "model": "xgboost",
        "fit_seasons": list(fit_seasons),
        "evaluation_seasons": list(evaluation_seasons),
        "temperature": float(temperature),
        "fit_diagnostics": fit_metrics,
        "evaluation": results,
        "interpretation": (
            "Temperature scaling preserves within-race driver ranking. "
            "Winner accuracy should remain unchanged; Log Loss and Brier "
            "measure probability-quality changes."
        ),
        "holdout_policy": (
            "Only 2021-2024 out-of-fold probabilities are used. "
            "The 2025 final holdout is excluded."
        ),
    }
    return report, evaluation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit temperature scaling on 2021-2022; evaluate on 2023-2024."
    )
    parser.add_argument(
        "--input", type=Path,
        default=Path("reports/pre_race_walk_forward/driver_probabilities.csv"),
    )
    parser.add_argument(
        "--report-output", type=Path,
        default=Path("reports/pre_race_calibration/metrics.json"),
    )
    parser.add_argument(
        "--predictions-output", type=Path,
        default=Path("reports/pre_race_calibration/evaluation_probabilities.csv"),
    )
    args = parser.parse_args()
    probabilities = pd.read_csv(args.input)
    report, evaluation = run_calibration_experiment(probabilities)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    evaluation.to_csv(args.predictions_output, index=False)
    print("XGBOOST TEMPERATURE CALIBRATION")
    print(f"Fit seasons: {report['fit_seasons']}")
    print(f"Evaluation seasons: {report['evaluation_seasons']}")
    print(f"Fitted temperature: {report['temperature']:.4f}")
    print("\nEvaluation metrics")
    for season, values in report["evaluation"].items():
        print(f"  {season}:")
        for method in ("baseline", "temperature_scaled"):
            metrics = values[method]
            print(
                f"    {method:<20} log_loss={metrics['log_loss']:.4f} "
                f"brier={metrics['multiclass_brier_score']:.4f} "
                f"accuracy={metrics['winner_accuracy']:.4f}"
            )
    print(f"Metrics report: {args.report_output}")
    print(f"Calibrated probabilities: {args.predictions_output}")
    print("2025 was not loaded or evaluated.")


if __name__ == "__main__":
    main()
