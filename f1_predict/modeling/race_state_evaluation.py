from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RaceStateMetrics:
    """Compact diagnostics for a normalized race-state winner model."""

    winner_accuracy: float
    log_loss: float
    brier_score: float
    snapshots: int
    races: int
    scorable_coverage: float
    current_leader_accuracy: float
    model_vs_leader_accuracy_delta: float


def _score_snapshot(group: pd.DataFrame) -> tuple[float, float, float, bool] | None:
    """Score one snapshot whose eventual winner is still in the candidate field."""
    actual = group.loc[group["winner"] == 1, "driver_number"]
    if len(actual) != 1:
        return None

    winner = actual.iloc[0]
    winner_probability = float(
        group.loc[group["driver_number"] == winner, "win_probability"].iloc[0]
    )
    predicted = group.loc[group["win_probability"].idxmax(), "driver_number"]
    current_leader = pd.to_numeric(
        group["position_position"], errors="coerce"
    ).idxmin()

    probabilities = group["win_probability"].to_numpy(dtype=float)
    target = group["winner"].to_numpy(dtype=float)
    brier = float(np.sum((probabilities - target) ** 2))
    log_loss = -float(np.log(np.clip(winner_probability, 1e-15, 1.0)))

    leader_position = group.loc[current_leader, "position_position"]
    leader_driver = group.loc[current_leader, "driver_number"]
    leader_valid = pd.notna(leader_position)

    return (
        float(predicted == winner),
        log_loss,
        brier,
        bool(leader_valid and leader_driver == winner),
    )


def evaluate_race_state_predictions(predictions: pd.DataFrame) -> RaceStateMetrics:
    """Evaluate normalized winner probabilities and compare them with current P1."""
    required = {
        "session_key",
        "lap_number",
        "driver_number",
        "winner",
        "win_probability",
        "position_position",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing evaluation columns: {sorted(missing)}")

    rows: list[tuple[float, float, float, bool]] = []
    total = 0
    for _, group in predictions.groupby(["session_key", "lap_number"], sort=False):
        total += 1
        scored = _score_snapshot(group)
        if scored is not None:
            rows.append(scored)

    if not rows:
        raise ValueError("No scorable race-state snapshots")

    values = np.asarray([row[:3] for row in rows], dtype=float)
    leader_hits = np.asarray([row[3] for row in rows], dtype=float)

    return RaceStateMetrics(
        winner_accuracy=float(values[:, 0].mean()),
        log_loss=float(values[:, 1].mean()),
        brier_score=float(values[:, 2].mean()),
        snapshots=len(rows),
        races=int(predictions["session_key"].nunique()),
        scorable_coverage=float(len(rows) / total),
        current_leader_accuracy=float(leader_hits.mean()),
        model_vs_leader_accuracy_delta=float(values[:, 0].mean() - leader_hits.mean()),
    )


def phase_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Break winner accuracy, LogLoss and Brier score into early/mid/late race."""
    frame = predictions.copy()
    frame["race_progress"] = frame.groupby("session_key")["lap_number"].transform(
        lambda s: s / max(float(s.max()), 1.0)
    )
    frame["phase"] = pd.cut(
        frame["race_progress"],
        bins=[-np.inf, 1 / 3, 2 / 3, np.inf],
        labels=["early", "mid", "late"],
    )

    rows: list[dict[str, Any]] = []
    for phase, phase_frame in frame.groupby("phase", observed=True):
        metrics = evaluate_race_state_predictions(phase_frame)
        rows.append(
            {
                "phase": str(phase),
                "snapshots": metrics.snapshots,
                "races": metrics.races,
                "winner_accuracy": metrics.winner_accuracy,
                "log_loss": metrics.log_loss,
                "brier_score": metrics.brier_score,
                "current_leader_accuracy": metrics.current_leader_accuracy,
                "model_vs_leader_accuracy_delta": metrics.model_vs_leader_accuracy_delta,
            }
        )
    return pd.DataFrame(rows)


def final_lap_diagnostics(predictions: pd.DataFrame) -> dict[str, float]:
    """Evaluate the last observed lap of each race separately from all snapshots."""
    frame = predictions.copy()
    max_lap = frame.groupby("session_key")["lap_number"].transform("max")
    final = frame[frame["lap_number"] == max_lap].copy()
    metrics = evaluate_race_state_predictions(final)
    return asdict(metrics)


def confidence_diagnostics(predictions: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Check whether the model's top probability corresponds to realized wins."""
    rows: list[dict[str, Any]] = []
    for _, group in predictions.groupby(["session_key", "lap_number"], sort=False):
        actual = group.loc[group["winner"] == 1, "driver_number"]
        if len(actual) != 1:
            continue
        top = group.loc[group["win_probability"].idxmax()]
        rows.append(
            {
                "confidence": float(top["win_probability"]),
                "correct": float(top["driver_number"] == actual.iloc[0]),
            }
        )

    if not rows:
        raise ValueError("No snapshots available for confidence diagnostics")

    frame = pd.DataFrame(rows)
    frame["confidence_bin"] = pd.cut(
        frame["confidence"],
        bins=np.linspace(0.0, 1.0, bins + 1),
        include_lowest=True,
    )
    return (
        frame.groupby("confidence_bin", observed=True)
        .agg(
            snapshots=("correct", "size"),
            mean_confidence=("confidence", "mean"),
            empirical_accuracy=("correct", "mean"),
        )
        .reset_index()
    )


def build_race_state_report(predictions: pd.DataFrame) -> dict[str, Any]:
    """Return one compact report suitable for JSON or notebook display."""
    overall = evaluate_race_state_predictions(predictions)
    return {
        "overall": asdict(overall),
        "phase": phase_diagnostics(predictions).to_dict(orient="records"),
        "final_lap": final_lap_diagnostics(predictions),
        "confidence": confidence_diagnostics(predictions).to_dict(orient="records"),
    }
