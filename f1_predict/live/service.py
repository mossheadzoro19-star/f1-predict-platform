from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import exp
from pathlib import Path
from typing import Any

import pandas as pd

from f1_predict.data_sources.openf1 import OpenF1Client
from f1_predict.modeling.baseline import fit_baseline, predict_race_probabilities


@dataclass(frozen=True)
class RaceSnapshot:
    """A frontend-ready race state and probability snapshot."""

    mode: str
    source: str
    session: dict[str, Any]
    updated_at: str
    drivers: list[dict[str, Any]]
    note: str


def live_position_weight(position: int | None) -> float:
    """Convert current track position into transparent live-state evidence."""
    if position is None or position < 1:
        return 1.0
    return exp(-0.55 * (position - 1))


class RaceIntelligenceService:
    """Combine a historical ML prior with current or replayed race state."""

    def __init__(self, feature_path: Path | None = None) -> None:
        self.feature_path = feature_path or Path(
            "data/processed/features/race_driver_features.parquet"
        )
        self.client = OpenF1Client()
        self.features = pd.read_parquet(self.feature_path)
        self.model_season = int(self.features["season"].max())
        self.prior_predictions = self._build_historical_prior()

    def _build_historical_prior(self) -> dict[str, float]:
        """Train on earlier seasons and predict the final race of the latest season."""
        train = self.features[self.features["season"] < self.model_season]
        latest = self.features[self.features["season"] == self.model_season]
        latest_round = int(latest["round"].max())
        snapshot = latest[latest["round"] == latest_round].copy()

        model = fit_baseline(train)
        predictions = predict_race_probabilities(model, snapshot)
        return dict(predictions.set_index("driver_id")["win_probability"])

    def _driver_identity(self) -> dict[str, dict[str, Any]]:
        """Load driver codes and names from normalized Jolpica data."""
        path = Path(f"data/processed/jolpica/{self.model_season}/drivers.parquet")
        if not path.exists():
            return {}
        drivers = pd.read_parquet(path)
        return {
            str(row.driver_id): {
                "code": row.code,
                "name": f"{row.given_name} {row.family_name}",
            }
            for row in drivers.itertuples()
        }

    def _find_race_session(self) -> dict[str, Any] | None:
        """Find the current race, otherwise the latest completed race for replay."""
        now = datetime.now(timezone.utc)
        sessions = self.client.get_sessions(now.year)
        races = [
            item for item in sessions
            if item.get("session_name") == "Race"
            and not item.get("is_cancelled", False)
        ]
        if not races:
            return None

        for session in races:
            start = _parse_datetime(session.get("date_start"))
            end = _parse_datetime(session.get("date_end"))
            if start and end and start <= now <= end:
                return session

        completed = [
            session
            for session in races
            if (start := _parse_datetime(session.get("date_start"))) and start <= now
        ]
        return max(completed, key=lambda item: item.get("date_start", "")) if completed else None

    def snapshot(self) -> RaceSnapshot:
        """Return live state when possible, otherwise a deterministic replay."""
        try:
            session = self._find_race_session()
            if session is None:
                return self._replay_fallback("No OpenF1 race session found.")

            session_key = int(session["session_key"])
            positions = self.client.get_positions(session_key)
            drivers = self.client.get_drivers(session_key)
            driver_map = {int(item["driver_number"]): item for item in drivers}

            now = datetime.now(timezone.utc)
            start = _parse_datetime(session.get("date_start"))
            end = _parse_datetime(session.get("date_end"))
            is_live = bool(start and end and start <= now <= end)

            identity = self._driver_identity()
            code_to_prior: dict[str, float] = {}
            for driver_id, probability in self.prior_predictions.items():
                info = identity.get(str(driver_id), {})
                code = info.get("code")
                if code:
                    code_to_prior[str(code).upper()] = float(probability)

            latest_positions: dict[int, int] = {}
            for item in positions:
                number = item.get("driver_number")
                position = item.get("position")
                if number is not None and position is not None:
                    latest_positions[int(number)] = int(position)

            if not latest_positions:
                return self._replay_fallback("OpenF1 returned no race positions.")

            raw: list[dict[str, Any]] = []
            field_size = len(latest_positions)
            for number, position in latest_positions.items():
                info = driver_map.get(number, {})
                code = str(info.get("name_acronym", "")).upper()
                prior = code_to_prior.get(code, 1.0 / field_size)
                raw.append(
                    {
                        "driver_number": number,
                        "driver": info.get("full_name", f"Driver {number}"),
                        "code": code or str(number),
                        "position": position,
                        "prior_probability": prior,
                        "raw_probability": prior * live_position_weight(position),
                        "team": info.get("team_name"),
                    }
                )

            total = sum(item["raw_probability"] for item in raw)
            for item in raw:
                item["win_probability"] = (
                    item["raw_probability"] / total if total > 0 else 1.0 / len(raw)
                )

            raw.sort(key=lambda item: item["win_probability"], reverse=True)
            mode = "LIVE" if is_live else "REPLAY"
            note = (
                "Historical ML prior + current OpenF1 race-position evidence."
                if is_live
                else "Historical OpenF1 race replay + historical ML prior."
            )
            return RaceSnapshot(
                mode=mode,
                source="OpenF1",
                session=session,
                updated_at=datetime.now(timezone.utc).isoformat(),
                drivers=raw,
                note=note,
            )
        except Exception as exc:
            return self._replay_fallback(
                f"Live provider unavailable; replay fallback active: {exc}"
            )

    def _replay_fallback(self, note: str) -> RaceSnapshot:
        """Serve the latest historical model prediction when live data is unavailable."""
        latest = self.features[
            self.features["season"].eq(self.model_season)
            & self.features["round"].eq(self.features["round"].max())
        ].copy()
        identity = self._driver_identity()
        rows = []
        for row in latest.itertuples():
            info = identity.get(str(row.driver_id), {})
            probability = float(self.prior_predictions.get(str(row.driver_id), 0.0))
            rows.append(
                {
                    "driver_number": None,
                    "driver": info.get("name", str(row.driver_id)),
                    "code": info.get("code") or str(row.driver_id),
                    "position": int(row.grid) if pd.notna(row.grid) else None,
                    "prior_probability": probability,
                    "win_probability": probability,
                    "team": None,
                }
            )
        rows.sort(key=lambda item: item["win_probability"], reverse=True)
        return RaceSnapshot(
            mode="REPLAY",
            source="historical dataset",
            session={
                "year": self.model_season,
                "session_name": "Historical pre-race replay",
                "session_key": None,
            },
            updated_at=datetime.now(timezone.utc).isoformat(),
            drivers=rows,
            note=note,
        )


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.to_pydatetime().astimezone(timezone.utc)
