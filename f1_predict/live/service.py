from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import exp
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    race_state: dict[str, Any]
    note: str
    replay_available: bool


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
            item
            for item in sessions
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
        return (
            max(completed, key=lambda item: item.get("date_start", ""))
            if completed
            else None
        )

    def _select_latest_by_driver(
        self,
        rows: list[dict[str, Any]],
        timestamp_key: str,
        target_time: datetime,
        driver_key: str = "driver_number",
    ) -> dict[int, dict[str, Any]]:
        """Select each driver's latest observation at or before target_time."""
        latest: dict[int, dict[str, Any]] = {}
        for item in rows:
            number = item.get(driver_key)
            timestamp = _parse_datetime(item.get(timestamp_key))
            if number is None or timestamp is None or timestamp > target_time:
                continue
            number = int(number)
            previous = latest.get(number)
            if previous is None or timestamp >= _parse_datetime(
                previous.get(timestamp_key)
            ):
                latest[number] = item
        return latest

    def _expected_laps(
        self,
        session: dict[str, Any],
        results: list[dict[str, Any]],
        laps: list[dict[str, Any]],
    ) -> tuple[int | None, str]:
        """Resolve race length from final results, then recent same-location history."""
        if results:
            counts = [
                int(item["number_of_laps"])
                for item in results
                if item.get("number_of_laps") is not None
            ]
            if counts:
                return max(counts), "OpenF1 session result"

        observed = [
            int(item["lap_number"])
            for item in laps
            if item.get("lap_number") is not None
        ]
        if observed and session.get("date_end"):
            end = _parse_datetime(session.get("date_end"))
            if end and end <= datetime.now(timezone.utc):
                return max(observed), "OpenF1 historical laps"

        # Do not perform additional network discovery inside a snapshot request.
        # A later cached circuit profile can supply this value without blocking
        # the live request path.
        return None, "Target unavailable in current snapshot"

    def _race_state(
        self,
        session: dict[str, Any],
        target_time: datetime,
        positions: list[dict[str, Any]],
        intervals: list[dict[str, Any]],
        laps: list[dict[str, Any]],
        stints: list[dict[str, Any]],
        weather: list[dict[str, Any]],
        results: list[dict[str, Any]],
    ) -> tuple[dict[int, int], dict[int, dict[str, Any]], dict[str, Any]]:
        latest_positions = self._select_latest_by_driver(
            positions, "date", target_time
        )
        latest_intervals = self._select_latest_by_driver(
            intervals, "date", target_time
        )

        latest_laps = self._select_latest_by_driver(
            laps, "date_start", target_time
        )
        current_lap = (
            max(
                int(item["lap_number"])
                for item in latest_laps.values()
                if item.get("lap_number") is not None
            )
            if latest_laps
            else 0
        )

        lap_target, target_source = self._expected_laps(session, results, laps)
        laps_to_go = (
            max(lap_target - current_lap, 0)
            if lap_target is not None and current_lap
            else None
        )

        latest_stints: dict[int, dict[str, Any]] = {}
        for item in stints:
            number = item.get("driver_number")
            lap_start = item.get("lap_start")
            if number is None or lap_start is None or int(lap_start) > current_lap:
                continue
            number = int(number)
            previous = latest_stints.get(number)
            if previous is None or int(item["lap_start"]) > int(previous["lap_start"]):
                latest_stints[number] = item

        latest_weather: dict[str, Any] | None = None
        for item in weather:
            timestamp = _parse_datetime(item.get("date"))
            if timestamp is None or timestamp > target_time:
                continue
            if latest_weather is None or timestamp >= _parse_datetime(
                latest_weather.get("date")
            ):
                latest_weather = item

        driver_state: dict[int, dict[str, Any]] = {}
        for number, position_item in latest_positions.items():
            interval_item = latest_intervals.get(number, {})
            stint = latest_stints.get(number, {})
            tyre_age = None
            if stint.get("tyre_age_at_start") is not None and stint.get("lap_start") is not None:
                tyre_age = (
                    int(stint["tyre_age_at_start"])
                    + max(current_lap - int(stint["lap_start"]), 0)
                )
            driver_state[number] = {
                "gap_to_leader": interval_item.get("gap_to_leader"),
                "interval": interval_item.get("interval"),
                "compound": stint.get("compound"),
                "tyre_age": tyre_age,
            }

        race_state = {
            "current_lap": current_lap or None,
            "total_laps": lap_target,
            "laps_to_go": laps_to_go,
            "lap_target_source": target_source,
            "snapshot_time": target_time.isoformat(),
            "weather": (
                {
                    "air_temperature": latest_weather.get("air_temperature"),
                    "track_temperature": latest_weather.get("track_temperature"),
                    "humidity": latest_weather.get("humidity"),
                    "rainfall": latest_weather.get("rainfall"),
                    "wind_speed": latest_weather.get("wind_speed"),
                }
                if latest_weather
                else None
            ),
        }
        return (
            {number: int(item["position"]) for number, item in latest_positions.items()},
            driver_state,
            race_state,
        )

    def snapshot(self, replay_offset_seconds: float | None = None) -> RaceSnapshot:
        """Return live state or a point-in-time historical replay snapshot."""
        session: dict[str, Any] | None = None
        try:
            session = self._find_race_session()
            if session is None:
                return self._replay_fallback("No OpenF1 race session found.")

            session_key = int(session["session_key"])
            positions = self.client.get_positions(session_key)
            try:
                drivers = self.client.get_drivers(session_key)
            except Exception:
                return self._replay_fallback(
                    "OpenF1 driver metadata unavailable; model fallback active.",
                    session=session,
                )

            optional_fetchers = {
                "intervals": self.client.get_intervals,
                "laps": self.client.get_laps,
                "stints": self.client.get_stints,
                "weather": self.client.get_weather,
                "results": self.client.get_session_result,
            }
            optional_data: dict[str, list[dict[str, Any]]] = {
                key: [] for key in optional_fetchers
            }
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {
                    executor.submit(fetcher, session_key): key
                    for key, fetcher in optional_fetchers.items()
                }
                for future in as_completed(futures):
                    key = futures[future]
                    try:
                        optional_data[key] = future.result()
                    except Exception:
                        # Optional race-state signals must never block the dashboard.
                        optional_data[key] = []

            intervals = optional_data["intervals"]
            laps = optional_data["laps"]
            stints = optional_data["stints"]
            weather = optional_data["weather"]
            results = optional_data["results"]

            now = datetime.now(timezone.utc)
            start = _parse_datetime(session.get("date_start"))
            end = _parse_datetime(session.get("date_end"))
            is_live = bool(start and end and start <= now <= end)

            if is_live:
                target_time = now
            elif start and replay_offset_seconds is not None:
                max_offset = max(
                    0.0,
                    (
                        (end - start).total_seconds()
                        if end and end > start
                        else 0.0
                    ),
                )
                target_time = start + pd.to_timedelta(
                    min(max(replay_offset_seconds, 0.0), max_offset), unit="s"
                ).to_pytimedelta()
            else:
                target_time = end or now

            latest_positions, driver_state, race_state = self._race_state(
                session,
                target_time,
                positions,
                intervals,
                laps,
                stints,
                weather,
                results,
            )
            driver_map = {int(item["driver_number"]): item for item in drivers}

            identity = self._driver_identity()
            code_to_prior: dict[str, float] = {}
            for driver_id, probability in self.prior_predictions.items():
                info = identity.get(str(driver_id), {})
                code = info.get("code")
                if code:
                    code_to_prior[str(code).upper()] = float(probability)

            if not latest_positions:
                return self._replay_fallback(
                    "OpenF1 returned no race positions; model fallback active.",
                    session=session,
                )

            raw: list[dict[str, Any]] = []
            field_size = len(latest_positions)
            for number, position in latest_positions.items():
                info = driver_map.get(number, {})
                code = str(info.get("name_acronym", "")).upper()
                prior = code_to_prior.get(code, 1.0 / field_size)
                state = driver_state.get(number, {})
                raw.append(
                    {
                        "driver_number": number,
                        "driver": info.get("full_name", f"Driver {number}"),
                        "code": code or str(number),
                        "position": position,
                        "prior_probability": prior,
                        "raw_probability": prior * live_position_weight(position),
                        "team": info.get("team_name"),
                        **state,
                    }
                )

            total = sum(item["raw_probability"] for item in raw)
            for item in raw:
                item["win_probability"] = (
                    item["raw_probability"] / total if total > 0 else 1.0 / len(raw)
                )

            raw.sort(key=lambda item: item["win_probability"], reverse=True)
            mode = "LIVE" if is_live and replay_offset_seconds is None else "REPLAY"
            note = (
                "Historical ML prior + current OpenF1 race-state evidence."
                if mode == "LIVE"
                else "Historical OpenF1 point-in-time replay + historical ML prior."
            )
            session_view = dict(session)
            session_view["replay_offset_seconds"] = (
                replay_offset_seconds if mode == "REPLAY" else None
            )
            return RaceSnapshot(
                mode=mode,
                source="OpenF1",
                session=session_view,
                updated_at=datetime.now(timezone.utc).isoformat(),
                drivers=raw,
                race_state=race_state,
                note=note,
                replay_available=True,
            )
        except Exception as exc:
            return self._replay_fallback(
                f"OpenF1 unavailable; model fallback active: {exc}",
                session=session,
            )

    def _replay_fallback(
        self,
        note: str,
        session: dict[str, Any] | None = None,
    ) -> RaceSnapshot:
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
                    "gap_to_leader": None,
                    "interval": None,
                    "compound": None,
                    "tyre_age": None,
                }
            )
        rows.sort(key=lambda item: item["win_probability"], reverse=True)
        fallback_session = dict(session) if session is not None else {
            "year": self.model_season,
            "session_name": "Historical model fallback",
            "session_key": None,
        }
        fallback_session.setdefault("year", self.model_season)
        fallback_session.setdefault("session_name", "Historical model fallback")
        fallback_session.setdefault("session_key", None)

        return RaceSnapshot(
            mode="FALLBACK",
            source="historical dataset",
            session=fallback_session,
            updated_at=datetime.now(timezone.utc).isoformat(),
            drivers=rows,
            race_state={
                "current_lap": None,
                "total_laps": None,
                "laps_to_go": None,
                "lap_target_source": "Unknown",
                "snapshot_time": datetime.now(timezone.utc).isoformat(),
                "weather": None,
            },
            note=note,
            replay_available=False,
        )


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.to_pydatetime().astimezone(timezone.utc)
