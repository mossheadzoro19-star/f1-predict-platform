from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import httpx


@dataclass(frozen=True)
class JolpicaClient:
    """Adapter around Jolpica-F1's Ergast-compatible REST API."""

    base_url: str = "https://api.jolpi.ca/ergast/f1"
    user_agent: str = "F1PredictPlatform/0.1"
    request_delay_seconds: float = 1.0
    max_retries: int = 4

    def _get_json(
        self,
        client: httpx.Client,
        path: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Fetch one Jolpica page, retrying rate-limit responses with backoff."""
        url = f"{self.base_url}/{path.lstrip('/')}"
        params = {"limit": limit, "offset": offset}

        for attempt in range(self.max_retries + 1):
            if self.request_delay_seconds > 0:
                time.sleep(self.request_delay_seconds)

            response = client.get(
                url,
                params=params,
                headers={"User-Agent": self.user_agent},
            )

            if response.status_code != 429:
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise TypeError(f"Unexpected Jolpica response from {url}")
                return payload

            if attempt == self.max_retries:
                response.raise_for_status()

            retry_after = response.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                delay = float(retry_after)
            else:
                delay = min(2**attempt, 30)

            time.sleep(delay)

        raise RuntimeError("Unreachable Jolpica retry state")

    @staticmethod
    def _page_item_count(
        path: str,
        page_races: list[dict[str, Any]],
    ) -> int:
        """Count the records represented by a page for correct offset pagination."""
        if path.endswith("/results/"):
            return sum(
                len(race.get("Results", []))
                for race in page_races
                if isinstance(race.get("Results", []), list)
            )

        if path.endswith("/qualifying/"):
            return sum(
                len(race.get("QualifyingResults", []))
                for race in page_races
                if isinstance(race.get("QualifyingResults", []), list)
            )

        if path.endswith("/laps/"):
            # Jolpica's /laps/ endpoint paginates driver timing records,
            # while those records are nested inside Laps objects.
            return sum(
                sum(
                    len(lap.get("Timings", []))
                    for lap in race.get("Laps", [])
                    if isinstance(lap, dict)
                )
                for race in page_races
                if isinstance(race.get("Laps", []), list)
            )

        if path.endswith("/pitstops/"):
            return sum(
                len(race.get("PitStops", []))
                for race in page_races
                if isinstance(race.get("PitStops", []), list)
            )

        return len(page_races)

    @staticmethod
    def _merge_race_pages(
        path: str,
        race_pages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge nested records split across pagination pages."""
        if not (
            path.endswith("/results/")
            or path.endswith("/qualifying/")
            or path.endswith("/laps/")
            or path.endswith("/pitstops/")
        ):
            return race_pages

        collection_key = {
            "/results/": "Results",
            "/qualifying/": "QualifyingResults",
            "/laps/": "Laps",
            "/pitstops/": "PitStops",
        }[next(suffix for suffix in ("/results/", "/qualifying/", "/laps/", "/pitstops/") if path.endswith(suffix))]

        merged: dict[str, dict[str, Any]] = {}
        for race in race_pages:
            key = str(race.get("round") or race.get("raceName"))
            if key not in merged:
                merged[key] = dict(race)
                merged[key][collection_key] = list(race.get(collection_key, []))
                continue

            target = merged[key]
            target[collection_key].extend(race.get(collection_key, []))

        if collection_key == "Laps":
            for race in merged.values():
                by_lap: dict[str, dict[str, Any]] = {}
                for lap in race["Laps"]:
                    lap_key = str(lap.get("number"))
                    if lap_key not in by_lap:
                        by_lap[lap_key] = dict(lap)
                        by_lap[lap_key]["Timings"] = list(lap.get("Timings", []))
                    else:
                        by_lap[lap_key]["Timings"].extend(lap.get("Timings", []))
                race["Laps"] = sorted(
                    by_lap.values(),
                    key=lambda lap: int(lap.get("number", 0)),
                )

        return list(merged.values())

    def _get_all(self, path: str) -> dict[str, Any]:
        """Fetch every page and merge nested RaceTable records."""
        race_pages: list[dict[str, Any]] = []
        first_payload: dict[str, Any] | None = None
        offset = 0

        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            while True:
                payload = self._get_json(client, path, limit=100, offset=offset)
                if first_payload is None:
                    first_payload = payload

                mrdata = payload.get("MRData", {})
                race_table = mrdata.get("RaceTable", {})
                page_races = race_table.get("Races", [])
                if not isinstance(page_races, list):
                    raise TypeError(f"Unexpected RaceTable.Races payload from {path}")

                page_count = self._page_item_count(path, page_races)
                total = int(mrdata.get("total", offset + page_count))
                race_pages.extend(page_races)

                if page_count == 0 or offset + page_count >= total:
                    break

                offset += page_count

        assert first_payload is not None
        races = self._merge_race_pages(path, race_pages)
        first_payload["MRData"]["limit"] = str(len(races))
        first_payload["MRData"]["offset"] = "0"
        first_payload["MRData"]["total"] = str(len(races))
        first_payload["MRData"]["RaceTable"]["Races"] = races
        return first_payload

    def get_season_schedule(self, year: int) -> dict[str, Any]:
        """Return all races scheduled in a season."""
        return self._get_all(f"{year}/races/")

    def get_season_results(self, year: int) -> dict[str, Any]:
        """Return all driver race results for a season."""
        return self._get_all(f"{year}/results/")

    def get_season_qualifying(self, year: int) -> dict[str, Any]:
        """Return all qualifying results for a season."""
        return self._get_all(f"{year}/qualifying/")

    def get_race_laps(self, year: int, round_number: int) -> dict[str, Any]:
        """Return lap-by-lap timing for one race."""
        return self._get_all(f"{year}/{round_number}/laps/")

    def get_race_pitstops(self, year: int, round_number: int) -> dict[str, Any]:
        """Return pit-stop records for one race."""
        return self._get_all(f"{year}/{round_number}/pitstops/")
