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

        return len(page_races)

    def _get_all(self, path: str) -> dict[str, Any]:
        """Fetch every page and combine the returned RaceTable.Races objects."""
        races: list[dict[str, Any]] = []
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

                races.extend(page_races)
                page_count = self._page_item_count(path, page_races)
                total = int(mrdata.get("total", offset + page_count))

                if page_count == 0 or offset + page_count >= total:
                    break

                offset += page_count

        assert first_payload is not None
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
