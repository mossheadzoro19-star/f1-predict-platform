from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class JolpicaClient:
    """Adapter around Jolpica-F1's Ergast-compatible REST API."""

    base_url: str = "https://api.jolpi.ca/ergast/f1"
    user_agent: str = "F1PredictPlatform/0.1 FastF1/0.0"

    def _get_json(
        self,
        client: httpx.Client,
        path: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        response = client.get(
            url,
            params={"limit": limit, "offset": offset},
            headers={"User-Agent": self.user_agent},
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise TypeError(f"Unexpected Jolpica response from {url}")
        return payload

    def _get_all(self, path: str) -> dict[str, Any]:
        """Fetch every page for an endpoint and combine RaceTable.Races."""
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

                total = int(mrdata.get("total", len(races)))
                if not page_races or len(races) >= total:
                    break

                offset += len(page_races)

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
