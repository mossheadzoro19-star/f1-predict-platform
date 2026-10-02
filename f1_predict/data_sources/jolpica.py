from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class JolpicaClient:
    """Small adapter around the Jolpica-F1 REST API."""

    base_url: str = "https://api.jolpi.ca/ergast/f1"

    def get_season_results(self, year: int) -> dict[str, Any]:
        url = f"{self.base_url}/{year}/results/1.json"
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            response = client.get(url, headers={"User-Agent": "f1-predict-research/0.1"})
            response.raise_for_status()
            return response.json()
