from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class OpenF1Client:
    """Adapter for OpenF1's public historical HTTP API."""

    base_url: str = "https://api.openf1.org/v1"

    def get_sessions(self, year: int) -> list[dict[str, Any]]:
        params = {"year": year}
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            response = client.get(f"{self.base_url}/sessions", params=params)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                raise TypeError("OpenF1 /sessions returned an unexpected payload.")
            return payload
