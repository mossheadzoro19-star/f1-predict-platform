from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import os

import httpx


@dataclass(frozen=True)
class OpenF1Client:
    """Adapter for OpenF1's public historical and authenticated live HTTP API."""

    base_url: str = "https://api.openf1.org/v1"
    access_token: str | None = None

    def _headers(self) -> dict[str, str]:
        token = self.access_token or os.getenv("OPENF1_API_TOKEN")
        return {"Authorization": f"Bearer {token}"} if token else {}

    def _get_list(self, endpoint: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            response = client.get(
                f"{self.base_url}/{endpoint}",
                params=params,
                headers=self._headers(),
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                raise TypeError(f"OpenF1 /{endpoint} returned an unexpected payload.")
            return payload

    def get_sessions(self, year: int) -> list[dict[str, Any]]:
        return self._get_list("sessions", {"year": year})

    def get_positions(self, session_key: int) -> list[dict[str, Any]]:
        """Return position observations for a race session."""
        return self._get_list("position", {"session_key": session_key})

    def get_drivers(self, session_key: int) -> list[dict[str, Any]]:
        """Return driver identity metadata for a session."""
        return self._get_list("drivers", {"session_key": session_key})

    def get_intervals(self, session_key: int) -> list[dict[str, Any]]:
        """Return gap-to-leader and interval observations."""
        return self._get_list("intervals", {"session_key": session_key})

    def get_laps(self, session_key: int) -> list[dict[str, Any]]:
        """Return individual lap observations."""
        return self._get_list("laps", {"session_key": session_key})

    def get_stints(self, session_key: int) -> list[dict[str, Any]]:
        """Return tyre stint observations."""
        return self._get_list("stints", {"session_key": session_key})

    def get_weather(self, session_key: int) -> list[dict[str, Any]]:
        """Return track weather observations."""
        return self._get_list("weather", {"session_key": session_key})

    def get_session_result(self, session_key: int) -> list[dict[str, Any]]:
        """Return final session results when OpenF1 has published them."""
        return self._get_list("session_result", {"session_key": session_key})
