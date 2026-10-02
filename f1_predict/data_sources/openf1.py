from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import os

import httpx


@dataclass(frozen=True)
class OpenF1Client:
    """Adapter for OpenF1's public historical HTTP API."""

    base_url: str = "https://api.openf1.org/v1"
    access_token: str | None = None

    def _headers(self) -> dict[str, str]:
        token = self.access_token or os.getenv("OPENF1_API_TOKEN")
        return {"Authorization": f"Bearer {token}"} if token else {}

    def get_sessions(self, year: int) -> list[dict[str, Any]]:
        params = {"year": year}
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            response = client.get(f"{self.base_url}/sessions", params=params, headers=self._headers())
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                raise TypeError("OpenF1 /sessions returned an unexpected payload.")
            return payload


    def get_positions(self, session_key: int) -> list[dict[str, Any]]:
        """Return position observations for a race session."""
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            response = client.get(
                f"{self.base_url}/position",
                params={"session_key": session_key},
                headers=self._headers(),
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                raise TypeError("OpenF1 /position returned an unexpected payload.")
            return payload

    def get_drivers(self, session_key: int) -> list[dict[str, Any]]:
        """Return driver identity metadata for a session."""
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            response = client.get(
                f"{self.base_url}/drivers",
                params={"session_key": session_key},
                headers=self._headers(),
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                raise TypeError("OpenF1 /drivers returned an unexpected payload.")
            return payload
