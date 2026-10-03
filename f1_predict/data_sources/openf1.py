from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import os
import time

import httpx


@dataclass
class OpenF1Client:
    """Adapter for OpenF1's public historical and authenticated live HTTP API."""

    base_url: str = "https://api.openf1.org/v1"
    access_token: str | None = None
    min_request_interval_seconds: float = 2.1
    max_retries: int = 5
    retry_backoff_seconds: float = 5.0
    _last_request_at: float = field(default=0.0, init=False, repr=False)

    def _headers(self, include_auth: bool = True) -> dict[str, str]:
        if not include_auth:
            return {}
        token = self.access_token or os.getenv("OPENF1_API_TOKEN")
        return {"Authorization": f"Bearer {token}"} if token else {}

    def _wait_for_rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        wait = self.min_request_interval_seconds - elapsed
        if wait > 0:
            time.sleep(wait)

    @staticmethod
    def _retry_after_seconds(response: httpx.Response, fallback: float) -> float:
        value = response.headers.get("Retry-After")
        if value:
            try:
                return max(0.0, float(value))
            except ValueError:
                pass
        return fallback

    def _get_list(self, endpoint: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        url = f"{self.base_url}/{endpoint}"
        token = self.access_token or os.getenv("OPENF1_API_TOKEN")
        use_auth = bool(token)
        auth_fallback_used = False

        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            for attempt in range(self.max_retries + 1):
                self._wait_for_rate_limit()
                try:
                    response = client.get(
                        url,
                        params=params,
                        headers=self._headers(include_auth=use_auth),
                    )
                    self._last_request_at = time.monotonic()
                except httpx.HTTPError:
                    self._last_request_at = time.monotonic()
                    if attempt >= self.max_retries:
                        raise
                    time.sleep(self.retry_backoff_seconds * (2**attempt))
                    continue

                if response.status_code == 401 and use_auth and not auth_fallback_used:
                    # Historical OpenF1 data is publicly accessible. If a stale
                    # local token is configured, retry once without authentication.
                    use_auth = False
                    auth_fallback_used = True
                    print(
                        f"OpenF1 authentication rejected on /{endpoint}; "
                        "retrying without authentication."
                    )
                    continue

                if response.status_code == 429:
                    if attempt >= self.max_retries:
                        response.raise_for_status()
                    delay = self._retry_after_seconds(
                        response,
                        self.retry_backoff_seconds * (2**attempt),
                    )
                    print(
                        f"OpenF1 rate limit hit on /{endpoint}; "
                        f"retrying in {delay:.1f}s "
                        f"(attempt {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(delay)
                    continue

                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, list):
                    raise TypeError(
                        f"OpenF1 /{endpoint} returned an unexpected payload."
                    )
                return payload

        raise RuntimeError("OpenF1 request loop exited unexpectedly.")

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

    def get_pit_stops(self, session_key: int) -> list[dict[str, Any]]:
        """Return pit-lane events for a race session."""
        return self._get_list("pit", {"session_key": session_key})

    def get_stints(self, session_key: int) -> list[dict[str, Any]]:
        """Return tyre stint observations."""
        return self._get_list("stints", {"session_key": session_key})

    def get_weather(self, session_key: int) -> list[dict[str, Any]]:
        """Return track weather observations."""
        return self._get_list("weather", {"session_key": session_key})

    def get_session_result(self, session_key: int) -> list[dict[str, Any]]:
        """Return final session results when OpenF1 has published them."""
        return self._get_list("session_result", {"session_key": session_key})
