# Error Handling

## Provider errors
- Timeout/network failure → STALE or DISCONNECTED.
- Provider unavailable → FALLBACK only if historical prediction is actually available.
- Authentication failure → DISCONNECTED with actionable server-side diagnostics.
- Rate limit → back off; do not spam retries.

## Data errors
Reject malformed or ambiguous snapshots rather than silently producing predictions.

## UI states
- LIVE: fresh live provider data.
- REPLAY: explicit historical replay.
- FALLBACK: historical prediction available but live source unavailable.
- STALE: provider connection exists but data freshness threshold has been exceeded.
- DISCONNECTED: no usable provider connection.

## User-facing errors
Messages should be concise and explain whether the displayed prediction is live, replayed or historical fallback.
