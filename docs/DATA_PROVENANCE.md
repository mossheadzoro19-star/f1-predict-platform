# Data provenance and licensing

## Jolpica-F1
Purpose: structured historical Formula 1 records, race results, qualifying, lap timing, and pit-stop records where available.

Use only within the provider's current terms, attribution requirements, rate limits and licensing conditions. Race-state features reconstructed from lap records are approximations and must not be described as official live timing.

Reference: https://github.com/jolpica/jolpica-f1

## FastF1
Purpose: richer historical F1 session, lap, telemetry and weather processing when data is accessible.

FastF1 software is MIT licensed, but upstream motorsport data may have separate terms. Do not assume the software license grants unrestricted redistribution of retrieved data. In October 2026, FastF1's maintainer reported that the F1 live-timing archive requires authentication and an F1 TV Access subscription or higher. Therefore FastF1 must remain optional for this no-paid-API project; do not purchase access to unblock historical ingestion.

Reference: https://github.com/theOehrly/Fast-F1/issues/978

## Live timing provider
Purpose: free-access real-time race state.

The live provider must sit behind an adapter so it can be replaced if access, reliability or licensing changes. Free access does not automatically mean unrestricted commercial redistribution. If no compliant free provider is available, the service must report stale/disconnected status or use explicitly selected historical replay.

## OpenF1
OpenF1 is retained as historical project context and as the source of an existing experimental benchmark dataset. It must not be treated as the required provider for future ingestion because current live-session access may require authenticated/paid access.

Reference: https://openf1.org/

## Repository policy
1. Keep raw provider responses separate from derived ML datasets.
2. Store retrieval timestamps and source identifiers.
3. Never commit provider credentials.
4. Re-check provider terms before public/commercial deployment.
5. Preserve point-in-time correctness and clearly document reconstructed/approximate features.
6. Never label cached, historical, reconstructed, or fallback data as LIVE.


## Static historical archive fallback

When the Jolpica API returns persistent HTTP 429 responses, use the public TracingInsights/RaceData release archive instead of repeatedly retrying the API. Its release archive contains historical race metadata, results, lap times, and pit stops; the repository identifies the dataset as CC0 1.0. The ingestion downloads and caches the ZIP locally, then builds the same platform feature schema without calling Jolpica endpoints.

Source: https://github.com/TracingInsights/RaceData
Release archive: https://github.com/TracingInsights/RaceData/releases/latest/download/data.zip

The archive's lap-end positions and lap times are historical observations. The generated timestamp is an approximate race-day anchor plus cumulative lap time, not an official timing-feed timestamp. Tyre compound and weather remain missing unless a source explicitly provides them. Verify source freshness and licensing before public/commercial deployment.
