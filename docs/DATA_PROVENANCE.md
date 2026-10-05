# Data provenance and licensing

## Jolpica-F1
Purpose: structured historical Formula 1 records.

Use only within the provider's current terms, attribution requirements, rate limits and licensing conditions.

Reference: https://github.com/jolpica/jolpica-f1

## FastF1
Purpose: richer historical F1 session, lap, telemetry and weather processing.

FastF1 software is MIT licensed, but upstream motorsport data may have separate terms. Do not assume the software license grants unrestricted redistribution of retrieved data.

Reference: https://github.com/theOehrly/Fast-F1

## Live timing provider
Purpose: free-access real-time race state.

The live provider must sit behind an adapter so it can be replaced if access, reliability or licensing changes. Free access does not automatically mean unrestricted commercial redistribution.

## OpenF1
OpenF1 is retained as historical project context only. It must not be treated as the required provider for this project because current live-session access may require authenticated/paid access.

Reference: https://openf1.org/

## Repository policy
1. Keep raw provider responses separate from derived ML datasets.
2. Store retrieval timestamps and source identifiers.
3. Never commit provider credentials.
4. Re-check provider terms before public/commercial deployment.
5. Preserve point-in-time correctness.
6. Never label cached or fallback data as LIVE.
