# UX Flows

## Flow 1 — Open dashboard
1. User opens the dashboard.
2. API snapshot loads.
3. UI shows provider state.
4. Current race/session and driver probabilities render.
5. Last update time is visible.

## Flow 2 — LIVE
1. Live provider is reachable.
2. UI displays LIVE.
3. Current race state is fetched.
4. Prediction probabilities update.
5. Freshness timestamp updates.
6. If updates stop, status becomes STALE rather than pretending to be live.

## Flow 3 — REPLAY
1. Historical race-state data is selected.
2. UI displays REPLAY.
3. User scrubs a race timestamp.
4. API returns point-in-time state.
5. Probabilities update for that timestamp.

## Flow 4 — Provider failure
1. Provider request fails.
2. UI shows FALLBACK or DISCONNECTED according to actual state.
3. Historical prediction may remain visible.
4. UI explicitly states that it is not live.

## Flow 5 — Model interpretation
User can inspect position, gap, tyre, weather and P(P1). The UI must not imply that probability equals certainty.
