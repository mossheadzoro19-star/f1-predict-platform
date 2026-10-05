# Design System

## 1. Direction
**Style:** technical F1 telemetry + pixel-art racing interface.  
**Mood:** dark, precise, fast, cockpit/broadcast inspired.

The visual language uses the **Pixel Code** font as inspiration/reference: a readable monospace pixel-grid typeface with box-drawing and technical symbols. Reference: https://qwerasd205.github.io/PixelCode/

## 2. Principles
- Data first; decoration must never obscure race information.
- Dense but readable telemetry.
- Pixel geometry for accents, dividers and car illustrations.
- Minimal assets and dependencies.
- Clear distinction between live and historical/replay state.
- Motion should communicate race changes, not exist as decoration.

## 3. Palette
- Background: #08090D
- Surface: #101218
- Elevated surface: #151821
- Border: #242731
- Primary text: #F5F5F5
- Muted text: #9297A5
- Success/live: #42E06F
- Warning/stale: #F5C542
- Error/disconnected: #FF5A5F
- Accent: #E10600

## 4. Typography
- Preferred: Pixel Code for telemetry, labels and technical UI.
- Fallback: ui-monospace, SFMono-Regular, Consolas, monospace.
- Headings: bold/extra-bold pixel/mono treatment.
- Body: readable mono at normal weight.
- Avoid excessive letter spacing.

## 5. Components
- Status pill: LIVE, REPLAY, FALLBACK, STALE, DISCONNECTED.
- Probability bar: horizontal, normalized 0–100%.
- Driver row: position, driver, constructor, gap, tyre, P(P1).
- Metric tile: current lap, laps to go, leader, track temperature.
- Replay control: play/pause + timeline scrubber.
- Track panel: lightweight circuit outline or pixel drawing.
- F1 car motif: small inline pixel/ASCII car, never a large image asset.

Example lightweight car motif:

    ▄████████▄
   ███ ▄▄▄ ███
  ▀███▀███▀███▀
    O       O

The exact artwork may evolve, but it must remain code-rendered and lightweight.

## 6. Responsive
Desktop: telemetry-first multi-column layout.  
Tablet: collapse secondary panels.  
Mobile: single-column race order with sticky status and replay controls.

## 7. Accessibility
Use semantic HTML, keyboard-accessible controls, visible focus, readable contrast and text labels for status. Never rely on color alone.
