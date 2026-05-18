# Ground Truth — Verified Taal Parameters

## Recording
File: ICCR-1854-AC_SIDE_B.mp3

## Confirmed Taal
- Taal:         Teentaal
- Matras:       16
- Vibhag:       [4, 4, 4, 4]
- Khali:        matra 9
- Bhari:        matras 1, 5, 13

## Confirmed Tempo
- Global matra: 0.418s = 143.5 BPM (full-track autocorr, strongest 
                peak in 0.2–1.5s range)
- Cycle:        6.688s per avart
- Lay:          Madhya-Drut

## Verified Sam Positions (ground truth for testing)
Clip: no_vocals_clip_550_620.wav (offset 550s into main file)
In-clip times → absolute times:
  5.467s  → 555.467s
  12.155s → 562.155s
  18.843s → 568.843s  ← strongest onset, most reliable anchor
  25.531s → 575.531s
Spacing: exactly 6.688s between each

## Tabla Entry
- Tabla first enters at: ~225s into the recording
- Before 225s: alap (no tabla), sam detection should be skipped
- Silence gate threshold: mean onset strength < 0.05 (normalised)

## Sa Detection
- Sa frequency: detected from tanpura in no_vocals.wav
- Stored in: detected_sa.json
- Do not hardcode — always load from detected_sa.json
