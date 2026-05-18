# Sargam Extraction — Changes from Original PIPELINE.md Spec

These three changes were made on top of the original PIPELINE.md logic.
If extract_sargam_v3.py is ever lost or reimplemented, apply all three.

---

## Change 1: Sa loaded from file, not hardcoded

**Original:** `Sa = 87.0`

**Now:** 
```python
with open('detected_sa.json') as f:
    sa_data = json.load(f)
Sa = sa_data['sa_hz']   # = 98.0 Hz for this recording
```

**Why:**
The original PIPELINE.md guessed Sa = 87.0 Hz. The actual Sa for this
recording is 98.0 Hz, detected by running detect_sa.py which listens to
the tanpura drone in no_vocals.wav and finds the fundamental frequency.

Using the wrong Sa shifts every note by about a semitone — S becomes r,
R becomes G, etc. All 5,492 notes would be wrong.

The file detected_sa.json is produced by detect_sa.py and must exist
before running extract_sargam_v3.py. It contains:
  { "sa_hz": 98.0, "pa_hz": 146.8, "upper_sa_hz": 196.0,
    "confidence": "medium" }

---

## Change 2: Confidence threshold lowered from 0.85 to 0.75

**Original:** `CONFIDENCE_THRESHOLD = 0.85`

**Now:** `CONFIDENCE_THRESHOLD = 0.75`

**Why:**
CREPE (the pitch tracker) assigns a confidence score 0–1 to each frame.
The original spec used 0.85 to keep only high-confidence frames.

The problem: some genuine vocal passages (e.g. 4:39–4:44) had CREPE
confidence of 0.77–0.84 — real notes that were being silently dropped.
Lowering to 0.75 recovers these frames without opening the door to much
noise, because the VAD filter (Change 3) handles the real noise problem
independently.

In short: 0.85 was too aggressive for this recording. 0.75 keeps 
meaningful low-energy vocal frames while VAD handles bleed.

---

## Change 3: Input is vocals.f0.vad.csv not vocals.f0.csv

**Original:** reads `vocals.f0.csv`  (raw CREPE output)

**Now:** reads `vocals.f0.vad.csv`  (VAD-filtered CREPE output)

**Why:**
Even though Demucs separated vocals from instruments, the vocal track
(vocals.wav) still contains faint tanpura bleed — the drone hum leaks
through. CREPE confidently detects this drone as pitched audio at
~346 Hz with confidence 0.85–0.95, producing fake notes during silences
(e.g. 18:00–18:09 showed G notes where there is clearly no singing).

The fix was to run a two-stage filter (run_vad.py) that produces
vocals.f0.vad.csv:

  Stage 1 — Silero VAD (Voice Activity Detection):
    A neural network listens to vocals.wav and marks which time regions
    actually contain a human voice. Settings used:
      threshold = 0.35, min_silence = 1500ms, speech_pad = 500ms
    Frames inside VAD regions: keep if CREPE conf >= 0.75
    Frames outside VAD regions: discard (these are silence/bleed)

  Stage 2 — CREPE rescue for missed vocals:
    Sometimes Silero VAD misses a quiet vocal phrase. To recover these:
    Frames outside VAD regions are kept IF:
      - CREPE confidence >= 0.85  (very confident pitch)
      - AND pitch is NOT within ±15 Hz of tanpura fundamental or octaves
        (tanpura Hz = 346.0, tolerance = 15 Hz, check 173, 346, 692 Hz)
    This rescues real notes that VAD missed, while still blocking tanpura.

Result: 86,866 frames kept, 59,741 dropped.
Output file: vocals.f0.vad.csv  (same format as vocals.f0.csv)
Script that produces it: run_vad.py

vocals.f0.vad.csv must exist before running extract_sargam_v3.py.
If it does not exist, re-run: python3 run_vad.py

---

## Summary — files that must exist before running extract_sargam_v3.py

  detected_sa.json       ← produced by detect_sa.py
  vocals.f0.vad.csv      ← produced by run_vad.py
  (which itself needs vocals.f0.csv from CREPE and vocals.wav from Demucs)
