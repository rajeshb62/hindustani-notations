# Detect Taal from Tabla Track

## Goal
Analyse the tabla in no_vocals.wav to identify:
- Tempo (BPM / lay)
- Beat cycle length (matras)
- Probable taal name
- Sam position (beat 1 of each cycle)

## Common Hindustani taals to match against
| Taal      | Matras | Vibhag structure |
|-----------|--------|-----------------|
| Teentaal  | 16     | 4+4+4+4         |
| Ektaal    | 12     | 2+2+2+2+2+2     |
| Jhaptaal  | 10     | 2+3+2+3         |
| Rupak     | 7      | 3+2+2           |
| Keherwa   | 8      | 4+4             |
| Dadra     | 6      | 3+3             |
| Tilwada   | 16     | 4+4+4+4         |
| Jhoomra   | 14     | 3+4+3+4         |
| Chartal   | 12     | 2+2+2+2+2+2     |
| Deepchandi| 14     | 3+4+3+4         |

## Step 1 — Install dependencies
Run: pip install librosa numpy scipy madmom

Note: madmom is the best beat tracker for tabla — 
it handles irregular onsets far better than librosa alone.

## Step 2 — Create and run detect_taal.py

Create a file called detect_taal.py with this content:

---
import librosa
import numpy as np
from collections import Counter
import json
import warnings
warnings.filterwarnings('ignore')

TABLA_PATH = 'separated/htdemucs/ICCR-1854-AC_SIDE_B/no_vocals.wav'

# Known taals: name -> matras
TAALS = {
    "Teentaal":   16,
    "Tilwada":    16,
    "Jhoomra":    14,
    "Deepchandi": 14,
    "Ektaal":     12,
    "Chartal":    12,
    "Jhaptaal":   10,
    "Keherwa":    8,
    "Rupak":      7,
    "Dadra":      6,
}

print("Loading tabla track...")
y, sr = librosa.load(TABLA_PATH, mono=True)
total_duration = len(y) / sr
print(f"Duration: {total_duration:.1f}s ({total_duration/60:.1f} mins)")

# ── Step A: Tempo estimation ──────────────────────────────────────
print("\nEstimating tempo...")

# Use multiple windows and vote on tempo
window_sec = 30
hop = sr // 100  # 10ms hop

tempo_votes = []
windows = np.linspace(60*sr, len(y) - (window_sec+60)*sr, 8).astype(int)

for start in windows:
    segment = y[start:start + window_sec * sr]
    tempo, beats = librosa.beat.beat_track(
        y=segment, sr=sr,
        hop_length=hop,
        start_bpm=60,
        tightness=100
    )
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo[0])
    if 20 <= tempo <= 500:
        tempo_votes.append(tempo)

# Round to nearest 0.5 BPM and find consensus
rounded_tempos = [round(t * 2) / 2 for t in tempo_votes]
tempo_counts = Counter(rounded_tempos)
print(f"Tempo votes: {sorted(tempo_counts.items())}")

# Take the median as most robust estimate
median_tempo = float(np.median(tempo_votes))
print(f"Estimated tempo: {median_tempo:.1f} BPM")

# Also compute half and double tempo (accounts for matra vs beat confusion)
tempos_to_check = [
    median_tempo / 2,
    median_tempo,
    median_tempo * 2,
    median_tempo * 1.5,
    median_tempo / 1.5,
]

# ── Step B: Beat tracking on full track ──────────────────────────
print("\nTracking beats across full recording...")
tempo_full, beat_frames = librosa.beat.beat_track(
    y=y, sr=sr,
    hop_length=hop,
    bpm=median_tempo,
    tightness=100
)
beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)
print(f"Total beats detected: {len(beat_times)}")

if len(beat_times) > 1:
    inter_beat = np.diff(beat_times)
    median_ibi = float(np.median(inter_beat))
    beat_bpm = 60.0 / median_ibi
    print(f"Inter-beat interval: {median_ibi:.3f}s ({beat_bpm:.1f} BPM)")
else:
    beat_bpm = median_tempo
    median_ibi = 60.0 / median_tempo

# ── Step C: Onset detection for accent pattern ───────────────────
print("\nDetecting onsets...")
onset_frames = librosa.onset.onset_detect(
    y=y, sr=sr,
    hop_length=hop,
    backtrack=True,
    units='frames'
)
onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop)
onset_strength = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
print(f"Total onsets: {len(onset_times)}")

# ── Step D: Cycle length detection ───────────────────────────────
print("\nDetecting cycle length (matras per cycle)...")

# Autocorrelate onset strength to find repeating cycle
oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
tempogram = librosa.feature.tempogram(
    onset_envelope=oenv, sr=sr, hop_length=hop
)

# Look for periodicity in inter-onset-interval
if len(onset_times) > 4:
    iois = np.diff(onset_times)
    median_ioi = float(np.median(iois))
    
    # How many matras fit in one cycle?
    # Try each taal's matra count
    matra_duration = median_ioi  # approximate single matra duration
    
    print(f"\nApproximate matra duration: {matra_duration:.3f}s")
    print(f"Matras per minute: {60/matra_duration:.1f}")

# ── Step E: Match to taal ─────────────────────────────────────────
print("\nMatching to known taals...")

# The beat tracker gives us the matra duration
# Cycle duration = matra_duration * num_matras
# We score each taal by how well its cycle fits the detected beat grid

matra_dur = 60.0 / beat_bpm  # seconds per matra

print(f"\nMatra duration: {matra_dur:.3f}s")
print(f"Matra tempo:    {beat_bpm:.1f} BPM")

taal_scores = {}
for taal_name, matras in TAALS.items():
    cycle_dur = matra_dur * matras
    cycle_bpm = 60.0 / cycle_dur
    
    # Score: how many strong onsets land on cycle boundaries?
    cycle_onsets = 0
    for t in onset_times:
        phase = (t % cycle_dur) / cycle_dur
        # Within 5% of cycle start = on the sam
        if phase < 0.05 or phase > 0.95:
            cycle_onsets += 1
    
    score = cycle_onsets / max(len(onset_times), 1)
    taal_scores[taal_name] = {
        "score": round(score, 4),
        "matras": matras,
        "cycle_duration_sec": round(cycle_dur, 3),
        "cycle_bpm": round(cycle_bpm, 2),
    }

# Sort by score
ranked = sorted(taal_scores.items(), key=lambda x: -x[1]["score"])

print("\nTaal ranking:")
for name, info in ranked[:6]:
    print(f"  {name:<14} {info['matras']} matras  "
          f"cycle={info['cycle_duration_sec']:.2f}s  "
          f"score={info['score']:.4f}")

best_taal = ranked[0][0]
best_info = ranked[0][1]

# ── Step F: Lay (tempo feel) classification ──────────────────────
# Hindustani lay: vilambit (<60 BPM), madhya (60-120), drut (>120)
if beat_bpm < 60:
    lay = "Vilambit (slow)"
elif beat_bpm < 120:
    lay = "Madhya (medium)"
else:
    lay = "Drut (fast)"

print(f"\nLay: {lay} ({beat_bpm:.1f} BPM)")

# ── Step G: Sam detection (beat 1 of each cycle) ─────────────────
cycle_dur = best_info['cycle_duration_sec']

# Find strong onsets that could be sam (beat 1)
# Sam typically has the strongest onset in the cycle
sam_times = []
t = beat_times[0] if len(beat_times) > 0 else 0.0

while t < total_duration:
    # Find the strongest onset within ±10% of expected sam position
    window = cycle_dur * 0.10
    candidates = [ot for ot in onset_times if abs(ot - t) < window]
    if candidates:
        # Pick the one closest to expected
        sam = min(candidates, key=lambda x: abs(x - t))
        sam_times.append(round(float(sam), 3))
    t += cycle_dur

print(f"\nSam positions detected: {len(sam_times)}")
print(f"First few sams: {sam_times[:8]}")

# ── Save results ──────────────────────────────────────────────────
result = {
    "detected_taal": best_taal,
    "matras": best_info["matras"],
    "cycle_duration_sec": best_info["cycle_duration_sec"],
    "matra_duration_sec": round(matra_dur, 4),
    "matra_bpm": round(beat_bpm, 2),
    "lay": lay,
    "all_taal_scores": {k: v for k, v in ranked},
    "sam_times": sam_times[:200],  # first 200 sams
    "total_cycles": len(sam_times),
}

with open('detected_taal.json', 'w') as f:
    json.dump(result, f, indent=2)

print(f"\n{'='*50}")
print(f"RESULT: {best_taal} ({best_info['matras']} matras)")
print(f"Tempo:  {beat_bpm:.1f} BPM — {lay}")
print(f"Cycle:  {best_info['cycle_duration_sec']:.2f}s per avart")
print(f"Saved:  detected_taal.json")
---

Run: python detect_taal.py


## Step 3 — Expected output: detected_taal.json
Example of what it should look like:
{
  "detected_taal": "Teentaal",
  "matras": 16,
  "cycle_duration_sec": 12.4,
  "matra_duration_sec": 0.775,
  "matra_bpm": 77.4,
  "lay": "Madhya (medium)",
  "sam_times": [4.2, 16.6, 29.0, ...],
  "total_cycles": 118
}


## Step 4 — Add taal display to the visualiser app (index.html)

Once detected_taal.json exists, update index.html to:

1. Load detected_taal.json at startup alongside the sargam txt file

2. Display a taal cycle strip at the bottom of the app showing:
   - The taal name and matra count (e.g. "Teentaal — 16 matras")
   - Lay / tempo feel (e.g. "Madhya — 77 BPM")
   - A row of matra boxes (one box per matra in the cycle)
   - The current matra highlighted as audio plays
   - Sam (matra 1) highlighted in a distinct colour (gold)
   - Khali (empty beat, typically matra 9 in Teentaal) 
     highlighted in a different colour (grey)

3. Matra tracking logic:
   - Use sam_times array from the JSON to know when each 
     avart (cycle) starts
   - Interpolate between sam times to calculate current 
     matra position
   - Formula:
       current avart index = find last sam_time <= audio.currentTime
       elapsed = audio.currentTime - sam_times[avart_index]
       current_matra = floor(elapsed / matra_duration_sec) + 1

4. Vibhag (section) markers:
   - Teentaal:   mark matras 1, 5, 9(khali), 13
   - Ektaal:     mark matras 1, 3, 5, 7(khali), 9, 11
   - Jhaptaal:   mark matras 1, 3, 6, 8
   - Rupak:      mark matras 1(khali), 4, 6
   - Keherwa:    mark matras 1, 5
   - Dadra:      mark matras 1, 4
   For other taals show equal divisions only


## Notes for Claude Code
- detect_taal.py may take 3-5 minutes on a long recording
- madmom may fail to install on some systems — if so remove it,
  the script uses librosa only and still works
- After running, print the full contents of detected_taal.json 
  before updating index.html
- If detected_taal shows low scores across all taals (all < 0.02),
  report this — it may mean the recording is in a rare taal or 
  the tabla is mixed very quietly
- The sam_times array is the most important output — it drives
  the real-time matra counter in the app
