# Detect True Sa from Tanpura and Re-extract Sargam

## Goal
Use the tanpura drone in no_vocals.wav to accurately detect the Sa 
frequency, then re-run the sargam extraction with the correct Sa.

## Input files (must already exist from previous pipeline run)
- separated/htdemucs/ICCR-1854-AC_SIDE_B/no_vocals.wav
- separated/htdemucs/ICCR-1854-AC_SIDE_B/vocals.wav
- vocals.f0.csv  (from CREPE)

## Step 1 — Install additional dependency
Run: pip install librosa numpy scipy

## Step 2 — Create and run detect_sa.py

Create a file called detect_sa.py with this content:

---
import librosa
import numpy as np
from collections import Counter
import json
import math

print("Loading tanpura track...")
y, sr = librosa.load(
    'separated/htdemucs/ICCR-1854-AC_SIDE_B/no_vocals.wav',
    mono=True
)

total_duration = len(y) / sr
print(f"Total duration: {total_duration:.1f}s")

# Sample 5 stable windows spread across the recording
# Avoid first and last 60 seconds (intro/outro)
window_size = sr * 20  # 20 second windows
starts = np.linspace(60 * sr, len(y) - 60 * sr - window_size, 5).astype(int)

all_pitches = []

for i, start in enumerate(starts):
    segment = y[start:start + window_size]
    pitches, magnitudes = librosa.piptrack(
        y=segment, sr=sr,
        fmin=50, fmax=500,
        threshold=0.1
    )
    for t in range(pitches.shape[1]):
        idx = magnitudes[:, t].argmax()
        p = pitches[idx, t]
        mag = magnitudes[idx, t]
        if p > 50 and mag > 0.01:
            all_pitches.append(p)
    print(f"  Window {i+1}/5 processed — {len(all_pitches)} pitch frames so far")

all_pitches = np.array(all_pitches)

# Round to nearest 0.5 Hz for clustering
rounded = [round(p * 2) / 2 for p in all_pitches]
counts = Counter(rounded)

print("\nTop 20 pitch clusters in tanpura:")
top = counts.most_common(20)
for hz, count in top:
    print(f"  {hz:.1f} Hz — {count} frames")

# Find Sa: the tanpura plays Sa(low), Pa, Sa(mid), Sa(high)
# Pa is always ~1.5x Sa (perfect fifth)
# Strategy: find the most common low pitch — that is Sa

# Filter to reasonable vocal Sa range (60–200 Hz for Indian classical)
candidates = [(hz, cnt) for hz, cnt in counts.most_common(50) if 60 <= hz <= 200]

# The strongest cluster in that range is most likely Sa or Pa
# Check: if two strong clusters exist where cluster2 / cluster1 ≈ 1.498 (Pa/Sa ratio)
# then cluster1 is Sa

best_sa = None
best_score = 0

for sa_hz, sa_count in candidates:
    # Look for Pa at sa_hz * 1.498 (perfect fifth)
    pa_target = sa_hz * 1.498
    pa_tolerance = 4  # Hz
    pa_count = sum(cnt for hz, cnt in candidates 
                   if abs(hz - pa_target) < pa_tolerance)
    
    # Score = presence of both Sa and Pa
    score = sa_count + pa_count * 0.5
    if score > best_score:
        best_score = score
        best_sa = sa_hz

print(f"\nDetected Sa = {best_sa:.2f} Hz")
print(f"Expected Pa = {best_sa * 1.498:.2f} Hz")

# Verify by checking if Pa is actually present
pa_hz = best_sa * 1.498
pa_found = sum(cnt for hz, cnt in candidates if abs(hz - pa_hz) < 5)
print(f"Pa presence in tanpura: {pa_found} frames")

# Also check upper Sa (octave above)
upper_sa = best_sa * 2
upper_found = sum(cnt for hz, cnt in counts.items() 
                  if abs(hz - upper_sa) < 5)
print(f"Upper Sa ({upper_sa:.1f} Hz) presence: {upper_found} frames")

# Save result
result = {
    "sa_hz": round(best_sa, 2),
    "pa_hz": round(best_sa * 1.498, 2),
    "upper_sa_hz": round(best_sa * 2, 2),
    "confidence": "high" if pa_found > 100 else "medium"
}

with open('detected_sa.json', 'w') as f:
    json.dump(result, f, indent=2)

print(f"\nSaved to detected_sa.json")
print(json.dumps(result, indent=2))
---

Run: python detect_sa.py


## Step 3 — Create and run extract_sargam_v3.py

Create a file called extract_sargam_v3.py with this content:

---
import math
import json

# Load detected Sa
with open('detected_sa.json') as f:
    sa_data = json.load(f)

Sa = sa_data['sa_hz']
print(f"Using Sa = {Sa} Hz (detected from tanpura)")
print(f"Confidence: {sa_data['confidence']}")

CONFIDENCE_THRESHOLD = 0.85

sargam = [
    (0,    "S"),
    (100,  "r"), (200,  "R"),
    (300,  "g"), (400,  "G"),
    (500,  "M"), (600,  "M+"),
    (700,  "P"),
    (800,  "d"), (900,  "D"),
    (1000, "n"), (1100, "N"),
]

NOTE_NAMES = {
    "S":  "Sa",
    "R":  "Re shuddh",  "r":  "Re komal",
    "G":  "Ga shuddh",  "g":  "Ga komal",
    "M":  "Ma shuddh",  "M+": "Ma tivra",
    "P":  "Pa",
    "D":  "Dha shuddh", "d":  "Dha komal",
    "N":  "Ni shuddh",  "n":  "Ni komal",
}

def hz_to_swara(hz, sa):
    if hz < 50:
        return None, 0
    ratio = hz / sa
    octave = 0
    while ratio < 1.0:
        ratio *= 2;  octave -= 1
    while ratio >= 2.0:
        ratio /= 2;  octave += 1
    cents = 1200 * math.log2(ratio)
    best = min(sargam, key=lambda x: min(
        abs(x[0] - cents),
        abs(1200 - cents) if x[1] == "S" else 9999
    ))
    return best[1], octave

# Load CREPE output
raw = []
with open('vocals.f0.csv') as f:
    next(f)  # skip header
    for line in f:
        parts = line.strip().split(',')
        if len(parts) == 3:
            try:
                t    = float(parts[0])
                hz   = float(parts[1])
                conf = float(parts[2])
                if conf >= CONFIDENCE_THRESHOLD and hz > 50:
                    swara, octave = hz_to_swara(hz, Sa)
                    if swara is None:
                        continue
                    if octave < 0:
                        label = swara.lower() + "."
                    elif octave > 0:
                        label = swara + "'"
                    else:
                        label = swara
                    raw.append((t, hz, conf, label))
            except:
                pass

print(f"Raw voiced frames: {len(raw)}")

# Group consecutive frames into discrete notes
grouped = []
if raw:
    cur_label  = raw[0][3]
    cur_start  = raw[0][0]
    cur_end    = raw[0][0]
    cur_hz_sum = raw[0][1]
    cur_count  = 1

    for t, hz, conf, label in raw[1:]:
        gap = t - cur_end
        if label == cur_label and gap <= 0.05:
            cur_end    = t
            cur_hz_sum += hz
            cur_count  += 1
        else:
            dur = cur_end - cur_start + 0.01
            if dur >= 0.04:
                avg_hz   = cur_hz_sum / cur_count
                base     = cur_label.replace("'", "").replace(".", "")
                fullname = NOTE_NAMES.get(base, base)
                octave_str = (
                    " — taar saptak"   if "'" in cur_label else
                    " — mandra saptak" if "." in cur_label else
                    " — madhya saptak"
                )
                grouped.append((
                    cur_start, dur, cur_label,
                    fullname + octave_str, avg_hz
                ))
            cur_label  = label
            cur_start  = t
            cur_end    = t
            cur_hz_sum = hz
            cur_count  = 1

# Final note
if cur_count > 0:
    dur = cur_end - cur_start + 0.01
    if dur >= 0.04:
        base = cur_label.replace("'", "").replace(".", "")
        grouped.append((
            cur_start, dur, cur_label,
            NOTE_NAMES.get(base, base) + (
                " — taar saptak"   if "'" in cur_label else
                " — mandra saptak" if "." in cur_label else
                " — madhya saptak"
            ),
            cur_hz_sum / cur_count
        ))

print(f"Discrete notes after grouping: {len(grouped)}")

# Write output
lines = []
lines.append("SARGAM NOTATION v3 — ICCR-1854-AC SIDE B")
lines.append("=" * 65)
lines.append("Source: Demucs vocals + CREPE + tanpura Sa detection")
lines.append(f"Detected Sa = {Sa} Hz | Pa = {sa_data['pa_hz']} Hz")
lines.append(f"Confidence: {sa_data['confidence']}")
lines.append(f"CREPE confidence threshold = {CONFIDENCE_THRESHOLD}")
lines.append(f"Total notes: {len(grouped)}")
lines.append("")
lines.append("KEY:")
lines.append("  S=Sa  R=Re(sh) r=Re(ko)  G=Ga(sh) g=Ga(ko)  M=Ma(sh)")
lines.append("  M+=Ma(tivra)  P=Pa  D=Dha(sh) d=Dha(ko)  N=Ni(sh)  n=Ni(ko)")
lines.append("  ' = taar (upper octave)   . = mandra (lower octave)")
lines.append("")
lines.append("FORMAT: [MM:SS.ss]  note  duration  full_name")
lines.append("=" * 65)

current_minute = -1
for start, dur, label, full, hz in grouped:
    m = int(start // 60)
    s = start % 60
    if m != current_minute:
        current_minute = m
        lines.append(f"\n--- Minute {m:02d} ---")
    lines.append(f"  [{m:02d}:{s:05.2f}]  {label:<6}  {dur:.3f}s   {full}")

lines.append("\n\n" + "=" * 65)
lines.append("COMPACT NOTATION (16 notes per line)")
lines.append("=" * 65)
for i in range(0, len(grouped), 16):
    block = grouped[i:i+16]
    m = int(block[0][0] // 60)
    s = block[0][0] % 60
    note_str = "  ".join(f"{n:<4}" for _, _, n, _, _ in block)
    lines.append(f"[{m:02d}:{s:04.1f}]  {note_str}")

outfile = "sargam_notation_v3_ICCR1854.txt"
with open(outfile, 'w') as f:
    f.write("\n".join(lines))

print(f"\nDone! Written to {outfile}")
---

Run: python extract_sargam_v3.py


## Step 4 — Expected outputs
- detected_sa.json          ← Sa/Pa frequencies detected from tanpura
- sargam_notation_v3_ICCR1854.txt  ← final sargam file for the app

## Step 5 — Update the visualiser
Once sargam_notation_v3_ICCR1854.txt is created, update index.html 
to load this file instead of any previous sargam txt file.
The file format is identical so no parser changes are needed.

## Notes for Claude Code
- Run steps in order: detect_sa.py first, then extract_sargam_v3.py
- detect_sa.py will print the detected Sa Hz — verify it looks 
  reasonable (typically 60–200 Hz for Indian classical male/female voice)
- If confidence shows "medium", the Sa detection may be less reliable —
  report the detected_sa.json contents back to the user before proceeding
- The whole pipeline assumes vocals.f0.csv already exists from the 
  previous CREPE run — do not re-run CREPE unless asked
