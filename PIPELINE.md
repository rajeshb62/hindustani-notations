# Hindustani Vocal Sargam Extraction Pipeline

## Goal
Process a Hindustani classical music MP3 to extract accurate sargam
notation for the vocal track only, ignoring tabla and tanpura.

## Input file
ICCR-1854-AC_SIDE_B.mp3 (in the current folder)

## Steps to build and run

### 1. Install dependencies
Run: pip install -r requirements.txt

### 2. Separate vocals from tabla/tanpura using Demucs
Run this shell command:
  python -m demucs --two-stems=vocals "ICCR-1854-AC_SIDE_B.mp3"

This creates a folder: separated/htdemucs/ICCR-1854-AC_SIDE_B/
containing two files:
  - vocals.wav      ← use this
  - no_vocals.wav   ← ignore

### 3. Run CREPE pitch tracking on vocals only
Run this shell command:
  crepe separated/htdemucs/ICCR-1854-AC_SIDE_B/vocals.wav
    --output .
    --step-size 10
    --model full
    --viterbi

This produces: vocals.f0.csv
Columns: time, frequency, confidence

### 4. Run the Python script below to convert to sargam
Create and run a file called extract_sargam.py with this content:

---

import math

Sa = 87.0
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
    "R":  "Re shuddh",  "r": "Re komal",
    "G":  "Ga shuddh",  "g": "Ga komal",
    "M":  "Ma shuddh",  "M+": "Ma tivra",
    "P":  "Pa",
    "D":  "Dha shuddh", "d": "Dha komal",
    "N":  "Ni shuddh",  "n": "Ni komal",
}

def hz_to_swara(hz, sa):
    if hz < 50:
        return None, 0
    ratio = hz / sa
    octave = 0
    while ratio < 1.0:
        ratio *= 2; octave -= 1
    while ratio >= 2.0:
        ratio /= 2; octave += 1
    cents = 1200 * math.log2(ratio)
    best = min(sargam, key=lambda x: min(
        abs(x[0] - cents),
        abs(1200 - cents) if x[1] == "S" else 9999
    ))
    return best[1], octave

raw = []
with open('vocals.f0.csv') as f:
    next(f)
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

# Group consecutive frames of the same swara into discrete notes
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
                fullname = NOTE_NAMES.get(
                    cur_label.replace("'","").replace(".",""),
                    cur_label
                )
                octave_str = (
                    " — taar saptak"   if "'" in cur_label else
                    " — mandra saptak" if "." in cur_label else
                    " — madhya saptak"
                )
                grouped.append((cur_start, dur, cur_label,
                                 fullname + octave_str, avg_hz))
            cur_label  = label
            cur_start  = t
            cur_end    = t
            cur_hz_sum = hz
            cur_count  = 1

# Write timestamped notation file
lines = []
lines.append("SARGAM NOTATION v3 — ICCR-1854-AC SIDE B")
lines.append("=" * 65)
lines.append("Source: Demucs vocals + CREPE pitch tracking")
lines.append(f"Sa = {Sa} Hz | Confidence threshold = {CONFIDENCE_THRESHOLD}")
lines.append(f"Total notes: {len(grouped)}")
lines.append("")
lines.append("KEY:")
lines.append("  S=Sa  R=Re(sh) r=Re(ko)  G=Ga(sh)  g=Ga(ko)  M=Ma(sh)")
lines.append("  M+=Ma(tivra)  P=Pa  D=Dha(sh)  d=Dha(ko)  N=Ni(sh)  n=Ni(ko)")
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

# Compact block notation
lines.append("\n\n" + "=" * 65)
lines.append("COMPACT NOTATION (16 notes per line)")
lines.append("=" * 65)
for i in range(0, len(grouped), 16):
    block = grouped[i:i+16]
    m = int(block[0][0] // 60)
    s = block[0][0] % 60
    note_str = "  ".join(f"{n:<4}" for _,_,n,_,_ in block)
    lines.append(f"[{m:02d}:{s:04.1f}]  {note_str}")

outfile = "sargam_notation_v3_ICCR1854.txt"
with open(outfile, 'w') as f:
    f.write("\n".join(lines))

print(f"Done! {len(grouped)} notes written to {outfile}")
print(f"Duration covered: {grouped[0][0]:.1f}s — {grouped[-1][0]:.1f}s")

---

### 5. Output
The script produces: sargam_notation_v3_ICCR1854.txt
This file is used by the visualiser app (index.html) as the sargam data source.

### 6. Notes for Claude Code
- All commands should be run in sequence in the project folder
- Demucs may take 5–10 minutes on a long audio file — this is normal
- CREPE may take 10–20 minutes on a 24-minute file — this is normal
- If tensorflow gives GPU warnings, ignore them — CPU mode works fine
- After generating sargam_notation_v3_ICCR1854.txt, update index.html
  to reference this new file instead of the old sargam txt file
- The new file format is identical to v2 so no parser changes needed
