"""
VAD step: detect voiced segments in vocals.wav using Silero VAD,
then filter vocals.f0.csv to keep only frames where voice is active.
Writes vocals.f0.vad.csv for use by extract_sargam_v3.py.
"""
import torch
import torchaudio
import json
import csv

VOCALS_WAV = 'separated/htdemucs/ICCR-1854-AC_SIDE_B/vocals.wav'
F0_CSV     = 'vocals.f0.csv'
OUT_CSV    = 'vocals.f0.vad.csv'
VAD_JSON   = 'vad_segments.json'

# ── 1. Load Silero VAD ────────────────────────────────────────────
print("Loading Silero VAD model...")
model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,
    trust_repo=True
)
(get_speech_timestamps, _, read_audio, *_) = utils

# ── 2. Run VAD on vocals.wav ──────────────────────────────────────
print(f"Running VAD on {VOCALS_WAV} ...")
# Silero VAD works at 16kHz
wav = read_audio(VOCALS_WAV, sampling_rate=16000)

speech_timestamps = get_speech_timestamps(
    wav, model,
    sampling_rate=16000,
    threshold=0.35,               # sensitive to quieter/breathy voice
    min_speech_duration_ms=150,
    min_silence_duration_ms=1500, # must be 1.5s of silence to split segments
    speech_pad_ms=500,            # pad 500ms each side
    return_seconds=True
)

print(f"Found {len(speech_timestamps)} voiced segments")
total_voiced = sum(s['end'] - s['start'] for s in speech_timestamps)
print(f"Total voiced duration: {total_voiced:.1f}s")

# Save segments for inspection
with open(VAD_JSON, 'w') as f:
    json.dump(speech_timestamps, f, indent=2)
print(f"Saved segments to {VAD_JSON}")

# ── 3. Build a fast lookup: is time T voiced? ─────────────────────
# Merge into sorted list of (start, end) pairs
segments = [(s['start'], s['end']) for s in speech_timestamps]

def is_voiced(t):
    # Binary search
    lo, hi = 0, len(segments) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        s, e = segments[mid]
        if t < s:
            hi = mid - 1
        elif t > e:
            lo = mid + 1
        else:
            return True
    return False

# ── 4. Filter vocals.f0.csv ───────────────────────────────────────
# Tanpura dominant frequency — frames near this are bleed, not voice
TANPURA_HZ   = 346.0
TANPURA_TOL  = 15.0   # ±15 Hz around tanpura fundamental
# Also its sub-octave (173 Hz) and upper octave (692 Hz)
TANPURA_HARMONICS = [TANPURA_HZ / 2, TANPURA_HZ, TANPURA_HZ * 2]

def is_tanpura(hz):
    return any(abs(hz - h) < TANPURA_TOL for h in TANPURA_HARMONICS)

print(f"Filtering {F0_CSV} ...")
kept_vad = kept_crepe = dropped = 0

with open(F0_CSV) as fin, open(OUT_CSV, 'w', newline='') as fout:
    reader = csv.reader(fin)
    writer = csv.writer(fout)
    header = next(reader)
    writer.writerow(header)
    for row in reader:
        if len(row) < 3:
            continue
        t    = float(row[0])
        hz   = float(row[1])
        conf = float(row[2])

        if is_voiced(t):
            # VAD confirmed voiced region — keep if conf > 0.75
            if conf >= 0.75:
                writer.writerow(row)
                kept_vad += 1
            else:
                dropped += 1
        else:
            # VAD says silent — only keep if CREPE is very confident
            # AND pitch is not tanpura bleed
            if conf >= 0.85 and not is_tanpura(hz):
                writer.writerow(row)
                kept_crepe += 1
            else:
                dropped += 1

total = kept_vad + kept_crepe + dropped
print(f"Kept (VAD region):    {kept_vad:,} frames")
print(f"Kept (CREPE rescue):  {kept_crepe:,} frames  ← voice VAD missed")
print(f"Dropped:              {dropped:,} frames ({dropped/total*100:.1f}%) — silence/tanpura bleed")
print(f"Written to {OUT_CSV}")
