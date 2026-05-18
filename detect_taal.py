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

rounded_tempos = [round(t * 2) / 2 for t in tempo_votes]
tempo_counts = Counter(rounded_tempos)
print(f"Tempo votes: {sorted(tempo_counts.items())}")

median_tempo = float(np.median(tempo_votes))
print(f"Estimated tempo: {median_tempo:.1f} BPM")

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

# ── Step C: Onset detection ───────────────────────────────────────
print("\nDetecting onsets...")
onset_frames = librosa.onset.onset_detect(
    y=y, sr=sr,
    hop_length=hop,
    backtrack=True,
    units='frames'
)
onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop)
print(f"Total onsets: {len(onset_times)}")

# ── Step D: Cycle length detection ───────────────────────────────
print("\nDetecting cycle length (matras per cycle)...")

oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)

if len(onset_times) > 4:
    iois = np.diff(onset_times)
    median_ioi = float(np.median(iois))
    print(f"\nApproximate matra duration: {median_ioi:.3f}s")
    print(f"Matras per minute: {60/median_ioi:.1f}")

# ── Step E: Match to taal ─────────────────────────────────────────
print("\nMatching to known taals...")

matra_dur = 60.0 / beat_bpm

print(f"\nMatra duration: {matra_dur:.3f}s")
print(f"Matra tempo:    {beat_bpm:.1f} BPM")

taal_scores = {}
for taal_name, matras in TAALS.items():
    cycle_dur = matra_dur * matras
    cycle_bpm = 60.0 / cycle_dur

    cycle_onsets = 0
    for t in onset_times:
        phase = (t % cycle_dur) / cycle_dur
        if phase < 0.05 or phase > 0.95:
            cycle_onsets += 1

    score = cycle_onsets / max(len(onset_times), 1)
    taal_scores[taal_name] = {
        "score": round(score, 4),
        "matras": matras,
        "cycle_duration_sec": round(cycle_dur, 3),
        "cycle_bpm": round(cycle_bpm, 2),
    }

ranked = sorted(taal_scores.items(), key=lambda x: -x[1]["score"])

print("\nTaal ranking:")
for name, info in ranked[:6]:
    print(f"  {name:<14} {info['matras']} matras  "
          f"cycle={info['cycle_duration_sec']:.2f}s  "
          f"score={info['score']:.4f}")

best_taal = ranked[0][0]
best_info = ranked[0][1]

# ── Step F: Lay classification ────────────────────────────────────
if beat_bpm < 60:
    lay = "Vilambit (slow)"
elif beat_bpm < 120:
    lay = "Madhya (medium)"
else:
    lay = "Drut (fast)"

print(f"\nLay: {lay} ({beat_bpm:.1f} BPM)")

# ── Step G: Sam detection ─────────────────────────────────────────
cycle_dur = best_info['cycle_duration_sec']

sam_times = []
t = beat_times[0] if len(beat_times) > 0 else 0.0

while t < total_duration:
    window = cycle_dur * 0.10
    candidates = [ot for ot in onset_times if abs(ot - t) < window]
    if candidates:
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
    "sam_times": sam_times[:200],
    "total_cycles": len(sam_times),
}

with open('detected_taal.json', 'w') as f:
    json.dump(result, f, indent=2)

print(f"\n{'='*50}")
print(f"RESULT: {best_taal} ({best_info['matras']} matras)")
print(f"Tempo:  {beat_bpm:.1f} BPM — {lay}")
print(f"Cycle:  {best_info['cycle_duration_sec']:.2f}s per avart")
print(f"Saved:  detected_taal.json")
