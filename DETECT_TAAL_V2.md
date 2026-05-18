# Taal Detection V2 — Tabla Isolated from Tanpura

## What was wrong with V1
- no_vocals.wav contains both tabla AND tanpura
- Tanpura is a loud sustained drone that confused the beat tracker
- librosa's beat tracker latched onto tanpura periodicity, not tabla strokes
- Scores were near-identical across taals = unreliable

## What V2 does differently
1. Applies Harmonic-Percussive Source Separation (HPSS) to no_vocals.wav
   - Harmonic component = tanpura drone (discarded)
   - Percussive component = tabla strokes only (used for detection)
2. Runs onset detection on the percussive component only
3. Uses autocorrelation of the onset envelope to find cycle length
   - Much more reliable than beat_track() for non-Western rhythms
4. Scores taals against the full inter-onset-interval histogram
   rather than just sam positions

## Step 1 — Create and run detect_taal_v2.py

Create a file called detect_taal_v2.py with this content:

---
import librosa
import numpy as np
from collections import Counter
import json
import warnings
warnings.filterwarnings('ignore')

TABLA_PATH = 'separated/htdemucs/ICCR-1854-AC_SIDE_B/no_vocals.wav'

TAALS = {
    "Teentaal":   {"matras": 16, "vibhag": [4,4,4,4],   "khali": [9]},
    "Tilwada":    {"matras": 16, "vibhag": [4,4,4,4],   "khali": [9]},
    "Jhoomra":    {"matras": 14, "vibhag": [3,4,3,4],   "khali": [8]},
    "Deepchandi": {"matras": 14, "vibhag": [3,4,3,4],   "khali": [8]},
    "Ektaal":     {"matras": 12, "vibhag": [2,2,2,2,2,2],"khali": [7,11]},
    "Chartal":    {"matras": 12, "vibhag": [2,2,2,2,2,2],"khali": [7,11]},
    "Jhaptaal":   {"matras": 10, "vibhag": [2,3,2,3],   "khali": [6]},
    "Keherwa":    {"matras": 8,  "vibhag": [4,4],        "khali": [5]},
    "Rupak":      {"matras": 7,  "vibhag": [3,2,2],      "khali": [1]},
    "Dadra":      {"matras": 6,  "vibhag": [3,3],        "khali": [4]},
}

# ── Step 1: Load and separate ─────────────────────────────────────
print("Loading no_vocals track...")
y, sr = librosa.load(TABLA_PATH, mono=True)
total_dur = len(y) / sr
print(f"Duration: {total_dur:.1f}s")

print("Separating tabla from tanpura using HPSS...")
# margin > 1 makes the separation more aggressive
# kernel_size controls the time/frequency resolution of separation
y_harmonic, y_percussive = librosa.effects.hpss(
    y,
    kernel_size=31,
    margin=4.0
)

# Verify separation worked
harm_rms = float(np.sqrt(np.mean(y_harmonic**2)))
perc_rms = float(np.sqrt(np.mean(y_percussive**2)))
print(f"Harmonic RMS (tanpura): {harm_rms:.4f}")
print(f"Percussive RMS (tabla): {perc_rms:.4f}")
print(f"Separation ratio: {harm_rms/max(perc_rms,0.0001):.1f}x")

if perc_rms < 0.001:
    print("WARNING: Percussive signal very weak — tabla may be quiet in this mix")

# ── Step 2: Onset detection on percussive only ────────────────────
print("\nDetecting tabla onsets on percussive component...")
hop = 512

# Use multiple onset detectors and combine
onset_env = librosa.onset.onset_strength(
    y=y_percussive,
    sr=sr,
    hop_length=hop,
    aggregate=np.median,
    fmax=8000   # tabla has energy up to ~8kHz
)

# Onset times
onset_frames = librosa.onset.onset_detect(
    onset_envelope=onset_env,
    sr=sr,
    hop_length=hop,
    backtrack=True,
    units='frames',
    pre_max=3,
    post_max=3,
    pre_avg=10,
    post_avg=10,
    delta=0.07,
    wait=4
)
onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop)
print(f"Onsets detected: {len(onset_times)}")

if len(onset_times) < 10:
    print("ERROR: Too few onsets detected. The tabla may be very quiet.")
    print("Falling back to full mix for onset detection...")
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr, hop_length=hop,
        backtrack=True, units='frames'
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop)
    print(f"Onsets from full mix: {len(onset_times)}")

# ── Step 3: Tempo estimation from percussive only ─────────────────
print("\nEstimating tempo from tabla...")
tempo_estimates = []

# Sample 10 windows across the recording (skip first and last 2 mins)
n_windows = 10
start_sec = 120
end_sec = total_dur - 120
if end_sec <= start_sec:
    start_sec = 30
    end_sec = total_dur - 30

window_sec = 20
window_starts = np.linspace(
    start_sec * sr,
    (end_sec - window_sec) * sr,
    n_windows
).astype(int)

for i, ws in enumerate(window_starts):
    seg = y_percussive[ws:ws + window_sec * sr]
    try:
        t, _ = librosa.beat.beat_track(
            y=seg, sr=sr,
            hop_length=hop,
            start_bpm=80,
            tightness=80
        )
        if isinstance(t, np.ndarray):
            t = float(t[0])
        if 20 <= t <= 600:
            tempo_estimates.append(t)
    except:
        pass

print(f"Tempo estimates across {len(tempo_estimates)} windows: "
      f"{[round(t,1) for t in tempo_estimates]}")

if tempo_estimates:
    median_tempo = float(np.median(tempo_estimates))
else:
    median_tempo = 80.0
    print("WARNING: Using default 80 BPM")

print(f"Median matra tempo: {median_tempo:.1f} BPM")

# ── Step 4: Autocorrelation to find cycle length ──────────────────
print("\nFinding taal cycle length via autocorrelation...")

# Compute onset envelope for autocorrelation
hop_ac = 256
onset_env_ac = librosa.onset.onset_strength(
    y=y_percussive, sr=sr, hop_length=hop_ac
)

# Autocorrelate — peaks reveal the cycle length
ac = librosa.autocorrelate(onset_env_ac, max_size=sr * 30 // hop_ac)
ac_times = np.arange(len(ac)) * hop_ac / sr  # convert to seconds

# Find peaks in autocorrelation
from scipy.signal import find_peaks
peaks, props = find_peaks(
    ac,
    height=np.max(ac) * 0.1,
    distance=sr * 3 // hop_ac   # minimum 3 second cycle
)

print("\nAutocorrelation peaks (possible cycle lengths):")
peak_times = ac_times[peaks]
peak_strengths = ac[peaks]

# Sort by strength
sorted_peaks = sorted(
    zip(peak_times, peak_strengths),
    key=lambda x: -x[1]
)
for t, strength in sorted_peaks[:10]:
    print(f"  {t:.2f}s  (strength: {strength:.2f})")

# ── Step 5: Match cycle lengths to taals ─────────────────────────
print("\nMatching cycle lengths to taals...")

matra_dur = 60.0 / median_tempo
print(f"Matra duration: {matra_dur:.3f}s")

taal_scores = {}
for taal_name, info in TAALS.items():
    matras = info["matras"]
    expected_cycle = matra_dur * matras

    # Score 1: how well does expected cycle match autocorrelation peaks?
    ac_score = 0.0
    for peak_t, peak_s in sorted_peaks[:10]:
        # Check if this peak is near expected cycle or its multiples/fractions
        for mult in [0.5, 1.0, 1.5, 2.0]:
            if abs(peak_t - expected_cycle * mult) < expected_cycle * 0.08:
                ac_score += float(peak_s) * (1.0 / mult)
    
    # Score 2: inter-onset-interval histogram fit
    # How many IOIs are close to matura duration?
    if len(onset_times) > 1:
        iois = np.diff(onset_times)
        ioi_score = 0.0
        for ioi in iois:
            # Check if IOI is close to 1, 2, or 4 matras
            for n_matras in [1, 2, 3, 4]:
                expected_ioi = matra_dur * n_matras
                if abs(ioi - expected_ioi) < matra_dur * 0.12:
                    ioi_score += 1.0 / n_matras
        ioi_score /= max(len(iois), 1)
    else:
        ioi_score = 0.0
    
    # Score 3: onset density — expected bols per matra
    # Tabla plays more bols per matra in some taals
    avg_onsets_per_sec = len(onset_times) / max(total_dur, 1)
    expected_bols_per_matra = 2.0  # conservative estimate
    density_score = 1.0 / (1.0 + abs(
        avg_onsets_per_sec - expected_bols_per_matra / matra_dur
    ))
    
    combined = ac_score * 0.5 + ioi_score * 0.35 + density_score * 0.15
    
    taal_scores[taal_name] = {
        "matras": matras,
        "expected_cycle_sec": round(expected_cycle, 3),
        "ac_score": round(ac_score, 4),
        "ioi_score": round(ioi_score, 4),
        "combined_score": round(combined, 4),
        "vibhag": info["vibhag"],
        "khali": info["khali"],
    }

ranked = sorted(taal_scores.items(),
                key=lambda x: -x[1]["combined_score"])

print("\nTaal ranking:")
print(f"{'Taal':<14} {'Matras':>6}  {'Cycle':>7}  "
      f"{'AC':>7}  {'IOI':>7}  {'Total':>7}")
print("-" * 58)
for name, info in ranked:
    print(f"{name:<14} {info['matras']:>6}  "
          f"{info['expected_cycle_sec']:>6.2f}s  "
          f"{info['ac_score']:>7.4f}  "
          f"{info['ioi_score']:>7.4f}  "
          f"{info['combined_score']:>7.4f}")

best_taal = ranked[0][0]
best_info = ranked[0][1]

# ── Step 6: Lay classification ────────────────────────────────────
if median_tempo < 50:
    lay = "Ati-Vilambit (very slow)"
elif median_tempo < 80:
    lay = "Vilambit (slow)"
elif median_tempo < 150:
    lay = "Madhya (medium)"
elif median_tempo < 250:
    lay = "Drut (fast)"
else:
    lay = "Ati-Drut (very fast)"

# ── Step 7: Sam time detection ────────────────────────────────────
print(f"\nDetecting sam positions for {best_taal}...")
cycle_dur = best_info["expected_cycle_sec"]

# Use beat times for sam estimation
tempo_full, beat_frames = librosa.beat.beat_track(
    y=y_percussive,
    sr=sr,
    hop_length=hop,
    bpm=median_tempo,
    tightness=80
)
beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)

# Sam = every Nth beat where N = matras
matras = best_info["matras"]
sam_times = []

if len(beat_times) >= matras:
    # Find the best phase alignment using onset strength at cycle positions
    best_phase_score = -1
    best_offset = 0
    
    for offset in range(min(matras, len(beat_times))):
        score = 0.0
        for i in range(offset, len(beat_times), matras):
            t = beat_times[i]
            # Find onset strength at this time
            frame = int(t * sr / hop)
            if frame < len(onset_env):
                score += onset_env[frame]
        if score > best_phase_score:
            best_phase_score = score
            best_offset = offset
    
    # Extract sam times
    for i in range(best_offset, len(beat_times), matras):
        sam_times.append(round(float(beat_times[i]), 3))

print(f"Sam positions: {len(sam_times)} cycles detected")
print(f"First 8 sams: {sam_times[:8]}")

# ── Step 8: Save ──────────────────────────────────────────────────
result = {
    "detected_taal": best_taal,
    "matras": best_info["matras"],
    "vibhag": best_info["vibhag"],
    "khali_matras": best_info["khali"],
    "cycle_duration_sec": best_info["expected_cycle_sec"],
    "matra_duration_sec": round(matra_dur, 4),
    "matra_bpm": round(median_tempo, 2),
    "lay": lay,
    "confidence": (
        "high"   if best_info["combined_score"] > 0.3  else
        "medium" if best_info["combined_score"] > 0.15 else
        "low"
    ),
    "all_taal_scores": {
        k: v["combined_score"] for k, v in ranked
    },
    "top_autocorr_peaks_sec": [
        round(float(t), 3) for t, _ in sorted_peaks[:5]
    ],
    "sam_times": sam_times[:300],
    "total_cycles_detected": len(sam_times),
    "separation_quality": {
        "harmonic_rms": round(harm_rms, 5),
        "percussive_rms": round(perc_rms, 5),
        "ratio": round(harm_rms / max(perc_rms, 0.0001), 2)
    }
}

with open('detected_taal.json', 'w') as f:
    json.dump(result, f, indent=2)

print(f"\n{'='*55}")
print(f"RESULT:    {best_taal} ({best_info['matras']} matras)")
print(f"Tempo:     {median_tempo:.1f} BPM — {lay}")
print(f"Cycle:     {best_info['expected_cycle_sec']:.2f}s per avart")
print(f"Vibhag:    {best_info['vibhag']}")
print(f"Khali at:  matra {best_info['khali']}")
print(f"Confidence: {result['confidence']}")
print(f"\nTop autocorr peaks: {result['top_autocorr_peaks_sec']}")
print(f"\nSaved to detected_taal.json")
print(f"\nNOTE: If confidence is low or taal seems wrong,")
print(f"check top_autocorr_peaks_sec — the true cycle length")
print(f"in seconds should appear as one of those peaks.")
print(f"Divide by matra_duration_sec ({matra_dur:.3f}s) to get matras.")
---

Run: python detect_taal_v2.py


## Step 2 — After running, do this check

Look at the output:
- top_autocorr_peaks_sec shows the strongest periodicities found
- Divide each peak by matra_duration_sec
- The result that is closest to a whole number from this list
  [6, 7, 8, 10, 12, 14, 16] is the most likely matra count

Example: peaks = [9.8s, 4.9s], matra_dur = 0.77s
  9.8 / 0.77 = 12.7 ≈ not clean
  9.8 / 0.61 = 16.0 ← clean! → Teentaal

Report this calculation to the user so they can verify.


## Step 3 — If confidence is low, ask the user

If the JSON shows confidence = "low" OR all_taal_scores are 
very close together (difference < 0.05), print this message:

  "Automatic taal detection has low confidence.
   The most likely candidates based on tempo are:
   [list top 3 from ranking]
   
   Can you confirm the taal from the liner notes or by ear?
   If you tell me the taal name I will hardcode it and the
   matra tracking will be precise."

Then wait for user confirmation before updating index.html.


## Step 4 — If user confirms taal manually

If the user tells you the correct taal, run this patch:

  import json
  with open('detected_taal.json') as f:
      data = json.load(f)
  
  TAALS = {
      "Teentaal":   {"matras":16,"vibhag":[4,4,4,4],"khali":[9]},
      "Tilwada":    {"matras":16,"vibhag":[4,4,4,4],"khali":[9]},
      "Jhoomra":    {"matras":14,"vibhag":[3,4,3,4],"khali":[8]},
      "Deepchandi": {"matras":14,"vibhag":[3,4,3,4],"khali":[8]},
      "Ektaal":     {"matras":12,"vibhag":[2,2,2,2,2,2],"khali":[7,11]},
      "Chartal":    {"matras":12,"vibhag":[2,2,2,2,2,2],"khali":[7,11]},
      "Jhaptaal":   {"matras":10,"vibhag":[2,3,2,3],"khali":[6]},
      "Keherwa":    {"matras":8, "vibhag":[4,4],"khali":[5]},
      "Rupak":      {"matras":7, "vibhag":[3,2,2],"khali":[1]},
      "Dadra":      {"matras":6, "vibhag":[3,3],"khali":[4]},
  }
  
  correct_taal = "TAAL_NAME_HERE"  # replace with user input
  info = TAALS[correct_taal]
  
  data["detected_taal"] = correct_taal
  data["matras"] = info["matras"]
  data["vibhag"] = info["vibhag"]
  data["khali_matras"] = info["khali"]
  data["cycle_duration_sec"] = round(
      data["matra_duration_sec"] * info["matras"], 3
  )
  data["confidence"] = "high (user confirmed)"
  
  with open('detected_taal.json', 'w') as f:
      json.dump(data, f, indent=2)
  print(f"Updated to {correct_taal}")


## Step 5 — Update index.html
Only update index.html after taal is confirmed (auto or manual).
The taal display requirements are the same as in DETECT_TAAL.md Step 4.


## Notes for Claude Code
- HPSS is the critical improvement — always verify the separation
  ratio printed at the start. If harmonic_rms / percussive_rms > 20
  the tanpura is overwhelming and results may still be unreliable
- The autocorrelation peaks are the most honest output — 
  always show them to the user
- Do NOT silently assume a taal if confidence is low — 
  always ask the user to confirm
- madmom is not required in this version
