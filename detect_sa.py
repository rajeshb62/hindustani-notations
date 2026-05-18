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
