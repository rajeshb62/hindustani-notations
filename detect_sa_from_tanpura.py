"""
Detect Sa from the tanpura drone track.

The tanpura continuously drones Sa + Pa (+ octave Sa), making it
the most reliable source for Sa detection. We look for the two
strongest low-frequency peaks and check if their ratio is ~3:2 (Sa:Pa).

Usage:
    python detect_sa_from_tanpura.py
"""

import numpy as np
import wave, struct, math

TANPURA = "separated/htdemucs/ICCR-1854-AC_SIDE_B/tanpura_only.wav"

# ── Load a chunk of the tanpura (first 60s is plenty) ──────
def load_wav_mono(path, max_seconds=60):
    with wave.open(path, 'rb') as w:
        sr        = w.getframerate()
        n_chan     = w.getnchannels()
        sampwidth  = w.getsampwidth()
        n_frames   = min(w.getnframes(), sr * max_seconds)
        raw        = w.readframes(n_frames)

    if sampwidth == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    if n_chan > 1:
        samples = samples.reshape(-1, n_chan).mean(axis=1)

    return samples, sr

print("Loading tanpura track (first 60s)...")
samples, sr = load_wav_mono(TANPURA)
print(f"  Sample rate: {sr} Hz  |  Samples: {len(samples)}  |  Duration: {len(samples)/sr:.1f}s")

# ── FFT over the full chunk ─────────────────────────────────
print("Running FFT...")
N    = len(samples)
fft  = np.abs(np.fft.rfft(samples * np.hanning(N)))
freq = np.fft.rfftfreq(N, 1.0 / sr)

# Focus on 50–250 Hz (tanpura Sa range for typical Hindustani)
lo = np.searchsorted(freq, 50)
hi = np.searchsorted(freq, 250)
fft_band = fft[lo:hi]
freq_band = freq[lo:hi]

# ── Find top peaks in the band ──────────────────────────────
def find_peaks(spectrum, freqs, n_peaks=20, min_gap_hz=2.0):
    peaks = []
    min_gap = int(min_gap_hz / (freqs[1] - freqs[0]))
    s = spectrum.copy()
    for _ in range(n_peaks):
        idx = np.argmax(s)
        peaks.append((freqs[idx], s[idx]))
        lo_g = max(0, idx - min_gap)
        hi_g = min(len(s), idx + min_gap)
        s[lo_g:hi_g] = 0
    return sorted(peaks, key=lambda x: -x[1])

peaks = find_peaks(fft_band, freq_band, n_peaks=30)

print(f"\nTop 10 peaks (50–250 Hz):")
for hz, amp in peaks[:10]:
    print(f"  {hz:7.2f} Hz   amplitude={amp:.0f}")

# ── Look for Sa+Pa pairs (ratio ~1.5) ──────────────────────
print(f"\nSearching for Sa+Pa pairs (ratio 1.490–1.510)...")
candidates = []
top_peaks = peaks[:20]
for i, (f1, a1) in enumerate(top_peaks):
    for j, (f2, a2) in enumerate(top_peaks):
        if i == j: continue
        ratio = f2 / f1
        if 1.490 <= ratio <= 1.510:   # Pa/Sa = 3/2 = 1.5
            combined = a1 + a2
            candidates.append((f1, f2, ratio, combined))
            print(f"  Sa candidate={f1:.2f} Hz  Pa={f2:.2f} Hz  ratio={ratio:.4f}  combined_amp={combined:.0f}")

# Also check for upper Sa (octave) confirming Sa
print(f"\nSearching for Sa+octave Sa pairs (ratio ~2.0)...")
for i, (f1, a1) in enumerate(top_peaks):
    for j, (f2, a2) in enumerate(top_peaks):
        if i == j: continue
        ratio = f2 / f1
        if 1.95 <= ratio <= 2.05:
            print(f"  Sa={f1:.2f} Hz  upper Sa={f2:.2f} Hz  ratio={ratio:.4f}")

# ── Best Sa estimate ────────────────────────────────────────
if candidates:
    # Weight by combined amplitude
    best = max(candidates, key=lambda x: x[3])
    sa_est = best[0]
    pa_est = best[1]
    print(f"\n{'='*50}")
    print(f"  BEST Sa ESTIMATE FROM TANPURA: {sa_est:.2f} Hz")
    print(f"  Corresponding Pa:              {pa_est:.2f} Hz")
    print(f"  Ratio:                         {best[2]:.4f}  (ideal=1.5000)")
    print(f"{'='*50}")
    print(f"\n  Comparison:")
    print(f"    detected_sa.json:  98.0 Hz")
    print(f"    optimizer (v2):    92.5 Hz")
    print(f"    tanpura analysis:  {sa_est:.2f} Hz")

    # Cents difference from both candidates
    def cents_diff(a, b):
        return abs(1200 * math.log2(a / b))

    print(f"\n  Cents from 98.0 Hz:  {cents_diff(sa_est, 98.0):.1f} cents")
    print(f"  Cents from 92.5 Hz:  {cents_diff(sa_est, 92.5):.1f} cents")
else:
    print("\nNo clear Sa+Pa pair found. Showing top 5 peaks for manual inspection:")
    for hz, amp in peaks[:5]:
        print(f"  {hz:.2f} Hz")
