"""
Deeper analysis of tanpura peaks to identify Sa via harmonic series.
The tanpura's 2nd harmonic (octave) is often stronger than the fundamental.
"""
import numpy as np
import wave, math

TANPURA = "separated/htdemucs/ICCR-1854-AC_SIDE_B/tanpura_only.wav"

def load_wav_mono(path, max_seconds=120):
    with wave.open(path, 'rb') as w:
        sr       = w.getframerate()
        n_chan   = w.getnchannels()
        sw       = w.getsampwidth()
        n_frames = min(w.getnframes(), sr * max_seconds)
        raw      = w.readframes(n_frames)
    fmt = np.int16 if sw == 2 else np.int32
    div = 32768.0 if sw == 2 else 2147483648.0
    s = np.frombuffer(raw, dtype=fmt).astype(np.float32) / div
    if n_chan > 1:
        s = s.reshape(-1, n_chan).mean(axis=1)
    return s, sr

print("Loading 120s of tanpura...")
samples, sr = load_wav_mono(TANPURA, max_seconds=120)

# Use longer FFT for better frequency resolution
N   = len(samples)
fft = np.abs(np.fft.rfft(samples * np.hanning(N)))
freq = np.fft.rfftfreq(N, 1.0 / sr)
res  = freq[1] - freq[0]
print(f"  FFT resolution: {res:.4f} Hz/bin  (N={N})")

def find_peaks(fft, freq, f_lo, f_hi, n=30, min_gap_hz=1.5):
    lo = np.searchsorted(freq, f_lo)
    hi = np.searchsorted(freq, f_hi)
    s = fft[lo:hi].copy()
    f = freq[lo:hi]
    gap = max(1, int(min_gap_hz / res))
    peaks = []
    for _ in range(n):
        idx = np.argmax(s)
        peaks.append((f[idx], s[idx]))
        s[max(0, idx-gap):idx+gap] = 0
    return sorted(peaks, key=lambda x: -x[1])

# ── Score each Sa candidate ─────────────────────────────────
# For each candidate Sa, check how well its harmonic series (×1, ×2, ×3)
# aligns with the actual spectrum.
candidates = [88.0, 90.0, 92.0, 92.28, 93.0, 94.6, 96.0, 96.97, 98.0, 99.0, 100.0]

peaks_low  = find_peaks(fft, freq,  50,  260)
peaks_all  = find_peaks(fft, freq,  50, 1000, n=60)

def peak_power_near(hz, tolerance_cents=25):
    """Sum of FFT power within tolerance_cents of hz."""
    low  = hz * 2**(-tolerance_cents/1200)
    high = hz * 2**( tolerance_cents/1200)
    lo   = np.searchsorted(freq, low)
    hi   = np.searchsorted(freq, high)
    return float(np.sum(fft[lo:hi]))

print(f"\nTop 15 raw peaks (50–260 Hz):")
for hz, amp in peaks_low[:15]:
    print(f"  {hz:8.3f} Hz   amp={amp:,.0f}")

print(f"\nSa candidate scoring (harmonics 1–4):")
print(f"  {'Sa':>8}   {'H1':>12}  {'H2':>12}  {'H3':>12}  {'H4':>12}  {'total':>12}")
scores = []
for sa in candidates:
    h = [peak_power_near(sa * k) for k in [1, 2, 3, 4]]
    total = sum(h)
    scores.append((sa, h, total))
    print(f"  {sa:8.2f}   {h[0]:12,.0f}  {h[1]:12,.0f}  {h[2]:12,.0f}  {h[3]:12,.0f}  {total:12,.0f}")

best = max(scores, key=lambda x: x[2])
print(f"\n{'='*55}")
print(f"  STRONGEST HARMONIC SUPPORT:  Sa = {best[0]:.2f} Hz")
print(f"{'='*55}")

# Compare cents distance from our two main candidates
def cents(a, b):
    return 1200 * math.log2(a / b)

print(f"\n  vs 98.0 Hz:  {cents(best[0], 98.0):+.1f} cents")
print(f"  vs 92.5 Hz:  {cents(best[0], 92.5):+.1f} cents")
print(f"\n  (100 cents = 1 semitone)")
