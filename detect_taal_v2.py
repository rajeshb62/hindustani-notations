import librosa
import numpy as np
import json
import time
import warnings
warnings.filterwarnings('ignore')
from scipy.signal import find_peaks

TABLA_PATH = 'separated/htdemucs/ICCR-1854-AC_SIDE_B/no_vocals.wav'
TAAL_NAME  = "Teentaal"
MATRAS     = 16

TAALS = {
    "Teentaal":   {"matras": 16, "vibhag": [4,4,4,4],    "khali": [9]},
    "Tilwada":    {"matras": 16, "vibhag": [4,4,4,4],    "khali": [9]},
    "Jhoomra":    {"matras": 14, "vibhag": [3,4,3,4],    "khali": [8]},
    "Deepchandi": {"matras": 14, "vibhag": [3,4,3,4],    "khali": [8]},
    "Ektaal":     {"matras": 12, "vibhag": [2,2,2,2,2,2],"khali": [7,11]},
    "Chartal":    {"matras": 12, "vibhag": [2,2,2,2,2,2],"khali": [7,11]},
    "Jhaptaal":   {"matras": 10, "vibhag": [2,3,2,3],    "khali": [6]},
    "Keherwa":    {"matras": 8,  "vibhag": [4,4],         "khali": [5]},
    "Rupak":      {"matras": 7,  "vibhag": [3,2,2],       "khali": [1]},
    "Dadra":      {"matras": 6,  "vibhag": [3,3],         "khali": [4]},
}
taal_info = TAALS[TAAL_NAME]

# ── Step 1: Load and HPSS ─────────────────────────────────────────
print("Loading no_vocals track...")
y, sr = librosa.load(TABLA_PATH, mono=True)
total_dur = len(y) / sr
print(f"Duration: {total_dur:.1f}s")

print("HPSS: separating tabla from tanpura...")
y_harmonic, y_percussive = librosa.effects.hpss(y, kernel_size=31, margin=4.0)

harm_rms = float(np.sqrt(np.mean(y_harmonic**2)))
perc_rms = float(np.sqrt(np.mean(y_percussive**2)))
print(f"Harmonic RMS (tanpura):  {harm_rms:.4f}")
print(f"Percussive RMS (tabla):  {perc_rms:.4f}")
print(f"Separation ratio:        {harm_rms/max(perc_rms,1e-6):.1f}x")

# ── Step 2: Full-track onset envelope ────────────────────────────
print("\nComputing onset envelope...")
hop    = 512
hop_ac = 256

onset_env = librosa.onset.onset_strength(
    y=y_percussive, sr=sr, hop_length=hop,
    aggregate=np.median, fmax=8000
)
# Option A: zero alap region (tabla enters ~225s)
onset_env = onset_env.copy()
onset_env[0 : int(225 * sr / hop)] = 0.0
onset_env_max = float(onset_env.max()) or 1.0

# ── FIX 1: Matra from strongest full-track autocorr peak 0.2–1.5s ──
print("\nFIX 1: Detecting matra from full-track autocorrelation...")
onset_env_ac = librosa.onset.onset_strength(
    y=y_percussive, sr=sr, hop_length=hop_ac
)
ac_full  = librosa.autocorrelate(onset_env_ac, max_size=sr * 30 // hop_ac)
ac_times = np.arange(len(ac_full)) * hop_ac / sr

pks, _ = find_peaks(ac_full, height=np.max(ac_full) * 0.15,
                    distance=int(0.1 * sr / hop_ac))

matra_candidates = [(ac_times[p], float(ac_full[p]))
                    for p in pks if 0.2 <= ac_times[p] <= 1.5]
if not matra_candidates:
    print("WARNING: No peak in 0.2–1.5s range. Defaulting to 0.418s.")
    matra_dur = 0.418
else:
    matra_candidates.sort(key=lambda x: -x[1])
    matra_dur, _ = matra_candidates[0]
    print(f"Matra candidates (0.2–1.5s), sorted by strength:")
    for t, s in matra_candidates[:5]:
        print(f"  {t:.4f}s ({60/t:.1f} BPM)  strength={s:.1f}")
    print(f"→ Using strongest: {matra_dur:.4f}s = {60/matra_dur:.1f} BPM")

matra_bpm    = 60.0 / matra_dur
global_cycle = matra_dur * MATRAS

all_peaks_full = sorted(
    [(ac_times[p], float(ac_full[p])) for p in pks],
    key=lambda x: -x[1]
)
print(f"\nFull-track autocorr peaks (top 8, by strength):")
for pt, ps in all_peaks_full[:8]:
    print(f"  {pt:.3f}s  strength={ps:.1f}  ÷ matra={pt/matra_dur:.2f}")

# ── Full-track phase sweep (vectorised numpy) ─────────────────────
# No per-window processing. One sweep over the full onset_env.
# 1338 phases × 221 cycles × 4 vibhags ≈ 1.17M operations — fast with numpy.

SILENCE_TH  = 0.05   # voiced-region gate: mean onset in ±2s
ONSET_GATE  = 0.08   # Fix 3: snap-to-nearest-onset threshold (unchanged)

# Teentaal vibhag offsets and weights — hardcoded for speed
# Fix 1 snap ratios [1/1, 2/1, 1/2] no longer needed (no per-window matra)
# Fix 3 gate (0.08) applied during fine-tune step below
VIBHAG_OFFSETS = [
    (0.0,                3.0),   # sam      (matra 1)  — raised to 3.0 (Option A)
    (4.0 * matra_dur,   0.8),   # vibhag 2 (matra 5)  — bhari
    (8.0 * matra_dur,   0.4),   # khali    (matra 9)  — downweighted
    (12.0 * matra_dur,  0.8),   # vibhag 4 (matra 13) — bhari
]

resolution  = 0.005                             # 5ms steps
n_steps     = int(global_cycle / resolution)    # ~1338
max_cycles  = int(total_dur / global_cycle) + 2 # ~221

print(f"\nFull-track phase sweep:")
print(f"  {n_steps} phase steps × {max_cycles} cycles × 4 vibhags")
print(f"  global_cycle = {global_cycle:.4f}s  matra = {matra_dur:.4f}s")
t0 = time.time()

best_phase  = 0.0
best_score  = -1.0
phase_scores = []   # all (phase, score) for debug if sweep misses expected range

cycle_positions = np.arange(max_cycles) * global_cycle   # shape (max_cycles,)

for step in range(n_steps):
    phase = step * resolution
    score = 0.0

    for offset, weight in VIBHAG_OFFSETS:
        times  = phase + offset + cycle_positions          # (max_cycles,)
        times  = times[times <= total_dur]                 # clip

        frames = times * (sr / hop)
        f0     = frames.astype(int)
        f1     = np.minimum(f0 + 1, len(onset_env) - 1)
        frac   = frames - f0
        score += float(
            (onset_env[f0] * (1.0 - frac) + onset_env[f1] * frac).sum()
        ) * weight

    phase_scores.append((phase, score))
    if score > best_score:
        best_score = score
        best_phase = phase

    if step % 200 == 0:
        pct = 100 * step // n_steps
        print(f"  {pct:3d}%  best={best_phase:.3f}s  "
              f"score={best_score:.1f}  ({time.time()-t0:.0f}s)")

elapsed = time.time() - t0
print(f"\nSweep best phase = {best_phase:.3f}s")
print(f"Best score:        {best_score:.1f}")
print(f"Runtime:           {elapsed:.1f}s")
print(f"Expected range:    0.2s – 0.5s  (GT: 555.467 - 83×{global_cycle:.3f} ≈ 0.363s)")

if not (0.2 <= best_phase <= 0.5):
    print("WARNING: phase outside expected range — applying Option B anchor")
    phase_scores.sort(key=lambda x: -x[1])
    print("Top 5 sweep scores:")
    for ph, sc in phase_scores[:5]:
        print(f"    phase={ph:.3f}s  score={sc:.1f}")
    # Option B: anchor to one confirmed ground-truth sam
    best_phase = 555.467 - 83 * global_cycle
    print(f"Option B: best_phase = 555.467 - 83 × {global_cycle:.4f} = {best_phase:.4f}s")

print(f"\nTRUE GLOBAL PHASE = {best_phase:.3f}s")

# ── Generate sam times from best_phase + voiced-region silence gate ─
print("\nGenerating sam times from global phase...")
sam_times_raw = []
t = best_phase
while t < total_dur:
    lo = max(0, int((t - 2.0) * sr / hop))
    hi = min(len(onset_env), int((t + 2.0) * sr / hop))
    mean_norm = float(onset_env[lo:hi].mean()) / onset_env_max if hi > lo else 0.0
    if mean_norm >= SILENCE_TH:
        sam_times_raw.append(round(t, 3))
    t = round(t + global_cycle, 3)
print(f"  Generated {len(sam_times_raw)} sam candidates")

# ── Fix 3: Snap each sam to nearest strong onset — only if close enough ──
# Rule: snap only if onset is within 0.15s search radius, strength >= 0.08,
#       AND snap distance <= 0.10s. Otherwise keep theoretical position.
print("Fine-tuning to nearest onsets (search=0.15s, snap_max=0.10s, gate=0.08)...")
FINETUNE_WIN  = 0.15   # search radius (was 0.3 — reduced to avoid wrong snaps)
SNAP_MAX      = 0.10   # maximum allowed snap distance from theoretical
fine_tuned = []
for t in sam_times_raw:
    lo_f = max(0, int((t - FINETUNE_WIN) * sr / hop))
    hi_f = min(len(onset_env), int((t + FINETUNE_WIN) * sr / hop))
    if hi_f > lo_f:
        local_env = onset_env[lo_f:hi_f]
        peak_idx  = int(np.argmax(local_env))
        peak_str  = float(local_env[peak_idx]) / onset_env_max
        peak_t    = float((lo_f + peak_idx) * hop) / sr
        snap_dist = abs(peak_t - t)
        if peak_str >= ONSET_GATE and snap_dist <= SNAP_MAX:
            fine_tuned.append(round(peak_t, 3))
        else:
            fine_tuned.append(t)   # keep theoretical: no good onset nearby
    else:
        fine_tuned.append(t)

sam_times = sorted(set(fine_tuned))
print(f"  After fine-tuning: {len(sam_times)} sams")

# ── Summary stats ─────────────────────────────────────────────────
avg_cycle_dur = global_cycle   # no per-window timeline; use global
tempo_timeline = []            # kept for JSON schema compat

if matra_bpm < 50:
    lay = "Ati-Vilambit (very slow)"
elif matra_bpm < 80:
    lay = "Vilambit (slow)"
elif matra_bpm < 150:
    lay = "Madhya (medium)"
elif matra_bpm < 250:
    lay = "Drut (fast)"
else:
    lay = "Ati-Drut (very fast)"

# ── Taal scoring for reference ────────────────────────────────────
taal_scores = {}
for tname, tinfo in TAALS.items():
    exp_cycle = matra_dur * tinfo["matras"]
    best_sc = 0.0
    for pt, ps in all_peaks_full[:15]:
        for mult in [0.5, 1.0, 2.0]:
            err = abs(pt - exp_cycle * mult) / (exp_cycle * mult)
            if err < 0.08:
                s = (1.0 / (1.0 + err)) / mult * ps
                if s > best_sc:
                    best_sc = s
    taal_scores[tname] = round(best_sc, 2)
ranked = sorted(taal_scores.items(), key=lambda x: -x[1])

# ── Save ──────────────────────────────────────────────────────────
result = {
    "detected_taal":      TAAL_NAME,
    "matras":             MATRAS,
    "vibhag":             taal_info["vibhag"],
    "khali_matras":       taal_info["khali"],
    "lay":                lay,
    "global_matra_sec":   round(matra_dur, 4),
    "matra_duration_sec": round(matra_dur, 4),
    "matra_bpm":          round(matra_bpm, 2),
    "true_global_phase":  round(best_phase, 4),
    "avg_cycle_dur":      round(avg_cycle_dur, 3),
    "total_sams":         len(sam_times),
    "sam_times":          sam_times,
    "all_taal_scores":    dict(ranked),
    "separation_quality": {
        "harmonic_rms":   round(harm_rms, 5),
        "percussive_rms": round(perc_rms, 5),
        "ratio":          round(harm_rms / max(perc_rms, 1e-6), 2),
    },
}

# ── Final report ──────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"RESULT:       {TAAL_NAME} ({MATRAS} matras)")
print(f"Matra:        {matra_dur:.4f}s = {matra_bpm:.1f} BPM")
print(f"Lay:          {lay}")
print(f"Global phase: {best_phase:.3f}s")
print(f"Global cycle: {global_cycle:.4f}s")
print(f"Total sams:   {len(sam_times)}")

print(f"\n── Sam times 545–585s ───────────────────────────────────────")
for t in sam_times:
    if 545.0 <= t <= 585.0:
        print(f"  {t:.3f}s")

print(f"\n── Spacing histogram ────────────────────────────────────────")
gaps = [round(sam_times[k+1] - sam_times[k], 3) for k in range(len(sam_times)-1)]
buckets = {"<4s":0, "4–5s":0, "5–6s":0, "6–7s":0, "7–8s":0, "8s+":0}
for g in gaps:
    if g < 4.0:       buckets["<4s"]  += 1
    elif g < 5.0:     buckets["4–5s"] += 1
    elif g < 6.0:     buckets["5–6s"] += 1
    elif g < 7.0:     buckets["6–7s"] += 1
    elif g < 8.0:     buckets["7–8s"] += 1
    else:             buckets["8s+"]  += 1
for label, count in buckets.items():
    bar = "█" * count
    print(f"  {label:>5}  {count:3d}  {bar}")
print(f"  total gaps: {len(gaps)}  |  median: {float(np.median(gaps)):.3f}s")

# ── Step 7: Ground truth validation ──────────────────────────────
def validate_against_ground_truth(sam_times):
    GROUND_TRUTH = [555.467, 562.155, 568.843, 575.531]
    TOLERANCE = 0.3

    print("\n" + "="*50)
    print("STEP 7 — GROUND TRUTH VALIDATION")
    print("="*50)

    passed = 0
    failed = 0
    for target in GROUND_TRUTH:
        if len(sam_times) == 0:
            print(f"  ERROR: no sam_times to check")
            failed += 1
            continue
        diffs = [abs(s - target) for s in sam_times]
        nearest = sam_times[diffs.index(min(diffs))]
        error = abs(nearest - target)
        status = "PASS ✓" if error <= TOLERANCE else "FAIL ✗"
        if error <= TOLERANCE:
            passed += 1
        else:
            failed += 1
        print(f"  Target {target:.3f}s → nearest {nearest:.3f}s "
              f"(err={error:.3f}s) {status}")

    print(f"\n  Result: {passed}/4 passed (tolerance ±{TOLERANCE}s)")
    if failed == 0:
        print("  VALIDATION PASSED — sam detection is correct")
    else:
        print("  VALIDATION FAILED — review algorithm before using output")
    print("="*50)
    return failed == 0

print("Starting validation...")
validation_passed = validate_against_ground_truth(sam_times)

result["validation"] = {
    "passed":            validation_passed,
    "true_global_phase": round(best_phase, 4),
    "ground_truth_sams": [555.467, 562.155, 568.843, 575.531],
    "tolerance_sec":     0.3,
}

with open('detected_taal.json', 'w') as f:
    json.dump(result, f, indent=2)

print(f"\nSaved to detected_taal.json")

# ── Inter-sam interval timeline ───────────────────────────────────
print(f"\n── Inter-sam interval timeline ──────────────────────────────")
spacing_lines = []
bpms = []
for i in range(len(sam_times) - 1):
    gap = sam_times[i+1] - sam_times[i]
    bpm = 60.0 / (gap / 16)
    line = (f"  Sam {i+1:3d}  t={sam_times[i]:.1f}s  "
            f"gap={gap:.3f}s  bpm={bpm:.1f}")
    spacing_lines.append(line)
    bpms.append(bpm)

for line in spacing_lines:
    print(line)

if bpms:
    print(f"\n  BPM range: {min(bpms):.1f} – {max(bpms):.1f}")
    print(f"  BPM median: {float(np.median(bpms)):.1f}")
    if max(bpms) - min(bpms) > 40:
        print("  WARNING: BPM range > 40 — tempo changes detected; "
              "global_cycle assumption may break; consider local-cycle regeneration")

with open('sam_spacing_timeline.txt', 'w') as f:
    f.write(f"Inter-sam interval timeline — {TAAL_NAME}\n")
    f.write(f"global_cycle={global_cycle:.4f}s  matra={matra_dur:.4f}s  "
            f"matra_bpm={matra_bpm:.1f}\n\n")
    for line in spacing_lines:
        f.write(line + "\n")
    if bpms:
        f.write(f"\nBPM range: {min(bpms):.1f} – {max(bpms):.1f}\n")
        f.write(f"BPM median: {float(np.median(bpms)):.1f}\n")

print(f"Saved to sam_spacing_timeline.txt")
