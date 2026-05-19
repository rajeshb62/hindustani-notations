"""
Self-improving sargam extraction v2.

Three upgrades over v1:
  1. Sub-cycle scoring  — each taal cycle split into 4 quarters → 4× more
                          comparison signal from the same fixed sam marks
  2. DTW contour score  — compares melodic shape (up/down movement) not just
                          which notes appear; catches ornaments and glides
  3. Differential evolution + iterative refinement — global optimizer that
                          zooms into the best region across 3 rounds

Combined score = 0.65 × sub-cycle chroma  +  0.35 × DTW contour

Usage:
    python self_improve.py

Each run warm-starts from best_params.json if it exists.
Outputs:
    best_params.json               updated best parameters
    sargam_notation_improved.txt   notation re-extracted with best params
    improvement_log.csv            score for every evaluation
"""

import math, json, csv, time, bisect
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cosine as cosine_distance

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
SARGAM = [
    (0,    "S"),
    (100,  "r"), (200,  "R"),
    (300,  "g"), (400,  "G"),
    (500,  "M"), (600,  "M+"),
    (700,  "P"),
    (800,  "d"), (900,  "D"),
    (1000, "n"), (1100, "N"),
]
SWARA_SEMITONE = {
    "S":0, "r":1, "R":2, "g":3, "G":4,
    "M":5, "M+":6, "P":7, "d":8, "D":9, "n":10, "N":11,
}
NOTE_NAMES = {
    "S":"Sa", "R":"Re shuddh", "r":"Re komal",
    "G":"Ga shuddh", "g":"Ga komal",
    "M":"Ma shuddh", "M+":"Ma tivra", "P":"Pa",
    "D":"Dha shuddh", "d":"Dha komal",
    "N":"Ni shuddh",  "n":"Ni komal",
}

SUB_WINDOWS   = 4          # quarters per taal cycle
DTW_SAMPLES   = 24         # pitch-sequence length per sub-window for DTW
DTW_BAND      = 6          # Sakoe-Chiba band width (limits DTW search)
SCORE_WEIGHTS = (0.65, 0.35)  # (chroma_weight, dtw_weight)

# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────
def load_crepe(path="vocals.f0.vad.csv"):
    frames = []   # list of (time, hz, conf)
    with open(path) as f:
        next(f)
        for line in f:
            p = line.strip().split(",")
            if len(p) == 3:
                try:
                    t, hz, c = float(p[0]), float(p[1]), float(p[2])
                    if hz > 0:
                        frames.append((t, hz, c))
                except ValueError:
                    pass
    frames.sort(key=lambda x: x[0])
    return frames

def load_sams(path="sam_times_manual.json"):
    with open(path) as f:
        data = json.load(f)
    return sorted(float(t) for t in data.get("sam_times", data))

def load_prev_best(path="best_params.json"):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None

# ─────────────────────────────────────────────────────────────
# FAST FRAME LOOKUP  (binary-search index into sorted frames)
# ─────────────────────────────────────────────────────────────
def make_time_index(frames):
    """Return sorted list of timestamps for fast bisect lookups."""
    return [f[0] for f in frames]

def frames_in_range(frames, times_idx, t0, t1):
    lo = bisect.bisect_left(times_idx, t0)
    hi = bisect.bisect_right(times_idx, t1)
    return frames[lo:hi]

# ─────────────────────────────────────────────────────────────
# CHROMA UTILITIES
# ─────────────────────────────────────────────────────────────
def hz_to_chroma(hz):
    if hz < 30: return None
    return int(round(69 + 12 * math.log2(hz / 440.0))) % 12

def sa_bin(sa_hz):
    return hz_to_chroma(sa_hz)

def swara_to_chroma(swara, sa_hz):
    offset = SWARA_SEMITONE.get(swara)
    if offset is None: return None
    sb = sa_bin(sa_hz)
    return (sb + offset) % 12 if sb is not None else None

def gt_chroma_hist(frames, times_idx, t0, t1):
    """Chroma histogram from CREPE frames in window, confidence-weighted."""
    h = np.zeros(12)
    for t, hz, c in frames_in_range(frames, times_idx, t0, t1):
        if c < 0.5: continue
        b = hz_to_chroma(hz)
        if b is not None:
            h[b] += c
    n = np.linalg.norm(h)
    return h / n if n > 0 else h

def synth_chroma_hist(notes, sa_hz, t0, t1):
    """Chroma histogram from note sequence in window, duration-weighted."""
    h = np.zeros(12)
    for start, dur, swara, octave in notes:
        end = start + dur
        overlap = min(end, t1) - max(start, t0)
        if overlap <= 0: continue
        b = swara_to_chroma(swara, sa_hz)
        if b is not None:
            h[b] += overlap
    n = np.linalg.norm(h)
    return h / n if n > 0 else h

# ─────────────────────────────────────────────────────────────
# DTW CONTOUR SCORING
# ─────────────────────────────────────────────────────────────
def chroma_dist(a, b):
    """Circular chroma distance, 0 (identical) to 1 (tritone)."""
    if a is None or b is None: return 1.0
    d = abs(int(a) - int(b))
    return min(d, 12 - d) / 6.0

def build_pitch_sequence(frames, times_idx, t0, t1, n_samples):
    """
    Resample the GT pitch track to n_samples evenly-spaced points.
    Returns list of chroma bins (None for silent/unvoiced frames).
    """
    if t0 >= t1 or n_samples < 2: return [None] * n_samples
    step = (t1 - t0) / n_samples
    seq = []
    for i in range(n_samples):
        tc = t0 + (i + 0.5) * step
        # Find nearest CREPE frame within ±step/2
        lo = bisect.bisect_left(times_idx, tc - step)
        hi = bisect.bisect_right(times_idx, tc + step)
        best_c, best_b = 0.0, None
        for t, hz, c in frames[lo:hi]:
            if c > best_c and c >= 0.5:
                b = hz_to_chroma(hz)
                if b is not None:
                    best_c, best_b = c, b
        seq.append(best_b)
    return seq

def build_note_sequence(notes, sa_hz, t0, t1, n_samples):
    """Resample the synthesised note track to n_samples points."""
    if t0 >= t1 or n_samples < 2: return [None] * n_samples
    step = (t1 - t0) / n_samples
    seq = []
    for i in range(n_samples):
        tc = t0 + (i + 0.5) * step
        b = None
        for start, dur, swara, octave in notes:
            if start <= tc < start + dur:
                b = swara_to_chroma(swara, sa_hz)
                break
        seq.append(b)
    return seq

def dtw_similarity(seq_a, seq_b, band=DTW_BAND):
    """
    DTW similarity (1 - normalised distance) between two pitch sequences.
    Uses Sakoe-Chiba band to constrain the warping path.
    """
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0: return 0.0

    INF = float('inf')
    dp = np.full((n + 1, m + 1), INF)
    dp[0, 0] = 0.0

    for i in range(1, n + 1):
        j_lo = max(1, i - band)
        j_hi = min(m, i + band)
        for j in range(j_lo, j_hi + 1):
            cost = chroma_dist(seq_a[i - 1], seq_b[j - 1])
            dp[i, j] = cost + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])

    raw = dp[n, m]
    if raw == INF: return 0.0
    normalised = raw / (n + m)        # 0 = perfect, ~1 = bad
    return max(0.0, 1.0 - normalised) # flip to similarity

# ─────────────────────────────────────────────────────────────
# SARGAM EXTRACTION
# ─────────────────────────────────────────────────────────────
def hz_to_swara(hz, sa_hz):
    if hz < 50: return None, 0
    ratio = hz / sa_hz
    octave = 0
    while ratio < 1.0: ratio *= 2;  octave -= 1
    while ratio >= 2.0: ratio /= 2; octave += 1
    cents = 1200 * math.log2(ratio)
    best = min(SARGAM, key=lambda x: min(
        abs(x[0] - cents),
        abs(1200 - cents) if x[1] == "S" else 9999,
    ))
    return best[1], octave

def extract_notes(frames, sa_hz, conf_thresh, gap_tol, min_dur):
    raw = []
    for t, hz, c in frames:
        if c < conf_thresh or hz <= 50: continue
        sw, oct = hz_to_swara(hz, sa_hz)
        if sw: raw.append((t, sw, oct))
    if not raw: return []

    notes, cur_sw, cur_oct = [], raw[0][1], raw[0][2]
    cur_start = cur_end = raw[0][0]
    for t, sw, oct in raw[1:]:
        if sw == cur_sw and oct == cur_oct and t - cur_end <= gap_tol:
            cur_end = t
        else:
            dur = cur_end - cur_start + 0.01
            if dur >= min_dur:
                notes.append((cur_start, dur, cur_sw, cur_oct))
            cur_sw, cur_oct, cur_start, cur_end = sw, oct, t, t
    dur = cur_end - cur_start + 0.01
    if dur >= min_dur:
        notes.append((cur_start, dur, cur_sw, cur_oct))
    return notes

# ─────────────────────────────────────────────────────────────
# COMBINED SCORING
# ─────────────────────────────────────────────────────────────
def score_params(frames, times_idx, notes, sa_hz, sams,
                 n_sub=SUB_WINDOWS, n_dtw=DTW_SAMPLES):
    """
    Score using sub-cycle chroma similarity + DTW contour similarity.
    Returns combined score in [0, 1].
    """
    chroma_sims, dtw_sims = [], []

    for i in range(len(sams) - 1):
        t0, t1 = sams[i], sams[i + 1]
        dur = t1 - t0
        if dur < 0.5: continue   # skip spuriously short cycles

        step = dur / n_sub
        for q in range(n_sub):
            qa, qb = t0 + q * step, t0 + (q + 1) * step

            # ── Chroma similarity ──
            gt_h    = gt_chroma_hist(frames, times_idx, qa, qb)
            syn_h   = synth_chroma_hist(notes, sa_hz, qa, qb)
            if np.any(gt_h) and np.any(syn_h):
                chroma_sims.append(1.0 - cosine_distance(gt_h, syn_h))

            # ── DTW contour ──
            gt_seq  = build_pitch_sequence(frames, times_idx, qa, qb, n_dtw)
            syn_seq = build_note_sequence(notes, sa_hz, qa, qb, n_dtw)
            dtw_sims.append(dtw_similarity(gt_seq, syn_seq))

    if not chroma_sims: return 0.0
    cw, dw = SCORE_WEIGHTS
    return cw * float(np.mean(chroma_sims)) + dw * float(np.mean(dtw_sims))

# ─────────────────────────────────────────────────────────────
# NOTATION WRITER
# ─────────────────────────────────────────────────────────────
def write_notation(notes, sa_hz, params, score, outfile="sargam_notation_improved.txt"):
    conf, gap, min_dur = params
    lines = [
        "SARGAM NOTATION — IMPROVED (self_improve.py v2)",
        "=" * 65,
        f"Sa = {sa_hz} Hz",
        f"Confidence = {conf}  Gap = {gap}s  Min duration = {min_dur}s",
        f"Combined score = {score:.5f}  (0.65×chroma + 0.35×DTW)",
        f"Total notes: {len(notes)}", "",
        "KEY:",
        "  S=Sa  R=Re(sh) r=Re(ko)  G=Ga(sh) g=Ga(ko)  M=Ma(sh)",
        "  M+=Ma(tivra)  P=Pa  D=Dha(sh) d=Dha(ko)  N=Ni(sh)  n=Ni(ko)",
        "  ' = taar (upper octave)   . = mandra (lower octave)", "",
        "FORMAT: [MM:SS.ss]  note  duration  full_name",
        "=" * 65,
    ]
    cur_min = -1
    for start, dur, sw, oct in notes:
        label = (sw + "'") if oct > 0 else (sw.lower() + ".") if oct < 0 else sw
        full  = NOTE_NAMES.get(sw, sw)
        ostr  = " — taar saptak" if oct > 0 else " — mandra saptak" if oct < 0 else " — madhya saptak"
        m, s  = int(start // 60), start % 60
        if m != cur_min:
            cur_min = m
            lines.append(f"\n--- Minute {m:02d} ---")
        lines.append(f"  [{m:02d}:{s:05.2f}]  {label:<6}  {dur:.3f}s   {full}{ostr}")

    lines += ["", "", "=" * 65, "COMPACT NOTATION (16 notes per line)", "=" * 65]
    for i in range(0, len(notes), 16):
        block = notes[i:i+16]
        m, s  = int(block[0][0] // 60), block[0][0] % 60
        nstr  = "  ".join(
            f"{(sw + chr(39) if o > 0 else sw.lower() + '.' if o < 0 else sw):<4}"
            for _, _, sw, o in block
        )
        lines.append(f"[{m:02d}:{s:04.1f}]  {nstr}")

    with open(outfile, "w") as f:
        f.write("\n".join(lines))
    print(f"  Written → {outfile}")

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def run():
    print("=" * 62)
    print("  SELF-IMPROVING SARGAM  v2  (sub-cycle + DTW)")
    print("=" * 62)

    frames    = load_crepe()
    times_idx = make_time_index(frames)
    sams      = load_sams()
    prev      = load_prev_best()

    n_cycles   = len(sams) - 1
    n_windows  = n_cycles * SUB_WINDOWS
    print(f"\nData:  {len(frames)} CREPE frames  |  {len(sams)} sams  "
          f"→  {n_cycles} cycles  →  {n_windows} scoring windows")

    if n_cycles < 2:
        print("Need ≥2 sam marks. Mark sams in the visualiser, then re-run.")
        return

    # ── Warm-start bounds from previous best ────────────────
    if prev:
        sa0    = prev["sa_hz"]
        conf0  = prev["confidence_threshold"]
        gap0   = prev["gap_tolerance"]
        mdur0  = prev["min_duration"]
        prev_score = prev["best_score"]
        print(f"Warm-start from previous best: score={prev_score:.5f}  "
              f"sa={sa0}  conf={conf0}  gap={gap0}  min_dur={mdur0}")
    else:
        sa0, conf0, gap0, mdur0 = 98.0, 0.75, 0.05, 0.04
        prev_score = 0.0

    # ── Baseline with previous best params ──────────────────
    t0 = time.time()
    baseline_notes = extract_notes(frames, sa0, conf0, gap0, mdur0)
    baseline_score = score_params(frames, times_idx, baseline_notes, sa0, sams)
    print(f"\nBaseline (v2 scorer):  {baseline_score:.5f}  "
          f"({len(baseline_notes)} notes)  [{time.time()-t0:.1f}s]")

    log_rows   = []
    eval_count = [0]

    def objective(p):
        sa, conf, gap, min_dur = p
        notes = extract_notes(frames, sa, conf, gap, min_dur)
        if len(notes) < 20:
            return 1.0
        sc = score_params(frames, times_idx, notes, sa, sams)
        log_rows.append((round(sa, 2), round(conf, 3), round(gap, 3),
                         round(min_dur, 3), len(notes), round(sc, 6)))
        eval_count[0] += 1
        return -sc   # minimise negative = maximise score

    best_score  = baseline_score
    best_params = (sa0, conf0, gap0, mdur0)
    best_notes  = baseline_notes

    # ── Three rounds of differential evolution ──────────────
    rounds = [
        # (Sa spread, conf spread, gap spread, mindur spread, popsize, maxiter)
        (6.0, 0.15, 0.06, 0.03, 5, 12),   # Round 1: broad
        (2.0, 0.08, 0.03, 0.02, 4,  8),   # Round 2: refined
        (0.8, 0.04, 0.015, 0.01, 3, 6),   # Round 3: tight
    ]

    for rnd, (dsa, dconf, dgap, dmd, popsize, maxiter) in enumerate(rounds, 1):
        sa_c, conf_c, gap_c, md_c = best_params
        bounds = [
            (max(80, sa_c - dsa),   sa_c + dsa),
            (max(0.5, conf_c - dconf), min(0.95, conf_c + dconf)),
            (max(0.01, gap_c - dgap),  gap_c + dgap),
            (max(0.01, md_c - dmd),    md_c + dmd),
        ]
        n_evals = popsize * (4 + 1) * (maxiter + 1)
        print(f"\nRound {rnd}/3  bounds={[(round(a,2),round(b,2)) for a,b in bounds]}"
              f"  ~{n_evals} evals")
        t0 = time.time()

        res = differential_evolution(
            objective, bounds,
            popsize=popsize, maxiter=maxiter,
            tol=0.0005, seed=42 + rnd,
            disp=False, polish=True,
        )
        elapsed = time.time() - t0
        rnd_score = -res.fun
        rnd_params = tuple(res.x)

        print(f"  Round {rnd} best:  score={rnd_score:.5f}  "
              f"Δ={rnd_score - best_score:+.5f}  [{elapsed:.1f}s  "
              f"{eval_count[0]} evals total]")

        if rnd_score > best_score + 0.0002:
            best_score  = rnd_score
            best_params = rnd_params
            best_notes  = extract_notes(frames, *rnd_params)
            sa_c, conf_c, gap_c, md_c = [round(v, 4) for v in rnd_params]
            print(f"  → New best!  sa={sa_c}  conf={conf_c}  "
                  f"gap={gap_c}  min_dur={md_c}  notes={len(best_notes)}")
        else:
            print(f"  → No meaningful improvement this round. Stopping early.")
            break

    # ── Final report ─────────────────────────────────────────
    sa, conf, gap, min_dur = [round(v, 4) for v in best_params]
    improvement = best_score - prev_score

    print(f"\n{'='*62}")
    print(f"  RESULT")
    print(f"{'='*62}")
    print(f"  Previous best score  : {prev_score:.5f}")
    print(f"  New best score       : {best_score:.5f}  ({improvement:+.5f})")
    print(f"  Sa                   : {sa} Hz")
    print(f"  Confidence threshold : {conf}")
    print(f"  Gap tolerance        : {gap}s")
    print(f"  Min duration         : {min_dur}s")
    print(f"  Note count           : {len(best_notes)}")
    print(f"  Total evaluations    : {eval_count[0]}")

    # ── Save outputs ─────────────────────────────────────────
    result = {
        "previous_score"        : round(prev_score, 6),
        "best_score"            : round(best_score, 6),
        "improvement_this_run"  : round(improvement, 6),
        "sa_hz"                 : sa,
        "confidence_threshold"  : conf,
        "gap_tolerance"         : gap,
        "min_duration"          : min_dur,
        "note_count"            : len(best_notes),
        "scoring"               : f"sub-cycle(×{SUB_WINDOWS}) chroma + DTW",
    }
    with open("best_params.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved → best_params.json")

    write_notation(best_notes, sa, (conf, gap, min_dur), best_score)

    with open("improvement_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sa_hz","conf","gap","min_dur","notes","score"])
        for row in sorted(log_rows, key=lambda r: -r[-1]):
            w.writerow(row)
    print(f"  Saved → improvement_log.csv  ({len(log_rows)} evaluations)")

    if improvement <= 0.0002:
        print(f"\n  Score has plateaued — extraction params are near-optimal.")
        print(f"  Future gains require richer CREPE data or ornament modelling.")
    else:
        print(f"\n  Run again to continue refining.")

if __name__ == "__main__":
    run()
