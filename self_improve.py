"""
Self-improving sargam extraction.

Goal: find the extraction parameters (Sa Hz, confidence threshold,
gap tolerance, min note duration) that maximise the chroma similarity
between the synthesised note sequence and the CREPE pitch track,
measured cycle-by-cycle using the sam marks as taal anchors.

No audio re-generation needed — we compare chroma histograms directly
from the CREPE CSV and the extracted note sequence.

Usage:
    python self_improve.py

Outputs:
    best_params.json              — best parameters found
    sargam_notation_improved.txt  — notation re-extracted with best params
    improvement_log.csv           — score per parameter combination
"""

import math
import json
import csv
import itertools

import numpy as np
from scipy.spatial.distance import cosine as cosine_distance

# ─────────────────────────────────────────────
# SARGAM CONSTANTS
# ─────────────────────────────────────────────
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
    "S": 0,  "r": 1,  "R": 2,
    "g": 3,  "G": 4,
    "M": 5,  "M+": 6,
    "P": 7,
    "d": 8,  "D": 9,
    "n": 10, "N": 11,
}

NOTE_NAMES = {
    "S":  "Sa",
    "R":  "Re shuddh",  "r":  "Re komal",
    "G":  "Ga shuddh",  "g":  "Ga komal",
    "M":  "Ma shuddh",  "M+": "Ma tivra",
    "P":  "Pa",
    "D":  "Dha shuddh", "d":  "Dha komal",
    "N":  "Ni shuddh",  "n":  "Ni komal",
}

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def load_crepe(path="vocals.f0.vad.csv"):
    frames = []
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 3:
                try:
                    t, hz, conf = float(parts[0]), float(parts[1]), float(parts[2])
                    if hz > 0:
                        frames.append((t, hz, conf))
                except ValueError:
                    pass
    return frames

def load_sams(path="sam_times_manual.json"):
    with open(path) as f:
        data = json.load(f)
    times = data.get("sam_times", data)
    return sorted(float(t) for t in times)

def load_detected_sa(path="detected_sa.json"):
    with open(path) as f:
        return json.load(f)

# ─────────────────────────────────────────────
# CHROMA UTILITIES
# ─────────────────────────────────────────────
def hz_to_chroma_bin(hz):
    """Convert a frequency in Hz to a chroma bin (0–11)."""
    if hz < 30:
        return None
    midi = 69 + 12 * math.log2(hz / 440.0)
    return int(round(midi)) % 12

def sa_chroma_bin(sa_hz):
    return hz_to_chroma_bin(sa_hz)

def swara_to_chroma_bin(base_swara, sa_hz):
    """Map a base swara label to its chroma bin given Sa."""
    offset = SWARA_SEMITONE.get(base_swara)
    if offset is None:
        return None
    sa_bin = sa_chroma_bin(sa_hz)
    if sa_bin is None:
        return None
    return (sa_bin + offset) % 12

# ─────────────────────────────────────────────
# GROUND-TRUTH CHROMA  (from CREPE frames)
# ─────────────────────────────────────────────
def build_gt_chroma(frames, t_start, t_end):
    """
    Chroma histogram from raw CREPE pitch frames in [t_start, t_end].
    Weighted by confidence so uncertain frames contribute less.
    """
    chroma = np.zeros(12)
    for t, hz, conf in frames:
        if t < t_start or t >= t_end:
            continue
        if conf < 0.5:          # discard very low-confidence frames
            continue
        b = hz_to_chroma_bin(hz)
        if b is not None:
            chroma[b] += conf
    norm = np.linalg.norm(chroma)
    return chroma / norm if norm > 0 else chroma

# ─────────────────────────────────────────────
# SARGAM EXTRACTION  (parameterised)
# ─────────────────────────────────────────────
def hz_to_swara(hz, sa_hz):
    if hz < 50:
        return None, 0
    ratio = hz / sa_hz
    octave = 0
    while ratio < 1.0:
        ratio *= 2;  octave -= 1
    while ratio >= 2.0:
        ratio /= 2;  octave += 1
    cents = 1200 * math.log2(ratio)
    best = min(SARGAM, key=lambda x: min(
        abs(x[0] - cents),
        abs(1200 - cents) if x[1] == "S" else 9999,
    ))
    return best[1], octave

def extract_notes(frames, sa_hz, conf_thresh, gap_tol, min_dur):
    """
    Convert CREPE frames → discrete sargam notes.
    Returns list of (start, duration, base_swara, octave).
    """
    raw = []
    for t, hz, conf in frames:
        if conf < conf_thresh or hz <= 50:
            continue
        swara, octave = hz_to_swara(hz, sa_hz)
        if swara is not None:
            raw.append((t, swara, octave))

    if not raw:
        return []

    notes = []
    cur_swara, cur_octave = raw[0][1], raw[0][2]
    cur_start = cur_end = raw[0][0]

    for t, swara, octave in raw[1:]:
        gap = t - cur_end
        if swara == cur_swara and octave == cur_octave and gap <= gap_tol:
            cur_end = t
        else:
            dur = cur_end - cur_start + 0.01
            if dur >= min_dur:
                notes.append((cur_start, dur, cur_swara, cur_octave))
            cur_swara, cur_octave = swara, octave
            cur_start = cur_end = t

    dur = cur_end - cur_start + 0.01
    if dur >= min_dur:
        notes.append((cur_start, dur, cur_swara, cur_octave))

    return notes

# ─────────────────────────────────────────────
# SYNTHESISED CHROMA  (from extracted notes)
# ─────────────────────────────────────────────
def build_synth_chroma(notes, sa_hz, t_start, t_end):
    """
    Chroma histogram from synthesised note sequence in [t_start, t_end].
    Weighted by overlap duration.
    """
    chroma = np.zeros(12)
    for start, dur, swara, octave in notes:
        note_end = start + dur
        overlap = min(note_end, t_end) - max(start, t_start)
        if overlap <= 0:
            continue
        b = swara_to_chroma_bin(swara, sa_hz)
        if b is not None:
            chroma[b] += overlap
    norm = np.linalg.norm(chroma)
    return chroma / norm if norm > 0 else chroma

# ─────────────────────────────────────────────
# SCORING
# ─────────────────────────────────────────────
def score_params(frames, notes, sa_hz, sams):
    """
    Average cosine similarity between GT and synthesised chroma,
    measured per taal cycle (sam-to-sam window).
    Returns (mean_similarity, per_cycle_similarities).
    """
    sims = []
    for i in range(len(sams) - 1):
        t0, t1 = sams[i], sams[i + 1]
        gt    = build_gt_chroma(frames, t0, t1)
        synth = build_synth_chroma(notes, sa_hz, t0, t1)
        if np.all(gt == 0) or np.all(synth == 0):
            continue
        sim = 1.0 - cosine_distance(gt, synth)
        sims.append(sim)
    mean = float(np.mean(sims)) if sims else 0.0
    return mean, sims

# ─────────────────────────────────────────────
# NOTATION FILE WRITER
# ─────────────────────────────────────────────
def write_notation(notes, sa_hz, params, score, outfile="sargam_notation_improved.txt"):
    conf_thresh, gap_tol, min_dur = params
    lines = []
    lines.append("SARGAM NOTATION — IMPROVED (self_improve.py)")
    lines.append("=" * 65)
    lines.append(f"Sa = {sa_hz} Hz")
    lines.append(f"Confidence threshold = {conf_thresh}")
    lines.append(f"Gap tolerance = {gap_tol}s  |  Min duration = {min_dur}s")
    lines.append(f"Chroma similarity score = {score:.4f}")
    lines.append(f"Total notes: {len(notes)}")
    lines.append("")
    lines.append("KEY:")
    lines.append("  S=Sa  R=Re(sh) r=Re(ko)  G=Ga(sh) g=Ga(ko)  M=Ma(sh)")
    lines.append("  M+=Ma(tivra)  P=Pa  D=Dha(sh) d=Dha(ko)  N=Ni(sh)  n=Ni(ko)")
    lines.append("  ' = taar (upper octave)   . = mandra (lower octave)")
    lines.append("")
    lines.append("FORMAT: [MM:SS.ss]  note  duration  full_name")
    lines.append("=" * 65)

    current_minute = -1
    for start, dur, swara, octave in notes:
        if octave < 0:
            label = swara.lower() + "."
        elif octave > 0:
            label = swara + "'"
        else:
            label = swara
        base = swara
        fullname = NOTE_NAMES.get(base, base)
        octave_str = (
            " — taar saptak"   if octave > 0 else
            " — mandra saptak" if octave < 0 else
            " — madhya saptak"
        )
        m = int(start // 60)
        s = start % 60
        if m != current_minute:
            current_minute = m
            lines.append(f"\n--- Minute {m:02d} ---")
        lines.append(f"  [{m:02d}:{s:05.2f}]  {label:<6}  {dur:.3f}s   {fullname}{octave_str}")

    lines.append("\n\n" + "=" * 65)
    lines.append("COMPACT NOTATION (16 notes per line)")
    lines.append("=" * 65)
    for i in range(0, len(notes), 16):
        block = notes[i:i + 16]
        m = int(block[0][0] // 60)
        s = block[0][0] % 60
        note_str = "  ".join(
            f"{(sw + chr(39) if oct > 0 else sw.lower() + '.' if oct < 0 else sw):<4}"
            for _, _, sw, oct in block
        )
        lines.append(f"[{m:02d}:{s:04.1f}]  {note_str}")

    with open(outfile, "w") as f:
        f.write("\n".join(lines))
    print(f"  Notation written to {outfile}")

# ─────────────────────────────────────────────
# MAIN OPTIMISATION LOOP
# ─────────────────────────────────────────────
def run():
    print("=" * 60)
    print("  SELF-IMPROVING SARGAM EXTRACTION")
    print("=" * 60)

    frames = load_crepe()
    sams   = load_sams()
    sa_data = load_detected_sa()
    base_sa = sa_data["sa_hz"]

    print(f"\nData loaded:")
    print(f"  CREPE frames : {len(frames)}")
    print(f"  Sam marks    : {len(sams)}  ({len(sams)-1} complete cycles)")
    print(f"  Detected Sa  : {base_sa} Hz  (search ±6 Hz)")

    if len(sams) < 2:
        print("\nERROR: Need at least 2 sam marks to score cycles.")
        print("Mark sams in the visualiser app, then re-run.")
        return

    # ── Baseline score with current parameters ──────────────
    baseline_notes = extract_notes(frames, base_sa, 0.75, 0.05, 0.04)
    baseline_score, _ = score_params(frames, baseline_notes, base_sa, sams)
    print(f"\nBaseline score  : {baseline_score:.4f}  ({len(baseline_notes)} notes)")

    # ── Parameter grid ───────────────────────────────────────
    sa_range    = [round(base_sa + d, 1) for d in [-4, -2, -1, 0, 1, 2, 4]]
    conf_range  = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    gap_range   = [0.03, 0.05, 0.08, 0.12]
    mindur_range= [0.03, 0.04, 0.06]

    total = len(sa_range) * len(conf_range) * len(gap_range) * len(mindur_range)
    print(f"\nSearching {total} combinations...\n")

    best_score  = baseline_score
    best_params = (base_sa, 0.75, 0.05, 0.04)
    best_notes  = baseline_notes
    log_rows    = []

    for i, (sa, conf, gap, min_dur) in enumerate(
        itertools.product(sa_range, conf_range, gap_range, mindur_range), 1
    ):
        notes = extract_notes(frames, sa, conf, gap, min_dur)
        if len(notes) < 20:
            continue
        score, _ = score_params(frames, notes, sa, sams)
        log_rows.append((sa, conf, gap, min_dur, len(notes), score))

        if score > best_score:
            best_score  = score
            best_params = (sa, conf, gap, min_dur)
            best_notes  = notes
            improvement = score - baseline_score
            print(f"  [{i:>4}/{total}] ✓ New best  score={score:.4f}"
                  f"  Δ={improvement:+.4f}"
                  f"  sa={sa}  conf={conf}  gap={gap}  min_dur={min_dur}"
                  f"  notes={len(notes)}")

    sa, conf, gap, min_dur = best_params

    print(f"\n{'='*60}")
    print(f"  RESULT")
    print(f"{'='*60}")
    print(f"  Baseline score  : {baseline_score:.4f}")
    print(f"  Best score      : {best_score:.4f}  (+{best_score - baseline_score:.4f})")
    print(f"  Sa              : {sa} Hz  (was {base_sa})")
    print(f"  Confidence      : {conf}  (was 0.75)")
    print(f"  Gap tolerance   : {gap}s  (was 0.05)")
    print(f"  Min duration    : {min_dur}s  (was 0.04)")
    print(f"  Notes           : {len(best_notes)}  (was {len(baseline_notes)})")

    # ── Save outputs ─────────────────────────────────────────
    result = {
        "baseline_score"   : round(baseline_score, 6),
        "best_score"       : round(best_score, 6),
        "improvement"      : round(best_score - baseline_score, 6),
        "sa_hz"            : sa,
        "confidence_threshold": conf,
        "gap_tolerance"    : gap,
        "min_duration"     : min_dur,
        "note_count"       : len(best_notes),
    }
    with open("best_params.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Params saved to best_params.json")

    write_notation(best_notes, sa, (conf, gap, min_dur), best_score)

    # ── Write log ─────────────────────────────────────────────
    with open("improvement_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sa_hz", "conf_thresh", "gap_tol", "min_dur", "note_count", "score"])
        for row in sorted(log_rows, key=lambda r: -r[-1]):
            w.writerow(row)
    print(f"  Full log saved to improvement_log.csv")
    print(f"\nDone. To use the improved notation in the apps, replace")
    print(f"sargam_notation_v3_ICCR1854.txt with sargam_notation_improved.txt")

if __name__ == "__main__":
    run()
