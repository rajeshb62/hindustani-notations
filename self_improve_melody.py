#!/usr/bin/env python3
"""
Self-improving melodic-fidelity pipeline for the ICCR-1854 Hindustani notation app.

Goal
----
Improve the timestamped sargam notation so that a note-player rendering preserves
what the singer sang in vocals.wav: swara sequence, octave, timing, rests, and
melodic contour. Timbre is deliberately ignored.

This script treats the VAD-filtered CREPE track (vocals.f0.vad.csv) as the
reference vocal melody and scores any notation by comparing the notation-implied
pitch curve against that reference. It can:

  1. evaluate the currently deployed notation;
  2. search extraction/post-processing parameters;
  3. write a candidate notation and report, without replacing the deployed file
     unless --replace-current is explicitly passed.

Typical use
-----------
  python3 self_improve_melody.py evaluate
  python3 self_improve_melody.py search --iterations 150
  python3 self_improve_melody.py search --iterations 300 --write-candidate

Important files expected in the working directory
-------------------------------------------------
  vocals.f0.vad.csv            reference pitch track (CREPE + VAD)
  sargam_notation_cleaned.txt  current notation consumed by both HTML apps
  sam_times_manual.json        user-marked sam times for phrase/segment scoring
  best_params.json             warm-start parameters, if present
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import os
import random
import re
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

ROOT = Path(__file__).resolve().parent

# note-player.html uses the tanpura-supported Sa. The extraction Sa may be
# searched/tuned, but scoring should judge the pitch curve the browser will
# synthesize from labels.  If note-player.html changes, update this constant
# or make it a CLI argument.
NOTE_PLAYER_SA_HZ = 96.97

SARGAM_CENTS = [
    (0, "S"), (100, "r"), (200, "R"), (300, "g"), (400, "G"),
    (500, "M"), (600, "M+"), (700, "P"), (800, "d"), (900, "D"),
    (1000, "n"), (1100, "N"),
]
SWARA_TO_CENTS = {sw: cents for cents, sw in SARGAM_CENTS}
NOTE_NAMES = {
    "S": "Sa", "R": "Re shuddh", "r": "Re komal",
    "G": "Ga shuddh", "g": "Ga komal",
    "M": "Ma shuddh", "M+": "Ma tivra", "P": "Pa",
    "D": "Dha shuddh", "d": "Dha komal",
    "N": "Ni shuddh", "n": "Ni komal",
}


@dataclass(frozen=True)
class Frame:
    t: float
    hz: float
    conf: float


@dataclass(frozen=True)
class Note:
    start: float
    dur: float
    swara: str
    octave: int

    @property
    def end(self) -> float:
        return self.start + self.dur

    @property
    def label(self) -> str:
        if self.octave > 0:
            return self.swara + ("'" * self.octave)
        if self.octave < 0:
            return self.swara + ('.' * abs(self.octave))
        return self.swara


@dataclass(frozen=True)
class Params:
    sa_hz: float = 96.97
    conf_threshold: float = 0.628
    gap_tolerance: float = 0.075
    min_duration: float = 0.010
    min_keep: float = 0.020
    median_window: int = 3
    bridge_threshold: float = 0.020
    bridge_gap: float = 0.085


@dataclass
class Score:
    total: float
    pitch_similarity: float
    coverage: float
    swara_accuracy: float
    octave_accuracy: float
    rest_precision: float
    smoothness: float
    mean_abs_cents: float
    voiced_frames: int
    covered_frames: int
    note_count: int
    short_note_fraction: float
    notes_per_voiced_second: float


# ─────────────────────────────────────────────────────────────
# Loading / parsing
# ─────────────────────────────────────────────────────────────
def load_frames(path: Path) -> list[Frame]:
    frames: list[Frame] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        # CREPE usually writes time,frequency,confidence; tolerate no headers too.
        if reader.fieldnames and {"time", "frequency", "confidence"}.issubset(reader.fieldnames):
            for row in reader:
                try:
                    t = float(row["time"]); hz = float(row["frequency"]); c = float(row["confidence"])
                except Exception:
                    continue
                if hz > 0:
                    frames.append(Frame(t, hz, c))
        else:
            f.seek(0)
            next(f, None)
            for line in f:
                p = line.strip().split(",")
                if len(p) != 3:
                    continue
                try:
                    t, hz, c = map(float, p)
                except Exception:
                    continue
                if hz > 0:
                    frames.append(Frame(t, hz, c))
    frames.sort(key=lambda x: x.t)
    return frames


def load_sams(path: Path) -> list[float]:
    try:
        data = json.loads(path.read_text())
        times = data.get("sam_times", data)
        return sorted(float(x) for x in times)
    except Exception:
        return []


def load_warm_start() -> Params:
    sa = 96.97
    conf = 0.628
    gap = 0.075
    md = 0.010
    bp = ROOT / "best_params.json"
    if bp.exists():
        try:
            data = json.loads(bp.read_text())
            sa = float(data.get("sa_hz", sa))
            conf = float(data.get("confidence_threshold", data.get("conf_threshold", conf)))
            gap = float(data.get("gap_tolerance", gap))
            md = float(data.get("min_duration", md))
        except Exception:
            pass
    return Params(sa_hz=sa, conf_threshold=conf, gap_tolerance=gap, min_duration=md)


def parse_label(raw: str) -> Optional[tuple[str, int]]:
    raw = raw.strip()
    if not raw:
        return None
    base = "M+" if raw.startswith("M+") else raw[0]
    if base not in SWARA_TO_CENTS:
        return None
    marker_text = raw[2:] if base == "M+" else raw[1:]
    up = marker_text.count("'")
    down = marker_text.count(".")
    octave = up - down
    # Legacy lowercase s/p without explicit markers mean mandra; keep compatibility.
    if base in {"s", "p"} and up == 0 and down == 0:
        octave = -1
    return base, octave


def parse_notation(path: Path) -> list[Note]:
    pat = re.compile(r"^\s*\[(\d{1,2}):([\d.]+)\]\s+(\S+)\s+([\d.]+)s?")
    notes: list[Note] = []
    for line in path.read_text(errors="ignore").splitlines():
        m = pat.match(line)
        if not m:
            continue
        parsed = parse_label(m.group(3))
        if not parsed:
            continue
        sw, octv = parsed
        start = int(m.group(1)) * 60 + float(m.group(2))
        dur = float(m.group(4))
        notes.append(Note(start, dur, sw, octv))
    notes.sort(key=lambda n: n.start)
    return notes


# ─────────────────────────────────────────────────────────────
# Pitch / extraction helpers
# ─────────────────────────────────────────────────────────────
def hz_to_rel_cents(hz: float, sa_hz: float) -> float:
    return 1200.0 * math.log2(hz / sa_hz)


def normalise_cents_to_octave(cents: float) -> tuple[float, int]:
    octave = math.floor(cents / 1200.0)
    local = cents - octave * 1200.0
    if local < 0:
        local += 1200.0
        octave -= 1
    # Treat values close to upper Sa as Sa in the next octave.
    if local >= 1150:
        local -= 1200.0
        octave += 1
    return local, octave


def hz_to_swara(hz: float, sa_hz: float) -> tuple[str, int]:
    local, octave = normalise_cents_to_octave(hz_to_rel_cents(hz, sa_hz))
    cents, sw = min(SARGAM_CENTS, key=lambda x: abs(x[0] - local))
    return sw, octave


def note_abs_cents(note: Note) -> float:
    return SWARA_TO_CENTS[note.swara] + 1200.0 * note.octave


def frame_abs_cents(frame: Frame, sa_hz: float) -> float:
    return hz_to_rel_cents(frame.hz, sa_hz)


def cents_distance(a: float, b: float) -> float:
    return abs(a - b)


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def octave_deglitch_cents(times: list[float], cents: list[float], window_sec: float = 1.5) -> list[float]:
    """Fold likely octave-tracker jumps back toward the local melodic register.

    CREPE occasionally locks to a harmonic at 2x/4x the sung pitch. For notation,
    a short isolated octave leap surrounded by the prior register should usually be
    written in the surrounding saptak, not as ati-taar. Real long high-register
    passages survive because the local median moves with them.
    """
    if not cents:
        return []
    out: list[float] = []
    left = 0
    right = 0
    n = len(cents)
    for i, t in enumerate(times):
        while left < n and times[left] < t - window_sec:
            left += 1
        while right < n and times[right] <= t + window_sec:
            right += 1
        local = cents[left:right]
        ref = median(local) if local else cents[i]
        candidates = [cents[i] + 1200.0 * k for k in range(-3, 4)]
        best = min(candidates, key=lambda x: abs(x - ref))
        # Only fold clear octave/harmonic glitches; leave normal melodic jumps alone.
        if abs(cents[i] - ref) > 650 and abs(best - ref) + 120 < abs(cents[i] - ref):
            out.append(best)
        else:
            out.append(cents[i])
    return out


def octave_deglitch_frames(frames: list[Frame], sa_hz: float = NOTE_PLAYER_SA_HZ) -> list[Frame]:
    """Return frames with harmonic octave glitches folded for notation/evaluation.

    This treats the pitch track as evidence, not gospel: isolated 2x/4x jumps near
    note 269 were audible/visible as tracker harmonics, so we score and extract
    against the locally continuous melodic contour.
    """
    if not frames:
        return []
    times = [f.t for f in frames]
    cents = [frame_abs_cents(f, sa_hz) for f in frames]
    fixed = octave_deglitch_cents(times, cents)
    return [Frame(f.t, sa_hz * (2 ** (c / 1200.0)), f.conf) for f, c in zip(frames, fixed)]


def extract_notes(frames: list[Frame], params: Params) -> list[Note]:
    voiced = [f for f in frames if f.conf >= params.conf_threshold and f.hz > 50]
    if not voiced:
        return []

    # Optional median smoothing in cent-space before quantisation.
    times = [f.t for f in voiced]
    cents = [frame_abs_cents(f, params.sa_hz) for f in voiced]
    cents = octave_deglitch_cents(times, cents)
    win = max(1, int(params.median_window))
    if win % 2 == 0:
        win += 1
    half = win // 2

    raw: list[tuple[float, str, int]] = []
    for i, f in enumerate(voiced):
        if win > 1:
            smoothed_cents = median(cents[max(0, i - half): min(len(cents), i + half + 1)])
            hz = params.sa_hz * (2 ** (smoothed_cents / 1200.0))
        else:
            hz = f.hz
        sw, octv = hz_to_swara(hz, params.sa_hz)
        raw.append((f.t, sw, octv))

    notes: list[Note] = []
    cs, csw, coct = raw[0]
    ce = cs
    for t, sw, octv in raw[1:]:
        # Merge octave flicker for same swara at octave boundary (common around Sa).
        same_pitch_class = (sw == csw and abs(octv - coct) <= 1)
        if same_pitch_class and t - ce <= params.gap_tolerance:
            ce = t
        else:
            dur = ce - cs + 0.01
            if dur >= params.min_duration:
                notes.append(Note(cs, dur, csw, coct))
            cs, ce, csw, coct = t, t, sw, octv
    dur = ce - cs + 0.01
    if dur >= params.min_duration:
        notes.append(Note(cs, dur, csw, coct))

    notes = merge_bridge_fragments(notes, params.bridge_threshold, params.bridge_gap)
    notes = [n for n in notes if n.dur >= params.min_keep]
    return notes


def merge_bridge_fragments(notes: list[Note], bridge_threshold: float, bridge_gap: float) -> list[Note]:
    changed = True
    while changed:
        changed = False
        out: list[Note] = []
        i = 0
        while i < len(notes):
            if i + 2 < len(notes):
                a, b, c = notes[i], notes[i + 1], notes[i + 2]
                if (a.swara == c.swara and a.octave == c.octave and
                    b.dur <= bridge_threshold and c.start - a.end <= bridge_gap):
                    out.append(Note(a.start, c.end - a.start, a.swara, a.octave))
                    i += 3
                    changed = True
                    continue
            out.append(notes[i])
            i += 1
        notes = out
    return notes


# ─────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────
def build_note_index(notes: list[Note]) -> tuple[list[float], list[Note]]:
    return [n.start for n in notes], notes


def note_at(t: float, starts: list[float], notes: list[Note]) -> Optional[Note]:
    i = bisect.bisect_right(starts, t) - 1
    if i < 0:
        return None
    n = notes[i]
    return n if n.start <= t < n.end else None


def voiced_intervals(frames: list[Frame], conf_floor: float = 0.50, max_gap: float = 0.16) -> list[tuple[float, float]]:
    voiced = [f for f in frames if f.conf >= conf_floor and f.hz > 50]
    if not voiced:
        return []
    intervals: list[tuple[float, float]] = []
    start = end = voiced[0].t
    for f in voiced[1:]:
        if f.t - end <= max_gap:
            end = f.t
        else:
            intervals.append((start, end + 0.01))
            start = end = f.t
    intervals.append((start, end + 0.01))
    return intervals


def interval_overlap(a0: float, a1: float, intervals: list[tuple[float, float]]) -> float:
    total = 0.0
    # Intervals are sorted; simple scan is fine for this file size.
    for b0, b1 in intervals:
        if b1 <= a0:
            continue
        if b0 >= a1:
            break
        total += max(0.0, min(a1, b1) - max(a0, b0))
    return total


def score_notes(notes: list[Note], frames: list[Frame], params: Params, start: Optional[float] = None, end: Optional[float] = None) -> Score:
    starts, indexed_notes = build_note_index(notes)
    eval_frames = [f for f in frames if f.conf >= 0.50 and f.hz > 50 and (start is None or f.t >= start) and (end is None or f.t <= end)]
    if not eval_frames:
        return Score(0, 0, 0, 0, 0, 0, 0, 999, 0, 0, len(notes), 1.0, 999)

    covered = 0
    sw_ok = 0
    oct_ok = 0
    abs_errs: list[float] = []
    for f in eval_frames:
        n = note_at(f.t, starts, indexed_notes)
        if n is None:
            continue
        covered += 1
        gt_sw, gt_oct = hz_to_swara(f.hz, NOTE_PLAYER_SA_HZ)
        if n.swara == gt_sw:
            sw_ok += 1
        if n.swara == gt_sw and abs(n.octave - gt_oct) <= 0:
            oct_ok += 1
        abs_errs.append(cents_distance(frame_abs_cents(f, NOTE_PLAYER_SA_HZ), note_abs_cents(n)))

    coverage = covered / len(eval_frames)
    swara_acc = sw_ok / covered if covered else 0.0
    octave_acc = oct_ok / covered if covered else 0.0
    mean_abs = statistics.mean(abs_errs) if abs_errs else 999.0
    pitch_similarity = max(0.0, 1.0 - min(mean_abs, 220.0) / 220.0)

    intervals = voiced_intervals([f for f in frames if (start is None or f.t >= start) and (end is None or f.t <= end)])
    note_dur = sum(n.dur for n in notes if (start is None or n.end >= start) and (end is None or n.start <= end))
    false_dur = 0.0
    for n in notes:
        if start is not None and n.end < start:
            continue
        if end is not None and n.start > end:
            continue
        a0 = max(n.start, start) if start is not None else n.start
        a1 = min(n.end, end) if end is not None else n.end
        if a1 <= a0:
            continue
        false_dur += (a1 - a0) - interval_overlap(a0, a1, intervals)
    rest_precision = 1.0 - (false_dur / note_dur if note_dur > 0 else 0.0)
    rest_precision = max(0.0, min(1.0, rest_precision))

    voiced_sec = sum(b - a for a, b in intervals)
    selected_notes = [n for n in notes if (start is None or n.end >= start) and (end is None or n.start <= end)]
    short_fraction = sum(1 for n in selected_notes if n.dur < 0.030) / len(selected_notes) if selected_notes else 1.0
    notes_per_voiced_second = len(selected_notes) / max(1.0, voiced_sec)
    # A good notation captures ornaments but should not be almost all micro-jitter.
    frag_penalty = min(0.45, short_fraction * 0.30 + max(0.0, notes_per_voiced_second - 14.0) * 0.015)
    smoothness = max(0.0, 1.0 - frag_penalty)

    total = (
        0.42 * pitch_similarity +
        0.20 * coverage +
        0.15 * swara_acc +
        0.08 * octave_acc +
        0.10 * rest_precision +
        0.05 * smoothness
    )
    return Score(total, pitch_similarity, coverage, swara_acc, octave_acc, rest_precision, smoothness,
                 mean_abs, len(eval_frames), covered, len(selected_notes), short_fraction, notes_per_voiced_second)


def worst_segments(notes: list[Note], frames: list[Frame], params: Params, sams: list[float], n: int = 12) -> list[dict]:
    segments: list[tuple[float, float]] = []
    if len(sams) >= 2:
        # Score avart-sized windows. Manual sams may have sub-second imprecision; this is OK for diagnostics.
        segments = [(sams[i], sams[i + 1]) for i in range(len(sams) - 1) if sams[i + 1] - sams[i] >= 1.0]
    if not segments:
        if frames:
            t0, t1 = frames[0].t, frames[-1].t
            segments = [(x, min(x + 10.0, t1)) for x in frange(t0, t1, 10.0)]
    rows = []
    for a, b in segments:
        sc = score_notes(notes, frames, params, a, b)
        if sc.voiced_frames < 8:
            continue
        rows.append({
            "start": round(a, 2), "end": round(b, 2), "score": round(sc.total, 4),
            "coverage": round(sc.coverage, 4), "mean_abs_cents": round(sc.mean_abs_cents, 1),
            "note_count": sc.note_count,
        })
    return sorted(rows, key=lambda r: r["score"])[:n]


def frange(a: float, b: float, step: float) -> Iterable[float]:
    x = a
    while x < b:
        yield x
        x += step


# ─────────────────────────────────────────────────────────────
# Search
# ─────────────────────────────────────────────────────────────
def mutate_params(base: Params, rng: random.Random, broad: bool = False) -> Params:
    if broad:
        sa = rng.uniform(max(94.0, base.sa_hz - 4.0), min(101.0, base.sa_hz + 4.0))
        conf = rng.uniform(0.50, 0.90)
        gap = rng.uniform(0.025, 0.140)
        md = rng.uniform(0.005, 0.060)
        keep = rng.uniform(0.010, 0.050)
        bridge = rng.uniform(0.000, 0.055)
        bridge_gap = rng.uniform(0.040, 0.140)
    else:
        sa = rng.gauss(base.sa_hz, 0.70)
        conf = rng.gauss(base.conf_threshold, 0.055)
        gap = rng.gauss(base.gap_tolerance, 0.020)
        md = rng.gauss(base.min_duration, 0.012)
        keep = rng.gauss(base.min_keep, 0.010)
        bridge = rng.gauss(base.bridge_threshold, 0.012)
        bridge_gap = rng.gauss(base.bridge_gap, 0.025)
    return Params(
        sa_hz=clamp(sa, 94.0, 101.0),
        conf_threshold=clamp(conf, 0.50, 0.92),
        gap_tolerance=clamp(gap, 0.015, 0.160),
        min_duration=clamp(md, 0.004, 0.080),
        min_keep=clamp(keep, 0.008, 0.070),
        median_window=int(rng.choice([1, 3, 5, 7])),
        bridge_threshold=clamp(bridge, 0.0, 0.070),
        bridge_gap=clamp(bridge_gap, 0.030, 0.180),
    )


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def search(frames: list[Frame], current_notes: list[Note], base: Params, iterations: int, seed: int, sams: list[float]) -> tuple[Params, list[Note], Score, list[dict]]:
    rng = random.Random(seed)
    baseline = score_notes(current_notes, frames, base)
    best_params = base
    best_notes = current_notes
    best_score = baseline
    log: list[dict] = []

    # Include current extraction parameters as a candidate, then random-mutated candidates.
    candidates = [base]
    candidates.extend(mutate_params(base, rng, broad=(i < iterations * 0.35)) for i in range(iterations))

    for i, p in enumerate(candidates):
        notes = extract_notes(frames, p)
        sc = score_notes(notes, frames, p)
        row = {"i": i, **round_params(p), **round_score(sc)}
        log.append(row)
        if sc.total > best_score.total:
            best_params, best_notes, best_score = p, notes, sc

    log.sort(key=lambda r: r["total"], reverse=True)
    return best_params, best_notes, best_score, log


def round_params(p: Params) -> dict:
    d = asdict(p)
    for k, v in list(d.items()):
        if isinstance(v, float):
            d[k] = round(v, 5)
    return d


def round_score(s: Score) -> dict:
    d = asdict(s)
    for k, v in list(d.items()):
        if isinstance(v, float):
            d[k] = round(v, 6)
    return d


# ─────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────
def write_notation(notes: list[Note], params: Params, score: Score, outfile: Path) -> None:
    lines = [
        "SARGAM NOTATION — MELODIC-FIDELITY CANDIDATE (self_improve_melody.py)",
        "=" * 78,
        f"Sa = {params.sa_hz:.4f} Hz",
        f"Confidence = {params.conf_threshold:.4f}  Gap = {params.gap_tolerance:.4f}s  Min duration = {params.min_duration:.4f}s",
        f"Post-process: median_window={params.median_window}  min_keep={params.min_keep:.4f}s  bridge={params.bridge_threshold:.4f}s/{params.bridge_gap:.4f}s",
        f"Melody score = {score.total:.5f}  pitch={score.pitch_similarity:.4f} coverage={score.coverage:.4f} swara={score.swara_accuracy:.4f} rest={score.rest_precision:.4f}",
        f"Mean abs cents = {score.mean_abs_cents:.1f}  Total notes: {len(notes)}",
        "",
        "KEY:",
        "  S=Sa  R=Re(sh) r=Re(ko)  G=Ga(sh) g=Ga(ko)  M=Ma(sh)",
        "  M+=Ma(tivra)  P=Pa  D=Dha(sh) d=Dha(ko)  N=Ni(sh)  n=Ni(ko)",
        "  ' = one taar octave; '' = two taar octaves; . = one mandra octave; .. = two mandra octaves",
        "",
        "FORMAT: [MM:SS.ss]  note  duration  full_name",
        "=" * 78,
    ]
    cur_min = -1
    for n in notes:
        m, s = int(n.start // 60), n.start % 60
        if m != cur_min:
            cur_min = m
            lines.append(f"\n--- Minute {m:02d} ---")
        full = NOTE_NAMES.get(n.swara, n.swara)
        if n.octave == 0:
            ostr = " — madhya saptak"
        elif n.octave == 1:
            ostr = " — taar saptak"
        elif n.octave == -1:
            ostr = " — mandra saptak"
        elif n.octave > 1:
            ostr = f" — ati-taar saptak +{n.octave}"
        else:
            ostr = f" — ati-mandra saptak {n.octave}"
        lines.append(f"  [{m:02d}:{s:05.2f}]  {n.label:<6}  {n.dur:.3f}s   {full}{ostr}")
    lines += ["", "", "=" * 78, "COMPACT NOTATION (16 notes per line)", "=" * 78]
    for i in range(0, len(notes), 16):
        blk = notes[i:i + 16]
        if not blk:
            continue
        m, s = int(blk[0].start // 60), blk[0].start % 60
        lines.append(f"[{m:02d}:{s:04.1f}]  " + "  ".join(f"{x.label:<4}" for x in blk))
    outfile.write_text("\n".join(lines))


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def print_score(label: str, score: Score) -> None:
    print(f"{label}: total={score.total:.5f} pitch={score.pitch_similarity:.4f} coverage={score.coverage:.4f} "
          f"swara={score.swara_accuracy:.4f} octave={score.octave_accuracy:.4f} rest={score.rest_precision:.4f} "
          f"smooth={score.smoothness:.4f} cents={score.mean_abs_cents:.1f} notes={score.note_count}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate/search notation for melodic fidelity against vocals.f0.vad.csv")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ev = sub.add_parser("evaluate")
    ev.add_argument("--notation", default="sargam_notation_cleaned.txt")
    ev.add_argument("--report", default="melody_evaluation_report.json")

    se = sub.add_parser("search")
    se.add_argument("--iterations", type=int, default=120)
    se.add_argument("--seed", type=int, default=1854)
    se.add_argument("--notation", default="sargam_notation_cleaned.txt")
    se.add_argument("--output-dir", default="melody_improve_runs")
    se.add_argument("--write-candidate", action="store_true", help="write best candidate notation into output-dir")
    se.add_argument("--replace-current", action="store_true", help="replace sargam_notation_cleaned.txt if candidate beats current")

    args = ap.parse_args()
    os.chdir(ROOT)

    frames_path = ROOT / "vocals.f0.vad.csv"
    notation_path = ROOT / args.notation
    sams_path = ROOT / "sam_times_manual.json"
    if not frames_path.exists():
        raise SystemExit(f"Missing {frames_path}")
    if not notation_path.exists():
        raise SystemExit(f"Missing {notation_path}")

    raw_frames = load_frames(frames_path)
    frames = octave_deglitch_frames(raw_frames, NOTE_PLAYER_SA_HZ)
    sams = load_sams(sams_path)
    warm = load_warm_start()
    current_notes = parse_notation(notation_path)
    current_score = score_notes(current_notes, frames, warm)

    print(f"Loaded {len(frames)} VAD-filtered CREPE frames ({sum(1 for a, b in zip(raw_frames, frames) if abs(a.hz - b.hz) > 1.0)} octave-glitch frames folded)")
    print(f"Loaded {len(current_notes)} notes from {notation_path.name}")
    print(f"Loaded {len(sams)} manual sam marks")
    print_score("Current notation", current_score)

    if args.cmd == "evaluate":
        report = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "notation": notation_path.name,
            "note_player_sa_hz": NOTE_PLAYER_SA_HZ,
            "params_for_reference": round_params(warm),
            "score": round_score(current_score),
            "worst_sam_segments": worst_segments(current_notes, frames, warm, sams),
        }
        out = ROOT / args.report
        write_json(out, report)
        print(f"Wrote {out.name}")
        return 0

    outdir = ROOT / args.output_dir / time.strftime("%Y%m%d_%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    best_params, best_notes, best_score, log = search(frames, current_notes, warm, args.iterations, args.seed, sams)
    print_score("Best candidate", best_score)
    print(f"Search finished in {time.time() - t0:.1f}s; output dir: {outdir}")

    candidate_file = outdir / "sargam_notation_candidate.txt"
    if args.write_candidate or args.replace_current:
        write_notation(best_notes, best_params, best_score, candidate_file)
        print(f"Wrote {candidate_file.relative_to(ROOT)}")

    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "iterations": args.iterations,
        "seed": args.seed,
        "note_player_sa_hz": NOTE_PLAYER_SA_HZ,
        "current": {"params": round_params(warm), "score": round_score(current_score), "note_count": len(current_notes)},
        "best": {"params": round_params(best_params), "score": round_score(best_score), "note_count": len(best_notes)},
        "delta_total": round(best_score.total - current_score.total, 6),
        "candidate_file": str(candidate_file.relative_to(ROOT)) if (args.write_candidate or args.replace_current) else None,
        "worst_sam_segments_current": worst_segments(current_notes, frames, warm, sams),
        "worst_sam_segments_best": worst_segments(best_notes, frames, best_params, sams),
    }
    write_json(outdir / "report.json", report)
    write_csv(outdir / "search_log.csv", log[: min(len(log), 500)])
    print(f"Wrote {outdir.relative_to(ROOT) / 'report.json'}")
    print(f"Wrote {outdir.relative_to(ROOT) / 'search_log.csv'}")

    if args.replace_current:
        if best_score.total > current_score.total:
            backup = ROOT / f"sargam_notation_cleaned.backup_{time.strftime('%Y%m%d_%H%M%S')}.txt"
            backup.write_text(notation_path.read_text())
            write_notation(best_notes, best_params, best_score, notation_path)
            print(f"Replaced {notation_path.name}; backup at {backup.name}")
        else:
            print("Did not replace current notation because candidate did not improve total score.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
