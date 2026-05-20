#!/usr/bin/env python3
"""Post-20:00 diagnostics and conservative octave-continuity candidate.

This script does not overwrite production notation. It focuses on the second
composition (20:00-end), reports weak sam/window regions, and writes an audition
candidate with only high-confidence octave-only corrections.
"""
from __future__ import annotations

import bisect
import importlib.util
import json
import math
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
START_SEC = 20 * 60
END_SEC = 1466.076


def load_melody_module():
    spec = importlib.util.spec_from_file_location("self_improve_melody", ROOT / "self_improve_melody.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["self_improve_melody"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


sim = load_melody_module()


def fmt_time(t: float) -> str:
    m = int(t // 60)
    s = t - m * 60
    return f"{m:02d}:{s:05.2f}"


def octave_errors(notes: list, frames: list, start: float, end: float) -> int:
    starts, indexed = sim.build_note_index(notes)
    errors = 0
    for f in frames:
        if f.t < start or f.t > end or f.conf < 0.5 or f.hz <= 50:
            continue
        n = sim.note_at(f.t, starts, indexed)
        if n is None:
            continue
        gt_sw, gt_oct = sim.hz_to_swara(f.hz, sim.NOTE_PLAYER_SA_HZ)
        if n.swara == gt_sw and n.octave != gt_oct:
            errors += 1
    return errors


def segment_row(notes: list, frames: list, params, start: float, end: float) -> dict:
    sc = sim.score_notes(notes, frames, params, start, end)
    return {
        "start": round(start, 2),
        "end": round(end, 2),
        "label": f"{fmt_time(start)}-{fmt_time(end)}",
        "score": round(sc.total, 5),
        "mean_abs_cents": round(sc.mean_abs_cents, 1),
        "coverage": round(sc.coverage, 4),
        "swara_accuracy": round(sc.swara_accuracy, 4),
        "octave_accuracy": round(sc.octave_accuracy, 4),
        "octave_error_frames": octave_errors(notes, frames, start, end),
        "note_count": sc.note_count,
        "short_note_fraction": round(sc.short_note_fraction, 4),
        "notes_per_voiced_second": round(sc.notes_per_voiced_second, 2),
    }


def make_window_rows(notes: list, frames: list, params, start: float, end: float, seconds: float = 10.0) -> list[dict]:
    rows = []
    t = start
    while t < end:
        b = min(t + seconds, end)
        row = segment_row(notes, frames, params, t, b)
        if row["note_count"] or row["coverage"]:
            rows.append(row)
        t += seconds
    return rows


def make_sam_rows(notes: list, frames: list, params, sams: list[float], start: float, end: float) -> list[dict]:
    rows = []
    for a, b in zip(sams, sams[1:]):
        if b <= start or a >= end or b - a < 1.0:
            continue
        rows.append(segment_row(notes, frames, params, max(a, start), min(b, end)))
    return rows


def frames_for_note(frames: list, frame_times: list[float], note) -> list:
    i = bisect.bisect_left(frame_times, note.start)
    j = bisect.bisect_right(frame_times, note.end)
    return [f for f in frames[i:j] if f.conf >= 0.75 and f.hz > 50]


def conservative_octave_candidate(notes: list, frames: list) -> tuple[list, list[dict]]:
    """Return notes with only well-supported octave-only fixes after START_SEC.

    A note is changed only when:
    - it is in the post-20 section and at least 40ms long;
    - at least 4 confident reference frames fall inside it;
    - reference frames strongly agree on the same swara as the note;
    - those same-swara frames strongly agree on a different octave;
    - the octave fold improves mean cents by at least 300 cents.
    """
    frame_times = [f.t for f in frames]
    out = []
    changes = []
    for n in notes:
        if n.start < START_SEC or n.dur < 0.040:
            out.append(n)
            continue
        fs = frames_for_note(frames, frame_times, n)
        if len(fs) < 4:
            out.append(n)
            continue
        labels = [sim.hz_to_swara(f.hz, sim.NOTE_PLAYER_SA_HZ) for f in fs]
        same = [(sw, octv) for sw, octv in labels if sw == n.swara]
        if len(same) / len(fs) < 0.70:
            out.append(n)
            continue
        oct_counts = Counter(octv for _sw, octv in same)
        target_oct, target_count = oct_counts.most_common(1)[0]
        if target_oct == n.octave or target_count / len(same) < 0.70:
            out.append(n)
            continue
        old_errs = [sim.cents_distance(sim.frame_abs_cents(f, sim.NOTE_PLAYER_SA_HZ), sim.note_abs_cents(n)) for f in fs]
        candidate_note = sim.Note(n.start, n.dur, n.swara, target_oct)
        new_errs = [sim.cents_distance(sim.frame_abs_cents(f, sim.NOTE_PLAYER_SA_HZ), sim.note_abs_cents(candidate_note)) for f in fs]
        old_mean = statistics.mean(old_errs)
        new_mean = statistics.mean(new_errs)
        if old_mean - new_mean < 300.0:
            out.append(n)
            continue
        out.append(candidate_note)
        changes.append({
            "start": round(n.start, 2),
            "end": round(n.end, 2),
            "time": fmt_time(n.start),
            "old_label": n.label,
            "new_label": candidate_note.label,
            "duration": round(n.dur, 3),
            "frames": len(fs),
            "same_swara_fraction": round(len(same) / len(fs), 3),
            "target_octave_fraction": round(target_count / len(same), 3),
            "old_mean_cents": round(old_mean, 1),
            "new_mean_cents": round(new_mean, 1),
            "improvement_cents": round(old_mean - new_mean, 1),
        })
    return out, changes


def main() -> int:
    raw_frames = sim.load_frames(ROOT / "vocals.f0.vad.csv")
    frames = sim.octave_deglitch_frames(raw_frames, sim.NOTE_PLAYER_SA_HZ)
    notes = sim.parse_notation(ROOT / "sargam_notation_cleaned.txt")
    params = sim.load_warm_start()
    sams = sim.load_sams(ROOT / "sam_times_manual.json")

    candidate_notes, changes = conservative_octave_candidate(notes, frames)

    current_late = sim.score_notes(notes, frames, params, START_SEC, END_SEC)
    candidate_late = sim.score_notes(candidate_notes, frames, params, START_SEC, END_SEC)
    current_all = sim.score_notes(notes, frames, params)
    candidate_all = sim.score_notes(candidate_notes, frames, params)

    outdir = ROOT / "melody_improve_runs" / time.strftime("%Y%m%d_%H%M%S_post20")
    outdir.mkdir(parents=True, exist_ok=True)
    candidate_path = outdir / "sargam_notation_post20_octave_candidate.txt"
    report_path = outdir / "post20_diagnostic_report.json"

    sim.write_notation(candidate_notes, params, candidate_all, candidate_path)

    ten_sec_rows = make_window_rows(notes, frames, params, START_SEC, END_SEC, 10.0)
    sam_rows = make_sam_rows(notes, frames, params, sams, START_SEC, END_SEC)
    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "start_sec": START_SEC,
        "end_sec": END_SEC,
        "input_notation": "sargam_notation_cleaned.txt",
        "candidate_file": str(candidate_path.relative_to(ROOT)),
        "note_player_sa_hz": sim.NOTE_PLAYER_SA_HZ,
        "current_late_score": sim.round_score(current_late),
        "candidate_late_score": sim.round_score(candidate_late),
        "current_all_score": sim.round_score(current_all),
        "candidate_all_score": sim.round_score(candidate_all),
        "octave_corrections": changes,
        "worst_10s_windows": sorted(ten_sec_rows, key=lambda r: r["score"])[:12],
        "worst_sam_cycles": sorted(sam_rows, key=lambda r: r["score"])[:12],
        "high_fragmentation_windows": sorted(ten_sec_rows, key=lambda r: r["short_note_fraction"], reverse=True)[:12],
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"Loaded {len(notes)} notes and {len(frames)} deglitched vocal frames")
    sim.print_score("Current post-20", current_late)
    sim.print_score("Candidate post-20", candidate_late)
    print(f"Octave corrections: {len(changes)}")
    for ch in changes[:20]:
        print(f"  {ch['time']} {ch['old_label']} -> {ch['new_label']} ({ch['improvement_cents']} cents improvement)")
    print(f"Wrote {candidate_path.relative_to(ROOT)}")
    print(f"Wrote {report_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
