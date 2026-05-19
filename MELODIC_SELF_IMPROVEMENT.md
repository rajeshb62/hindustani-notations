# Self-improving melodic-fidelity pipeline

This project already has earlier `self_improve*.py` scripts that tune extraction parameters against chroma/DTW windows. `self_improve_melody.py` is a safer, app-facing pipeline: it measures whether the notation consumed by `sargam_visualiser.html` and `note-player.html` preserves the sung melody in `vocals.f0.vad.csv`.

## Goal

Regenerate `sargam_notation_cleaned.txt` so that the melody implied by the notation, when played in `note-player.html`, matches the vocal line from `vocals.wav` in:

- swara sequence
- octave
- onset/timing coverage
- note durations and rests
- phrase/melodic contour

Timbre is ignored: a flute-like note-player rendering does not need to sound like the singer; it needs to trace the same pitch-time melody.

## Inputs

Required:

- `vocals.f0.vad.csv` — VAD-filtered CREPE pitch track; reference melody
- `sargam_notation_cleaned.txt` — current notation used by both HTML apps
- `sam_times_manual.json` — manual sam marks, used for diagnostic phrase/avart scoring

Useful/warm-start:

- `best_params.json` — previous Sa/confidence/gap/min-duration parameters
- `separated/htdemucs/ICCR-1854-AC_SIDE_B/vocals.wav`
- `separated/htdemucs/ICCR-1854-AC_SIDE_B/tanpura_only.wav`
- `separated/htdemucs/ICCR-1854-AC_SIDE_B/tabla_only.wav`
- `ICCR-1854-AC_SIDE_B.mp3`

## Scoring model

The script compares the pitch curve implied by notation against CREPE/VAD frames. `note-player.html` now uses the tanpura-supported `SA_HZ = 96.97`, so scoring judges notation labels as the browser will synthesize them at 96.97 Hz. The notation/player grammar supports repeated octave markers (`S'`, `S''`, `S.`, `S..`), so real high-register material is preserved rather than clamped to one dot/quote marker. The extraction Sa can still be searched because it changes which labels are produced, but the final objective is note-player melody fidelity.

Weighted total:

```text
0.42 pitch similarity   mean absolute cents error, capped at 220 cents
0.20 coverage           fraction of voiced CREPE frames covered by notation
0.15 swara accuracy     swara match on covered frames
0.08 octave accuracy    swara + octave match on covered frames
0.10 rest precision     notation should not create notes outside vocal regions
0.05 smoothness         penalty for excessive micro-note fragmentation
```

It also reports worst sam-to-sam segments using `sam_times_manual.json`. The manual sam marks can be off by sub-second amounts; they are used for diagnostics, not as hard rhythmic truth.

## Commands

Evaluate current deployed notation:

```bash
python3 self_improve_melody.py evaluate
```

Search for better extraction/post-processing parameters without writing notation:

```bash
python3 self_improve_melody.py search --iterations 150
```

Search and write a candidate file under `melody_improve_runs/<timestamp>/`:

```bash
python3 self_improve_melody.py search --iterations 300 --write-candidate
```

Replace `sargam_notation_cleaned.txt` only if the candidate improves the score:

```bash
python3 self_improve_melody.py search --iterations 300 --write-candidate --replace-current
```

The replace mode creates a timestamped backup before overwriting.

## Output files

Evaluation:

- `melody_evaluation_report.json`

Search:

- `melody_improve_runs/<timestamp>/report.json`
- `melody_improve_runs/<timestamp>/search_log.csv`
- `melody_improve_runs/<timestamp>/sargam_notation_candidate.txt` if `--write-candidate` is used

## Parameters searched

- `sa_hz`
- `conf_threshold`
- `gap_tolerance`
- `min_duration`
- `min_keep`
- `median_window`
- `bridge_threshold`
- `bridge_gap`

The search is intentionally conservative and does **not** deploy/replace by default.

## How to audition an improvement

1. Run search with `--write-candidate`.
2. Inspect `report.json`, especially `delta_total` and worst segments.
3. Temporarily copy the candidate to `sargam_notation_cleaned.txt` or run with `--replace-current` if the score improves.
4. Open `sargam_visualiser.html` to inspect notation against the original audio.
5. Open `note-player.html` to audition whether the melody is recognizable from notation alone.

## Current caveat

This is an objective improvement loop against the CREPE/VAD pitch reference. It still cannot know every Hindustani musical intention by itself: meend, gamak, kan-swar, andolan, and phrase grammar may eventually need richer notation than discrete note blocks. But this script creates the measurable loop needed to keep improving without guessing by ear every time.
