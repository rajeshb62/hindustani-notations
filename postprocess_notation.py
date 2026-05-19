"""
Post-process the extracted notation:
  1. Re-extract using best_params.json (9.6ms min_dur, score 0.902)
  2. Merge same-swara+octave notes separated only by sub-20ms fragments
  3. Remove any residual notes still shorter than MIN_KEEP after merging
  4. Write cleaned sargam_notation_cleaned.txt

This preserves short ornaments (kan swara, gamak) that ARE musical,
while removing repeated micro-flickers of the same note.
"""
import math, json, bisect
import numpy as np

SA_HZ    = 96.97
MIN_KEEP = 0.020   # 20ms — anything shorter after merging is noise

SARGAM = [
    (0,"S"),(100,"r"),(200,"R"),(300,"g"),(400,"G"),
    (500,"M"),(600,"M+"),(700,"P"),(800,"d"),(900,"D"),
    (1000,"n"),(1100,"N"),
]
SWARA_SEMITONE = {
    "S":0,"r":1,"R":2,"g":3,"G":4,"M":5,"M+":6,
    "P":7,"d":8,"D":9,"n":10,"N":11,
}
NOTE_NAMES = {
    "S":"Sa","R":"Re shuddh","r":"Re komal",
    "G":"Ga shuddh","g":"Ga komal",
    "M":"Ma shuddh","M+":"Ma tivra","P":"Pa",
    "D":"Dha shuddh","d":"Dha komal",
    "N":"Ni shuddh","n":"Ni komal",
}

def hz_to_chroma(hz):
    if hz < 30: return None
    return int(round(69 + 12*math.log2(hz/440.0))) % 12

def hz_to_swara(hz):
    if hz < 50: return None, 0
    ratio = hz / SA_HZ
    octave = 0
    while ratio < 1.0:  ratio *= 2;  octave -= 1
    while ratio >= 2.0: ratio /= 2;  octave += 1
    cents = 1200 * math.log2(ratio)
    best = min(SARGAM, key=lambda x: min(
        abs(x[0]-cents), abs(1200-cents) if x[1]=="S" else 9999))
    return best[1], octave

def load_crepe():
    frames = []
    with open("vocals.f0.vad.csv") as f:
        next(f)
        for line in f:
            p = line.strip().split(",")
            if len(p) == 3:
                try:
                    t,hz,c = float(p[0]),float(p[1]),float(p[2])
                    if hz > 0: frames.append((t,hz,c))
                except: pass
    frames.sort(key=lambda x: x[0])
    return frames

def extract_notes(frames, conf, gap, min_dur):
    raw = []
    for t,hz,c in frames:
        if c < conf or hz <= 50: continue
        sw,oct = hz_to_swara(hz)
        if sw: raw.append((t,sw,oct))
    if not raw: return []
    notes, csw, coct = [], raw[0][1], raw[0][2]
    cs = ce = raw[0][0]
    for t,sw,oct in raw[1:]:
        if sw==csw and oct==coct and t-ce<=gap: ce=t
        else:
            dur = ce-cs+0.01
            if dur >= min_dur: notes.append((cs,dur,csw,coct))
            csw,coct,cs,ce = sw,oct,t,t
    dur = ce-cs+0.01
    if dur >= min_dur: notes.append((cs,dur,csw,coct))
    return notes

def merge_same_swara_runs(notes, bridge_thresh=0.020, gap_thresh=0.085):
    """
    Merge note[i] into note[i+2] when:
      - note[i] and note[i+2] are same swara+octave
      - note[i+1] (the bridge) is shorter than bridge_thresh
      - total gap between note[i] end and note[i+2] start <= gap_thresh
    The bridge note is absorbed; result is one longer note.
    Repeat until stable.
    """
    changed = True
    while changed:
        changed = False
        out = []
        i = 0
        while i < len(notes):
            if i+2 < len(notes):
                s0,d0,sw0,o0 = notes[i]
                s1,d1,sw1,o1 = notes[i+1]
                s2,d2,sw2,o2 = notes[i+2]
                gap = s2 - (s0+d0)
                if sw0==sw2 and o0==o2 and d1 < bridge_thresh and gap <= gap_thresh:
                    # merge i and i+2, absorbing i+1
                    new_start = s0
                    new_dur   = (s2+d2) - s0
                    out.append((new_start, new_dur, sw0, o0))
                    i += 3
                    changed = True
                    continue
            out.append(notes[i])
            i += 1
        notes = out
    return notes

def filter_short(notes, min_keep):
    return [(s,d,sw,o) for s,d,sw,o in notes if d >= min_keep]

def write_notation(notes, params, outfile="sargam_notation_cleaned.txt"):
    conf = round(params['confidence_threshold'], 4)
    gap  = round(params['gap_tolerance'], 4)
    md   = round(params['min_duration'], 4)
    sc   = params['best_score']

    lines = [
        "SARGAM NOTATION — TANPURA-ANCHORED, POST-PROCESSED (postprocess_notation.py)",
        "="*72,
        f"Sa = {SA_HZ} Hz  (tanpura harmonic analysis)",
        f"Confidence = {conf}  Gap = {gap}s  Min duration (optimizer) = {md}s",
        f"Post-process: same-swara bridge merge + {MIN_KEEP*1000:.0f}ms floor filter",
        f"Score = {sc:.5f}  (0.65×chroma + 0.35×DTW, sub-cycle)",
        f"Total notes: {len(notes)}", "",
        "KEY:",
        "  S=Sa  R=Re(sh) r=Re(ko)  G=Ga(sh) g=Ga(ko)  M=Ma(sh)",
        "  M+=Ma(tivra)  P=Pa  D=Dha(sh) d=Dha(ko)  N=Ni(sh)  n=Ni(ko)",
        "  ' = taar (upper octave)   . = mandra (lower octave)", "",
        "FORMAT: [MM:SS.ss]  note  duration  full_name",
        "="*72,
    ]
    cur_min = -1
    for start,dur,sw,oct in notes:
        label = (sw+"'") if oct>0 else (sw.lower()+".") if oct<0 else sw
        full  = NOTE_NAMES.get(sw,sw)
        ostr  = " — taar saptak" if oct>0 else " — mandra saptak" if oct<0 else " — madhya saptak"
        m,s   = int(start//60), start%60
        if m != cur_min:
            cur_min = m
            lines.append(f"\n--- Minute {m:02d} ---")
        lines.append(f"  [{m:02d}:{s:05.2f}]  {label:<6}  {dur:.3f}s   {full}{ostr}")
    lines += ["","","="*72,"COMPACT NOTATION (16 notes per line)","="*72]
    for i in range(0,len(notes),16):
        blk = notes[i:i+16]
        m,s = int(blk[0][0]//60), blk[0][0]%60
        ns  = "  ".join(
            f"{(sw+chr(39) if o>0 else sw.lower()+'.' if o<0 else sw):<4}"
            for _,_,sw,o in blk)
        lines.append(f"[{m:02d}:{s:04.1f}]  {ns}")
    with open(outfile,"w") as f:
        f.write("\n".join(lines))
    print(f"  Written → {outfile}")

# ── Main ────────────────────────────────────────────────────
params = json.load(open("best_params.json"))
print(f"Loaded best_params: score={params['best_score']:.5f}  "
      f"conf={params['confidence_threshold']}  "
      f"gap={params['gap_tolerance']}  min_dur={params['min_duration']}")

frames = load_crepe()
print(f"Loaded {len(frames)} CREPE frames")

raw_notes = extract_notes(frames,
                          params['confidence_threshold'],
                          params['gap_tolerance'],
                          params['min_duration'])
print(f"\nRaw notes (optimizer output): {len(raw_notes)}")

# Step 1: merge same-swara runs bridged by short notes
merged = merge_same_swara_runs(raw_notes)
print(f"After same-swara bridge merge: {len(merged)}  "
      f"(removed {len(raw_notes)-len(merged)} bridge fragments)")

# Step 2: filter residual shorts
cleaned = filter_short(merged, MIN_KEEP)
print(f"After {MIN_KEEP*1000:.0f}ms floor filter: {len(cleaned)}  "
      f"(removed {len(merged)-len(cleaned)} residual short notes)")

short_raw = sum(1 for _,d,_,_ in raw_notes if d < MIN_KEEP)
print(f"\nSummary:")
print(f"  Raw notes         : {len(raw_notes)}")
print(f"  Notes < {MIN_KEEP*1000:.0f}ms (raw): {short_raw}  ({100*short_raw/len(raw_notes):.1f}%)")
print(f"  Cleaned notes     : {len(cleaned)}")
print(f"  Total removed     : {len(raw_notes)-len(cleaned)}")

write_notation(cleaned, params)
print("Done.")
