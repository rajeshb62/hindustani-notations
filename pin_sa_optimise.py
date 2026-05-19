"""
Re-optimise with Sa pinned to 96.97 Hz (from tanpura harmonic analysis).
Only tunes confidence, gap tolerance, min duration.
Sa is treated as ground truth — not a free parameter.
"""
import math, json, csv, bisect, time
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cosine as cosine_dist

SA_HZ = 96.97   # pinned from tanpura analysis

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
SUB_WINDOWS = 4
DTW_SAMPLES = 24
DTW_BAND    = 6
W_CHROMA    = 0.65
W_DTW       = 0.35

# ── Data loading ────────────────────────────────────────────
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

def load_sams():
    with open("sam_times_manual.json") as f:
        d = json.load(f)
    return sorted(float(t) for t in d.get("sam_times", d))

frames    = load_crepe()
times_idx = [f[0] for f in frames]
sams      = load_sams()

# ── Chroma + DTW helpers (same as self_improve v2) ──────────
def hz_to_chroma(hz):
    if hz < 30: return None
    return int(round(69 + 12*math.log2(hz/440.0))) % 12

def swara_to_chroma(sw):
    off = SWARA_SEMITONE.get(sw)
    sb  = hz_to_chroma(SA_HZ)
    return (sb + off) % 12 if (off is not None and sb is not None) else None

def hz_to_swara(hz):
    if hz < 50: return None, 0
    ratio = hz / SA_HZ
    octave = 0
    while ratio < 1.0: ratio *= 2;  octave -= 1
    while ratio >= 2.0: ratio /= 2; octave += 1
    cents = 1200 * math.log2(ratio)
    best = min(SARGAM, key=lambda x: min(
        abs(x[0]-cents), abs(1200-cents) if x[1]=="S" else 9999))
    return best[1], octave

def frames_in(t0, t1):
    lo = bisect.bisect_left(times_idx, t0)
    hi = bisect.bisect_right(times_idx, t1)
    return frames[lo:hi]

def gt_chroma(t0, t1):
    h = np.zeros(12)
    for _,hz,c in frames_in(t0,t1):
        if c < 0.5: continue
        b = hz_to_chroma(hz)
        if b is not None: h[b] += c
    n = np.linalg.norm(h)
    return h/n if n > 0 else h

def synth_chroma(notes, t0, t1):
    h = np.zeros(12)
    for start,dur,sw,_ in notes:
        ov = min(start+dur,t1) - max(start,t0)
        if ov <= 0: continue
        b = swara_to_chroma(sw)
        if b is not None: h[b] += ov
    n = np.linalg.norm(h)
    return h/n if n > 0 else h

def chroma_dist_circ(a, b):
    if a is None or b is None: return 1.0
    d = abs(int(a)-int(b))
    return min(d, 12-d) / 6.0

def gt_seq(t0, t1, n):
    step = (t1-t0)/n
    seq = []
    for i in range(n):
        tc = t0+(i+0.5)*step
        lo = bisect.bisect_left(times_idx, tc-step)
        hi = bisect.bisect_right(times_idx, tc+step)
        bc, bb = 0.0, None
        for _,hz,c in frames[lo:hi]:
            if c > bc and c >= 0.5:
                b = hz_to_chroma(hz)
                if b is not None: bc,bb = c,b
        seq.append(bb)
    return seq

def synth_seq(notes, t0, t1, n):
    step = (t1-t0)/n
    seq = []
    for i in range(n):
        tc = t0+(i+0.5)*step
        b = None
        for start,dur,sw,_ in notes:
            if start <= tc < start+dur:
                b = swara_to_chroma(sw); break
        seq.append(b)
    return seq

def dtw_sim(sa, sb, band=DTW_BAND):
    n, m = len(sa), len(sb)
    if n == 0 or m == 0: return 0.0
    dp = np.full((n+1,m+1), float('inf'))
    dp[0,0] = 0
    for i in range(1,n+1):
        for j in range(max(1,i-band), min(m,i+band)+1):
            cost = chroma_dist_circ(sa[i-1], sb[j-1])
            dp[i,j] = cost + min(dp[i-1,j], dp[i,j-1], dp[i-1,j-1])
    raw = dp[n,m]
    return max(0.0, 1.0 - raw/(n+m)) if raw != float('inf') else 0.0

def extract_notes(conf, gap, min_dur):
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

def score(notes):
    ch_sims, dt_sims = [], []
    for i in range(len(sams)-1):
        t0,t1 = sams[i], sams[i+1]
        if t1-t0 < 0.5: continue
        step = (t1-t0)/SUB_WINDOWS
        for q in range(SUB_WINDOWS):
            qa,qb = t0+q*step, t0+(q+1)*step
            gh = gt_chroma(qa,qb);  sh = synth_chroma(notes,qa,qb)
            if np.any(gh) and np.any(sh):
                ch_sims.append(1.0 - cosine_dist(gh,sh))
            dt_sims.append(dtw_sim(gt_seq(qa,qb,DTW_SAMPLES),
                                   synth_seq(notes,qa,qb,DTW_SAMPLES)))
    if not ch_sims: return 0.0
    return W_CHROMA*float(np.mean(ch_sims)) + W_DTW*float(np.mean(dt_sims))

# ── Baseline ────────────────────────────────────────────────
print("="*58)
print(f"  OPTIMISE WITH Sa PINNED = {SA_HZ} Hz (from tanpura)")
print("="*58)
prev = json.load(open("best_params.json"))
print(f"\nPrevious best score : {prev['best_score']:.5f}  "
      f"(Sa was {prev['sa_hz']} Hz)")

t0 = time.time()
baseline_notes = extract_notes(prev['confidence_threshold'],
                                prev['gap_tolerance'], prev['min_duration'])
baseline_score = score(baseline_notes)
print(f"Baseline (pinned Sa): {baseline_score:.5f}  "
      f"({len(baseline_notes)} notes)  [{time.time()-t0:.1f}s]")

log_rows = []
eval_n   = [0]

def objective(p):
    conf, gap, min_dur = p
    notes = extract_notes(conf, gap, min_dur)
    if len(notes) < 20: return 1.0
    sc = score(notes)
    log_rows.append((round(conf,4), round(gap,4), round(min_dur,4),
                     len(notes), round(sc,6)))
    eval_n[0] += 1
    return -sc

best_score  = baseline_score
best_params = (prev['confidence_threshold'],
               prev['gap_tolerance'], prev['min_duration'])
best_notes  = baseline_notes

rounds = [
    # conf spread, gap spread, mindur spread, popsize, maxiter
    (0.12, 0.05, 0.02, 5, 12),
    (0.06, 0.025, 0.01, 4, 8),
    (0.03, 0.012, 0.005, 3, 6),
]
conf0, gap0, md0 = best_params
for rnd, (dc,dg,dm,pop,mi) in enumerate(rounds, 1):
    bounds = [
        (max(0.50, conf0-dc), min(0.95, conf0+dc)),
        (max(0.01, gap0-dg),  gap0+dg),
        (max(0.005, md0-dm),  md0+dm),
    ]
    n_est = pop*(3+1)*(mi+1)
    print(f"\nRound {rnd}/3  ~{n_est} evals  "
          f"conf={[round(x,3) for x in bounds[0]]}  "
          f"gap={[round(x,3) for x in bounds[1]]}  "
          f"min_dur={[round(x,4) for x in bounds[2]]}")
    t0 = time.time()
    res = differential_evolution(objective, bounds,
                                 popsize=pop, maxiter=mi,
                                 tol=0.0003, seed=7+rnd,
                                 disp=False, polish=True)
    rnd_sc = -res.fun
    conf0, gap0, md0 = res.x
    print(f"  Round {rnd}: score={rnd_sc:.5f}  Δ={rnd_sc-best_score:+.5f}"
          f"  [{time.time()-t0:.1f}s  {eval_n[0]} evals total]")
    if rnd_sc > best_score + 0.0002:
        best_score  = rnd_sc
        best_params = tuple(res.x)
        best_notes  = extract_notes(*best_params)
        conf0, gap0, md0 = [round(v,4) for v in best_params]
        print(f"  → New best!  conf={conf0}  gap={gap0}  min_dur={md0}"
              f"  notes={len(best_notes)}")
    else:
        print(f"  → Plateaued. Stopping early.")
        break

conf, gap, md = [round(v,4) for v in best_params]
print(f"\n{'='*58}")
print(f"  RESULT")
print(f"{'='*58}")
print(f"  Sa (pinned)          : {SA_HZ} Hz")
print(f"  Confidence           : {conf}")
print(f"  Gap tolerance        : {gap}s")
print(f"  Min duration         : {md}s")
print(f"  Score                : {best_score:.5f}")
print(f"  Notes                : {len(best_notes)}")
print(f"  Evaluations          : {eval_n[0]}")

# ── Write notation ──────────────────────────────────────────
def write_notation(notes, outfile="sargam_notation_improved.txt"):
    lines = [
        "SARGAM NOTATION — TANPURA-ANCHORED (pin_sa_optimise.py)",
        "="*65,
        f"Sa = {SA_HZ} Hz  (from tanpura harmonic analysis)",
        f"Confidence = {conf}  Gap = {gap}s  Min duration = {md}s",
        f"Score = {best_score:.5f}  (0.65×chroma + 0.35×DTW, sub-cycle)",
        f"Total notes: {len(notes)}", "",
        "KEY:",
        "  S=Sa  R=Re(sh) r=Re(ko)  G=Ga(sh) g=Ga(ko)  M=Ma(sh)",
        "  M+=Ma(tivra)  P=Pa  D=Dha(sh) d=Dha(ko)  N=Ni(sh)  n=Ni(ko)",
        "  ' = taar (upper octave)   . = mandra (lower octave)", "",
        "FORMAT: [MM:SS.ss]  note  duration  full_name",
        "="*65,
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
    lines += ["","","="*65,"COMPACT NOTATION (16 notes per line)","="*65]
    for i in range(0,len(notes),16):
        blk = notes[i:i+16]
        m,s = int(blk[0][0]//60), blk[0][0]%60
        ns  = "  ".join(
            f"{(sw+chr(39) if o>0 else sw.lower()+'.' if o<0 else sw):<4}"
            for _,_,sw,o in blk)
        lines.append(f"[{m:02d}:{s:04.1f}]  {ns}")
    with open(outfile,"w") as f:
        f.write("\n".join(lines))
    print(f"\n  Written → {outfile}")

write_notation(best_notes)

result = {
    "previous_score"        : round(prev["best_score"], 6),
    "best_score"            : round(best_score, 6),
    "improvement_this_run"  : round(best_score - baseline_score, 6),
    "sa_hz"                 : SA_HZ,
    "sa_source"             : "tanpura harmonic analysis (pinned)",
    "confidence_threshold"  : conf,
    "gap_tolerance"         : gap,
    "min_duration"          : md,
    "note_count"            : len(best_notes),
}
with open("best_params.json","w") as f:
    json.dump(result, f, indent=2)
print(f"  Saved  → best_params.json")

with open("improvement_log.csv","w",newline="") as f:
    w = csv.writer(f)
    w.writerow(["conf","gap","min_dur","notes","score"])
    for row in sorted(log_rows, key=lambda r: -r[-1]):
        w.writerow(row)
print(f"  Saved  → improvement_log.csv  ({len(log_rows)} evals)")
