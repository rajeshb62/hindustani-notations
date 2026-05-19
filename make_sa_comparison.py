"""
Generates compare_sa.html — side-by-side comparison of
Sa=98.0 Hz vs Sa=92.5 Hz notation for the first 45 seconds.

Both players sync to the same audio. Compare which labels
feel more correct against the vocal melody.

Usage:
    python make_sa_comparison.py
Output:
    compare_sa.html   (open directly in browser)
"""

import math, json

CLIP_END    = 45.0   # seconds to compare
SA_A        = 98.0   # original detected Sa
SA_B        = 92.525 # optimizer-suggested Sa
# Use run-2 params for everything except Sa (fair comparison)
CONF        = 0.7136
GAP         = 0.07
MIN_DUR     = 0.011

SARGAM = [
    (0,    "S"),
    (100,  "r"), (200,  "R"),
    (300,  "g"), (400,  "G"),
    (500,  "M"), (600,  "M+"),
    (700,  "P"),
    (800,  "d"), (900,  "D"),
    (1000, "n"), (1100, "N"),
]
NOTE_NAMES = {
    "S":"Sa", "R":"Re shuddh", "r":"Re komal",
    "G":"Ga shuddh", "g":"Ga komal",
    "M":"Ma shuddh", "M+":"Ma tivra", "P":"Pa",
    "D":"Dha shuddh", "d":"Dha komal",
    "N":"Ni shuddh", "n":"Ni komal",
}
NOTE_COLORS = {
    "S":"#E63946", "r":"#F4845F", "R":"#F4845F",
    "g":"#E9C46A", "G":"#E9C46A",
    "M":"#52B788", "M+":"#52B788",
    "P":"#2EC4B6",
    "d":"#4895EF", "D":"#4895EF",
    "n":"#C77DFF", "N":"#C77DFF",
}

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

def extract(sa_hz):
    raw = []
    with open("vocals.f0.vad.csv") as f:
        next(f)
        for line in f:
            p = line.strip().split(",")
            if len(p) != 3: continue
            try:
                t, hz, c = float(p[0]), float(p[1]), float(p[2])
            except ValueError:
                continue
            if t > CLIP_END: break
            if c < CONF or hz <= 50: continue
            sw, oct = hz_to_swara(hz, sa_hz)
            if sw: raw.append((t, sw, oct))

    if not raw: return []
    notes, cur_sw, cur_oct = [], raw[0][1], raw[0][2]
    cur_start = cur_end = raw[0][0]
    for t, sw, oct in raw[1:]:
        if sw == cur_sw and oct == cur_oct and t - cur_end <= GAP:
            cur_end = t
        else:
            dur = cur_end - cur_start + 0.01
            if dur >= MIN_DUR:
                notes.append((cur_start, dur, cur_sw, cur_oct))
            cur_sw, cur_oct, cur_start, cur_end = sw, oct, t, t
    dur = cur_end - cur_start + 0.01
    if dur >= MIN_DUR:
        notes.append((cur_start, dur, cur_sw, cur_oct))
    return notes

def to_json(notes):
    out = []
    for start, dur, sw, oct in notes:
        label = (sw + "'") if oct > 0 else (sw.lower() + ".") if oct < 0 else sw
        full  = NOTE_NAMES.get(sw, sw)
        ostr  = "taar" if oct > 0 else "mandra" if oct < 0 else "madhya"
        color = NOTE_COLORS.get(sw, "#aaa")
        komal = sw in ("r","g","d","n")
        out.append({
            "t": round(start, 3),
            "d": round(dur,   3),
            "label": label,
            "full":  f"{full} ({ostr})",
            "color": color,
            "komal": komal,
        })
    return json.dumps(out)

notes_a = extract(SA_A)
notes_b = extract(SA_B)
json_a  = to_json(notes_a)
json_b  = to_json(notes_b)

print(f"Sa={SA_A} Hz → {len(notes_a)} notes in {CLIP_END}s")
print(f"Sa={SA_B} Hz → {len(notes_b)} notes in {CLIP_END}s")

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>Sa Comparison: 98 Hz vs 92.5 Hz</title>
<link href="https://fonts.googleapis.com/css2?family=EB+Garamond:wght@400;700&family=Space+Grotesk:wght@400;500;600&display=swap" rel="stylesheet"/>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
:root{{--bg:#07060f;--border:rgba(255,255,255,0.09);--dim:#5a5060}}
body{{background:var(--bg);color:#e8e0d8;font-family:'Space Grotesk',sans-serif;min-height:100vh;display:flex;flex-direction:column}}

/* ── AUDIO BAR ── */
#audio-bar{{padding:12px 24px;background:rgba(0,0,0,0.6);border-bottom:1px solid var(--border);display:flex;align-items:center;gap:14px;flex-shrink:0}}
#play-btn{{width:42px;height:42px;border-radius:50%;background:rgba(255,255,255,0.1);border:1px solid rgba(255,255,255,0.2);color:#fff;font-size:15px;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:background .15s}}
#play-btn:hover{{background:rgba(255,255,255,0.2)}}
#scrub-wrap{{flex:1;display:flex;flex-direction:column;gap:4px}}
#scrub{{width:100%;height:3px;border-radius:2px;accent-color:#c8b89a;cursor:pointer}}
#time-row{{display:flex;justify-content:space-between;font-size:.7rem;color:var(--dim)}}
#title-label{{font-size:.75rem;letter-spacing:.1em;color:rgba(200,184,154,.7);text-transform:uppercase;flex-shrink:0}}

/* ── PANELS ── */
#panels{{flex:1;display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--border)}}
.panel{{background:var(--bg);display:flex;flex-direction:column;align-items:center;justify-content:center;padding:24px 16px;gap:12px;position:relative;min-height:300px}}
.panel-label{{position:absolute;top:14px;left:50%;transform:translateX(-50%);font-size:.68rem;letter-spacing:.14em;text-transform:uppercase;color:var(--dim);white-space:nowrap}}
.note-char{{font-family:'EB Garamond',Georgia,serif;font-size:clamp(80px,15vw,140px);font-weight:700;line-height:1;color:rgba(255,255,255,.12);transition:color .12s}}
.note-char.komal{{text-decoration:underline;text-decoration-thickness:3px;text-underline-offset:5px}}
.note-name{{font-size:.8rem;letter-spacing:.12em;text-transform:uppercase;color:rgba(232,224,216,.35);transition:color .2s;text-align:center}}
.oct-dot{{font-size:16px;height:20px;opacity:0;transition:opacity .1s}}
.oct-dot.on{{opacity:1}}

/* ── MINI TIMELINE ── */
.mini-timeline{{width:100%;height:36px;background:rgba(255,255,255,.03);border-radius:6px;overflow:hidden;position:relative;margin-top:8px;cursor:pointer}}
.mt-fill{{position:absolute;left:0;top:0;bottom:0;background:rgba(255,255,255,.07);transition:width .1s linear}}
.mt-cursor{{position:absolute;top:0;bottom:0;width:2px;background:rgba(255,255,255,.35);transform:translateX(-50%)}}
.mt-notes{{position:absolute;inset:0;overflow:hidden}}
.mt-note{{position:absolute;top:6px;bottom:6px;border-radius:2px;opacity:.7}}

/* ── NOTE COUNT ── */
.note-count{{font-size:.65rem;color:var(--dim);letter-spacing:.06em}}

/* ── DIFF BADGE ── */
#diff-bar{{padding:8px 24px;background:rgba(0,0,0,.5);border-top:1px solid var(--border);text-align:center;font-size:.72rem;color:var(--dim);letter-spacing:.05em}}
#diff-bar span{{color:#c8b89a;font-weight:500}}
</style>
</head>
<body>

<div id="audio-bar">
  <button id="play-btn" onclick="togglePlay()">▶</button>
  <div id="scrub-wrap">
    <input id="scrub" type="range" min="0" max="{CLIP_END}" step="0.05" value="0" oninput="seek(this.value)"/>
    <div id="time-row"><span id="cur-time">0:00</span><span>First {int(CLIP_END)}s only</span></div>
  </div>
  <div id="title-label">Sa comparison · ICCR-1854</div>
</div>

<div id="panels">
  <!-- Panel A: Sa=98 Hz -->
  <div class="panel" id="panel-a">
    <div class="panel-label">Sa = 98.0 Hz &nbsp;·&nbsp; original &nbsp;·&nbsp; <span id="count-a">{len(notes_a)} notes</span></div>
    <div class="oct-dot" id="oct-a-above">●</div>
    <div class="note-char" id="note-a">—</div>
    <div class="oct-dot" id="oct-a-below">●</div>
    <div class="note-name" id="name-a">press play</div>
    <div class="mini-timeline" id="mt-a" onclick="seekMT(event, 'a')">
      <div class="mt-fill" id="mtf-a"></div>
      <div class="mt-cursor" id="mtc-a"></div>
      <div class="mt-notes" id="mtn-a"></div>
    </div>
  </div>

  <!-- Panel B: Sa=92.5 Hz -->
  <div class="panel" id="panel-b">
    <div class="panel-label">Sa = 92.5 Hz &nbsp;·&nbsp; optimizer &nbsp;·&nbsp; <span id="count-b">{len(notes_b)} notes</span></div>
    <div class="oct-dot" id="oct-b-above">●</div>
    <div class="note-char" id="note-b">—</div>
    <div class="oct-dot" id="oct-b-below">●</div>
    <div class="note-name" id="name-b">press play</div>
    <div class="mini-timeline" id="mt-b" onclick="seekMT(event, 'b')">
      <div class="mt-fill" id="mtf-b"></div>
      <div class="mt-cursor" id="mtc-b"></div>
      <div class="mt-notes" id="mtn-b"></div>
    </div>
  </div>
</div>

<div id="diff-bar">
  Current time: <span id="diff-notes">—</span>
</div>

<audio id="audio" src="./ICCR-1854-AC_SIDE_B.mp3" preload="auto"></audio>

<script>
const CLIP_END = {CLIP_END};
const notesA = {json_a};
const notesB = {json_b};
const audio  = document.getElementById('audio');

// ── Build mini timelines ────────────────────────────────
function buildMT(notes, containerId) {{
  const c = document.getElementById(containerId);
  notes.forEach(n => {{
    const el = document.createElement('div');
    el.className = 'mt-note';
    el.style.left    = (n.t / CLIP_END * 100) + '%';
    el.style.width   = Math.max(0.3, n.d / CLIP_END * 100) + '%';
    el.style.background = n.color;
    c.appendChild(el);
  }});
}}
buildMT(notesA, 'mtn-a');
buildMT(notesB, 'mtn-b');

// ── Note lookup ─────────────────────────────────────────
function findNote(notes, t) {{
  let lo = 0, hi = notes.length - 1, found = -1;
  while (lo <= hi) {{
    const mid = (lo + hi) >> 1;
    if (notes[mid].t <= t) {{ found = mid; lo = mid + 1; }}
    else hi = mid - 1;
  }}
  if (found === -1) return null;
  const n = notes[found];
  return (t < n.t + n.d + 0.05) ? n : null;
}}

// ── Update panel ────────────────────────────────────────
function updatePanel(side, note) {{
  const nc  = document.getElementById('note-' + side);
  const nm  = document.getElementById('name-' + side);
  const oca = document.getElementById('oct-' + side + '-above');
  const ocb = document.getElementById('oct-' + side + '-below');
  if (!note) {{
    nc.textContent = '—';
    nc.style.color = 'rgba(255,255,255,.1)';
    nc.className   = 'note-char';
    nm.textContent = 'silence';
    nm.style.color = '';
    oca.className  = 'oct-dot';
    ocb.className  = 'oct-dot';
    return;
  }}
  nc.textContent = note.label.replace(/['.']/g,'');
  nc.style.color = note.color;
  nc.className   = 'note-char' + (note.komal ? ' komal' : '');
  nm.textContent = note.full;
  nm.style.color = note.color;
  const isTaar   = note.label.includes("'");
  const isMandra = note.label.includes('.');
  oca.className  = 'oct-dot' + (isTaar   ? ' on' : '');
  oca.style.color= note.color;
  ocb.className  = 'oct-dot' + (isMandra ? ' on' : '');
  ocb.style.color= note.color;
}}

// ── Diff bar ────────────────────────────────────────────
function updateDiff(nA, nB) {{
  const la = nA ? nA.label : '—';
  const lb = nB ? nB.label : '—';
  const el = document.getElementById('diff-notes');
  if (!nA && !nB) {{ el.textContent = 'silence in both'; return; }}
  if (la === lb) {{
    el.innerHTML = `Both: <span style="color:${{(nA||nB).color}}">${{la}}</span> — agreement ✓`;
  }} else {{
    const ca = nA ? nA.color : '#aaa';
    const cb = nB ? nB.color : '#aaa';
    el.innerHTML = `98 Hz: <span style="color:${{ca}}">${{la}}</span> &nbsp;|&nbsp; 92.5 Hz: <span style="color:${{cb}}">${{lb}}</span> — <span style="color:#e94560">disagreement</span>`;
  }}
}}

// ── Scrub ───────────────────────────────────────────────
audio.addEventListener('timeupdate', () => {{
  const t = audio.currentTime;
  if (t >= CLIP_END) {{ audio.pause(); audio.currentTime = CLIP_END; }}

  const pct = t / CLIP_END * 100;
  document.getElementById('mtf-a').style.width = pct + '%';
  document.getElementById('mtf-b').style.width = pct + '%';
  document.getElementById('mtc-a').style.left  = pct + '%';
  document.getElementById('mtc-b').style.left  = pct + '%';
  document.getElementById('scrub').value        = t;

  const m = Math.floor(t/60), s = Math.floor(t%60);
  document.getElementById('cur-time').textContent = m+':'+(s<10?'0':'')+s;

  const nA = findNote(notesA, t);
  const nB = findNote(notesB, t);
  updatePanel('a', nA);
  updatePanel('b', nB);
  updateDiff(nA, nB);
}});
audio.addEventListener('play',  () => {{ document.getElementById('play-btn').textContent = '⏸'; }});
audio.addEventListener('pause', () => {{ document.getElementById('play-btn').textContent = '▶'; }});

function togglePlay() {{
  if (audio.paused) {{ audio.play(); }} else {{ audio.pause(); }}
}}
function seek(v) {{
  audio.currentTime = parseFloat(v);
}}
function seekMT(e, side) {{
  const rect = e.currentTarget.getBoundingClientRect();
  const frac = (e.clientX - rect.left) / rect.width;
  audio.currentTime = frac * CLIP_END;
}}

document.addEventListener('keydown', e => {{
  if (e.code === 'Space') {{ e.preventDefault(); togglePlay(); }}
  if (e.code === 'ArrowRight') audio.currentTime = Math.min(CLIP_END, audio.currentTime + 2);
  if (e.code === 'ArrowLeft')  audio.currentTime = Math.max(0, audio.currentTime - 2);
}});
</script>
</body>
</html>"""

with open("compare_sa.html", "w") as f:
    f.write(HTML)

print(f"Generated compare_sa.html")
print(f"Open it in a browser alongside the sargam visualiser.")
print(f"The bottom bar shows agreement/disagreement at each moment.")
