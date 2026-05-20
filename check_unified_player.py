#!/usr/bin/env python3
"""Static smoke checks for integrated Hindustani music player UX."""
from pathlib import Path
import re

HTML = Path('sargam_visualiser.html').read_text(encoding='utf-8')

REQUIRED_SNIPPETS = [
    'id="mode-toggle-btn"',
    "let playbackMode = 'audio'",
    'function getTransportTime()',
    'function switchPlaybackMode()',
    'function startNotationPlayback',
    'function stopNotationPlayback',
    'function updateTransportUI()',
    'requestAnimationFrame(updateTransportUI)',
    'const SA_HZ = 96.97',
    'function noteFreq',
    'function playSynthNote',
    'id="volume-slider"',
    'function setUnifiedVolume',
    'audio.volume = unifiedVolume',
    'masterGain.gain.value = unifiedVolume',
]

VERCELIGNORE = Path('.vercelignore').read_text(encoding='utf-8') if Path('.vercelignore').exists() else ''
for required_runtime_file in ['ICCR-1854-AC_SIDE_B.mp3', 'sargam_notation_cleaned.txt', 'sam_times_manual.json']:
    assert required_runtime_file not in [line.strip() for line in VERCELIGNORE.splitlines()], f'{required_runtime_file} must be deployed for the production player'


missing = [snippet for snippet in REQUIRED_SNIPPETS if snippet not in HTML]
assert not missing, 'Missing integrated-player hooks: ' + ', '.join(missing)

assert re.search(r'function\s+togglePlay\s*\(\)\s*{[^}]*isTransportPlaying', HTML, re.S), 'togglePlay should use unified transport state'
assert 'audio.addEventListener(\'timeupdate\'' not in HTML and 'audio.addEventListener("timeupdate"' not in HTML, 'UI updates should not depend solely on audio timeupdate'
assert 'div.onclick = () => { seekTransport(n.time); if (!isTransportPlaying) togglePlay(); };' in HTML, 'timeline clicks should use unified transport seek'

print('unified player static checks passed')
