#!/bin/bash
cd "/Users/rajeshbhat/claudecode/claude_experiments/hindustani notations"

echo "[1/3] Starting Demucs..." >> pipeline.log
venv/bin/python -m demucs --two-stems=vocals "ICCR-1854-AC_SIDE_B.mp3" >> pipeline.log 2>&1
echo "[1/3] Demucs DONE" >> pipeline.log

echo "[2/3] Starting CREPE..." >> pipeline.log
venv/bin/crepe "separated/htdemucs/ICCR-1854-AC_SIDE_B/vocals.wav" --output . --step-size 10 --model full --viterbi >> pipeline.log 2>&1
echo "[2/3] CREPE DONE" >> pipeline.log

echo "[3/3] Running extract_sargam.py..." >> pipeline.log
venv/bin/python extract_sargam.py >> pipeline.log 2>&1
echo "[3/3] PIPELINE COMPLETE" >> pipeline.log
