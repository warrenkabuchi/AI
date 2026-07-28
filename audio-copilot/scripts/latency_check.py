"""Phase 1 exit criteria: measure STT and first-chunk latency with real models.

Run from audio-copilot/:  .venv/bin/python scripts/latency_check.py
Makes a few real Claude calls. Requires a Claude Code login.

Reports both paths, because they differ by ~4x:
  cold  - first question after startup (what a warmup absorbs)
  warm  - every question after that (what the user actually experiences)
"""
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilot.brain import Brain
from copilot.modes.copilot_mode import SYSTEM_PROMPT, USER_TEMPLATE
from copilot.stt import Transcriber

fixture = Path(__file__).resolve().parent.parent / "tests/fixtures/question.wav"
with wave.open(str(fixture), "rb") as w:
    audio = np.frombuffer(w.readframes(w.getnframes()),
                          dtype=np.int16).astype(np.float32) / 32768.0

print("loading whisper base.en...")
transcriber = Transcriber("base.en")

t0 = time.monotonic()
text = transcriber.transcribe(audio)
stt_s = time.monotonic() - t0
print(f"STT: {stt_s:.2f}s -> {text!r}\n")

brain = Brain(warm=False)  # measure the cold call ourselves
try:
    prompts = [text, "What is a container?", "Explain blue-green deployment."]
    for i, prompt in enumerate(prompts):
        t1 = time.monotonic()
        chunks = brain.stream(SYSTEM_PROMPT, USER_TEMPLATE.format(context=prompt))
        first = next(chunks)
        first_s = time.monotonic() - t1
        rest = "".join(chunks)
        total_s = time.monotonic() - t1
        label = "cold" if i == 0 else "warm"
        print(f"[{label}] first chunk {first_s:.2f}s | complete {total_s:.2f}s")
        if i == 0:
            print(f"--- sample answer ---\n{first}{rest}\n")
        print(f"       end-of-utterance -> first chunk: {stt_s + first_s:.2f}s")
finally:
    brain.close()
