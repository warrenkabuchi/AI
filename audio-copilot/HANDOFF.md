# Handoff prompt — meeting copilot with hovering teleprompter UI

Copy everything below the line into a new session.

---

Build me a macOS app that listens to my meetings and shows live AI guidance
in a floating, always-on-top window styled like a teleprompter — large,
readable text I can glance at while talking, not a dense chat log.

## Start from the existing code, don't rewrite it

There is a working Python implementation at:

    /Users/kabuchi/Documents/Evi BTC/AI/audio-copilot

on branch `master` (also checked out at `.claude/worktrees/audio-copilot`).
Read its `README.md` and
`docs/superpowers/specs/2026-07-28-audio-copilot-design.md` first. It has 80
passing tests. Run them with `.venv/bin/python -m pytest tests -q` from the
`audio-copilot` directory.

What already works and should be reused as-is:

- `copilot/capture.py` — mic and system-audio capture. Verified working with
  BlackHole; PortAudio handles 48kHz-stereo to 16kHz-mono conversion, so no
  resampling code is needed. Includes a watchdog that warns when a device is
  connected but receiving digital silence.
- `copilot/devices.py` — device lookup by partial name, `--list-devices`.
- `copilot/vad.py` — energy-based utterance chunking.
- `copilot/stt.py` — whisper.cpp via pywhispercpp, Metal-accelerated.
  Transcribes a 1.7s utterance in ~0.12s on an M4.
- `copilot/brain.py` — `Brain`, a persistent Claude Agent SDK session. This
  one matters: a fresh one-shot `query()` costs ~12s per call because it
  spawns a subprocess, while a reused session answers in ~3.3s. Keep the
  persistent session and the startup warmup.
- `copilot/modes/copilot_mode.py` — rolling 90s transcript, detects questions,
  streams up to three bullet points of talking points.
- `copilot/display/overlay.py` + `overlay.html` — borderless always-on-top
  pywebview window.

## What to change

The overlay currently renders guidance as small body text. Make it
teleprompter-style: large type, generous line height, high contrast, one idea
per line, comfortable to read at a glance from normal sitting distance while
you are talking to someone. The transcript should be de-emphasised or
collapsible — the guidance is the thing you look at, the transcript is
reference. Keep the window draggable and always-on-top.

Judge the design by whether you could actually read it mid-sentence without
losing your place. Iterate against the running app, not a mockup.

## Two known bugs to fix

1. **Audio feedback loop.** In practice mode the app speaks questions with
   `say`, the microphone captures that audio, and Whisper transcribes the
   app's own speech as the user's answer. Gate capture while the app is
   speaking AND discard audio buffered during that window — blocking on
   `subprocess.run(["say", ...])` is not sufficient, because queued blocks are
   still transcribed after it returns. This matters for any mode that produces
   audio output.
2. **`Progress: 0%` / `Progress: 100%` spam.** pywhispercpp writes progress to
   stdout on every transcription and it floods the terminal. Suppress it.

## Constraints

- macOS on Apple Silicon (M4). BlackHole is already installed.
- Transcription stays local: whisper.cpp, no cloud STT. Free, private, offline.
- Claude access via the Claude Agent SDK, which uses the existing Claude Code
  login — no API key, no per-token billing.
- Python. UI is pywebview + HTML/CSS. Do not introduce Electron, Rust, or a
  second language without saying why first.
- No emoji anywhere in the UI. Use inline SVG icons (Lucide/Heroicons style).
- Skip mockups and formal design docs. Make sensible defaults and start
  coding, then iterate against the running app.
- Do not build features whose purpose is hiding the app from other meeting
  participants or from screen sharing.

## How to verify

The previous session's blind spot was verifying only through a recorded WAV
file (`--source file:tests/fixtures/question.wav`), which cannot reproduce
feedback loops or real-world audio. Test with a real microphone and a real
floating window before claiming anything works. If you cannot do that
yourself, say so plainly rather than implying it was checked.

Useful commands, run from `audio-copilot/`:

    .venv/bin/python -m copilot --list-devices
    .venv/bin/python -m copilot --display overlay          # mic
    .venv/bin/python -m copilot --source blackhole --display overlay
    .venv/bin/python scripts/latency_check.py

For system audio, a Multi-Output Device containing both the speakers and
BlackHole must be selected as the system output in Sound settings, or
BlackHole receives silence.
