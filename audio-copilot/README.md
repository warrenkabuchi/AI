# Audio Copilot

Local audio -> whisper.cpp -> Claude assistant, in three modes: teleprompter,
interview practice, and meeting copilot. See
`docs/superpowers/specs/2026-07-28-audio-copilot-design.md`.

Everything runs on your machine: transcription is whisper.cpp with Metal
acceleration, and Claude access uses your existing Claude Code login, so
there is no API key to configure and no per-token bill.

## Setup

    cd audio-copilot
    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt

The whisper model (`base.en`, ~141 MB) downloads on first run.

## Usage

    .venv/bin/python -m copilot                          # copilot mode, terminal
    .venv/bin/python -m copilot --display overlay        # floating always-on-top window
    .venv/bin/python -m copilot --mode teleprompter --script talk.txt --display overlay
    .venv/bin/python -m copilot --mode practice --topic "cloud architecture"
    .venv/bin/python -m copilot --source "BlackHole 2ch" # transcribe system audio
    .venv/bin/python -m copilot --source file:tests/fixtures/question.wav  # replay a WAV
    .venv/bin/python scripts/latency_check.py            # measure STT + Claude latency

**Teleprompter:** the on-window buttons scroll (up / down / back to top), and
reading the script aloud advances the highlight on its own.

**Practice:** Claude asks you an interview question on `--topic`, speaks it
aloud, then listens. Stop talking for ~2.5 seconds and it scores the answer
out of 10 with specific critiques, then moves to the next question
(`--questions`, default 5). A session summary closes it out. Commands:
`done` scores immediately, `skip` moves on unscored, `stop` ends early.

**Copilot:** transcribes continuously and answers any question it hears with
up to three bullet points, using the last 90 seconds as context.

## Latency

Measured on an M4, `base.en`, using `scripts/latency_check.py`:

| Stage | Time |
|---|---|
| Transcription (1.7 s utterance) | 0.12 s |
| Claude first chunk, session warm | 3.3 s |
| Claude first chunk, cold start | 8.7 s |

The cold start is a one-time cost when the agent session spins up. `copilot`
mode warms the session during whisper's model load, so questions asked after
startup land on the ~3.4 s warm path. Transcription is not the bottleneck.

## Capturing system audio (BlackHole)

To transcribe other apps' audio (Teams, Zoom, a browser) you need BlackHole
(already installed if you use it with rekordbox):

1. Open Audio MIDI Setup -> "+" -> Create Multi-Output Device.
2. Check both your speakers/headphones AND "BlackHole 2ch".
3. Set the Multi-Output Device as system output (Sound settings).
4. Run with `--source "BlackHole 2ch"`.

You keep hearing audio through your speakers; the app reads the BlackHole
side. If the device disappears the app shows `audio_lost` and retries every
2 seconds rather than exiting.

## Transcription accuracy

Whisper handles ordinary speech well but mangles rare proper nouns,
especially from synthetic voices — `say "What is Kubernetes?"` transcribes as
"Qubernetes" on `tiny.en` and "Cuba needs" on `base.en` and `small.en` alike.
Real speech does considerably better. Test fixtures deliberately use
common-vocabulary phrases so the suite tests our wiring, not Whisper's
vocabulary.

## Tests

    .venv/bin/python -m pytest tests -v

69 tests, all offline apart from a one-time `tiny.en` model download. Claude
is faked in every test; no API calls are made.
