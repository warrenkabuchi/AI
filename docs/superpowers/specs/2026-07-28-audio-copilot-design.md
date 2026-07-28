# Audio Copilot — Design

**Date:** 2026-07-28
**Status:** Approved by user (brainstorming session)

## Summary

A macOS (Apple Silicon) desktop tool that captures audio, transcribes it locally, and
uses Claude to assist the speaker. Three modes share one pipeline:

1. **Teleprompter** — always-on-top overlay scrolls a prepared script; later phases
   auto-scroll by matching live transcription against the script.
2. **Practice mode** — the app asks interview-style questions aloud, transcribes the
   user's spoken answer, and Claude scores/critiques it.
3. **Meeting copilot** — captures system audio during calls the user openly
   participates in, maintains a rolling transcript, and surfaces talking points.

Goal: real-time reminders of well-known technical topics (DevOps, MLOps, cloud
architectures) during live conversations — copilot mode surfaces concise talking
points so the user isn't fishing for terminology mid-discussion.

Out of scope: concealment features (e.g., hiding the overlay from screen capture).

## Decisions made

| Decision | Choice | Why |
|---|---|---|
| Scope | All three modes, phased | Shared pipeline amortizes across modes |
| STT | Local whisper.cpp (`pywhispercpp`) | Free, private, offline, fast on M4 via Metal |
| LLM access | Claude Agent SDK | Uses existing Claude Code login; no separate API billing |
| Overlay | Python + pywebview | One language, one process; HTML/CSS UI like existing dashboards |
| Process model | Single process, queue-connected modules | Least machinery that supports all phases (Approach A) |
| First mode after pipeline | Teleprompter | User preference |

## Architecture

New top-level directory `audio-copilot/` in this repo (matches the existing
multi-project layout). Single Python process, four layers connected by queues:

```
capture (thread) → audio queue → stt (thread) → segment queue → active mode → display
  sounddevice        raw PCM       whisper.cpp     Segment(text,    teleprompter /    pywebview overlay
  mic OR BlackHole                 via pywhispercpp start, end)     practice /        OR terminal printer
                                                                    copilot
```

Entry point: `python -m copilot --mode <teleprompter|practice|copilot> --source <device> --display <terminal|overlay>`.
Phase 1 runs the identical pipeline with `--display terminal`; the overlay is just
another display consumer.

## Components

- **`capture.py`** — wraps `sounddevice.InputStream`, 16 kHz mono, ~100 ms blocks
  into a bounded queue. Source is a device name (mic, or "BlackHole 2ch" for system
  audio). Includes `FileSource`, which replays a WAV at real-time speed for tests.
- **`stt.py`** — whisper.cpp via `pywhispercpp` (Metal-accelerated; model
  auto-downloads on first run). Energy-based VAD cuts audio into 3–8 s utterances;
  each is transcribed and emitted as `Segment(text, start, end)`. Chunked
  pseudo-streaming; expected ~1–2 s behind live speech. No true streaming in v1.
- **`brain.py`** — `stream_response(prompt, context) -> iterator[str]` implemented
  with the Claude Agent SDK. Modes own their prompts; brain owns plumbing only.
- **`modes/`** — one class per mode implementing `on_segment(seg)` and
  `on_command(cmd)`:
  - *Teleprompter*: script display + scroll; later, auto-scroll by fuzzy-matching
    segments against script position. Mostly ignores the brain.
  - *Practice*: Claude generates questions on a user-supplied topic, reads them via
    macOS `say`, listens to the answer, scores it with specific critique; session
    summary at the end.
  - *Copilot*: rolling transcript window; requests talking points when a question is
    detected (interrogative heuristics) or every ~15 s.
- **`display/`** — `TerminalDisplay` (prints) and `OverlayDisplay` (one borderless
  always-on-top pywebview window, single HTML page). Updates pushed via
  `evaluate_js(view_state)`; clicks/keys return through pywebview's `js_api`.

## Data flow

Capture pushes PCM blocks into a bounded queue → STT thread runs VAD, accumulates an
utterance, transcribes, emits a `Segment` → active mode consumes segments on its own
thread, maintains state (script position / rolling transcript / current Q&A), and
calls `display.update(view_state)` with a plain dict. Claude responses stream
token-by-token through the same `update` path. Commands flow the reverse way:
keypress or overlay button → display → `mode.on_command()`.

## Error handling

- **Audio device vanishes**: capture emits a status event; display shows "audio
  lost"; capture retries every 2 s until the device returns. The app never exits on
  audio errors.
- **STT falls behind**: bounded audio queue drops oldest blocks; display shows a
  "lagging" badge so transcript gaps are visible rather than silent drift.
- **Claude call fails** (offline, rate limit): mode shows the error inline where the
  answer would appear, keeps transcribing, retries on the next trigger.
  Transcription never depends on the network.
- **First run**: missing whisper model downloads with a progress line; missing
  BlackHole device produces a message pointing at the multi-output setup steps in
  the README.

## Testing

- `FileSource` + recorded WAV fixtures test capture → STT end-to-end without a mic.
- Mode logic tested with scripted `Segment` lists and a fake brain (canned streaming
  responses). No API calls in tests.
- `scripts/latency_check.py` measures wall-clock from end-of-utterance to first
  Claude token — the number Phase 1 exists to validate.
- Overlay: manual smoke checklist (one window, one page); no automated UI tests.

## Phases

1. **Core pipeline** — mic → whisper.cpp → Claude, terminal display; question
   heuristic (interrogatives + rolling 15 s window). Exit criteria: an
   end-of-utterance → first-token latency the user accepts.
2. **Teleprompter** — overlay window; load a markdown/text script; manual scroll via
   keys/buttons; then transcript-tracked auto-scroll.
3. **Practice mode** — question generation, `say` playback, answer scoring, session
   summary.
4. **Meeting copilot** — BlackHole multi-output capture, rolling transcript, live
   talking points, plus an audio-routing setup guide.
