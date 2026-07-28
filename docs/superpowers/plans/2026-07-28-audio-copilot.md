# Audio Copilot Implementation Plan (Phases 1–2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the core audio → whisper.cpp → Claude pipeline with a terminal display (Phase 1), then the always-on-top teleprompter overlay (Phase 2), per `docs/superpowers/specs/2026-07-28-audio-copilot-design.md`.

**Architecture:** Single Python process. A capture thread pushes 16 kHz mono PCM blocks into a bounded queue; an STT thread runs energy-based VAD, transcribes utterances with pywhispercpp, and emits `Segment`/`Status` events onto an event queue; the active mode consumes events and pushes small event dicts to a display (terminal printer or pywebview overlay). Claude is called through the Claude Agent SDK behind one streaming function.

**Tech Stack:** Python 3.11+, sounddevice, numpy, pywhispercpp, claude-agent-sdk, pywebview, pytest. macOS Apple Silicon.

**Out of scope for this plan:** Practice mode and meeting copilot (spec Phases 3–4) get their own plans after Phase 1 latency is validated. Also out of scope: concealment features of any kind (e.g., hiding the overlay from screen capture).

## Global Constraints

- All commands run from the `audio-copilot/` directory at the repo root unless a step says otherwise.
- Run tests with `python3 -m pytest` (adds cwd to `sys.path`; the package is not installed).
- Audio format everywhere: 16 kHz, mono, float32 numpy arrays in [-1, 1].
- Whisper model: default `base.en` at runtime; tests use `tiny.en` (auto-downloads ~75 MB on first test run — this is the only network access tests may perform; Claude is always faked in tests).
- No emoji anywhere in UI copy or icons — inline SVG only (user preference).
- Dependencies limited to the Tech Stack list; stdlib otherwise (`wave`, `difflib`, `queue`, `threading`, `argparse`, `json`).
- Commit messages: imperative sentence style matching repo history (e.g., "Add VAD utterance chunker").

## File Structure

```
audio-copilot/
  requirements.txt
  README.md                      # Task 13
  copilot/
    __init__.py
    __main__.py                  # python -m copilot
    cli.py                       # build_parser(), main()
    events.py                    # Segment, Status dataclasses
    vad.py                       # UtteranceChunker
    capture.py                   # FileSource, MicSource
    stt.py                       # Transcriber, stt_worker
    question.py                  # looks_like_question()
    brain.py                     # stream_response()
    runner.py                    # run_mode() event loop
    modes/
      __init__.py
      copilot_mode.py            # CopilotMode (Phase 1 mode)
      teleprompter.py            # TeleprompterMode + best_match_line()
    display/
      __init__.py
      terminal.py                # TerminalDisplay
      overlay.py                 # OverlayDisplay, OverlayApi, start_overlay()
      overlay.html
  scripts/
    latency_check.py
  tests/
    fixtures/question.wav        # generated in Task 4
    test_events.py
    test_vad.py
    test_capture.py
    test_stt.py
    test_question.py
    test_brain.py
    test_runner_display.py
    test_copilot_mode.py
    test_cli.py
    test_teleprompter.py
```

**Display protocol (used by every mode and display):** `display.update(ev: dict)` where `ev` is one of:
- `{"event": "segment", "text": str}` — a new transcript line
- `{"event": "answer_start"}` / `{"event": "answer_chunk", "text": str}` / `{"event": "answer_end"}`
- `{"event": "status", "text": str}`
- `{"event": "prompter", "lines": list[str], "pos": int}` — teleprompter view

---

### Task 1: Project scaffold and event types

**Files:**
- Create: `audio-copilot/requirements.txt`
- Create: `audio-copilot/copilot/__init__.py` (empty)
- Create: `audio-copilot/copilot/modes/__init__.py` (empty)
- Create: `audio-copilot/copilot/display/__init__.py` (empty)
- Create: `audio-copilot/copilot/events.py`
- Test: `audio-copilot/tests/test_events.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Segment(text: str, start: float, end: float)` and `Status(kind: str, detail: str = "")` dataclasses in `copilot.events`. Every later task imports these.

- [ ] **Step 1: Create scaffold and requirements**

`requirements.txt`:

```
sounddevice
numpy
pywhispercpp
claude-agent-sdk
pywebview
pytest
```

Create the empty `__init__.py` files listed above. Then install:

Run: `python3 -m pip install -r requirements.txt`

- [ ] **Step 2: Write the failing test**

`tests/test_events.py`:

```python
from copilot.events import Segment, Status


def test_segment_fields():
    s = Segment(text="hello", start=1.0, end=2.5)
    assert (s.text, s.start, s.end) == ("hello", 1.0, 2.5)


def test_status_default_detail():
    st = Status(kind="audio_lost")
    assert st.kind == "audio_lost"
    assert st.detail == ""
```

Run: `python3 -m pytest tests/test_events.py -v` — Expected: FAIL (ModuleNotFoundError or ImportError).

- [ ] **Step 3: Implement `copilot/events.py`**

```python
from dataclasses import dataclass


@dataclass
class Segment:
    text: str
    start: float
    end: float


@dataclass
class Status:
    kind: str  # "audio_lost" | "audio_ok" | "lagging" | "info"
    detail: str = ""
```

- [ ] **Step 4: Run tests** — `python3 -m pytest tests/test_events.py -v` — Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add audio-copilot
git commit -m "Add audio-copilot scaffold and event types"
```

---

### Task 2: Energy-based VAD utterance chunker

**Files:**
- Create: `audio-copilot/copilot/vad.py`
- Test: `audio-copilot/tests/test_vad.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `UtteranceChunker(sample_rate=16000, threshold_rms=0.01, silence_close_s=0.6, min_speech_s=0.3, max_utterance_s=8.0)` with `feed(block: np.ndarray) -> list[np.ndarray]` and `flush() -> np.ndarray | None`, plus a public `sample_rate` attribute. Used by `stt_worker` in Task 4.

- [ ] **Step 1: Write the failing tests**

`tests/test_vad.py`:

```python
import numpy as np
from copilot.vad import UtteranceChunker

SR = 16000
BLOCK = 1600  # 100 ms


def silence(seconds):
    return np.zeros(int(SR * seconds), dtype=np.float32)


def tone(seconds):
    t = np.arange(int(SR * seconds)) / SR
    return (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def feed_all(chunker, audio):
    out = []
    for i in range(0, len(audio), BLOCK):
        out.extend(chunker.feed(audio[i:i + BLOCK]))
    tail = chunker.flush()
    if tail is not None:
        out.append(tail)
    return out


def test_silence_yields_nothing():
    assert feed_all(UtteranceChunker(), silence(3.0)) == []


def test_single_utterance_detected():
    audio = np.concatenate([silence(0.5), tone(1.5), silence(1.0)])
    utts = feed_all(UtteranceChunker(), audio)
    assert len(utts) == 1
    assert len(utts[0]) >= int(SR * 1.5)  # includes the speech


def test_long_speech_splits_at_max():
    utts = feed_all(UtteranceChunker(max_utterance_s=4.0), tone(10.0))
    assert len(utts) >= 2


def test_short_blip_ignored():
    audio = np.concatenate([silence(0.5), tone(0.1), silence(1.0)])
    assert feed_all(UtteranceChunker(), audio) == []
```

Run: `python3 -m pytest tests/test_vad.py -v` — Expected: FAIL (no module `copilot.vad`).

- [ ] **Step 2: Implement `copilot/vad.py`**

```python
import numpy as np


class UtteranceChunker:
    """Cuts a stream of PCM blocks into utterances using RMS energy."""

    def __init__(self, sample_rate=16000, threshold_rms=0.01,
                 silence_close_s=0.6, min_speech_s=0.3, max_utterance_s=8.0):
        self.sample_rate = sample_rate
        self.threshold_rms = threshold_rms
        self.silence_close_s = silence_close_s
        self.min_speech_s = min_speech_s
        self.max_utterance_s = max_utterance_s
        self._reset()

    def _reset(self):
        self._blocks = []
        self._speech = False
        self._speech_s = 0.0
        self._silence_run_s = 0.0
        self._total_s = 0.0

    def feed(self, block):
        out = []
        dur = len(block) / self.sample_rate
        rms = float(np.sqrt(np.mean(block.astype(np.float64) ** 2)))
        if rms >= self.threshold_rms:
            self._speech = True
            self._silence_run_s = 0.0
            self._speech_s += dur
            self._blocks.append(block)
            self._total_s += dur
        elif self._speech:
            self._silence_run_s += dur
            self._blocks.append(block)
            self._total_s += dur
            if self._silence_run_s >= self.silence_close_s:
                out.extend(self._close())
        if self._speech and self._total_s >= self.max_utterance_s:
            out.extend(self._close())
        return out

    def flush(self):
        closed = self._close()
        return closed[0] if closed else None

    def _close(self):
        utt = np.concatenate(self._blocks) if self._blocks else None
        keep = self._speech and self._speech_s >= self.min_speech_s
        self._reset()
        return [utt] if (keep and utt is not None) else []
```

- [ ] **Step 3: Run tests** — `python3 -m pytest tests/test_vad.py -v` — Expected: 4 PASS.

- [ ] **Step 4: Commit**

```bash
git add audio-copilot/copilot/vad.py audio-copilot/tests/test_vad.py
git commit -m "Add energy-based VAD utterance chunker"
```

---

### Task 3: Audio capture sources

**Files:**
- Create: `audio-copilot/copilot/capture.py`
- Test: `audio-copilot/tests/test_capture.py`

**Interfaces:**
- Consumes: `Status` from `copilot.events`.
- Produces:
  - `FileSource(path, audio_q, block_s=0.1, realtime=False)` with `run()` — pushes float32 blocks then a `None` sentinel. Requires a 16 kHz mono 16-bit WAV.
  - `MicSource(device, audio_q, event_q, block_s=0.1, sample_rate=16000)` with `run()` and `stop()`. `device=None` means system default input; a string selects by name (e.g., `"BlackHole 2ch"`). Emits `Status("audio_ok"/"audio_lost"/"lagging")` on `event_q`, retries a lost device every 2 s, pushes `None` to `audio_q` on stop.

- [ ] **Step 1: Write the failing tests**

`tests/test_capture.py`:

```python
import queue
import wave

import numpy as np

from copilot.capture import FileSource, MicSource
from copilot.events import Status


def write_wav(path, audio_f32, sr=16000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((audio_f32 * 32767).astype(np.int16).tobytes())


def test_file_source_replays_blocks(tmp_path):
    audio = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)  # 1 s
    write_wav(tmp_path / "a.wav", audio)
    q = queue.Queue()
    FileSource(tmp_path / "a.wav", q, block_s=0.1, realtime=False).run()
    blocks, item = [], q.get_nowait()
    while item is not None:
        blocks.append(item)
        item = q.get_nowait()
    total = np.concatenate(blocks)
    assert len(total) == 16000
    assert total.dtype == np.float32
    assert abs(float(total[0]) - (-0.5)) < 0.001


def test_mic_source_drops_oldest_when_full():
    audio_q = queue.Queue(maxsize=2)
    event_q = queue.Queue()
    src = MicSource(None, audio_q, event_q)
    b1, b2, b3 = (np.full(10, i, dtype=np.float32) for i in (1, 2, 3))
    src._push(b1)
    src._push(b2)
    src._push(b3)  # queue full -> drops b1, emits lagging status once
    assert float(audio_q.get_nowait()[0]) == 2.0
    assert float(audio_q.get_nowait()[0]) == 3.0
    st = event_q.get_nowait()
    assert isinstance(st, Status) and st.kind == "lagging"
    src._push(b1)  # still lagging episode: no duplicate status
    assert event_q.empty()
```

Run: `python3 -m pytest tests/test_capture.py -v` — Expected: FAIL (no module `copilot.capture`).

- [ ] **Step 2: Implement `copilot/capture.py`**

```python
import queue
import threading
import time
import wave

import numpy as np

from .events import Status


class FileSource:
    """Replays a 16 kHz mono 16-bit WAV into audio_q. Test/latency harness."""

    def __init__(self, path, audio_q, block_s=0.1, realtime=False):
        self.path = path
        self.audio_q = audio_q
        self.block_s = block_s
        self.realtime = realtime

    def run(self):
        with wave.open(str(self.path), "rb") as w:
            if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (16000, 1, 2):
                raise ValueError(f"{self.path}: need 16 kHz mono 16-bit WAV")
            raw = w.readframes(w.getnframes())
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        n = int(16000 * self.block_s)
        for i in range(0, len(audio), n):
            self.audio_q.put(audio[i:i + n])
            if self.realtime:
                time.sleep(self.block_s)
        self.audio_q.put(None)


class MicSource:
    """Captures a live input device; retries when the device disappears."""

    def __init__(self, device, audio_q, event_q, block_s=0.1, sample_rate=16000):
        self.device = device
        self.audio_q = audio_q
        self.event_q = event_q
        self.block_s = block_s
        self.sample_rate = sample_rate
        self._stop = threading.Event()
        self._lagging = False

    def run(self):
        import sounddevice as sd
        while not self._stop.is_set():
            try:
                with sd.InputStream(device=self.device,
                                    samplerate=self.sample_rate, channels=1,
                                    dtype="float32",
                                    blocksize=int(self.sample_rate * self.block_s),
                                    callback=self._callback):
                    self.event_q.put(Status("audio_ok", str(self.device or "default")))
                    while not self._stop.is_set():
                        time.sleep(0.2)
            except Exception as e:
                self.event_q.put(Status("audio_lost", str(e)))
                if self._stop.wait(2.0):
                    break
        self.audio_q.put(None)

    def stop(self):
        self._stop.set()

    def _callback(self, indata, frames, time_info, status):
        self._push(indata[:, 0].copy())

    def _push(self, block):
        try:
            self.audio_q.put_nowait(block)
            self._lagging = False
        except queue.Full:
            try:
                self.audio_q.get_nowait()
            except queue.Empty:
                pass
            self.audio_q.put_nowait(block)
            if not self._lagging:
                self._lagging = True
                self.event_q.put(Status("lagging", "audio backlog; dropping oldest"))
```

- [ ] **Step 3: Run tests** — `python3 -m pytest tests/test_capture.py -v` — Expected: 2 PASS.

- [ ] **Step 4: Commit**

```bash
git add audio-copilot/copilot/capture.py audio-copilot/tests/test_capture.py
git commit -m "Add file and microphone audio capture sources"
```

---

### Task 4: Whisper transcriber and STT worker

**Files:**
- Create: `audio-copilot/copilot/stt.py`
- Create: `audio-copilot/tests/fixtures/question.wav` (generated)
- Test: `audio-copilot/tests/test_stt.py`

**Interfaces:**
- Consumes: `UtteranceChunker` (Task 2), `Segment`/`Status` (Task 1).
- Produces:
  - `Transcriber(model_name="base.en")` with `transcribe(audio: np.ndarray) -> str` (pads input to ≥1.1 s; whisper.cpp rejects sub-second audio).
  - `stt_worker(audio_q, event_q, transcriber, chunker)` — consumes blocks until `None`, emits `Segment` per utterance, forwards final `None` to `event_q`. `Segment.start`/`.end` are seconds of stream time (samples consumed / sample rate).

- [ ] **Step 1: Generate the speech fixture**

```bash
mkdir -p tests/fixtures
say -o tests/fixtures/question.aiff "What is Kubernetes?"
afconvert -f WAVE -d LEI16@16000 -c 1 tests/fixtures/question.aiff tests/fixtures/question.wav
rm tests/fixtures/question.aiff
```

- [ ] **Step 2: Write the failing tests**

`tests/test_stt.py`:

```python
import queue
import wave
from pathlib import Path

import numpy as np

from copilot.events import Segment
from copilot.stt import Transcriber, stt_worker
from copilot.vad import UtteranceChunker

FIXTURE = Path(__file__).parent / "fixtures" / "question.wav"


def load_fixture():
    with wave.open(str(FIXTURE), "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def test_transcribe_fixture():
    # First run downloads the tiny.en model (~75 MB); allowed by global constraints.
    text = Transcriber("tiny.en").transcribe(load_fixture())
    assert "kubernetes" in text.lower()


class FakeTranscriber:
    def transcribe(self, audio):
        return "hello world"


def test_stt_worker_emits_segments():
    audio_q, event_q = queue.Queue(), queue.Queue()
    sr = 16000
    t = np.arange(sr) / sr
    tone = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    audio = np.concatenate([tone, np.zeros(sr, dtype=np.float32)])
    for i in range(0, len(audio), 1600):
        audio_q.put(audio[i:i + 1600])
    audio_q.put(None)
    stt_worker(audio_q, event_q, FakeTranscriber(), UtteranceChunker())
    seg = event_q.get_nowait()
    assert isinstance(seg, Segment)
    assert seg.text == "hello world"
    assert seg.end > seg.start >= 0.0
    assert event_q.get_nowait() is None
```

Run: `python3 -m pytest tests/test_stt.py -v` — Expected: FAIL (no module `copilot.stt`).

- [ ] **Step 3: Implement `copilot/stt.py`**

```python
import numpy as np

from .events import Segment


class Transcriber:
    def __init__(self, model_name="base.en"):
        from pywhispercpp.model import Model
        self._model = Model(model_name)

    def transcribe(self, audio):
        sr = 16000
        min_len = int(sr * 1.1)  # whisper.cpp rejects audio under 1 s
        if len(audio) < min_len:
            audio = np.pad(audio, (0, min_len - len(audio)))
        segments = self._model.transcribe(audio)
        return " ".join(s.text.strip() for s in segments).strip()


def stt_worker(audio_q, event_q, transcriber, chunker):
    """Consumes PCM blocks from audio_q until None; emits Segment to event_q."""
    sr = chunker.sample_rate
    samples_seen = 0

    def emit(utterance):
        end = samples_seen / sr
        start = max(0.0, end - len(utterance) / sr)
        text = transcriber.transcribe(utterance)
        if text:
            event_q.put(Segment(text=text, start=start, end=end))

    while True:
        block = audio_q.get()
        if block is None:
            tail = chunker.flush()
            if tail is not None:
                emit(tail)
            event_q.put(None)
            return
        samples_seen += len(block)
        for utterance in chunker.feed(block):
            emit(utterance)
```

- [ ] **Step 4: Run tests** — `python3 -m pytest tests/test_stt.py -v` — Expected: 2 PASS (first run is slow: model download).

- [ ] **Step 5: Commit**

```bash
git add audio-copilot/copilot/stt.py audio-copilot/tests/test_stt.py audio-copilot/tests/fixtures/question.wav
git commit -m "Add whisper transcriber, STT worker, and speech fixture"
```

---

### Task 5: Question detection heuristic

**Files:**
- Create: `audio-copilot/copilot/question.py`
- Test: `audio-copilot/tests/test_question.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `looks_like_question(text: str) -> bool`. Used by `CopilotMode` (Task 8).

- [ ] **Step 1: Write the failing tests**

`tests/test_question.py`:

```python
import pytest

from copilot.question import looks_like_question


@pytest.mark.parametrize("text", [
    "What is Kubernetes?",
    "What is Kubernetes",           # whisper often drops the question mark
    "how would you scale this service",
    "Can you walk me through your CI pipeline",
    "Explain the difference between ECS and EKS.",
    "Really?",
])
def test_questions(text):
    assert looks_like_question(text)


@pytest.mark.parametrize("text", [
    "",
    "I deployed it yesterday.",
    "We use Terraform for everything.",
    "The what-if analysis is done.",  # interrogative not in first position
])
def test_non_questions(text):
    assert not looks_like_question(text)
```

Run: `python3 -m pytest tests/test_question.py -v` — Expected: FAIL.

- [ ] **Step 2: Implement `copilot/question.py`**

```python
import re

_INTERROGATIVES = {
    "what", "why", "how", "when", "where", "who", "which",
    "can", "could", "would", "should", "do", "does", "did",
    "is", "are", "will", "explain", "describe", "tell",
}
_PHRASES = ("can you", "could you", "walk me through", "tell me about",
            "what about", "talk about")


def looks_like_question(text):
    t = text.strip().lower()
    if not t:
        return False
    if t.endswith("?"):
        return True
    words = re.findall(r"[a-z']+", t)
    if words and words[0] in _INTERROGATIVES and len(words) >= 3:
        return True
    return any(p in t for p in _PHRASES)
```

- [ ] **Step 3: Run tests** — `python3 -m pytest tests/test_question.py -v` — Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add audio-copilot/copilot/question.py audio-copilot/tests/test_question.py
git commit -m "Add question detection heuristic"
```

---

### Task 6: Claude brain (Agent SDK streaming wrapper)

**Files:**
- Create: `audio-copilot/copilot/brain.py`
- Test: `audio-copilot/tests/test_brain.py`

**Interfaces:**
- Consumes: nothing internal.
- Produces: `stream_response(system_prompt: str, user_prompt: str, _query=None) -> Iterator[str]` — yields text chunks; raises the underlying exception on failure. `_query` is a test seam: a callable `(prompt: str, system_prompt: str) -> AsyncIterator` of message objects whose `.content` blocks may carry `.text`. Modes receive this function as `brain_fn` (Task 8).

- [ ] **Step 1: Write the failing tests**

`tests/test_brain.py`:

```python
import pytest

from copilot.brain import stream_response


class Block:
    def __init__(self, text):
        self.text = text


class Msg:
    def __init__(self, texts):
        self.content = [Block(t) for t in texts]


def test_yields_text_blocks_in_order():
    async def fake_query(prompt, system_prompt):
        assert system_prompt == "sys"
        assert "hello" in prompt
        yield Msg(["One. "])
        yield Msg(["Two."])

    assert list(stream_response("sys", "hello", _query=fake_query)) == ["One. ", "Two."]


def test_ignores_non_text_messages():
    class ToolMsg:
        content = None

    async def fake_query(prompt, system_prompt):
        yield ToolMsg()
        yield Msg(["ok"])

    assert list(stream_response("s", "u", _query=fake_query)) == ["ok"]


def test_raises_on_error():
    async def bad_query(prompt, system_prompt):
        raise RuntimeError("boom")
        yield  # pragma: no cover

    with pytest.raises(RuntimeError, match="boom"):
        list(stream_response("s", "u", _query=bad_query))
```

Run: `python3 -m pytest tests/test_brain.py -v` — Expected: FAIL.

- [ ] **Step 2: Implement `copilot/brain.py`**

```python
import asyncio
import queue
import threading

_DONE = object()


def _sdk_query(prompt, system_prompt):
    from claude_agent_sdk import query, ClaudeAgentOptions
    options = ClaudeAgentOptions(system_prompt=system_prompt, max_turns=1,
                                 allowed_tools=[])
    return query(prompt=prompt, options=options)


def stream_response(system_prompt, user_prompt, _query=None):
    """Streams Claude's text response as chunks. Blocks the calling thread.

    Granularity is one chunk per text block in v1 (no partial-message
    streaming); latency to the first chunk is what latency_check measures.
    """
    query_fn = _query or _sdk_query
    q = queue.Queue()
    worker = threading.Thread(target=_run, args=(query_fn, system_prompt,
                                                 user_prompt, q), daemon=True)
    worker.start()
    while True:
        item = q.get()
        if item is _DONE:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


def _run(query_fn, system_prompt, user_prompt, q):
    async def consume():
        async for message in query_fn(prompt=user_prompt,
                                      system_prompt=system_prompt):
            for block in getattr(message, "content", None) or []:
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    q.put(text)

    try:
        asyncio.run(consume())
    except BaseException as e:
        q.put(e)
    finally:
        q.put(_DONE)
```

- [ ] **Step 3: Run tests** — `python3 -m pytest tests/test_brain.py -v` — Expected: 3 PASS.

- [ ] **Step 4: Commit**

```bash
git add audio-copilot/copilot/brain.py audio-copilot/tests/test_brain.py
git commit -m "Add Claude Agent SDK streaming brain wrapper"
```

---

### Task 7: Mode event loop and terminal display

**Files:**
- Create: `audio-copilot/copilot/runner.py`
- Create: `audio-copilot/copilot/display/terminal.py`
- Test: `audio-copilot/tests/test_runner_display.py`

**Interfaces:**
- Consumes: `Segment`, `Status` (Task 1).
- Produces:
  - `run_mode(mode, event_q)` in `copilot.runner` — dispatches `Segment` → `mode.on_segment(seg)`, `Status` → `mode.on_status(st)`, returns on `None`.
  - `TerminalDisplay(out=None)` with `update(ev: dict)` implementing the display protocol from the File Structure section (`segment`, `answer_start`, `answer_chunk`, `answer_end`, `status`, `prompter`).

- [ ] **Step 1: Write the failing tests**

`tests/test_runner_display.py`:

```python
import io
import queue

from copilot.display.terminal import TerminalDisplay
from copilot.events import Segment, Status
from copilot.runner import run_mode


class RecordingMode:
    def __init__(self):
        self.calls = []

    def on_segment(self, seg):
        self.calls.append(("segment", seg.text))

    def on_status(self, st):
        self.calls.append(("status", st.kind))


def test_run_mode_dispatches_until_none():
    q = queue.Queue()
    q.put(Segment("hi", 0.0, 1.0))
    q.put(Status("lagging"))
    q.put(None)
    mode = RecordingMode()
    run_mode(mode, q)
    assert mode.calls == [("segment", "hi"), ("status", "lagging")]


def test_terminal_display_renders_events():
    out = io.StringIO()
    d = TerminalDisplay(out=out)
    d.update({"event": "segment", "text": "hello there"})
    d.update({"event": "answer_start"})
    d.update({"event": "answer_chunk", "text": "Point one."})
    d.update({"event": "answer_end"})
    d.update({"event": "status", "text": "audio_ok: default"})
    text = out.getvalue()
    assert "hello there" in text
    assert "Point one." in text
    assert "audio_ok" in text


def test_terminal_display_renders_prompter_window():
    out = io.StringIO()
    d = TerminalDisplay(out=out)
    lines = [f"line {i}" for i in range(10)]
    d.update({"event": "prompter", "lines": lines, "pos": 4})
    text = out.getvalue()
    assert "> line 4" in text
    assert "line 5" in text
    assert "line 0" not in text  # only a window around pos is shown
```

Run: `python3 -m pytest tests/test_runner_display.py -v` — Expected: FAIL.

- [ ] **Step 2: Implement `copilot/runner.py`**

```python
from .events import Segment, Status


def run_mode(mode, event_q):
    """Blocks consuming events until the None sentinel arrives."""
    while True:
        ev = event_q.get()
        if ev is None:
            return
        if isinstance(ev, Segment):
            mode.on_segment(ev)
        elif isinstance(ev, Status):
            mode.on_status(ev)
```

- [ ] **Step 3: Implement `copilot/display/terminal.py`**

```python
import sys


class TerminalDisplay:
    def __init__(self, out=None):
        self.out = out or sys.stdout

    def update(self, ev):
        kind = ev.get("event")
        if kind == "segment":
            print(f"» {ev['text']}", file=self.out)
        elif kind == "answer_start":
            print("--- answer ---", file=self.out)
        elif kind == "answer_chunk":
            self.out.write(ev["text"])
            self.out.flush()
        elif kind == "answer_end":
            print("\n--------------", file=self.out)
        elif kind == "status":
            print(f"[{ev['text']}]", file=self.out)
        elif kind == "prompter":
            lines, pos = ev["lines"], ev["pos"]
            lo, hi = max(0, pos - 1), min(len(lines), pos + 5)
            print("\n" * 2, file=self.out)
            for i in range(lo, hi):
                marker = ">" if i == pos else " "
                print(f"{marker} {lines[i]}", file=self.out)
```

- [ ] **Step 4: Run tests** — `python3 -m pytest tests/test_runner_display.py -v` — Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add audio-copilot/copilot/runner.py audio-copilot/copilot/display/terminal.py audio-copilot/tests/test_runner_display.py
git commit -m "Add mode event loop and terminal display"
```

---

### Task 8: Copilot mode

**Files:**
- Create: `audio-copilot/copilot/modes/copilot_mode.py`
- Test: `audio-copilot/tests/test_copilot_mode.py`

**Interfaces:**
- Consumes: `Segment`/`Status` (Task 1), `looks_like_question` (Task 5), a `brain_fn` with the `stream_response(system_prompt, user_prompt)` signature (Task 6), a display with `update(ev)` (Task 7).
- Produces: `CopilotMode(display, brain_fn, window_s=90.0)` with `on_segment(seg)`, `on_status(st)`, `on_command(cmd)` (no-op), and module constant `SYSTEM_PROMPT`. Used by the CLI (Task 9) and `latency_check` (Task 9).

- [ ] **Step 1: Write the failing tests**

`tests/test_copilot_mode.py`:

```python
from copilot.events import Segment, Status
from copilot.modes.copilot_mode import CopilotMode


class FakeDisplay:
    def __init__(self):
        self.events = []

    def update(self, ev):
        self.events.append(ev)


def fake_brain(system_prompt, user_prompt):
    assert "Kubernetes" in user_prompt  # rolling context reaches the brain
    yield "One. "
    yield "Two."


def test_answers_detected_questions():
    d = FakeDisplay()
    mode = CopilotMode(d, fake_brain)
    mode.on_segment(Segment("I set up the cluster.", 0.0, 2.0))
    mode.on_segment(Segment("What is Kubernetes?", 2.0, 4.0))
    kinds = [e["event"] for e in d.events]
    assert kinds == ["segment", "segment", "answer_start",
                     "answer_chunk", "answer_chunk", "answer_end"]
    assert d.events[3]["text"] == "One. "


def test_statements_get_no_answer():
    d = FakeDisplay()
    CopilotMode(d, fake_brain).on_segment(Segment("We use Terraform.", 0.0, 1.0))
    assert [e["event"] for e in d.events] == ["segment"]


def test_old_segments_trimmed_from_context():
    seen = {}

    def spy_brain(system_prompt, user_prompt):
        seen["prompt"] = user_prompt
        yield "ok"

    d = FakeDisplay()
    mode = CopilotMode(d, spy_brain, window_s=10.0)
    mode.on_segment(Segment("ancient history", 0.0, 1.0))
    mode.on_segment(Segment("What is Kubernetes?", 100.0, 102.0))
    assert "ancient history" not in seen["prompt"]


def test_brain_error_reported_not_raised():
    def broken_brain(system_prompt, user_prompt):
        raise RuntimeError("offline")
        yield  # pragma: no cover

    d = FakeDisplay()
    mode = CopilotMode(d, broken_brain)
    mode.on_segment(Segment("What is Kubernetes?", 0.0, 1.0))
    kinds = [e["event"] for e in d.events]
    assert kinds == ["segment", "answer_start", "status", "answer_end"]
    assert "offline" in d.events[2]["text"]


def test_status_forwarded():
    d = FakeDisplay()
    mode = CopilotMode(d, fake_brain)
    mode.on_status(Status("audio_lost", "device gone"))
    assert d.events == [{"event": "status", "text": "audio_lost: device gone"}]
```

Run: `python3 -m pytest tests/test_copilot_mode.py -v` — Expected: FAIL.

- [ ] **Step 2: Implement `copilot/modes/copilot_mode.py`**

```python
from collections import deque

from ..question import looks_like_question

SYSTEM_PROMPT = (
    "You are a concise technical coach listening to a live conversation. "
    "Given the transcript, respond to the most recent question with at most "
    "three short bullet points of talking points. Plain text, no headers."
)

USER_TEMPLATE = "Transcript (oldest first):\n{context}\n\nRespond to the latest question."


class CopilotMode:
    def __init__(self, display, brain_fn, window_s=90.0):
        self.display = display
        self.brain_fn = brain_fn
        self.window_s = window_s
        self.segments = deque()

    def on_segment(self, seg):
        self.segments.append(seg)
        while self.segments and self.segments[0].end < seg.end - self.window_s:
            self.segments.popleft()
        self.display.update({"event": "segment", "text": seg.text})
        if looks_like_question(seg.text):
            self._answer()

    def on_status(self, st):
        self.display.update({"event": "status", "text": f"{st.kind}: {st.detail}"})

    def on_command(self, cmd):
        pass

    def _answer(self):
        context = "\n".join(s.text for s in self.segments)
        self.display.update({"event": "answer_start"})
        try:
            for chunk in self.brain_fn(SYSTEM_PROMPT,
                                       USER_TEMPLATE.format(context=context)):
                self.display.update({"event": "answer_chunk", "text": chunk})
        except Exception as e:
            self.display.update({"event": "status", "text": f"brain error: {e}"})
        self.display.update({"event": "answer_end"})
```

- [ ] **Step 3: Run tests** — `python3 -m pytest tests/test_copilot_mode.py -v` — Expected: 5 PASS.

- [ ] **Step 4: Commit**

```bash
git add audio-copilot/copilot/modes/copilot_mode.py audio-copilot/tests/test_copilot_mode.py
git commit -m "Add copilot mode with rolling context and question answering"
```

---

### Task 9: CLI wiring, latency check, Phase 1 smoke test

**Files:**
- Create: `audio-copilot/copilot/cli.py`
- Create: `audio-copilot/copilot/__main__.py`
- Create: `audio-copilot/scripts/latency_check.py`
- Test: `audio-copilot/tests/test_cli.py`

**Interfaces:**
- Consumes: everything from Tasks 1–8.
- Produces: `build_parser() -> argparse.ArgumentParser` and `main(argv=None)` in `copilot.cli`. `python -m copilot` runs `main()`. Task 12 extends `main()`'s display branch.

- [ ] **Step 1: Write the failing test**

`tests/test_cli.py`:

```python
from copilot.cli import build_parser


def test_defaults():
    args = build_parser().parse_args([])
    assert args.mode == "copilot"
    assert args.source == "default"
    assert args.display == "terminal"
    assert args.model == "base.en"
    assert args.script is None


def test_teleprompter_args():
    args = build_parser().parse_args(
        ["--mode", "teleprompter", "--script", "talk.md", "--display", "overlay"])
    assert args.mode == "teleprompter"
    assert args.script == "talk.md"
```

Run: `python3 -m pytest tests/test_cli.py -v` — Expected: FAIL.

- [ ] **Step 2: Implement `copilot/cli.py`**

```python
import argparse
import queue
import threading
from pathlib import Path

from .brain import stream_response
from .capture import FileSource, MicSource
from .display.terminal import TerminalDisplay
from .modes.copilot_mode import CopilotMode
from .runner import run_mode
from .stt import Transcriber, stt_worker
from .vad import UtteranceChunker


def build_parser():
    p = argparse.ArgumentParser(prog="copilot")
    p.add_argument("--mode", choices=["copilot", "teleprompter"], default="copilot")
    p.add_argument("--source", default="default",
                   help='"default", an input device name (e.g. "BlackHole 2ch"), '
                        'or "file:path.wav"')
    p.add_argument("--display", choices=["terminal", "overlay"], default="terminal")
    p.add_argument("--model", default="base.en", help="whisper model name")
    p.add_argument("--script", default=None,
                   help="script file for teleprompter mode")
    return p


def make_mode(args, display):
    if args.mode == "copilot":
        return CopilotMode(display, stream_response)
    from .modes.teleprompter import TeleprompterMode  # Task 10
    if not args.script:
        raise SystemExit("--mode teleprompter requires --script")
    return TeleprompterMode(display, Path(args.script).read_text())


def build_pipeline(args, audio_q, event_q):
    """Returns (source, threads) with threads not yet started."""
    transcriber = Transcriber(args.model)
    chunker = UtteranceChunker()
    if args.source.startswith("file:"):
        source = FileSource(args.source[5:], audio_q, realtime=True)
    else:
        device = None if args.source == "default" else args.source
        source = MicSource(device, audio_q, event_q)
    threads = [
        threading.Thread(target=source.run, daemon=True),
        threading.Thread(target=stt_worker,
                         args=(audio_q, event_q, transcriber, chunker),
                         daemon=True),
    ]
    return source, threads


def main(argv=None):
    args = build_parser().parse_args(argv)
    audio_q = queue.Queue(maxsize=50)  # ~5 s backlog cap
    event_q = queue.Queue()
    source, threads = build_pipeline(args, audio_q, event_q)

    if args.display == "terminal":
        mode = make_mode(args, TerminalDisplay())
        for t in threads:
            t.start()
        try:
            run_mode(mode, event_q)
        except KeyboardInterrupt:
            pass
        finally:
            if hasattr(source, "stop"):
                source.stop()
    else:
        from .display.overlay import start_overlay  # Task 12
        start_overlay(lambda display: make_mode(args, display),
                      lambda: [t.start() for t in threads], event_q)


if __name__ == "__main__":
    main()
```

`copilot/__main__.py`:

```python
from .cli import main

main()
```

- [ ] **Step 3: Run tests** — `python3 -m pytest tests/test_cli.py -v` and the full suite `python3 -m pytest tests -v` — Expected: all PASS.

- [ ] **Step 4: Implement `scripts/latency_check.py`**

```python
"""Phase 1 exit criteria: measure STT and first-chunk latency with real models.

Run from audio-copilot/:  python3 scripts/latency_check.py
Makes ONE real Claude call. Requires a Claude Code login.
"""
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilot.brain import stream_response
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
t1 = time.monotonic()
print(f"STT: {t1 - t0:.2f}s -> {text!r}")

chunks = stream_response(SYSTEM_PROMPT, USER_TEMPLATE.format(context=text))
first = next(chunks)
t2 = time.monotonic()
rest = "".join(chunks)
t3 = time.monotonic()

print(f"first Claude chunk: {t2 - t1:.2f}s after STT done")
print(f"full answer: {t3 - t1:.2f}s\n---\n{first}{rest}")
print(f"\nTOTAL utterance-end -> first chunk: {t2 - t0:.2f}s")
```

- [ ] **Step 5: Run the latency check** — `python3 scripts/latency_check.py`
Expected: prints an STT time, a first-chunk time, and a sensible three-bullet answer about Kubernetes. Record the TOTAL number in the commit message.

- [ ] **Step 6: Manual smoke test (Phase 1 gate)**

Run: `python3 -m copilot` — speak a question at your mic (e.g., "What is a Kubernetes operator?"), watch the transcript line appear and an answer stream in the terminal. Ctrl-C to exit. This is the spec's Phase 1 exit: the human decides the latency is acceptable.

- [ ] **Step 7: Commit**

```bash
git add audio-copilot/copilot/cli.py audio-copilot/copilot/__main__.py audio-copilot/scripts/latency_check.py audio-copilot/tests/test_cli.py
git commit -m "Add CLI entry point and latency check script (Phase 1 complete)"
```

---

### Task 10: Teleprompter mode with manual scroll

**Files:**
- Create: `audio-copilot/copilot/modes/teleprompter.py`
- Test: `audio-copilot/tests/test_teleprompter.py`

**Interfaces:**
- Consumes: display protocol (Task 7), `Segment` (Task 1).
- Produces: `TeleprompterMode(display, script_text, lookahead=8)` with `on_segment`, `on_status`, `on_command` (`"up"`, `"down"`, `"top"`); emits `{"event": "prompter", "lines", "pos"}`. Auto-scroll lands in Task 11 — in this task `on_segment` is a no-op.

- [ ] **Step 1: Write the failing tests**

`tests/test_teleprompter.py`:

```python
from copilot.events import Status
from copilot.modes.teleprompter import TeleprompterMode

SCRIPT = """Welcome everyone to the demo.

First we cover deployment pipelines.
Then we look at rollback strategy.
Finally, questions.
"""


class FakeDisplay:
    def __init__(self):
        self.events = []

    def update(self, ev):
        self.events.append(ev)


def make():
    d = FakeDisplay()
    return d, TeleprompterMode(d, SCRIPT)


def test_loads_nonempty_lines_and_renders_start():
    d, mode = make()
    assert d.events[-1]["event"] == "prompter"
    assert d.events[-1]["lines"] == [
        "Welcome everyone to the demo.",
        "First we cover deployment pipelines.",
        "Then we look at rollback strategy.",
        "Finally, questions.",
    ]
    assert d.events[-1]["pos"] == 0


def test_manual_scroll_commands():
    d, mode = make()
    mode.on_command("down")
    mode.on_command("down")
    assert d.events[-1]["pos"] == 2
    mode.on_command("up")
    assert d.events[-1]["pos"] == 1
    mode.on_command("top")
    assert d.events[-1]["pos"] == 0


def test_scroll_clamped_to_script():
    d, mode = make()
    mode.on_command("up")
    assert d.events[-1]["pos"] == 0
    for _ in range(10):
        mode.on_command("down")
    assert d.events[-1]["pos"] == 3


def test_status_forwarded():
    d, mode = make()
    mode.on_status(Status("audio_lost", "gone"))
    assert d.events[-1] == {"event": "status", "text": "audio_lost: gone"}
```

Run: `python3 -m pytest tests/test_teleprompter.py -v` — Expected: FAIL.

- [ ] **Step 2: Implement `copilot/modes/teleprompter.py`**

```python
class TeleprompterMode:
    def __init__(self, display, script_text, lookahead=8):
        self.display = display
        self.lines = [l.strip() for l in script_text.splitlines() if l.strip()]
        self.pos = 0
        self.lookahead = lookahead
        self._render()

    def on_segment(self, seg):
        pass  # auto-scroll arrives in the next task

    def on_status(self, st):
        self.display.update({"event": "status", "text": f"{st.kind}: {st.detail}"})

    def on_command(self, cmd):
        if cmd == "down":
            self.pos = min(self.pos + 1, len(self.lines) - 1)
        elif cmd == "up":
            self.pos = max(self.pos - 1, 0)
        elif cmd == "top":
            self.pos = 0
        self._render()

    def _render(self):
        self.display.update({"event": "prompter", "lines": self.lines,
                             "pos": self.pos})
```

- [ ] **Step 3: Run tests** — `python3 -m pytest tests/test_teleprompter.py -v` — Expected: 4 PASS.

- [ ] **Step 4: Commit**

```bash
git add audio-copilot/copilot/modes/teleprompter.py audio-copilot/tests/test_teleprompter.py
git commit -m "Add teleprompter mode with manual scroll"
```

---

### Task 11: Transcript-tracked auto-scroll

**Files:**
- Modify: `audio-copilot/copilot/modes/teleprompter.py`
- Test: `audio-copilot/tests/test_teleprompter.py` (append)

**Interfaces:**
- Consumes: `TeleprompterMode` (Task 10).
- Produces: `best_match_line(lines: list[str], pos: int, spoken: str, lookahead: int = 8, threshold: float = 0.55) -> int | None` in the same module; `TeleprompterMode.on_segment` now advances `pos` to the matched line.

- [ ] **Step 1: Append failing tests to `tests/test_teleprompter.py`**

```python
from copilot.events import Segment
from copilot.modes.teleprompter import best_match_line


def test_best_match_line_finds_upcoming_line():
    lines = ["Welcome everyone to the demo.",
             "First we cover deployment pipelines.",
             "Then we look at rollback strategy."]
    assert best_match_line(lines, 0, "first we cover deployment pipelines") == 1


def test_best_match_line_none_below_threshold():
    lines = ["Welcome everyone to the demo."]
    assert best_match_line(lines, 0, "completely unrelated words here") is None


def test_best_match_line_only_looks_ahead():
    lines = ["alpha beta gamma", "delta epsilon zeta"]
    assert best_match_line(lines, 1, "alpha beta gamma") is None


def test_on_segment_advances_position():
    d, mode = make()
    mode.on_segment(Segment("then we look at rollback strategy", 0.0, 2.0))
    assert d.events[-1]["pos"] == 2


def test_on_segment_ignores_chatter():
    d, mode = make()
    mode.on_segment(Segment("totally off script remark", 0.0, 2.0))
    assert d.events[-1]["pos"] == 0
```

Run: `python3 -m pytest tests/test_teleprompter.py -v` — Expected: new tests FAIL.

- [ ] **Step 2: Implement in `copilot/modes/teleprompter.py`**

Add at module top:

```python
import difflib
import re


def _norm(text):
    return " ".join(re.findall(r"[a-z']+", text.lower()))


def best_match_line(lines, pos, spoken, lookahead=8, threshold=0.55):
    """Best-matching line at or after pos within lookahead, else None."""
    spoken_n = _norm(spoken)
    if not spoken_n:
        return None
    best_i, best_r = None, threshold
    for i in range(pos, min(pos + lookahead, len(lines))):
        r = difflib.SequenceMatcher(None, spoken_n, _norm(lines[i])).ratio()
        if r > best_r:
            best_i, best_r = i, r
    return best_i
```

Replace `on_segment`:

```python
    def on_segment(self, seg):
        match = best_match_line(self.lines, self.pos, seg.text, self.lookahead)
        if match is not None and match != self.pos:
            self.pos = match
            self._render()
        elif match is None:
            self._render()  # keep view fresh even when off script
```

Note: `test_on_segment_ignores_chatter` asserts `pos == 0` after a non-match — the re-render with unchanged `pos` satisfies it.

- [ ] **Step 3: Run tests** — `python3 -m pytest tests/test_teleprompter.py -v` — Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add audio-copilot/copilot/modes/teleprompter.py audio-copilot/tests/test_teleprompter.py
git commit -m "Add transcript-tracked auto-scroll to teleprompter"
```

---

### Task 12: Overlay display (pywebview)

**Files:**
- Create: `audio-copilot/copilot/display/overlay.py`
- Create: `audio-copilot/copilot/display/overlay.html`

**Interfaces:**
- Consumes: display protocol events, `run_mode` (Task 7); called from `cli.main` (Task 9) as `start_overlay(make_mode: Callable[[display], mode], start_pipeline: Callable[[], None], event_q)`.
- Produces: `OverlayDisplay(window)` with `update(ev)`; `OverlayApi` with `command(cmd)` exposed to JS; `start_overlay(...)` which blocks until the window closes. No automated tests — verified by the manual checklist below.

- [ ] **Step 1: Implement `copilot/display/overlay.py`**

```python
import json
from pathlib import Path


class OverlayDisplay:
    def __init__(self, window):
        self.window = window

    def update(self, ev):
        self.window.evaluate_js(f"handleEvent({json.dumps(ev)})")


class OverlayApi:
    """Exposed to JS as pywebview.api; routes button clicks to the mode."""

    def __init__(self):
        self.mode = None

    def command(self, cmd):
        if self.mode is not None:
            self.mode.on_command(cmd)


def start_overlay(make_mode, start_pipeline, event_q):
    import webview

    from ..runner import run_mode

    api = OverlayApi()
    html = (Path(__file__).parent / "overlay.html").read_text()
    window = webview.create_window("Copilot", html=html, js_api=api,
                                   frameless=True, on_top=True,
                                   width=420, height=560, x=40, y=60)
    mode = make_mode(OverlayDisplay(window))
    api.mode = mode

    def boot():
        start_pipeline()
        run_mode(mode, event_q)

    webview.start(func=boot)
```

- [ ] **Step 2: Implement `copilot/display/overlay.html`**

```html
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * { margin: 0; box-sizing: border-box; }
  html, body { height: 100%; }
  body {
    font: 14px/1.45 -apple-system, "SF Pro Text", sans-serif;
    background: rgba(18, 18, 22, 0.92); color: #e8e8ec;
    display: flex; flex-direction: column; padding: 12px;
    -webkit-user-select: none;
  }
  #drag { height: 18px; cursor: move; flex: none; color: #6b6b74;
          font-size: 11px; letter-spacing: 0.08em; }
  #status { flex: none; color: #9a9aa4; font-size: 11px; min-height: 15px; }
  #prompter { flex: 1; overflow: hidden; display: none; font-size: 20px; }
  #prompter .line { padding: 3px 0; color: #7a7a84; }
  #prompter .line.current { color: #ffffff; font-weight: 600; }
  #transcript { flex: 1; overflow-y: auto; color: #b9b9c2; }
  #transcript div { padding: 2px 0; }
  #answer { flex: none; max-height: 45%; overflow-y: auto; white-space: pre-wrap;
            border-top: 1px solid #33333c; margin-top: 8px; padding-top: 8px; }
  #controls { flex: none; display: none; gap: 8px; padding-top: 8px; }
  #controls button { background: #2a2a32; border: 0; border-radius: 6px;
                     color: #e8e8ec; padding: 6px 14px; cursor: pointer; }
  svg { width: 16px; height: 16px; vertical-align: middle; }
</style>
</head>
<body>
  <div id="drag" class="pywebview-drag-region">COPILOT</div>
  <div id="status"></div>
  <div id="prompter"></div>
  <div id="transcript"></div>
  <div id="answer"></div>
  <div id="controls">
    <button onclick="pywebview.api.command('up')" aria-label="scroll up">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m18 15-6-6-6 6"/></svg>
    </button>
    <button onclick="pywebview.api.command('down')" aria-label="scroll down">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m6 9 6 6 6-6"/></svg>
    </button>
    <button onclick="pywebview.api.command('top')" aria-label="back to top">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m17 11-5-5-5 5"/><path d="M12 6v12"/></svg>
    </button>
  </div>
<script>
  const el = id => document.getElementById(id);
  function handleEvent(ev) {
    if (ev.event === "segment") {
      const d = document.createElement("div");
      d.textContent = "» " + ev.text;
      el("transcript").appendChild(d);
      el("transcript").scrollTop = el("transcript").scrollHeight;
    } else if (ev.event === "answer_start") {
      el("answer").textContent = "";
    } else if (ev.event === "answer_chunk") {
      el("answer").textContent += ev.text;
    } else if (ev.event === "status") {
      el("status").textContent = ev.text;
    } else if (ev.event === "prompter") {
      el("transcript").style.display = "none";
      el("answer").style.display = "none";
      el("prompter").style.display = "block";
      el("controls").style.display = "flex";
      const p = el("prompter");
      p.innerHTML = "";
      const lo = Math.max(0, ev.pos - 1);
      const hi = Math.min(ev.lines.length, ev.pos + 8);
      for (let i = lo; i < hi; i++) {
        const d = document.createElement("div");
        d.className = "line" + (i === ev.pos ? " current" : "");
        d.textContent = ev.lines[i];
        p.appendChild(d);
      }
    }
  }
</script>
</body>
</html>
```

- [ ] **Step 3: Run the full suite** — `python3 -m pytest tests -v` — Expected: all PASS (no new automated tests; ensures no regressions).

- [ ] **Step 4: Manual smoke checklist (Phase 2 gate)**

1. `printf 'Welcome to the demo.\nFirst we cover pipelines.\nThen rollback strategy.\nFinally questions.\n' > talk.txt` (in `audio-copilot/`; do not commit this file).
2. Run: `python3 -m copilot --mode teleprompter --script talk.txt --display overlay`
3. Verify: borderless dark window appears, stays on top of other apps, is draggable by its top strip.
4. Verify: script lines visible, current line highlighted; the chevron buttons scroll; "top" resets.
5. Read the second script line aloud — the highlight advances to it (auto-scroll).
6. Run: `python3 -m copilot --display overlay` (copilot mode) — speak a question, transcript and streamed answer appear in the window.
7. Close the window; the process exits cleanly.

- [ ] **Step 5: Commit**

```bash
git add audio-copilot/copilot/display/overlay.py audio-copilot/copilot/display/overlay.html
git commit -m "Add always-on-top pywebview overlay display"
```

---

### Task 13: README and audio-routing guide

**Files:**
- Create: `audio-copilot/README.md`

**Interfaces:**
- Consumes: everything; documents the CLI (Task 9).
- Produces: user-facing docs; the "missing BlackHole" error path in the spec points here.

- [ ] **Step 1: Write `audio-copilot/README.md`**

```markdown
# Audio Copilot

Local audio -> whisper.cpp -> Claude assistant with three planned modes:
teleprompter, practice, and meeting copilot. See
`docs/superpowers/specs/2026-07-28-audio-copilot-design.md`.

## Setup

    cd audio-copilot
    python3 -m pip install -r requirements.txt

First run downloads the whisper model (base.en). Claude access uses your
existing Claude Code login via the Claude Agent SDK.

## Usage

    python3 -m copilot                          # copilot mode, terminal display
    python3 -m copilot --display overlay        # floating always-on-top window
    python3 -m copilot --mode teleprompter --script talk.txt --display overlay
    python3 -m copilot --source "BlackHole 2ch" # transcribe system audio
    python3 scripts/latency_check.py            # measure STT + Claude latency

Teleprompter keys: on-window buttons scroll (up/down/top); reading the script
aloud auto-advances the highlight.

## Capturing system audio (BlackHole)

To transcribe other apps' audio (Teams, Zoom, a browser) you need BlackHole
(already installed if you use it with rekordbox):

1. Open Audio MIDI Setup -> "+" -> Create Multi-Output Device.
2. Check both your speakers/headphones AND "BlackHole 2ch".
3. Set the Multi-Output Device as system output (Sound settings).
4. Run with `--source "BlackHole 2ch"`.

You keep hearing audio through your speakers; the app reads the BlackHole
side. If the device is missing the app shows "audio_lost" and retries every
2 seconds.

## Tests

    python3 -m pytest tests -v

Tests are offline except the one-time whisper tiny.en model download.
```

- [ ] **Step 2: Verify docs against reality** — run each command in the Usage block (except the BlackHole one if no device is configured) and confirm they start without argument errors.

- [ ] **Step 3: Commit**

```bash
git add audio-copilot/README.md
git commit -m "Add audio-copilot README with BlackHole routing guide"
```

---

## Self-Review Notes

- **Spec coverage:** capture (Task 3), VAD+STT (Tasks 2, 4), brain (Task 6), question heuristic (Task 5), copilot mode + rolling window (Task 8), displays + event protocol (Tasks 7, 12), CLI + latency exit criteria (Task 9), teleprompter manual + auto-scroll (Tasks 10–11), error handling: device retry & lagging (Task 3), brain errors inline (Task 8), first-run guidance (Task 13). Spec Phases 3–4 intentionally deferred to follow-up plans.
- **Known simplifications (documented, accepted):** answer streaming is per-text-block, not per-token; the copilot mode's brain call blocks its event loop while streaming (segments queue up behind it) — acceptable for v1, revisit if latency check says otherwise.
- **Type consistency check:** `Segment(text, start, end)`, `Status(kind, detail)`, display protocol event names, `stream_response(system_prompt, user_prompt, _query)`, `on_command("up"/"down"/"top")` used identically across Tasks 1–12.
