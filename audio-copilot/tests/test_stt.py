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
    # The fixture avoids rare proper nouns: macOS `say` renders them poorly enough
    # that Whisper misreads them ("Kubernetes" -> "Qubernetes"), which would make
    # this a synthesis-quality benchmark rather than a check of our wiring.
    text = Transcriber("tiny.en").transcribe(load_fixture())
    assert "scale this service" in text.lower()


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


def test_transcriber_suppresses_whisper_progress_output(monkeypatch):
    import pywhispercpp.model as pw

    captured = {}

    class FakeModel:
        def __init__(self, name, **params):
            captured.update(params)

    monkeypatch.setattr(pw, "Model", FakeModel)
    from copilot.stt import Transcriber
    Transcriber("tiny.en")
    assert captured.get("print_progress") is False


def test_non_speech_markers_are_dropped(monkeypatch):
    import pywhispercpp.model as pw

    class FakeSeg:
        def __init__(self, text):
            self.text = text

    class FakeModel:
        def __init__(self, name, **params):
            pass

        def transcribe(self, audio):
            return [FakeSeg(" [BLANK_AUDIO] "), FakeSeg("real speech"),
                    FakeSeg("(wind blowing)"), FakeSeg("*click*")]

    monkeypatch.setattr(pw, "Model", FakeModel)
    from copilot.stt import Transcriber
    assert Transcriber("tiny.en").transcribe(np.zeros(20000)) == "real speech"
