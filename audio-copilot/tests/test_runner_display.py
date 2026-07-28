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
