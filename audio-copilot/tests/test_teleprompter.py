from copilot.events import Segment, Status
from copilot.modes.teleprompter import TeleprompterMode, best_match_line

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
