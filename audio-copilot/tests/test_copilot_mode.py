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
