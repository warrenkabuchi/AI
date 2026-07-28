from copilot.events import Segment, Status
from copilot.modes.practice import PracticeMode


class FakeDisplay:
    def __init__(self):
        self.events = []

    def update(self, ev):
        self.events.append(ev)

    def kinds(self):
        return [e["event"] for e in self.events]

    def last(self, kind):
        return [e for e in self.events if e["event"] == kind][-1]


class FakeScheduler:
    """Stands in for threading.Timer so tests drive the answer-gap timeout."""

    def __init__(self):
        self.pending = None

    def __call__(self, delay, fn):
        self.pending = fn
        return self

    def cancel(self):
        self.pending = None

    def fire(self):
        fn, self.pending = self.pending, None
        assert fn is not None, "no timer was scheduled"
        fn()


def scripted_brain(*replies):
    """Returns a brain_fn yielding each reply in turn, recording prompts."""
    calls = []
    queue = list(replies)

    def brain(system_prompt, user_prompt):
        calls.append(user_prompt)
        text = queue.pop(0) if queue else "fallback"
        for word in text.split(" "):
            yield word + " "

    brain.calls = calls
    return brain


def make(num_questions=2, **kw):
    d = FakeDisplay()
    sched = FakeScheduler()
    spoken = []
    brain = kw.pop("brain", scripted_brain(
        "What is idempotency?", "Score 7. Good definition, weak on examples.",
        "What is a rollback?", "Score 8. Clear and concrete.",
        "You averaged 7.5 across two questions."))
    mode = PracticeMode(d, brain, topic="devops", speak=spoken.append,
                        scheduler=sched, num_questions=num_questions, **kw)
    return d, mode, sched, spoken, brain


def test_start_asks_and_speaks_first_question():
    d, mode, sched, spoken, brain = make()
    mode.start()
    q = d.last("question")
    assert q["text"] == "What is idempotency?"
    assert q["index"] == 1 and q["total"] == 2
    assert spoken == ["What is idempotency?"]
    assert "devops" in brain.calls[0]


def test_answer_accumulates_and_scores_after_gap():
    d, mode, sched, spoken, brain = make()
    mode.start()
    mode.on_segment(Segment("Idempotency means", 0.0, 1.0))
    mode.on_segment(Segment("you can retry safely.", 1.0, 2.0))
    assert d.last("answer_progress")["text"] == "Idempotency means you can retry safely."
    assert "critique_start" not in d.kinds()  # still listening

    sched.fire()  # silence gap elapsed -> score the answer
    assert "critique_start" in d.kinds()
    scoring_prompt = brain.calls[1]
    assert "What is idempotency?" in scoring_prompt
    assert "you can retry safely." in scoring_prompt


def test_scoring_advances_to_next_question():
    d, mode, sched, spoken, brain = make()
    mode.start()
    mode.on_segment(Segment("an answer", 0.0, 1.0))
    sched.fire()
    assert d.last("question")["text"] == "What is a rollback?"
    assert d.last("question")["index"] == 2
    assert spoken[-1] == "What is a rollback?"


def test_session_ends_with_summary_after_last_question():
    d, mode, sched, spoken, brain = make(num_questions=1)
    mode.start()
    mode.on_segment(Segment("an answer", 0.0, 1.0))
    sched.fire()
    assert "summary" in d.kinds()
    assert "averaged" in d.last("summary")["text"] or d.last("summary")["text"]
    assert mode.finished


def test_segments_ignored_once_finished():
    d, mode, sched, spoken, brain = make(num_questions=1)
    mode.start()
    mode.on_segment(Segment("an answer", 0.0, 1.0))
    sched.fire()
    before = len(d.events)
    mode.on_segment(Segment("more talking", 5.0, 6.0))
    assert len(d.events) == before  # no further output after the summary


def test_done_command_scores_immediately():
    d, mode, sched, spoken, brain = make()
    mode.start()
    mode.on_segment(Segment("my answer", 0.0, 1.0))
    mode.on_command("done")
    assert "critique_start" in d.kinds()
    assert sched.pending is None  # pending gap timer was cancelled


def test_skip_moves_on_without_scoring():
    d, mode, sched, spoken, brain = make()
    mode.start()
    mode.on_command("skip")
    assert "critique_start" not in d.kinds()
    assert d.last("question")["index"] == 2


def test_silence_schedules_no_timer():
    d, mode, sched, spoken, brain = make()
    mode.start()
    # The gap timer only runs once speech arrives, so total silence simply
    # keeps the question open instead of scoring an empty answer.
    assert sched.pending is None


def test_empty_answer_is_not_scored():
    d, mode, sched, spoken, brain = make()
    mode.start()
    mode.on_command("done")  # signalled done without having said anything
    assert "critique_start" not in d.kinds()
    assert mode.listening  # still waiting for a real answer


def test_brain_failure_does_not_kill_session():
    def broken(system_prompt, user_prompt):
        raise RuntimeError("offline")
        yield  # pragma: no cover

    d, mode, sched, spoken, brain = make(brain=broken)
    mode.start()
    assert "status" in d.kinds()
    assert "offline" in d.last("status")["text"]


def test_on_end_scores_pending_answer_then_summarises():
    d, mode, sched, spoken, brain = make()
    mode.start()
    mode.on_segment(Segment("a partial answer", 0.0, 1.0))
    mode.on_end()  # audio stream ended before the silence timer fired
    assert "critique_start" in d.kinds()
    assert "summary" in d.kinds()
    assert mode.finished


def test_on_end_with_nothing_said_still_summarises():
    d, mode, sched, spoken, brain = make()
    mode.start()
    mode.on_end()
    assert "critique_start" not in d.kinds()
    assert "summary" in d.kinds()


def test_on_end_is_idempotent():
    d, mode, sched, spoken, brain = make(num_questions=1)
    mode.start()
    mode.on_segment(Segment("an answer", 0.0, 1.0))
    sched.fire()
    count = d.kinds().count("summary")
    mode.on_end()
    assert d.kinds().count("summary") == count


def test_status_forwarded():
    d, mode, sched, spoken, brain = make()
    mode.on_status(Status("audio_lost", "gone"))
    assert d.last("status")["text"] == "audio_lost: gone"
