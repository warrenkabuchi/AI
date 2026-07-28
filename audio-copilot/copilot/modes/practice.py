"""Practice mode: Claude interviews you, you answer aloud, Claude critiques.

Unlike copilot mode this drives the conversation, so it holds a small state
machine: ask -> listen -> score -> repeat -> summarise. An answer is finished
when the speaker goes quiet for `answer_gap_s`, or when they say so via the
"done" command.
"""

import subprocess
import threading

SYSTEM_PROMPT = (
    "You are a demanding but fair technical interviewer running a practice "
    "session. Be concise and specific. Never pad with encouragement."
)

QUESTION_TEMPLATE = (
    "Topic: {topic}\n"
    "Questions already asked:\n{asked}\n\n"
    "Ask ONE new interview question on this topic. Output only the question, "
    "no preamble, no numbering."
)

SCORE_TEMPLATE = (
    "Question: {question}\n"
    "Candidate's spoken answer (auto-transcribed, so expect small errors):\n"
    "{answer}\n\n"
    "Score it out of 10 and give at most three specific critiques. Say what "
    "was missing, not just what was good. Plain text, no headers."
)

SUMMARY_TEMPLATE = (
    "Practice session on {topic} just finished. Questions and answers:\n\n"
    "{transcript}\n\n"
    "Give a short overall verdict: the strongest pattern, the weakest, and one "
    "thing to drill next. Plain text, no headers."
)


def _say(text):
    subprocess.run(["say", text], check=False)


def _timer(delay, fn):
    t = threading.Timer(delay, fn)
    t.daemon = True
    t.start()
    return t


class PracticeMode:
    def __init__(self, display, brain_fn, topic, speak=None, scheduler=None,
                 num_questions=5, answer_gap_s=2.5):
        self.display = display
        self.brain_fn = brain_fn
        self.topic = topic
        self.speak = speak or _say
        self.scheduler = scheduler or _timer
        self.num_questions = num_questions
        self.answer_gap_s = answer_gap_s

        self.index = 0
        self.question = None
        self.answer_parts = []
        self.history = []
        self.listening = False
        self.finished = False
        self._gap_timer = None

    # --- lifecycle ---------------------------------------------------

    def start(self):
        self._next_question()

    def on_segment(self, seg):
        if not self.listening or self.finished:
            return
        self.answer_parts.append(seg.text)
        self.display.update({"event": "answer_progress",
                             "text": " ".join(self.answer_parts)})
        self._restart_gap_timer()

    def on_status(self, st):
        self.display.update({"event": "status", "text": f"{st.kind}: {st.detail}"})

    def on_end(self):
        """Audio ended: score whatever was said, then close out the session."""
        if self.finished:
            return
        self._cancel_gap_timer()
        if self.listening and self.answer_parts:
            self._finish_answer()
        self._summarise()

    def on_command(self, cmd):
        if self.finished:
            return
        if cmd == "done":
            self._cancel_gap_timer()
            self._finish_answer()
        elif cmd == "skip":
            self._cancel_gap_timer()
            self.listening = False
            self._next_question()
        elif cmd == "stop":
            self._cancel_gap_timer()
            self._summarise()

    # --- internals ---------------------------------------------------

    def _next_question(self):
        if self.index >= self.num_questions:
            self._summarise()
            return
        asked = "\n".join(q for q, _ in self.history) or "(none yet)"
        question = self._collect(QUESTION_TEMPLATE.format(topic=self.topic,
                                                          asked=asked))
        if question is None:
            return
        self.index += 1
        self.question = question.strip()
        self.answer_parts = []
        self.listening = True
        self.display.update({"event": "question", "text": self.question,
                             "index": self.index, "total": self.num_questions})
        self.speak(self.question)

    def _finish_answer(self):
        answer = " ".join(self.answer_parts).strip()
        self.listening = False
        if not answer:
            # Nothing was said; keep waiting rather than scoring silence.
            self.listening = True
            return
        self.history.append((self.question, answer))
        self.display.update({"event": "critique_start"})
        self._stream(SCORE_TEMPLATE.format(question=self.question, answer=answer),
                     chunk_event="critique_chunk")
        self.display.update({"event": "critique_end"})
        self._next_question()

    def _summarise(self):
        if self.finished:
            return
        self.finished = True
        self.listening = False
        if not self.history:
            self.display.update({"event": "summary", "text": "No answers recorded."})
            return
        transcript = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in self.history)
        text = self._collect(SUMMARY_TEMPLATE.format(topic=self.topic,
                                                     transcript=transcript))
        self.display.update({"event": "summary", "text": (text or "").strip()})

    def _restart_gap_timer(self):
        self._cancel_gap_timer()
        self._gap_timer = self.scheduler(self.answer_gap_s, self._on_gap)

    def _cancel_gap_timer(self):
        if self._gap_timer is not None:
            self._gap_timer.cancel()
            self._gap_timer = None

    def _on_gap(self):
        self._gap_timer = None
        if self.listening and not self.finished:
            self._finish_answer()

    def _collect(self, prompt):
        """Runs a prompt to completion, returning None if the call failed."""
        parts = []
        try:
            for chunk in self.brain_fn(SYSTEM_PROMPT, prompt):
                parts.append(chunk)
        except Exception as e:
            self.display.update({"event": "status", "text": f"brain error: {e}"})
            return None
        return "".join(parts)

    def _stream(self, prompt, chunk_event):
        try:
            for chunk in self.brain_fn(SYSTEM_PROMPT, prompt):
                self.display.update({"event": chunk_event, "text": chunk})
        except Exception as e:
            self.display.update({"event": "status", "text": f"brain error: {e}"})
