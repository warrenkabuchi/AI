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
