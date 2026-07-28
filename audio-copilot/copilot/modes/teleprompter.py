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


class TeleprompterMode:
    def __init__(self, display, script_text, lookahead=8):
        self.display = display
        self.lines = [l.strip() for l in script_text.splitlines() if l.strip()]
        self.pos = 0
        self.lookahead = lookahead
        self._render()

    def on_segment(self, seg):
        match = best_match_line(self.lines, self.pos, seg.text, self.lookahead)
        if match is not None and match != self.pos:
            self.pos = match
        self._render()

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
