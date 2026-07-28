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
        elif kind == "question":
            print(f"\nQ{ev['index']}/{ev['total']}: {ev['text']}\n", file=self.out)
        elif kind == "answer_progress":
            print(f"\r» {ev['text']}", end="", file=self.out)
            self.out.flush()
        elif kind == "critique_start":
            print("\n--- critique ---", file=self.out)
        elif kind == "critique_chunk":
            self.out.write(ev["text"])
            self.out.flush()
        elif kind == "critique_end":
            print("\n----------------", file=self.out)
        elif kind == "summary":
            print(f"\n=== session summary ===\n{ev['text']}", file=self.out)
        elif kind == "prompter":
            lines, pos = ev["lines"], ev["pos"]
            lo, hi = max(0, pos - 1), min(len(lines), pos + 5)
            print("\n" * 2, file=self.out)
            for i in range(lo, hi):
                marker = ">" if i == pos else " "
                print(f"{marker} {lines[i]}", file=self.out)
