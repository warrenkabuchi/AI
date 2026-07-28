import argparse
import queue
import threading
from pathlib import Path

from .brain import Brain
from .capture import FileSource, MicSource
from .display.terminal import TerminalDisplay
from .modes.copilot_mode import SYSTEM_PROMPT, CopilotMode
from .runner import run_mode
from .stt import Transcriber, stt_worker
from .vad import UtteranceChunker


def build_parser():
    p = argparse.ArgumentParser(prog="copilot")
    p.add_argument("--mode", choices=["copilot", "teleprompter", "practice"],
                   default="copilot")
    p.add_argument("--source", default="default",
                   help='"default", an input device name (e.g. "BlackHole 2ch"), '
                        'or "file:path.wav"')
    p.add_argument("--display", choices=["terminal", "overlay"], default="terminal")
    p.add_argument("--model", default="base.en", help="whisper model name")
    p.add_argument("--script", default=None,
                   help="script file for teleprompter mode")
    p.add_argument("--topic", default=None,
                   help="subject to be quizzed on in practice mode")
    p.add_argument("--questions", type=int, default=5,
                   help="how many questions practice mode asks")
    return p


def make_mode(args, display, brain=None):
    if args.mode == "copilot":
        return CopilotMode(display, brain.stream)
    if args.mode == "practice":
        from .modes.practice import PracticeMode
        if not args.topic:
            raise SystemExit("--mode practice requires --topic")
        return PracticeMode(display, brain.stream, topic=args.topic,
                            num_questions=args.questions)
    from .modes.teleprompter import TeleprompterMode
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

    # Warm the session while whisper loads, so the first real question does
    # not pay the ~10s cold start.
    needs_brain = args.mode in ("copilot", "practice")
    brain = Brain(warm_system_prompt=SYSTEM_PROMPT) if needs_brain else None

    try:
        if args.display == "terminal":
            mode = make_mode(args, TerminalDisplay(), brain)
            for t in threads:
                t.start()
            try:
                if hasattr(mode, "start"):
                    mode.start()
                run_mode(mode, event_q)
            except KeyboardInterrupt:
                pass
        else:
            from .display.overlay import start_overlay
            start_overlay(lambda display: make_mode(args, display, brain),
                          lambda: [t.start() for t in threads], event_q)
    finally:
        if hasattr(source, "stop"):
            source.stop()
        if brain is not None:
            brain.close()


if __name__ == "__main__":
    main()
