import argparse
import queue
import threading
from pathlib import Path

from .brain import Brain
from .capture import CaptureGate, FileSource, MicSource
from .devices import DeviceNotFound, format_device_list, resolve_device
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
    p.add_argument("--list-devices", action="store_true",
                   help="print available audio input devices and exit")
    return p


def make_mode(args, display, brain=None, gate=None):
    if args.mode == "copilot":
        return CopilotMode(display, brain.stream)
    if args.mode == "practice":
        from .modes.practice import PracticeMode
        if not args.topic:
            raise SystemExit("--mode practice requires --topic")
        return PracticeMode(display, brain.stream, topic=args.topic,
                            num_questions=args.questions, gate=gate)
    from .modes.teleprompter import TeleprompterMode
    if not args.script:
        raise SystemExit("--mode teleprompter requires --script")
    return TeleprompterMode(display, Path(args.script).read_text())


def build_pipeline(args, audio_q, event_q, gate=None):
    """Returns (source, threads) with threads not yet started."""
    # Resolve the audio source before loading whisper: a bad device name
    # should fail immediately, not after a 141 MB model load.
    if args.source.startswith("file:"):
        source = FileSource(args.source[5:], audio_q, realtime=True)
    else:
        try:
            device = resolve_device(args.source)
        except DeviceNotFound as e:
            raise SystemExit(str(e))
        source = MicSource(device, audio_q, event_q, gate=gate)

    transcriber = Transcriber(args.model)
    chunker = UtteranceChunker()
    threads = [
        threading.Thread(target=source.run, daemon=True),
        threading.Thread(target=stt_worker,
                         args=(audio_q, event_q, transcriber, chunker),
                         daemon=True),
    ]
    return source, threads


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.list_devices:
        print("Audio input devices:")
        print(format_device_list())
        return

    audio_q = queue.Queue(maxsize=50)  # ~5 s backlog cap
    event_q = queue.Queue()
    # The gate mutes capture while the app plays audio of its own, so the
    # mic never transcribes the app's voice back at it (practice mode).
    gate = CaptureGate()
    source, threads = build_pipeline(args, audio_q, event_q, gate)

    # Warm the session while whisper loads, so the first real question does
    # not pay the ~10s cold start.
    needs_brain = args.mode in ("copilot", "practice")
    brain = Brain(warm_system_prompt=SYSTEM_PROMPT) if needs_brain else None

    try:
        if args.display == "terminal":
            mode = make_mode(args, TerminalDisplay(), brain, gate)
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
            start_overlay(lambda display: make_mode(args, display, brain, gate),
                          lambda: [t.start() for t in threads], event_q)
    finally:
        if hasattr(source, "stop"):
            source.stop()
        if brain is not None:
            brain.close()


if __name__ == "__main__":
    main()
