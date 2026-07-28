from .events import Segment, Status


def run_mode(mode, event_q):
    """Blocks consuming events until the None sentinel arrives."""
    while True:
        ev = event_q.get()
        if ev is None:
            return
        if isinstance(ev, Segment):
            mode.on_segment(ev)
        elif isinstance(ev, Status):
            mode.on_status(ev)
