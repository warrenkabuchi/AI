from .events import Segment, Status


def run_mode(mode, event_q):
    """Blocks consuming events until the None sentinel arrives.

    On end-of-stream the mode gets an `on_end` call if it defines one, so
    work still in flight (a half-finished answer waiting on a silence timer)
    is completed rather than dropped.
    """
    while True:
        ev = event_q.get()
        if ev is None:
            on_end = getattr(mode, "on_end", None)
            if on_end is not None:
                on_end()
            return
        if isinstance(ev, Segment):
            mode.on_segment(ev)
        elif isinstance(ev, Status):
            mode.on_status(ev)
