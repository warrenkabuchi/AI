import json
from pathlib import Path


class OverlayDisplay:
    def __init__(self, window):
        self.window = window

    def update(self, ev):
        self.window.evaluate_js(f"handleEvent({json.dumps(ev)})")


class OverlayApi:
    """Exposed to JS as pywebview.api; routes button clicks to the mode."""

    def __init__(self):
        self.mode = None

    def command(self, cmd):
        if self.mode is not None:
            self.mode.on_command(cmd)


def start_overlay(make_mode, start_pipeline, event_q):
    import webview

    from ..runner import run_mode

    api = OverlayApi()
    html = (Path(__file__).parent / "overlay.html").read_text()
    window = webview.create_window("Copilot", html=html, js_api=api,
                                   frameless=True, on_top=True,
                                   width=420, height=560, x=40, y=60)
    mode = make_mode(OverlayDisplay(window))
    api.mode = mode

    def boot():
        start_pipeline()
        if hasattr(mode, "start"):
            mode.start()
        run_mode(mode, event_q)

    webview.start(func=boot)
