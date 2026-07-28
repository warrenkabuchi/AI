import json

from copilot.display.overlay import OverlayApi, OverlayDisplay


class FakeWindow:
    def __init__(self):
        self.calls = []

    def evaluate_js(self, script):
        self.calls.append(script)


class RecordingMode:
    def __init__(self):
        self.commands = []

    def on_command(self, cmd):
        self.commands.append(cmd)


def test_update_serializes_event_to_js():
    w = FakeWindow()
    OverlayDisplay(w).update({"event": "segment", "text": 'he said "hi"'})
    assert len(w.calls) == 1
    assert w.calls[0].startswith("handleEvent(")
    payload = json.loads(w.calls[0][len("handleEvent("):-1])
    assert payload == {"event": "segment", "text": 'he said "hi"'}


def test_api_routes_commands_to_mode():
    api = OverlayApi()
    mode = RecordingMode()
    api.mode = mode
    api.command("down")
    api.command("top")
    assert mode.commands == ["down", "top"]


def test_api_before_mode_is_attached_is_a_noop():
    OverlayApi().command("down")  # must not raise during window startup
