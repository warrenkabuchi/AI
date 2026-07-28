import pytest

from copilot.devices import DeviceNotFound, format_device_list, resolve_device

DEVICES = [
    {"name": "Warren's iPhone Microphone", "max_input_channels": 1},
    {"name": "BlackHole 2ch", "max_input_channels": 2},
    {"name": "MacBook Air Microphone", "max_input_channels": 1},
    {"name": "External Speakers", "max_input_channels": 0},  # output only
]


def test_default_resolves_to_none():
    assert resolve_device("default", _devices=DEVICES) is None
    assert resolve_device(None, _devices=DEVICES) is None


def test_exact_name_match():
    assert resolve_device("BlackHole 2ch", _devices=DEVICES) == "BlackHole 2ch"


def test_case_insensitive_partial_match():
    assert resolve_device("blackhole", _devices=DEVICES) == "BlackHole 2ch"
    assert resolve_device("macbook air", _devices=DEVICES) == "MacBook Air Microphone"


def test_output_only_devices_are_not_matchable():
    with pytest.raises(DeviceNotFound):
        resolve_device("External Speakers", _devices=DEVICES)


def test_unknown_device_lists_the_alternatives():
    with pytest.raises(DeviceNotFound) as excinfo:
        resolve_device("Blackhol 3ch", _devices=DEVICES)
    msg = str(excinfo.value)
    assert "Blackhol 3ch" in msg
    assert "BlackHole 2ch" in msg          # shows what IS available
    assert "External Speakers" not in msg  # only input-capable ones


def test_ambiguous_prefix_picks_nothing_and_explains():
    devices = [{"name": "Mic A", "max_input_channels": 1},
               {"name": "Mic B", "max_input_channels": 1}]
    with pytest.raises(DeviceNotFound) as excinfo:
        resolve_device("mic", _devices=devices)
    assert "matches more than one" in str(excinfo.value).lower()


def test_format_device_list_shows_only_inputs():
    text = format_device_list(_devices=DEVICES)
    assert "BlackHole 2ch" in text
    assert "MacBook Air Microphone" in text
    assert "External Speakers" not in text
