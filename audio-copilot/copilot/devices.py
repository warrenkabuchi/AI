"""Input-device lookup.

Device names are the app's only handle on audio hardware, and a typo
otherwise surfaces as a PortAudio error that names no alternatives. These
helpers turn that into a message listing what is actually available.
"""


class DeviceNotFound(Exception):
    pass


def _query():
    import sounddevice as sd
    return sd.query_devices()


def _inputs(devices=None):
    devices = _query() if devices is None else devices
    return [d for d in devices if d.get("max_input_channels", 0) > 0]


def format_device_list(_devices=None):
    lines = [f"  - {d['name']}" for d in _inputs(_devices)]
    return "\n".join(lines) if lines else "  (no input devices found)"


def resolve_device(name, _devices=None):
    """Maps a user-supplied name to a device sounddevice accepts.

    Returns None for the system default. Matches exactly first, then
    case-insensitively on substring.
    """
    if name is None or name == "default":
        return None

    inputs = _inputs(_devices)
    names = [d["name"] for d in inputs]

    if name in names:
        return name

    lowered = name.lower()
    matches = [n for n in names if lowered in n.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise DeviceNotFound(
            f'"{name}" matches more than one input device:\n'
            + "\n".join(f"  - {n}" for n in matches)
            + "\nUse the full name."
        )
    raise DeviceNotFound(
        f'No input device matching "{name}". Available input devices:\n'
        + format_device_list(_devices)
    )
