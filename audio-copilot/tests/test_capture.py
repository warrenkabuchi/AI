import queue
import wave

import numpy as np

from copilot.capture import FileSource, MicSource
from copilot.events import Status


def write_wav(path, audio_f32, sr=16000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((audio_f32 * 32767).astype(np.int16).tobytes())


def test_file_source_replays_blocks(tmp_path):
    audio = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)  # 1 s
    write_wav(tmp_path / "a.wav", audio)
    q = queue.Queue()
    FileSource(tmp_path / "a.wav", q, block_s=0.1, realtime=False).run()
    blocks, item = [], q.get_nowait()
    while item is not None:
        blocks.append(item)
        item = q.get_nowait()
    total = np.concatenate(blocks)
    assert len(total) == 16000
    assert total.dtype == np.float32
    assert abs(float(total[0]) - (-0.5)) < 0.001


def blocks_of(value, seconds, sr=16000, block_s=0.1):
    n = int(sr * block_s)
    return [np.full(n, value, dtype=np.float32)] * int(seconds / block_s)


def drain_statuses(event_q):
    out = []
    while not event_q.empty():
        out.append(event_q.get_nowait())
    return out


def test_warns_when_input_is_digitally_silent():
    audio_q, event_q = queue.Queue(), queue.Queue()
    src = MicSource("BlackHole 2ch", audio_q, event_q, silence_warn_s=1.0)
    for b in blocks_of(0.0, 1.5):
        src._push(b)
    kinds = [(s.kind, s.detail) for s in drain_statuses(event_q)]
    silent = [d for k, d in kinds if k == "silent"]
    assert len(silent) == 1                      # warns once, not every block
    assert "multi-output" in silent[0].lower()   # BlackHole-specific guidance


def test_silence_warning_names_the_device_generically_for_a_mic():
    audio_q, event_q = queue.Queue(), queue.Queue()
    src = MicSource("MacBook Air Microphone", audio_q, event_q, silence_warn_s=1.0)
    for b in blocks_of(0.0, 1.5):
        src._push(b)
    silent = [s for s in drain_statuses(event_q) if s.kind == "silent"]
    assert len(silent) == 1
    assert "multi-output" not in silent[0].detail.lower()


def test_signal_resets_the_silence_watchdog():
    audio_q, event_q = queue.Queue(), queue.Queue()
    src = MicSource(None, audio_q, event_q, silence_warn_s=1.0)
    for b in blocks_of(0.0, 0.8):
        src._push(b)
    src._push(np.full(1600, 0.05, dtype=np.float32))  # sound arrives
    for b in blocks_of(0.0, 0.8):
        src._push(b)
    assert [s for s in drain_statuses(event_q) if s.kind == "silent"] == []


def test_recovery_after_silence_is_reported():
    audio_q, event_q = queue.Queue(), queue.Queue()
    src = MicSource(None, audio_q, event_q, silence_warn_s=1.0)
    for b in blocks_of(0.0, 1.5):
        src._push(b)
    src._push(np.full(1600, 0.05, dtype=np.float32))
    kinds = [s.kind for s in drain_statuses(event_q)]
    assert kinds == ["silent", "audio_ok"]


def test_mic_source_drops_oldest_when_full():
    audio_q = queue.Queue(maxsize=2)
    event_q = queue.Queue()
    src = MicSource(None, audio_q, event_q)
    b1, b2, b3 = (np.full(10, i, dtype=np.float32) for i in (1, 2, 3))
    src._push(b1)
    src._push(b2)
    src._push(b3)  # queue full -> drops b1, emits lagging status once
    assert float(audio_q.get_nowait()[0]) == 2.0
    assert float(audio_q.get_nowait()[0]) == 3.0
    st = event_q.get_nowait()
    assert isinstance(st, Status) and st.kind == "lagging"
    src._push(b1)  # still lagging episode: no duplicate status
    assert event_q.empty()
