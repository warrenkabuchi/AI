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
