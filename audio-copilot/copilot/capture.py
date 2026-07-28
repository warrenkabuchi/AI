import queue
import threading
import time
import wave

import numpy as np

from .events import Status


class FileSource:
    """Replays a 16 kHz mono 16-bit WAV into audio_q. Test/latency harness."""

    def __init__(self, path, audio_q, block_s=0.1, realtime=False):
        self.path = path
        self.audio_q = audio_q
        self.block_s = block_s
        self.realtime = realtime

    def run(self):
        with wave.open(str(self.path), "rb") as w:
            if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (16000, 1, 2):
                raise ValueError(f"{self.path}: need 16 kHz mono 16-bit WAV")
            raw = w.readframes(w.getnframes())
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        n = int(16000 * self.block_s)
        for i in range(0, len(audio), n):
            self.audio_q.put(audio[i:i + n])
            if self.realtime:
                time.sleep(self.block_s)
        self.audio_q.put(None)


class MicSource:
    """Captures a live input device; retries when the device disappears."""

    def __init__(self, device, audio_q, event_q, block_s=0.1, sample_rate=16000):
        self.device = device
        self.audio_q = audio_q
        self.event_q = event_q
        self.block_s = block_s
        self.sample_rate = sample_rate
        self._stop = threading.Event()
        self._lagging = False

    def run(self):
        import sounddevice as sd
        while not self._stop.is_set():
            try:
                with sd.InputStream(device=self.device,
                                    samplerate=self.sample_rate, channels=1,
                                    dtype="float32",
                                    blocksize=int(self.sample_rate * self.block_s),
                                    callback=self._callback):
                    self.event_q.put(Status("audio_ok", str(self.device or "default")))
                    while not self._stop.is_set():
                        time.sleep(0.2)
            except Exception as e:
                self.event_q.put(Status("audio_lost", str(e)))
                if self._stop.wait(2.0):
                    break
        self.audio_q.put(None)

    def stop(self):
        self._stop.set()

    def _callback(self, indata, frames, time_info, status):
        self._push(indata[:, 0].copy())

    def _push(self, block):
        try:
            self.audio_q.put_nowait(block)
            self._lagging = False
        except queue.Full:
            try:
                self.audio_q.get_nowait()
            except queue.Empty:
                pass
            self.audio_q.put_nowait(block)
            if not self._lagging:
                self._lagging = True
                self.event_q.put(Status("lagging", "audio backlog; dropping oldest"))
