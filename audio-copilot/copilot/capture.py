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


class CaptureGate:
    """Mutes capture while the app plays its own audio (e.g. `say`).

    Held via `with gate:` around playback. Stays muted for `tail_s` after
    release because the last of the playback is still draining through the
    output device and the input buffers when the subprocess returns —
    blocks arriving in that window are the app's own voice, not the user.
    """

    def __init__(self, tail_s=0.4, clock=time.monotonic):
        self.tail_s = tail_s
        self._clock = clock
        self._holds = 0
        self._until = 0.0
        self._lock = threading.Lock()

    def acquire(self):
        with self._lock:
            self._holds += 1

    def release(self):
        with self._lock:
            self._holds = max(0, self._holds - 1)
            if self._holds == 0:
                self._until = self._clock() + self.tail_s

    def muted(self):
        with self._lock:
            return self._holds > 0 or self._clock() < self._until

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()


BLACKHOLE_HINT = (
    "no signal. Open Audio MIDI Setup, create a Multi-Output Device "
    "containing both your speakers and BlackHole, and select it as the "
    "system output in Sound settings."
)
GENERIC_SILENCE_HINT = (
    "no signal. Check the device is unmuted and is the one carrying the "
    "audio you want."
)

# A live mic picks up room noise well above this; a mis-routed virtual device
# delivers mathematical zero. The gap is what makes the warning trustworthy.
_SILENCE_RMS = 1e-5


class MicSource:
    """Captures a live input device; retries when the device disappears.

    Also watches for an input that is connected but delivering pure silence,
    which is the usual symptom of a virtual device with nothing routed into
    it — otherwise indistinguishable from "nobody is talking".
    """

    def __init__(self, device, audio_q, event_q, block_s=0.1, sample_rate=16000,
                 silence_warn_s=10.0, gate=None):
        self.device = device
        self.audio_q = audio_q
        self.event_q = event_q
        self.block_s = block_s
        self.sample_rate = sample_rate
        self.silence_warn_s = silence_warn_s
        self.gate = gate
        self._stop = threading.Event()
        self._lagging = False
        self._silent_s = 0.0
        self._warned_silent = False

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

    def _watch_silence(self, block):
        rms = float(np.sqrt(np.mean(block.astype(np.float64) ** 2)))
        if rms > _SILENCE_RMS:
            if self._warned_silent:
                self.event_q.put(Status("audio_ok", "signal restored"))
            self._silent_s = 0.0
            self._warned_silent = False
            return
        self._silent_s += len(block) / self.sample_rate
        if self._silent_s >= self.silence_warn_s and not self._warned_silent:
            self._warned_silent = True
            name = str(self.device or "default")
            hint = (BLACKHOLE_HINT if "blackhole" in name.lower()
                    else GENERIC_SILENCE_HINT)
            self.event_q.put(Status("silent", f"{name}: {hint}"))

    def _push(self, block):
        if self.gate is not None and self.gate.muted():
            return
        self._watch_silence(block)
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
