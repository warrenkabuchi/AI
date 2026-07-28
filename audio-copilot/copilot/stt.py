import numpy as np

from .events import Segment


class Transcriber:
    def __init__(self, model_name="base.en"):
        from pywhispercpp.model import Model
        self._model = Model(model_name)

    def transcribe(self, audio):
        sr = 16000
        min_len = int(sr * 1.1)  # whisper.cpp rejects audio under 1 s
        if len(audio) < min_len:
            audio = np.pad(audio, (0, min_len - len(audio)))
        segments = self._model.transcribe(audio)
        return " ".join(s.text.strip() for s in segments).strip()


def stt_worker(audio_q, event_q, transcriber, chunker):
    """Consumes PCM blocks from audio_q until None; emits Segment to event_q."""
    sr = chunker.sample_rate
    samples_seen = 0

    def emit(utterance):
        end = samples_seen / sr
        start = max(0.0, end - len(utterance) / sr)
        text = transcriber.transcribe(utterance)
        if text:
            event_q.put(Segment(text=text, start=start, end=end))

    while True:
        block = audio_q.get()
        if block is None:
            tail = chunker.flush()
            if tail is not None:
                emit(tail)
            event_q.put(None)
            return
        samples_seen += len(block)
        for utterance in chunker.feed(block):
            emit(utterance)
