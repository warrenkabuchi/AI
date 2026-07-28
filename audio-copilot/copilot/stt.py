import re

import numpy as np

from .events import Segment

# Whisper labels non-speech audio with bracketed markers — "[BLANK_AUDIO]",
# "(wind blowing)", "*click*" — which read as real answers downstream.
_NON_SPEECH = re.compile(r"^[\[\(\*].*[\]\)\*]$")


class Transcriber:
    def __init__(self, model_name="base.en"):
        from pywhispercpp.model import Model
        # print_progress defaults on and floods the terminal with
        # "Progress: 0%" lines on every transcription.
        self._model = Model(model_name, print_progress=False)

    def transcribe(self, audio):
        sr = 16000
        min_len = int(sr * 1.1)  # whisper.cpp rejects audio under 1 s
        if len(audio) < min_len:
            audio = np.pad(audio, (0, min_len - len(audio)))
        segments = self._model.transcribe(audio)
        texts = (s.text.strip() for s in segments)
        return " ".join(t for t in texts if t and not _NON_SPEECH.match(t))


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
