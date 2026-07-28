import numpy as np


class UtteranceChunker:
    """Cuts a stream of PCM blocks into utterances using RMS energy."""

    def __init__(self, sample_rate=16000, threshold_rms=0.01,
                 silence_close_s=0.6, min_speech_s=0.3, max_utterance_s=8.0):
        self.sample_rate = sample_rate
        self.threshold_rms = threshold_rms
        self.silence_close_s = silence_close_s
        self.min_speech_s = min_speech_s
        self.max_utterance_s = max_utterance_s
        self._reset()

    def _reset(self):
        self._blocks = []
        self._speech = False
        self._speech_s = 0.0
        self._silence_run_s = 0.0
        self._total_s = 0.0

    def feed(self, block):
        out = []
        dur = len(block) / self.sample_rate
        rms = float(np.sqrt(np.mean(block.astype(np.float64) ** 2)))
        if rms >= self.threshold_rms:
            self._speech = True
            self._silence_run_s = 0.0
            self._speech_s += dur
            self._blocks.append(block)
            self._total_s += dur
        elif self._speech:
            self._silence_run_s += dur
            self._blocks.append(block)
            self._total_s += dur
            if self._silence_run_s >= self.silence_close_s:
                out.extend(self._close())
        if self._speech and self._total_s >= self.max_utterance_s:
            out.extend(self._close())
        return out

    def flush(self):
        closed = self._close()
        return closed[0] if closed else None

    def _close(self):
        utt = np.concatenate(self._blocks) if self._blocks else None
        keep = self._speech and self._speech_s >= self.min_speech_s
        self._reset()
        return [utt] if (keep and utt is not None) else []
