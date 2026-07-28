import numpy as np
from copilot.vad import UtteranceChunker

SR = 16000
BLOCK = 1600  # 100 ms


def silence(seconds):
    return np.zeros(int(SR * seconds), dtype=np.float32)


def tone(seconds):
    t = np.arange(int(SR * seconds)) / SR
    return (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def feed_all(chunker, audio):
    out = []
    for i in range(0, len(audio), BLOCK):
        out.extend(chunker.feed(audio[i:i + BLOCK]))
    tail = chunker.flush()
    if tail is not None:
        out.append(tail)
    return out


def test_silence_yields_nothing():
    assert feed_all(UtteranceChunker(), silence(3.0)) == []


def test_single_utterance_detected():
    audio = np.concatenate([silence(0.5), tone(1.5), silence(1.0)])
    utts = feed_all(UtteranceChunker(), audio)
    assert len(utts) == 1
    assert len(utts[0]) >= int(SR * 1.5)  # includes the speech


def test_long_speech_splits_at_max():
    utts = feed_all(UtteranceChunker(max_utterance_s=4.0), tone(10.0))
    assert len(utts) >= 2


def test_short_blip_ignored():
    audio = np.concatenate([silence(0.5), tone(0.1), silence(1.0)])
    assert feed_all(UtteranceChunker(), audio) == []
