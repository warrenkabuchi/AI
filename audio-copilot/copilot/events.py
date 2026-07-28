from dataclasses import dataclass


@dataclass
class Segment:
    text: str
    start: float
    end: float


@dataclass
class Status:
    kind: str  # "audio_lost" | "audio_ok" | "lagging" | "info"
    detail: str = ""
