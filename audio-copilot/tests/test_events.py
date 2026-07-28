from copilot.events import Segment, Status


def test_segment_fields():
    s = Segment(text="hello", start=1.0, end=2.5)
    assert (s.text, s.start, s.end) == ("hello", 1.0, 2.5)


def test_status_default_detail():
    st = Status(kind="audio_lost")
    assert st.kind == "audio_lost"
    assert st.detail == ""
