import pytest

from copilot.brain import stream_response


class Block:
    def __init__(self, text):
        self.text = text


class Msg:
    def __init__(self, texts):
        self.content = [Block(t) for t in texts]


def test_yields_text_blocks_in_order():
    async def fake_query(prompt, system_prompt):
        assert system_prompt == "sys"
        assert "hello" in prompt
        yield Msg(["One. "])
        yield Msg(["Two."])

    assert list(stream_response("sys", "hello", _query=fake_query)) == ["One. ", "Two."]


def test_ignores_non_text_messages():
    class ToolMsg:
        content = None

    async def fake_query(prompt, system_prompt):
        yield ToolMsg()
        yield Msg(["ok"])

    assert list(stream_response("s", "u", _query=fake_query)) == ["ok"]


def test_raises_on_error():
    async def bad_query(prompt, system_prompt):
        raise RuntimeError("boom")
        yield  # pragma: no cover

    with pytest.raises(RuntimeError, match="boom"):
        list(stream_response("s", "u", _query=bad_query))
