import threading

import pytest

from copilot.brain import Brain


class Block:
    def __init__(self, text):
        self.text = text


class Msg:
    def __init__(self, *texts):
        self.content = [Block(t) for t in texts]


class FakeClient:
    """Stands in for ClaudeSDKClient: records connects and replays scripted turns."""

    connects = 0

    def __init__(self, options=None):
        self.options = options
        self.prompts = []
        self._replies = [Msg("chunk-a "), Msg("chunk-b")]

    async def connect(self):
        FakeClient.connects += 1

    async def disconnect(self):
        pass

    async def query(self, prompt):
        self.prompts.append(prompt)

    async def receive_response(self):
        for m in self._replies:
            yield m


@pytest.fixture(autouse=True)
def reset_counter():
    FakeClient.connects = 0


def test_streams_chunks():
    brain = Brain(client_factory=FakeClient, warm=False)
    try:
        assert list(brain.stream("sys", "hello")) == ["chunk-a ", "chunk-b"]
    finally:
        brain.close()


def test_reuses_one_connection_across_calls():
    brain = Brain(client_factory=FakeClient, warm=False)
    try:
        list(brain.stream("sys", "one"))
        list(brain.stream("sys", "two"))
        list(brain.stream("sys", "three"))
        assert FakeClient.connects == 1  # the whole point: no per-call subprocess
        assert brain._client.prompts == ["one", "two", "three"]
    finally:
        brain.close()


def test_reconnects_when_system_prompt_changes():
    brain = Brain(client_factory=FakeClient, warm=False)
    try:
        list(brain.stream("sys-a", "one"))
        list(brain.stream("sys-b", "two"))
        assert FakeClient.connects == 2
    finally:
        brain.close()


def test_errors_surface_to_caller():
    class BoomClient(FakeClient):
        async def query(self, prompt):
            raise RuntimeError("offline")

    brain = Brain(client_factory=BoomClient, warm=False)
    try:
        with pytest.raises(RuntimeError, match="offline"):
            list(brain.stream("sys", "hello"))
    finally:
        brain.close()


def test_concurrent_callers_are_serialized():
    brain = Brain(client_factory=FakeClient, warm=False)
    results = []

    def call(n):
        results.append("".join(brain.stream("sys", f"q{n}")))

    try:
        threads = [threading.Thread(target=call, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert results == ["chunk-a chunk-b"] * 4
        assert FakeClient.connects == 1
    finally:
        brain.close()


def test_close_is_idempotent():
    brain = Brain(client_factory=FakeClient, warm=False)
    list(brain.stream("sys", "hi"))
    brain.close()
    brain.close()
