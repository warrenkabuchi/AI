"""Claude access.

Two entry points:

- `Brain` keeps ONE agent session alive across questions. This matters a lot:
  a fresh one-shot query pays ~12s of subprocess startup every time, while a
  reused session answers in ~3s. Modes should use `Brain.stream`.
- `stream_response` is the one-shot form, kept for scripts and tests.
"""

import asyncio
import queue
import threading

_DONE = object()


def _sdk_query(prompt, system_prompt):
    from claude_agent_sdk import query, ClaudeAgentOptions
    options = ClaudeAgentOptions(system_prompt=system_prompt, max_turns=1,
                                 allowed_tools=[])
    return query(prompt=prompt, options=options)


def _texts(message):
    for block in getattr(message, "content", None) or []:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            yield text


def stream_response(system_prompt, user_prompt, _query=None):
    """Streams Claude's text response as chunks. Blocks the calling thread.

    One-shot: spawns a fresh agent session per call. Prefer `Brain` for
    anything interactive.
    """
    query_fn = _query or _sdk_query
    q = queue.Queue()
    threading.Thread(target=_run_once, args=(query_fn, system_prompt,
                                             user_prompt, q), daemon=True).start()
    yield from _drain(q)


def _run_once(query_fn, system_prompt, user_prompt, q):
    async def consume():
        async for message in query_fn(prompt=user_prompt,
                                      system_prompt=system_prompt):
            for text in _texts(message):
                q.put(text)

    try:
        asyncio.run(consume())
    except BaseException as e:
        q.put(e)
    finally:
        q.put(_DONE)


def _drain(q):
    while True:
        item = q.get()
        if item is _DONE:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


def _default_client_factory(options):
    from claude_agent_sdk import ClaudeSDKClient
    return ClaudeSDKClient(options=options)


class Brain:
    """A persistent Claude session usable from any thread.

    Owns an asyncio loop on a background thread; `stream` is a blocking
    generator so callers stay synchronous. Calls are serialized because one
    session handles one turn at a time.
    """

    def __init__(self, client_factory=None, warm=True,
                 warm_system_prompt=None, warm_prompt="ready?"):
        self._client_factory = client_factory or _default_client_factory
        self._client = None
        self._system_prompt = None
        self._lock = threading.Lock()
        self._closed = False
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        if warm:
            threading.Thread(target=self._warm, args=(warm_system_prompt or "",
                                                      warm_prompt),
                             daemon=True).start()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _warm(self, system_prompt, prompt):
        """Pays the cold-start cost before the user asks anything real."""
        try:
            for _ in self.stream(system_prompt, prompt):
                pass
        except Exception:
            pass  # warmup is best-effort; real calls report their own errors

    def stream(self, system_prompt, user_prompt):
        """Yields text chunks. Blocks the calling thread."""
        if self._closed:
            raise RuntimeError("Brain is closed")
        q = queue.Queue()
        with self._lock:
            future = asyncio.run_coroutine_threadsafe(
                self._ask(system_prompt, user_prompt, q), self._loop)
            try:
                yield from _drain(q)
            finally:
                future.cancel()

    async def _ask(self, system_prompt, user_prompt, q):
        try:
            await self._ensure_client(system_prompt)
            await self._client.query(user_prompt)
            async for message in self._client.receive_response():
                for text in _texts(message):
                    q.put(text)
        except BaseException as e:
            q.put(e)
        finally:
            q.put(_DONE)

    async def _ensure_client(self, system_prompt):
        if self._client is not None and self._system_prompt == system_prompt:
            return
        if self._client is not None:
            await self._client.disconnect()
        options = self._make_options(system_prompt)
        self._client = self._client_factory(options)
        await self._client.connect()
        self._system_prompt = system_prompt

    def _make_options(self, system_prompt):
        try:
            from claude_agent_sdk import ClaudeAgentOptions
        except ImportError:
            return {"system_prompt": system_prompt}
        return ClaudeAgentOptions(system_prompt=system_prompt, allowed_tools=[])

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._client is not None:
            asyncio.run_coroutine_threadsafe(self._client.disconnect(),
                                             self._loop).result(timeout=10)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
