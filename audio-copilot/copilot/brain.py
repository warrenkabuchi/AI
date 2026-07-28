import asyncio
import queue
import threading

_DONE = object()


def _sdk_query(prompt, system_prompt):
    from claude_agent_sdk import query, ClaudeAgentOptions
    options = ClaudeAgentOptions(system_prompt=system_prompt, max_turns=1,
                                 allowed_tools=[])
    return query(prompt=prompt, options=options)


def stream_response(system_prompt, user_prompt, _query=None):
    """Streams Claude's text response as chunks. Blocks the calling thread.

    Granularity is one chunk per text block in v1 (no partial-message
    streaming); latency to the first chunk is what latency_check measures.
    """
    query_fn = _query or _sdk_query
    q = queue.Queue()
    worker = threading.Thread(target=_run, args=(query_fn, system_prompt,
                                                 user_prompt, q), daemon=True)
    worker.start()
    while True:
        item = q.get()
        if item is _DONE:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


def _run(query_fn, system_prompt, user_prompt, q):
    async def consume():
        async for message in query_fn(prompt=user_prompt,
                                      system_prompt=system_prompt):
            for block in getattr(message, "content", None) or []:
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    q.put(text)

    try:
        asyncio.run(consume())
    except BaseException as e:
        q.put(e)
    finally:
        q.put(_DONE)
