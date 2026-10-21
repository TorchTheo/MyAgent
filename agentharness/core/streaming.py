"""Streaming: run any agent/workflow in the background and consume its events live.

    s = agent.stream("task")
    for delta in s.text():           # or: for event in s: ...
        print(delta, end="", flush=True)
    result = s.result                # RunResult (re-raises the run's exception)

Inside a stream every agent in the tree (sub-agents, Parallel branches...) calls
``llm.stream`` and emits ``stream`` events with token deltas. Leaving the iteration
early (``break``) or calling ``cancel()`` stops the run at the next delta / step.
"""

from __future__ import annotations

import contextvars
import queue
import threading
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Iterator

from agentharness.core.events import SPAN_START, STREAM, Event

if TYPE_CHECKING:
    from agentharness.core.agent import Runnable, RunResult
    from agentharness.core.context import RunContext
    from agentharness.core.runtime import Runtime


class RunCancelled(Exception):
    """Raised inside the loop when a streamed run is cancelled."""


class StreamState:
    def __init__(self) -> None:
        self.cancelled = threading.Event()


_stream_state: ContextVar[StreamState | None] = ContextVar("agentharness_stream", default=None)


def current_stream() -> StreamState | None:
    return _stream_state.get()


_DONE = object()


class RunStream:
    """Live handle on a background run. Iterate once; read ``result`` afterwards."""

    def __init__(self, runnable: "Runnable", input: Any, ctx: "RunContext | None", runtime: "Runtime"):
        from agentharness.core.tracing import current_span

        self.state = StreamState()
        self._runtime = runtime
        self._q: queue.Queue[Any] = queue.Queue()
        self._spans: set[str] = set()
        outer = current_span()
        self._outer_id = outer.id if outer else None
        self._result: "RunResult | None" = None
        self._error: BaseException | None = None
        self._done = threading.Event()
        self._consumed = False

        def work() -> None:
            _stream_state.set(self.state)
            try:
                self._result = runnable.run(input, ctx)
            except BaseException as e:  # surfaced to the consumer
                self._error = e
            finally:
                runtime.remove_listener(self._on_event)
                self._done.set()
                self._q.put(_DONE)

        self._thread = threading.Thread(target=contextvars.copy_context().run, args=(work,),
                                        name=f"stream:{getattr(runnable, 'name', 'run')}", daemon=True)
        runtime.add_listener(self._on_event)
        self._thread.start()

    def _on_event(self, e: Event) -> None:
        # Keep only this run's span subtree (other runs may share the runtime concurrently).
        if e.type == SPAN_START:
            is_root = threading.current_thread() is self._thread and e.parent_id == self._outer_id
            if e.span_id and (is_root or e.parent_id in self._spans):
                self._spans.add(e.span_id)
                self._q.put(e)
        elif e.span_id in self._spans:
            self._q.put(e)

    # ---- consumption -------------------------------------------------
    def __iter__(self) -> Iterator[Event]:
        if self._consumed:
            raise RuntimeError("a RunStream can only be iterated once")
        self._consumed = True
        try:
            while True:
                item = self._q.get()
                if item is _DONE:
                    break
                yield item
        finally:
            if not self._done.is_set():
                self.cancel()  # consumer stopped early
        if self._error is not None:
            raise self._error

    def text(self, agent: str | None = None) -> Iterator[str]:
        """Yield only assistant text deltas (optionally from one agent)."""
        for e in self:
            if e.type == STREAM and e.data.get("type") == "text" and (agent is None or e.data.get("agent") == agent):
                yield e.data.get("delta", "")

    def cancel(self) -> None:
        self.state.cancelled.set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._done.wait(timeout)

    @property
    def done(self) -> bool:
        return self._done.is_set()

    @property
    def result(self) -> "RunResult":
        self._thread.join()
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result

    def __enter__(self) -> "RunStream":
        return self

    def __exit__(self, *exc: Any) -> None:
        if not self._done.is_set():
            self.cancel()
        self._thread.join()
