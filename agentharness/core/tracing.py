"""Span-based tracing.

Spans nest through ``contextvars``, so a sub-agent invoked from inside a tool call
automatically becomes a child of that tool span, and the whole multi-agent run
shares one ``trace_id``. Thread pools must propagate context explicitly with
``contextvars.copy_context()`` (``Parallel`` does this).
"""

from __future__ import annotations

import dataclasses
import json
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from agentharness.core.events import LOG, SPAN_END, SPAN_START, Event
from agentharness.core.messages import new_id

if TYPE_CHECKING:
    from agentharness.core.runtime import Runtime


class Span:
    def __init__(self, kind: str, name: str, trace_id: str, parent_id: str | None):
        self.id = new_id("sp_")
        self.kind = kind
        self.name = name
        self.trace_id = trace_id
        self.parent_id = parent_id
        self.output: Any = None
        self.attrs: dict[str, Any] = {}
        self.error: str | None = None

    def set_output(self, output: Any) -> None:
        self.output = output

    def set(self, **attrs: Any) -> None:
        self.attrs.update(attrs)


_current: ContextVar[Span | None] = ContextVar("agentharness_span", default=None)


def current_span() -> Span | None:
    return _current.get()


def _default(o: Any) -> Any:
    if hasattr(o, "to_dict"):
        return o.to_dict()
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return dataclasses.asdict(o)
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    if isinstance(o, Path):
        return str(o)
    return repr(o)


def jsonable(obj: Any) -> Any:
    """Convert arbitrary data into JSON-safe structures (never raises)."""
    try:
        return json.loads(json.dumps(obj, default=_default, ensure_ascii=False))
    except (TypeError, ValueError):
        return repr(obj)


@contextmanager
def span(runtime: "Runtime", kind: str, name: str, input: Any = None, **attrs: Any) -> Iterator[Span]:
    """Open a span; emits span_start / span_end events through the runtime's plugins."""
    parent = _current.get()
    sp = Span(kind, name, parent.trace_id if parent else new_id("tr_"), parent.id if parent else None)
    sp.attrs.update(attrs)
    token = _current.set(sp)
    t0 = time.time()
    runtime.emit(Event(SPAN_START, sp.trace_id, sp.id, sp.parent_id, kind, name,
                       {"input": jsonable(input), "attrs": jsonable(attrs)}, ts=t0))
    try:
        yield sp
    except BaseException as e:
        sp.error = sp.error or f"{type(e).__name__}: {e}"
        raise
    finally:
        _current.reset(token)
        t1 = time.time()
        runtime.emit(Event(SPAN_END, sp.trace_id, sp.id, sp.parent_id, kind, name, {
            "output": jsonable(sp.output),
            "attrs": jsonable(sp.attrs),
            "error": sp.error,
            "duration": t1 - t0,
        }, ts=t1))


def log(runtime: "Runtime", message: str, **data: Any) -> None:
    """Attach a point-in-time log entry to the current span."""
    sp = _current.get()
    trace_id = sp.trace_id if sp else new_id("tr_")
    runtime.emit(Event(LOG, trace_id, sp.id if sp else None, sp.parent_id if sp else None, "log", message,
                       {"message": message, **jsonable(data)}))
