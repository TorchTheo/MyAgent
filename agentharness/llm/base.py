"""LLM backend interface.

Tool schemas use the OpenAI function-calling format
(``{"type": "function", "function": {name, description, parameters}}``);
backends for other APIs convert from it.

Backends implement ``complete``; streaming backends also override ``stream``.
The default ``stream`` calls ``complete`` and replays the result as deltas, so every
backend works in streaming mode.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator

from agentharness.core.messages import LLMResponse, Message


class LLMError(RuntimeError):
    pass


@dataclass
class LLMRequest:
    """What the agent is about to send. Plugins may rewrite it in ``before_llm``."""

    messages: list[Message]
    tools: list[dict[str, Any]] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamDelta:
    """One streamed piece of a response.

    ``type`` is ``text`` | ``reasoning`` | ``tool_call`` (an arguments fragment of tool call
    ``index``) | ``done`` (last item; ``response`` holds the complete ``LLMResponse``).
    """

    type: str
    delta: str = ""
    index: int = 0
    id: str | None = None
    name: str | None = None
    response: LLMResponse | None = None


def split_response(resp: LLMResponse, chunk_size: int | None = None, delay: float = 0.0) -> Iterator[StreamDelta]:
    """Replay a complete response as deltas (optionally chunked, with a per-chunk delay)."""

    def pieces(s: str) -> Iterator[str]:
        if not chunk_size:
            yield s
            return
        for i in range(0, len(s), chunk_size):
            if delay:
                time.sleep(delay)
            yield s[i:i + chunk_size]

    m = resp.message
    for kind, text in (("reasoning", m.reasoning), ("text", m.content)):
        if text:
            for p in pieces(text):
                yield StreamDelta(kind, p)
    for i, c in enumerate(m.tool_calls):
        for p in pieces(json.dumps(c.arguments, ensure_ascii=False)):
            yield StreamDelta("tool_call", p, index=i, id=c.id, name=c.name)
    yield StreamDelta("done", response=resp)


class LLM(ABC):
    model: str = "unknown"

    @abstractmethod
    def complete(self, messages: list[Message], tools: list[dict[str, Any]] | None = None, **params: Any) -> LLMResponse:
        """Return the next assistant message (text and/or tool calls)."""

    def stream(self, messages: list[Message], tools: list[dict[str, Any]] | None = None, **params: Any) -> Iterator[StreamDelta]:
        """Yield ``StreamDelta``s; the last one has ``type == "done"`` and the full response."""
        yield from split_response(self.complete(messages, tools, **params))
