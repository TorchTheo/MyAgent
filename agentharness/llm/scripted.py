"""Deterministic scripted LLM for tests, demos and offline debugging of agent logic."""

from __future__ import annotations

import copy
import threading
import time
from typing import Any, Callable, Union

from agentharness.core.messages import LLMResponse, Message, ToolCall, Usage
from agentharness.llm.base import LLM, LLMError

Step = Union[Message, Callable[[list[Message], list[dict[str, Any]]], Message]]


class ScriptedLLM(LLM):
    """Returns pre-scripted assistant messages in order.

    Each step is a ``Message`` or a callable ``(messages, tools) -> Message``.
    """

    def __init__(self, steps: list[Step], model: str = "scripted", latency: float = 0.0):
        self.model = model
        self.latency = latency
        self._steps = list(steps)
        self._lock = threading.Lock()
        self.requests: list[list[Message]] = []

    @staticmethod
    def text(content: str) -> Message:
        return Message.assistant(content)

    @staticmethod
    def tool_call(name: str, text: str | None = None, /, **arguments: Any) -> Message:
        """``tool_call("write_file", path=..., content=...)``; optional assistant text goes second, positionally."""
        return Message.assistant(text, [ToolCall(name, arguments)])

    @staticmethod
    def tool_calls(*calls: tuple[str, dict[str, Any]], content: str | None = None) -> Message:
        return Message.assistant(content, [ToolCall(n, a) for n, a in calls])

    def complete(self, messages: list[Message], tools: list[dict[str, Any]] | None = None, **params: Any) -> LLMResponse:
        with self._lock:
            if not self._steps:
                raise LLMError("ScriptedLLM: script exhausted")
            step = self._steps.pop(0)
            self.requests.append(list(messages))
        if self.latency:
            time.sleep(self.latency)
        msg = step(messages, tools or []) if callable(step) else copy.deepcopy(step)
        prompt = sum(len(m.content or "") for m in messages) // 4
        completion = len(msg.content or "") // 4 + 8 * len(msg.tool_calls)
        return LLMResponse(message=msg, usage=Usage(prompt, completion), model=self.model,
                           finish_reason="tool_calls" if msg.tool_calls else "stop")
