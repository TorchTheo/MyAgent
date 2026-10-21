"""LLM backend interface.

Tool schemas use the OpenAI function-calling format
(``{"type": "function", "function": {name, description, parameters}}``);
backends for other APIs convert from it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from agentharness.core.messages import LLMResponse, Message


class LLMError(RuntimeError):
    pass


@dataclass
class LLMRequest:
    """What the agent is about to send. Plugins may rewrite it in ``before_llm``."""

    messages: list[Message]
    tools: list[dict[str, Any]] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)


class LLM(ABC):
    model: str = "unknown"

    @abstractmethod
    def complete(self, messages: list[Message], tools: list[dict[str, Any]] | None = None, **params: Any) -> LLMResponse:
        """Return the next assistant message (text and/or tool calls)."""
