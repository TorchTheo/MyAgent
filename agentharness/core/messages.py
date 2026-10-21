"""Provider-neutral message types used across the harness."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:16]}"


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("call_"))

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ToolCall":
        return cls(name=d["name"], arguments=d.get("arguments") or {}, id=d.get("id") or new_id("call_"))


@dataclass
class Message:
    """A chat message. ``role`` is one of system / user / assistant / tool."""

    role: str
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    reasoning: str | None = None  # reasoning trace returned by some models; never sent back

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str | None = None, tool_calls: list[ToolCall] | None = None) -> "Message":
        return cls(role="assistant", content=content, tool_calls=list(tool_calls or []))

    @classmethod
    def tool(cls, content: str, tool_call_id: str, name: str | None = None) -> "Message":
        return cls(role="tool", content=content, tool_call_id=tool_call_id, name=name)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [c.to_dict() for c in self.tool_calls]
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.name:
            d["name"] = self.name
        if self.reasoning:
            d["reasoning"] = self.reasoning
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Message":
        return cls(
            role=d["role"],
            content=d.get("content"),
            tool_calls=[ToolCall.from_dict(c) for c in d.get("tool_calls") or []],
            tool_call_id=d.get("tool_call_id"),
            name=d.get("name"),
            reasoning=d.get("reasoning"),
        )


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(self.prompt_tokens + other.prompt_tokens, self.completion_tokens + other.completion_tokens)

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class LLMResponse:
    message: Message
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    finish_reason: str | None = None
    raw: Any = None
