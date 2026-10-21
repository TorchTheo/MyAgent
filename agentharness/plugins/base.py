"""Plugin interface. Override only the hooks you need.

Hook semantics (called in plugin load order):

- ``tools()``                             tools registered into the runtime while the plugin is loaded
- ``on_load(runtime)`` / ``on_unload``    lifecycle
- ``on_event(event)``                     every span start/end and log event (observability sinks)
- ``before_run(ctx, agent, messages)``    may mutate ``messages`` in place
- ``after_run(ctx, agent, result)``
- ``before_llm(ctx, agent, request)``     mutate the ``LLMRequest`` or return a new one; replacing
                                          ``request.messages`` persists for the rest of the run
                                          (useful for context compaction / memory injection)
- ``after_llm(ctx, agent, response)``     return a new ``LLMResponse`` to override
- ``before_tool(ctx, agent, call)``       return a ``ToolCall`` to rewrite it, or a ``str`` to skip
                                          execution and use that string as the tool result
- ``after_tool(ctx, agent, call, result)`` return a ``str`` to replace the result
- ``on_error(ctx, agent, error)``         an exception escaped the agent run

Hook exceptions are isolated (reported as log events) unless ``strict = True``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentharness.core.context import RunContext
    from agentharness.core.events import Event
    from agentharness.core.messages import LLMResponse, Message, ToolCall
    from agentharness.core.runtime import Runtime
    from agentharness.llm.base import LLMRequest
    from agentharness.tools.base import Tool


class Plugin:
    name: str = ""
    strict: bool = False

    def __init_subclass__(cls, **kw: Any) -> None:
        super().__init_subclass__(**kw)
        if not cls.__dict__.get("name"):
            cls.name = cls.__name__

    def tools(self) -> list["Tool"]:
        return []

    def on_load(self, runtime: "Runtime") -> None: ...

    def on_unload(self, runtime: "Runtime") -> None: ...

    def on_event(self, event: "Event") -> None: ...

    def before_run(self, ctx: "RunContext", agent: Any, messages: list["Message"]) -> None: ...

    def after_run(self, ctx: "RunContext", agent: Any, result: Any) -> None: ...

    def before_llm(self, ctx: "RunContext", agent: Any, request: "LLMRequest") -> "LLMRequest | None": ...

    def after_llm(self, ctx: "RunContext", agent: Any, response: "LLMResponse") -> "LLMResponse | None": ...

    def before_tool(self, ctx: "RunContext", agent: Any, call: "ToolCall") -> "ToolCall | str | None": ...

    def after_tool(self, ctx: "RunContext", agent: Any, call: "ToolCall", result: str) -> "str | None": ...

    def on_error(self, ctx: "RunContext", agent: Any, error: BaseException) -> None: ...
