"""The agent loop.

Inside ``agent.stream(...)`` the LLM is called through ``llm.stream`` and every delta is
emitted as a ``stream`` event; otherwise ``llm.complete`` is used. Either way the final
``LLMResponse`` (and therefore the trace) is identical.

    messages = [system, *history, user]
    repeat up to max_steps:
        request = before_llm(messages, tools)        # plugins may rewrite
        response = llm.complete(request)             # traced as an `llm` span
        response = after_llm(response)
        if no tool calls: return final answer
        for each tool call (concurrently if allowed):
            call = before_tool(call)                 # plugins may rewrite / block
            result = tool.run(...)                   # traced as a `tool` span; errors go back to the model
            result = after_tool(result)
        append tool results; stop early if ctx.request_stop() was called
"""

from __future__ import annotations

import contextvars
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from agentharness.core.context import RunContext
from agentharness.core.messages import LLMResponse, Message, ToolCall, Usage
from agentharness.core.runtime import Runtime, default_runtime
from agentharness.core.streaming import RunCancelled, RunStream, current_stream
from agentharness.core.tracing import emit_stream, span
from agentharness.llm.base import LLM, LLMError, LLMRequest
from agentharness.tools.base import Tool, ToolError

Instructions = str | Callable[[RunContext], str]


@dataclass
class RunResult:
    output: Any
    messages: list[Message] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    steps: int = 0
    stop_reason: str = "final"  # final | max_steps | <custom reason from ctx.request_stop>
    trace_id: str | None = None
    agent: str = ""

    def __str__(self) -> str:
        return str(self.output)


class Runnable:
    """Anything with ``run(input, ctx=None) -> RunResult``: agents and workflows compose freely."""

    name: str = "runnable"
    description: str = ""
    runtime: Runtime | None = None

    def run(self, input: Any, ctx: RunContext | None = None) -> RunResult:  # pragma: no cover
        raise NotImplementedError

    def _context(self, ctx: RunContext | None) -> RunContext:
        if ctx is not None:
            return ctx.fork(self.name)
        return RunContext(self.runtime or default_runtime(), self.name)

    def stream(self, input: Any, ctx: RunContext | None = None) -> "RunStream":
        """Run in a background thread, streaming span/log/token events. See ``agentharness.core.streaming``."""
        rt = ctx.runtime if ctx is not None else (self.runtime or default_runtime())
        return RunStream(self, input, ctx, rt)

    def as_tool(self, name: str | None = None, description: str | None = None) -> Tool:
        """Expose this agent/workflow as a tool for another agent (supervisor / sub-agent pattern)."""
        from agentharness.core.multi import as_tool

        return as_tool(self, name, description)


class Agent(Runnable):
    def __init__(
        self,
        name: str,
        llm: LLM,
        instructions: Instructions = "You are a helpful assistant.",
        tools: Sequence[Tool] = (),
        *,
        description: str = "",
        runtime: Runtime | None = None,
        max_steps: int = 30,
        parallel_tool_calls: bool = True,
        max_tool_output_chars: int = 50_000,
        use_runtime_tools: bool = True,
        llm_params: dict[str, Any] | None = None,
    ):
        self.name = name
        self.llm = llm
        self.instructions = instructions
        self.tools: dict[str, Tool] = {}
        for t in tools:
            self.add_tool(t)
        self.description = description or f"Agent {name}"
        self.runtime = runtime
        self.max_steps = max_steps
        self.parallel_tool_calls = parallel_tool_calls
        self.max_tool_output_chars = max_tool_output_chars
        self.use_runtime_tools = use_runtime_tools
        self.llm_params = dict(llm_params or {})

    # ---- tool management ---------------------------------------------
    def add_tool(self, t: Tool) -> Tool:
        if not isinstance(t, Tool):
            raise TypeError(f"expected Tool, got {type(t).__name__} (decorate the function with @tool)")
        self.tools[t.name] = t
        return t

    def remove_tool(self, name: str) -> None:
        self.tools.pop(name, None)

    def resolve_tools(self, ctx: RunContext) -> dict[str, Tool]:
        """Agent tools override runtime (plugin) tools of the same name. Resolved every step."""
        resolved = {t.name: t for t in ctx.runtime.tools.list()} if self.use_runtime_tools else {}
        resolved.update(self.tools)
        return resolved

    def system_prompt(self, ctx: RunContext) -> str:
        return self.instructions(ctx) if callable(self.instructions) else self.instructions

    # ---- loop ----------------------------------------------------------
    def run(self, input: str | Message | list[Message], ctx: RunContext | None = None) -> RunResult:
        ctx = self._context(ctx)
        rt = ctx.runtime
        messages = self._initial_messages(ctx, input)
        with span(rt, "agent", self.name, input=_preview_input(input),
                  model=getattr(self.llm, "model", "?"), depth=ctx.depth) as sp:
            try:
                rt.plugins.call("before_run", ctx, self, messages)
                result = self._loop(ctx, messages)
                result.trace_id = sp.trace_id
                sp.set_output(result.output)
                sp.set(steps=result.steps, stop_reason=result.stop_reason, usage=result.usage.to_dict())
                rt.plugins.call("after_run", ctx, self, result)
                return result
            except BaseException as e:
                rt.plugins.call("on_error", ctx, self, e)
                raise

    def _initial_messages(self, ctx: RunContext, input: str | Message | list[Message]) -> list[Message]:
        if isinstance(input, str):
            history = [Message.user(input)]
        elif isinstance(input, Message):
            history = [input]
        else:
            history = list(input)
        if history and history[0].role == "system":
            return history
        prompt = self.system_prompt(ctx)
        return ([Message.system(prompt)] if prompt else []) + history

    def _loop(self, ctx: RunContext, messages: list[Message]) -> RunResult:
        rt = ctx.runtime
        usage = Usage()
        last_text: str | None = None
        st = current_stream()
        for step in range(1, self.max_steps + 1):
            if st is not None and st.cancelled.is_set():
                raise RunCancelled("stream cancelled")
            tools = self.resolve_tools(ctx)
            request = LLMRequest(messages=messages, tools=[t.schema() for t in tools.values()],
                                 params=dict(self.llm_params))
            for _, r in rt.plugins.call("before_llm", ctx, self, request):
                if isinstance(r, LLMRequest):
                    request = r
            messages = request.messages  # plugins may compact / rewrite history

            response = self._call_llm(ctx, request, step)
            usage = usage + response.usage
            msg = response.message
            msg.role = "assistant"
            messages.append(msg)
            if msg.content:
                last_text = msg.content

            if not msg.tool_calls:
                return RunResult(msg.content or "", messages, usage, step, "final", agent=self.name)

            for m in self._execute_tools(ctx, msg.tool_calls, tools):
                messages.append(m)
            if ctx.stop_reason:
                return RunResult(last_text or "", messages, usage, step, ctx.stop_reason, agent=self.name)
        return RunResult(last_text or "", messages, usage, self.max_steps, "max_steps", agent=self.name)

    def _call_llm(self, ctx: RunContext, request: LLMRequest, step: int) -> LLMResponse:
        rt = ctx.runtime
        model = getattr(self.llm, "model", "?")
        st = current_stream()
        with span(rt, "llm", model, input={
            "messages": [m.to_dict() for m in request.messages],
            "tools": [t["function"]["name"] for t in request.tools],
            "params": request.params,
        }, step=step, agent=self.name, tool_schemas=request.tools, stream=st is not None) as sp:
            if st is None:
                response = self.llm.complete(list(request.messages), request.tools or None, **request.params)
            else:
                response = self._consume_stream(ctx, request, sp, st, step)
            for _, r in rt.plugins.call("after_llm", ctx, self, response):
                if isinstance(r, LLMResponse):
                    response = r
            sp.set_output(response.message.to_dict())
            sp.set(usage=response.usage.to_dict(), finish_reason=response.finish_reason,
                   model=response.model or model)
        return response

    def _consume_stream(self, ctx: RunContext, request: LLMRequest, sp: Any, st: Any, step: int) -> LLMResponse:
        rt = ctx.runtime
        response: LLMResponse | None = None
        it = self.llm.stream(list(request.messages), request.tools or None, **request.params)
        try:
            for d in it:
                if st.cancelled.is_set():
                    raise RunCancelled("stream cancelled")
                if d.type == "done":
                    response = d.response
                    break
                data: dict[str, Any] = {"type": d.type, "delta": d.delta, "agent": self.name, "step": step}
                if d.type == "tool_call":
                    data.update(index=d.index, id=d.id, name=d.name)
                emit_stream(rt, sp, data)
        finally:
            close = getattr(it, "close", None)
            if close:
                close()  # release the HTTP connection when stopping early
        if response is None:
            raise LLMError(f"{type(self.llm).__name__}.stream ended without a 'done' delta")
        return response

    def _execute_tools(self, ctx: RunContext, calls: list[ToolCall], tools: dict[str, Tool]) -> list[Message]:
        if self.parallel_tool_calls and len(calls) > 1:
            with ThreadPoolExecutor(max_workers=min(8, len(calls))) as pool:
                futures = [pool.submit(contextvars.copy_context().run, self._run_tool, ctx, c, tools) for c in calls]
                return [f.result() for f in futures]
        return [self._run_tool(ctx, c, tools) for c in calls]

    def _run_tool(self, ctx: RunContext, call: ToolCall, tools: dict[str, Tool]) -> Message:
        rt = ctx.runtime
        with span(rt, "tool", call.name, input=call.arguments, call_id=call.id, agent=self.name) as sp:
            original_id = call.id
            result: str | None = None
            for _, r in rt.plugins.call("before_tool", ctx, self, call):
                if isinstance(r, ToolCall):
                    call = r
                elif isinstance(r, str):
                    result = r
                    sp.set(intercepted=True)
                    break
            if result is None:
                t = tools.get(call.name)
                if t is None:
                    result = f"Error: unknown tool {call.name!r}. Available tools: {', '.join(tools) or 'none'}"
                    sp.error = "unknown tool"
                else:
                    try:
                        result = t.run(ctx, call.arguments)
                    except ToolError as e:
                        result = f"Error: {e}"
                        sp.error = str(e)
                    except Exception as e:  # surface to the model, keep the loop alive
                        result = f"Error: {type(e).__name__}: {e}"
                        sp.error = result
            for _, r in rt.plugins.call("after_tool", ctx, self, call, result):
                if isinstance(r, str):
                    result = r
            if len(result) > self.max_tool_output_chars:
                result = result[: self.max_tool_output_chars] + f"\n... (truncated, {len(result)} chars total)"
            sp.set_output(result)
        return Message.tool(result, original_id, call.name)


def _preview_input(input: Any) -> Any:
    if isinstance(input, Message):
        return input.to_dict()
    if isinstance(input, list):
        return [m.to_dict() if isinstance(m, Message) else m for m in input]
    return input
