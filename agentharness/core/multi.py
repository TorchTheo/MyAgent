"""Multi-agent composition.

Three building blocks cover the common topologies and nest arbitrarily:

- ``as_tool(agent)``          supervisor / orchestrator-worker: a sub-agent becomes a tool
- ``Sequential([a, b, c])``   pipeline: each step's output is the next step's input
- ``Parallel([a, b], ...)``   fan-out on the same input, then optional aggregation

All runs share one trace, so the viewer shows the full agent tree.
"""

from __future__ import annotations

import contextvars
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Sequence

from agentharness.core.agent import Runnable, RunResult
from agentharness.core.context import RunContext
from agentharness.core.messages import Usage
from agentharness.core.runtime import Runtime
from agentharness.core.tracing import span
from agentharness.tools.base import Tool

Step = Runnable | Callable[[Any, RunContext], Any]


def _run_step(step: Step, input: Any, ctx: RunContext) -> RunResult:
    if isinstance(step, Runnable):
        return step.run(input, ctx)
    name = getattr(step, "__name__", "step")
    with span(ctx.runtime, "step", name, input=input) as sp:
        out = step(input, ctx)
        sp.set_output(out)
    return out if isinstance(out, RunResult) else RunResult(out, agent=name)


class Sequential(Runnable):
    def __init__(self, steps: Sequence[Step], name: str = "sequential", description: str = "",
                 runtime: Runtime | None = None):
        self.steps = list(steps)
        self.name = name
        self.description = description or f"Pipeline: {' -> '.join(getattr(s, 'name', getattr(s, '__name__', 'step')) for s in steps)}"
        self.runtime = runtime

    def run(self, input: Any, ctx: RunContext | None = None) -> RunResult:
        ctx = self._context(ctx)
        with span(ctx.runtime, "workflow", self.name, input=input, type="sequential") as sp:
            usage, value, results = Usage(), input, []
            for step in self.steps:
                r = _run_step(step, value, ctx)
                results.append(r)
                usage = usage + r.usage
                value = r.output
                if r.stop_reason not in ("final", "max_steps"):
                    break  # a step explicitly requested a stop
            sp.set_output(value)
            sp.set(usage=usage.to_dict())
        return RunResult(value, usage=usage, steps=len(results), trace_id=sp.trace_id, agent=self.name,
                         stop_reason=results[-1].stop_reason if results else "final")


class Parallel(Runnable):
    """Run branches concurrently on the same input.

    ``aggregate`` may be a Runnable (e.g. a judge agent receiving a formatted summary of
    all branch outputs) or a function ``(dict[name, output], ctx) -> Any``. Without it the
    output is ``{branch_name: output}``.
    """

    def __init__(self, branches: Sequence[Runnable], aggregate: Runnable | Callable[[dict[str, Any], RunContext], Any] | None = None,
                 name: str = "parallel", description: str = "", max_workers: int = 8, runtime: Runtime | None = None):
        names = [b.name for b in branches]
        if len(set(names)) != len(names):
            raise ValueError(f"parallel branch names must be unique: {names}")
        self.branches = list(branches)
        self.aggregate = aggregate
        self.name = name
        self.description = description or f"Parallel: {', '.join(names)}"
        self.max_workers = max_workers
        self.runtime = runtime

    def run(self, input: Any, ctx: RunContext | None = None) -> RunResult:
        ctx = self._context(ctx)
        with span(ctx.runtime, "workflow", self.name, input=input, type="parallel") as sp:
            with ThreadPoolExecutor(max_workers=max(1, min(self.max_workers, len(self.branches)))) as pool:
                futures = {b.name: pool.submit(contextvars.copy_context().run, b.run, input, ctx) for b in self.branches}
                results = {n: f.result() for n, f in futures.items()}
            usage = Usage()
            for r in results.values():
                usage = usage + r.usage
            outputs = {n: r.output for n, r in results.items()}
            if self.aggregate is None:
                value: Any = outputs
            elif isinstance(self.aggregate, Runnable):
                summary = f"Original task:\n{input}\n\n" + "\n\n".join(f"## Result from {n}\n{o}" for n, o in outputs.items())
                agg = self.aggregate.run(summary, ctx)
                usage = usage + agg.usage
                value = agg.output
            else:
                value = _run_step(self.aggregate, outputs, ctx).output
            sp.set_output(value)
            sp.set(usage=usage.to_dict())
        return RunResult(value, usage=usage, steps=len(results), trace_id=sp.trace_id, agent=self.name)


def as_tool(runnable: Runnable, name: str | None = None, description: str | None = None) -> Tool:
    def delegate(ctx: RunContext, task: str) -> str:
        r = runnable.run(task, ctx)
        return r.output if isinstance(r.output, str) else str(r.output)

    params = {"type": "object", "properties": {
        "task": {"type": "string", "description": "A complete, self-contained task description for the sub-agent."}},
        "required": ["task"]}
    t = Tool(delegate, name=name or runnable.name, description=description or runnable.description,
             parameters=params, metadata={"sub_agent": runnable.name})
    t.ctx_param = "ctx"
    return t
