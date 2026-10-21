"""Offline demo (no API key): a supervisor delegates to researcher/writer sub-agents,
followed by a parallel review. Produces a rich trace for the viewer."""

from __future__ import annotations

import tempfile
from pathlib import Path

from agentharness import Agent, Parallel, Runtime, ScriptedLLM, Sequential, builtin_tools, tool

S = ScriptedLLM


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression.

    Args:
        expression: e.g. '(3 + 4) * 2'
    """
    import ast
    import operator as op

    ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg}

    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in ops:
            return ops[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in ops:
            return ops[type(n.op)](ev(n.operand))
        raise ValueError("unsupported expression")

    return str(ev(ast.parse(expression, mode="eval").body))


def run_demo(trace_dir: str | Path = "traces") -> str:
    ws = Path(tempfile.mkdtemp(prefix="agentharness-demo-"))
    (ws / "notes.md").write_text("# Notes\nHarness variance can exceed model variance on long-horizon tasks.\n", encoding="utf-8")
    rt = Runtime(ws, trace_dir=trace_dir)
    lat = 0.05

    researcher = Agent("researcher", S([
        S.tool_calls(("list_dir", {"path": "."}), ("grep", {"pattern": "variance"}), content="Let me look at the workspace."),
        S.tool_call("read_file", path="notes.md"),
        S.tool_call("calculator", expression="7.8 * 100"),
        S.text("Finding: the harness can matter more than the model (reported 7.8x variance ratio = 780%)."),
    ], model="demo-researcher", latency=lat), "You research facts using tools.",
        tools=[*builtin_tools("list_dir", "grep", "read_file"), calculator], runtime=rt,
        description="Researches a question using workspace files.")

    writer = Agent("writer", S([
        S.tool_call("write_file", path="summary.md", content="# Summary\nHarness design is a first-class research variable.\n"),
        S.tool_call("read_file", path="missing.md"),
        S.text("Wrote summary.md."),
    ], model="demo-writer", latency=lat), "You write concise reports.",
        tools=builtin_tools("write_file", "read_file"), runtime=rt, description="Writes reports to files.")

    supervisor = Agent("supervisor", S([
        S.tool_call("researcher", "I'll delegate research first.", task="Find evidence on harness vs model variance."),
        S.tool_call("writer", task="Write a one-paragraph summary of: harness matters more than model."),
        S.text("Done: research collected and summary.md written."),
    ], model="demo-supervisor", latency=lat), "You coordinate sub-agents.",
        tools=[researcher.as_tool(), writer.as_tool()], runtime=rt)

    reviewers = Parallel([
        Agent("critic", S([S.text("Clear, but cite the source paper.")], model="demo-critic", latency=lat), runtime=rt),
        Agent("fact_checker", S([S.tool_call("read_file", path="summary.md"), S.text("Facts consistent with notes.")],
                                model="demo-checker", latency=lat), tools=builtin_tools("read_file"), runtime=rt),
    ], aggregate=Agent("judge", S([S.text("APPROVED with minor revision: add citation.")], model="demo-judge", latency=lat), runtime=rt),
        name="review", runtime=rt)

    pipeline = Sequential([supervisor, reviewers], name="research_pipeline", runtime=rt)
    result = pipeline.run("Summarize why agent harness design matters for evaluation.")
    return f"output: {result.output}\ntrace: {Path(trace_dir).resolve() / (result.trace_id + '.jsonl')}\nview with: agentharness view --dir {trace_dir}"
