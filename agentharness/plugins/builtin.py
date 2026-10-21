"""Small ready-made plugins."""

from __future__ import annotations

import fnmatch
import sys
from typing import Any, Callable, TextIO

from agentharness.core.events import LOG, SPAN_END, SPAN_START, Event
from agentharness.plugins.base import Plugin


class ConsolePlugin(Plugin):
    """Pretty-prints the run to the terminal."""

    name = "console"

    def __init__(self, stream: TextIO | None = None, max_chars: int = 300):
        self.stream = stream or sys.stderr
        self.max_chars = max_chars
        self._depth: dict[str, int] = {}

    def _clip(self, s: Any) -> str:
        s = str(s).replace("\n", " ")
        return s if len(s) <= self.max_chars else s[: self.max_chars] + "..."

    def on_event(self, e: Event) -> None:
        depth = self._depth.get(e.parent_id or "", -1) + 1
        pad = "  " * depth
        if e.type == SPAN_START:
            self._depth[e.span_id or ""] = depth
            if e.kind in ("agent", "workflow"):
                print(f"{pad}▶ {e.kind} {e.name}: {self._clip(e.data.get('input'))}", file=self.stream)
            elif e.kind == "tool":
                print(f"{pad}🔧 {e.name}({self._clip(e.data.get('input'))})", file=self.stream)
        elif e.type == SPAN_END:
            self._depth.pop(e.span_id or "", None)
            d = e.data
            if d.get("error"):
                print(f"{pad}✖ {e.kind} {e.name}: {self._clip(d['error'])}", file=self.stream)
            elif e.kind == "llm":
                out = d.get("output") or {}
                u = (d.get("attrs") or {}).get("usage") or {}
                text = out.get("content") or ""
                calls = ", ".join(c["name"] for c in out.get("tool_calls") or [])
                print(f"{pad}💬 {e.name} [{u.get('total_tokens', 0)} tok, {d['duration']:.2f}s] "
                      f"{self._clip(text)}{' → ' + calls if calls else ''}", file=self.stream)
            elif e.kind in ("agent", "workflow"):
                print(f"{pad}◀ {e.name} done in {d['duration']:.2f}s", file=self.stream)
        elif e.type == LOG:
            print(f"{pad}· {self._clip(e.name)}", file=self.stream)


class ToolPolicyPlugin(Plugin):
    """Allow/deny tool calls by glob pattern, or via a custom ``approve(ctx, agent, call) -> bool``.

    Denied calls are not executed; the model receives an explanatory error instead.
    """

    name = "tool_policy"

    def __init__(self, allow: list[str] | None = None, deny: list[str] | None = None,
                 approve: Callable[[Any, Any, Any], bool] | None = None):
        self.allow = allow
        self.deny = deny or []
        self.approve = approve

    def before_tool(self, ctx, agent, call):
        if any(fnmatch.fnmatch(call.name, p) for p in self.deny) or (
            self.allow is not None and not any(fnmatch.fnmatch(call.name, p) for p in self.allow)
        ):
            return f"Error: tool {call.name!r} is not permitted by policy"
        if self.approve and not self.approve(ctx, agent, call):
            return f"Error: tool call {call.name!r} was rejected"
        return None
