"""Small ready-made plugins."""

from __future__ import annotations

import fnmatch
import sys
import threading
from typing import Any, Callable, TextIO

from agentharness.core.events import LOG, SPAN_END, SPAN_START, STREAM, Event
from agentharness.plugins.base import Plugin


class ConsolePlugin(Plugin):
    """Pretty-prints the run to the terminal.

    With ``stream=True`` assistant text / reasoning deltas are printed live as they arrive
    (only when running via ``agent.stream(...)``).
    """

    name = "console"

    def __init__(self, stream: TextIO | None = None, max_chars: int = 300, live: bool = True):
        self.stream = stream or sys.stderr
        self.max_chars = max_chars
        self.live = live
        self._depth: dict[str, int] = {}
        self._streamed: set[str] = set()
        self._open: str | None = None  # span currently printing a live line
        self._lock = threading.Lock()

    def _clip(self, s: Any) -> str:
        s = str(s).replace("\n", " ")
        return s if len(s) <= self.max_chars else s[: self.max_chars] + "..."

    def _line(self, text: str) -> None:
        if self._open is not None:
            print(file=self.stream)
            self._open = None
        print(text, file=self.stream)

    def on_event(self, e: Event) -> None:
        with self._lock:
            self._handle(e)

    def _handle(self, e: Event) -> None:
        depth = self._depth.get(e.parent_id or "", -1) + 1
        pad = "  " * depth
        if e.type == STREAM:
            if not self.live or e.data.get("type") not in ("text", "reasoning"):
                return
            sid = e.span_id or ""
            if self._open != sid:
                if self._open is not None:
                    print(file=self.stream)
                print(f"{pad}💬 {e.name}: ", end="", file=self.stream)
                self._open = sid
            self._streamed.add(sid)
            print(e.data.get("delta", ""), end="", file=self.stream, flush=True)
        elif e.type == SPAN_START:
            self._depth[e.span_id or ""] = depth
            if e.kind in ("agent", "workflow"):
                self._line(f"{pad}▶ {e.kind} {e.name}: {self._clip(e.data.get('input'))}")
            elif e.kind == "tool":
                self._line(f"{pad}🔧 {e.name}({self._clip(e.data.get('input'))})")
        elif e.type == SPAN_END:
            self._depth.pop(e.span_id or "", None)
            d = e.data
            streamed = (e.span_id or "") in self._streamed
            self._streamed.discard(e.span_id or "")
            if d.get("error"):
                self._line(f"{pad}✖ {e.kind} {e.name}: {self._clip(d['error'])}")
            elif e.kind == "llm":
                out = d.get("output") or {}
                u = (d.get("attrs") or {}).get("usage") or {}
                text = "" if streamed else self._clip(out.get("content") or "")
                calls = ", ".join(c["name"] for c in out.get("tool_calls") or [])
                self._line(f"{pad}{'  ↳' if streamed else '💬'} {e.name} [{u.get('total_tokens', 0)} tok, {d['duration']:.2f}s] "
                           f"{text}{' → ' + calls if calls else ''}")
            elif e.kind in ("agent", "workflow"):
                self._line(f"{pad}◀ {e.name} done in {d['duration']:.2f}s")
        elif e.type == LOG:
            self._line(f"{pad}· {self._clip(e.name)}")


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
