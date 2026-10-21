"""Runtime: the shared environment that agents, tools and plugins live in."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Iterable

from agentharness.core.events import Event
from agentharness.plugins.base import Plugin
from agentharness.plugins.manager import PluginManager
from agentharness.plugins.tracing import TracingPlugin
from agentharness.tools.registry import ToolRegistry


class Runtime:
    """Holds the workspace, the plugin manager and the global (plugin-provided) tool registry.

    Agents resolve their tool list on every step, so tools registered here (directly
    or by a plugin) become visible immediately: this is what makes hot-plugging work.
    """

    def __init__(
        self,
        workspace: str | Path = ".",
        *,
        trace_dir: str | Path = "traces",
        tracing: bool = True,
        allow_outside_workspace: bool = False,
        plugins: Iterable[Plugin] = (),
    ):
        self.workspace = Path(workspace).resolve()
        self.allow_outside_workspace = allow_outside_workspace
        self.trace_dir = Path(trace_dir).resolve()
        self.tools = ToolRegistry()
        self.plugins = PluginManager(self)
        if tracing:
            self.plugins.load(TracingPlugin(self.trace_dir))
        for p in plugins:
            self.plugins.load(p)

    def emit(self, event: Event) -> None:
        self.plugins.emit(event)


_default: Runtime | None = None
_default_lock = threading.Lock()


def default_runtime() -> Runtime:
    """Lazily created process-wide runtime (cwd workspace, ./traces)."""
    global _default
    with _default_lock:
        if _default is None:
            _default = Runtime()
        return _default
