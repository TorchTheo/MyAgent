"""Thread-safe tool registry with ownership (so a plugin's tools can be removed with it)."""

from __future__ import annotations

import threading

from agentharness.tools.base import Tool


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, tuple[Tool, str | None]] = {}
        self._lock = threading.RLock()
        for t in tools or []:
            self.register(t)

    def register(self, tool: Tool, owner: str | None = None, replace: bool = False) -> Tool:
        if not isinstance(tool, Tool):
            raise TypeError(f"expected Tool, got {type(tool).__name__} (decorate the function with @tool)")
        with self._lock:
            if tool.name in self._tools and not replace:
                raise ValueError(f"tool {tool.name!r} is already registered (owner={self._tools[tool.name][1]!r})")
            self._tools[tool.name] = (tool, owner)
        return tool

    def unregister(self, name: str) -> Tool | None:
        with self._lock:
            entry = self._tools.pop(name, None)
        return entry[0] if entry else None

    def unregister_owner(self, owner: str) -> list[str]:
        with self._lock:
            names = [n for n, (_, o) in self._tools.items() if o == owner]
            for n in names:
                del self._tools[n]
        return names

    def get(self, name: str) -> Tool | None:
        with self._lock:
            entry = self._tools.get(name)
        return entry[0] if entry else None

    def list(self) -> list[Tool]:
        with self._lock:
            return [t for t, _ in self._tools.values()]

    def names(self) -> list[str]:
        with self._lock:
            return list(self._tools)

    def __contains__(self, name: str) -> bool:
        with self._lock:
            return name in self._tools

    def __len__(self) -> int:
        with self._lock:
            return len(self._tools)
