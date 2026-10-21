"""Per-run context passed to tools and plugin hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentharness.core.runtime import Runtime


@dataclass
class RunContext:
    runtime: "Runtime"
    agent_name: str = ""
    # Shared blackboard: forked contexts (sub-agents, workflow steps) share the same dict.
    # Parallel branches share it too, so guard concurrent writes yourself if needed.
    state: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    depth: int = 0
    stop_reason: str | None = None

    @property
    def workspace(self) -> Path:
        return self.runtime.workspace

    def request_stop(self, reason: str = "stopped") -> None:
        """Ask the agent loop to stop after the current step (usable from tools/plugins)."""
        self.stop_reason = reason

    def fork(self, agent_name: str) -> "RunContext":
        return RunContext(self.runtime, agent_name, self.state, dict(self.metadata), self.depth + 1)
