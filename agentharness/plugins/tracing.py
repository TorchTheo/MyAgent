"""Persist every event as JSON Lines: ``<trace_dir>/<trace_id>.jsonl``.

The trace viewer (``agentharness view``) reads these files; they are also easy to
load with pandas for offline analysis of trajectories.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from agentharness.core.events import LOG, Event
from agentharness.plugins.base import Plugin


class TracingPlugin(Plugin):
    name = "tracing"

    def __init__(self, trace_dir: str | Path = "traces"):
        self.trace_dir = Path(trace_dir)
        self._lock = threading.Lock()

    def on_event(self, event: Event) -> None:
        if event.type == LOG and event.span_id is None:
            return  # logs outside any run (e.g. plugin watcher) have no trace to attach to
        line = json.dumps(event.to_dict(), ensure_ascii=False, default=repr)
        with self._lock:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            with open(self.trace_dir / f"{event.trace_id}.jsonl", "a", encoding="utf-8") as f:
                f.write(line + "\n")
