"""Observation events. Everything observable in the harness is a span start/end or a log."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

SPAN_START = "span_start"
SPAN_END = "span_end"
LOG = "log"


@dataclass
class Event:
    type: str
    trace_id: str
    span_id: str | None
    parent_id: str | None
    kind: str | None = None  # agent | llm | tool | workflow | step | custom
    name: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "kind": self.kind,
            "name": self.name,
            "data": self.data,
            "ts": self.ts,
        }
