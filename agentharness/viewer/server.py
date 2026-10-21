"""Trace viewer HTTP server (standard library only).

Read-only and unauthenticated: it binds to 127.0.0.1 by default. Trace files contain
full prompts and tool outputs, so don't expose it on a public interface.
"""

from __future__ import annotations

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_STATIC = Path(__file__).parent / "static"
_TRACE_ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")


def read_events(path: Path) -> list[dict[str, Any]]:
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # partially written last line of a running trace
    return events


def summarize(path: Path) -> dict[str, Any] | None:
    events = read_events(path)
    if not events:
        return None
    starts = {e["span_id"]: e for e in events if e["type"] == "span_start"}
    ends = {e["span_id"]: e for e in events if e["type"] == "span_end"}
    roots = [e for e in starts.values() if not e.get("parent_id")]
    root = roots[0] if roots else next(iter(starts.values()), events[0])
    root_end = ends.get(root.get("span_id"))
    tokens = 0
    for sid, e in ends.items():
        if e.get("kind") == "llm":
            tokens += ((e["data"].get("attrs") or {}).get("usage") or {}).get("total_tokens", 0)
    errors = sum(1 for e in ends.values() if e["data"].get("error"))
    inp = (root.get("data") or {}).get("input")
    return {
        "id": path.stem,
        "name": root.get("name"),
        "kind": root.get("kind"),
        "input": inp if isinstance(inp, str) else json.dumps(inp, ensure_ascii=False)[:300],
        "start": events[0]["ts"],
        "duration": root_end["data"].get("duration") if root_end else None,
        "running": root_end is None,
        "error": (root_end or {}).get("data", {}).get("error"),
        "errors": errors,
        "tokens": tokens,
        "llm_calls": sum(1 for e in starts.values() if e.get("kind") == "llm"),
        "tool_calls": sum(1 for e in starts.values() if e.get("kind") == "tool"),
        "agents": sorted({e["name"] for e in starts.values() if e.get("kind") == "agent"}),
    }


def make_handler(trace_dir: Path) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj: Any, code: int = 200) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                self._send(200, (_STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/traces":
                items = []
                if trace_dir.is_dir():
                    for p in trace_dir.glob("*.jsonl"):
                        s = summarize(p)
                        if s:
                            items.append(s)
                items.sort(key=lambda s: s["start"], reverse=True)
                self._json(items)
            elif path.startswith("/api/traces/"):
                tid = path.rsplit("/", 1)[-1]
                p = trace_dir / f"{tid}.jsonl"
                if not _TRACE_ID.match(tid) or not p.is_file():
                    self._json({"error": "not found"}, 404)
                else:
                    self._json(read_events(p))
            else:
                self._json({"error": "not found"}, 404)

    return Handler


def make_server(trace_dir: str | Path = "traces", host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(Path(trace_dir).resolve()))


def serve(trace_dir: str | Path = "traces", host: str = "127.0.0.1", port: int = 8765) -> None:
    srv = make_server(trace_dir, host, port)
    print(f"agentharness trace viewer: http://{host}:{srv.server_address[1]}  (traces: {Path(trace_dir).resolve()})")
    if host not in ("127.0.0.1", "localhost", "::1"):
        print("warning: the viewer has no authentication and traces contain full prompts/tool outputs")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
