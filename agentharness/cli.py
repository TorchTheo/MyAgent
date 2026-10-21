"""Command line: ``agentharness view`` (trace viewer) and ``agentharness demo`` (offline demo)."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="agentharness")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("view", help="serve the trace viewer")
    v.add_argument("--dir", default="traces", help="trace directory (default: ./traces)")
    v.add_argument("--host", default="127.0.0.1", help="bind address (the viewer has no auth; keep it local)")
    v.add_argument("--port", type=int, default=8765)
    d = sub.add_parser("demo", help="run an offline multi-agent demo that writes traces")
    d.add_argument("--dir", default="traces")
    args = ap.parse_args(argv)

    if args.cmd == "view":
        from agentharness.viewer.server import serve

        serve(args.dir, args.host, args.port)
    elif args.cmd == "demo":
        from agentharness.demo import run_demo

        print(run_demo(args.dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
