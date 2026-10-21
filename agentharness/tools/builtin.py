"""Built-in tools: file system, search and shell.

All paths are resolved against ``runtime.workspace``; escaping the workspace is
rejected unless ``Runtime(allow_outside_workspace=True)``. ``bash`` runs arbitrary
shell commands in the workspace; gate it with ``ToolPolicyPlugin`` or run inside a
container when the model is untrusted.
"""

from __future__ import annotations

import fnmatch
import os
import re
import subprocess
from pathlib import Path

from agentharness.core.context import RunContext
from agentharness.tools.base import Tool, ToolError, tool

_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache"}


def resolve_path(ctx: RunContext, path: str) -> Path:
    ws = ctx.runtime.workspace
    p = Path(os.path.expanduser(path))
    p = (p if p.is_absolute() else ws / p).resolve()
    if not ctx.runtime.allow_outside_workspace and p != ws and ws not in p.parents:
        raise ToolError(f"path {path!r} is outside the workspace {ws}")
    return p


def _rel(ctx: RunContext, p: Path) -> str:
    try:
        return str(p.relative_to(ctx.runtime.workspace))
    except ValueError:
        return str(p)


def _walk(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
        for f in sorted(filenames):
            yield Path(dirpath) / f


@tool
def read_file(ctx: RunContext, path: str, offset: int = 1, limit: int = 400) -> str:
    """Read a UTF-8 text file. Lines are prefixed with their line number.

    Args:
        path: File path, relative to the workspace.
        offset: 1-based line number to start from.
        limit: Maximum number of lines to return.
    """
    p = resolve_path(ctx, path)
    if not p.is_file():
        raise ToolError(f"not a file: {path}")
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    start = max(offset, 1) - 1
    chunk = lines[start:start + max(limit, 1)]
    body = "\n".join(f"{i + start + 1:>6}\t{line}" for i, line in enumerate(chunk))
    rest = len(lines) - (start + len(chunk))
    if rest > 0:
        body += f"\n... ({rest} more lines; use offset={start + len(chunk) + 1})"
    return body or "(empty file)"


@tool
def write_file(ctx: RunContext, path: str, content: str) -> str:
    """Create or overwrite a file with the given content (parent directories are created).

    Args:
        path: File path, relative to the workspace.
        content: Full file content.
    """
    p = resolve_path(ctx, path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} chars to {_rel(ctx, p)}"


@tool
def edit_file(ctx: RunContext, path: str, old: str, new: str, replace_all: bool = False) -> str:
    """Replace an exact text snippet in a file.

    Args:
        path: File path, relative to the workspace.
        old: Exact text to find. Must be unique unless replace_all is true.
        new: Replacement text.
        replace_all: Replace every occurrence instead of requiring a unique match.
    """
    p = resolve_path(ctx, path)
    if not p.is_file():
        raise ToolError(f"not a file: {path}")
    text = p.read_text(encoding="utf-8")
    n = text.count(old) if old else 0
    if n == 0:
        raise ToolError("old text not found")
    if n > 1 and not replace_all:
        raise ToolError(f"old text occurs {n} times; add context or set replace_all=true")
    p.write_text(text.replace(old, new) if replace_all else text.replace(old, new, 1), encoding="utf-8")
    return f"replaced {n if replace_all else 1} occurrence(s) in {_rel(ctx, p)}"


@tool
def list_dir(ctx: RunContext, path: str = ".") -> str:
    """List a directory (directories end with '/').

    Args:
        path: Directory path, relative to the workspace.
    """
    p = resolve_path(ctx, path)
    if not p.is_dir():
        raise ToolError(f"not a directory: {path}")
    entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name))
    return "\n".join(e.name + ("/" if e.is_dir() else "") for e in entries) or "(empty directory)"


@tool
def find_files(ctx: RunContext, pattern: str, path: str = ".", limit: int = 200) -> str:
    """Find files whose workspace-relative path or name matches a glob pattern (e.g. '*.py', 'src/**/test_*.py').

    Args:
        pattern: Glob pattern.
        path: Directory to search in.
        limit: Maximum number of results.
    """
    root = resolve_path(ctx, path)
    out = []
    for f in _walk(root):
        rel = _rel(ctx, f)
        if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(f.name, pattern):
            out.append(rel)
            if len(out) >= limit:
                out.append(f"... (limit {limit} reached)")
                break
    return "\n".join(out) or "no matches"


@tool
def grep(ctx: RunContext, pattern: str, path: str = ".", glob: str = "*", ignore_case: bool = False, limit: int = 100) -> str:
    """Search file contents with a regular expression. Returns 'file:line: text' rows.

    Args:
        pattern: Python regular expression.
        path: File or directory to search.
        glob: Only search files whose name matches this glob.
        ignore_case: Case-insensitive matching.
        limit: Maximum number of matching lines.
    """
    try:
        rx = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    except re.error as e:
        raise ToolError(f"invalid regex: {e}") from e
    root = resolve_path(ctx, path)
    files = [root] if root.is_file() else _walk(root)
    out: list[str] = []
    for f in files:
        if not fnmatch.fnmatch(f.name, glob):
            continue
        try:
            with open(f, encoding="utf-8") as fh:
                for i, line in enumerate(fh, 1):
                    if rx.search(line):
                        out.append(f"{_rel(ctx, f)}:{i}: {line.rstrip()[:300]}")
                        if len(out) >= limit:
                            return "\n".join(out) + f"\n... (limit {limit} reached)"
        except (UnicodeDecodeError, OSError):
            continue
    return "\n".join(out) or "no matches"


@tool
def bash(ctx: RunContext, command: str, timeout: int = 120) -> str:
    """Run a shell command in the workspace and return exit code, stdout and stderr.

    Args:
        command: The shell command.
        timeout: Timeout in seconds.
    """
    try:
        r = subprocess.run(command, shell=True, cwd=ctx.runtime.workspace, capture_output=True,
                           text=True, timeout=timeout, errors="replace")
    except subprocess.TimeoutExpired as e:
        raise ToolError(f"command timed out after {timeout}s") from e
    parts = [f"exit_code: {r.returncode}"]
    if r.stdout:
        parts.append(f"stdout:\n{r.stdout}")
    if r.stderr:
        parts.append(f"stderr:\n{r.stderr}")
    return "\n".join(parts)


ALL: dict[str, Tool] = {t.name: t for t in (read_file, write_file, edit_file, list_dir, find_files, grep, bash)}


def builtin_tools(*names: str, exclude: tuple[str, ...] = ()) -> list[Tool]:
    """Return built-in tools; all of them by default, or only ``names``."""
    unknown = set(names) - ALL.keys()
    if unknown:
        raise ValueError(f"unknown built-in tools: {sorted(unknown)}")
    return [t for n, t in ALL.items() if (not names or n in names) and n not in exclude]
