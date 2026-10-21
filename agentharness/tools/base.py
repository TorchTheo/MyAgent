"""Tool definition: a Python callable + JSON schema, inferred from type hints and docstring."""

from __future__ import annotations

import asyncio
import enum
import inspect
import json
import re
import types
from typing import Annotated, Any, Callable, Literal, Union, get_args, get_origin, get_type_hints

from agentharness.core.context import RunContext

INVALID_ARGS_KEY = "__invalid_arguments__"


class ToolError(Exception):
    """Raise inside a tool to return a clean error message to the model."""


def _schema_for(tp: Any) -> dict[str, Any]:
    if tp is inspect.Parameter.empty or tp is Any:
        return {}
    origin, args = get_origin(tp), get_args(tp)
    if origin is Annotated:
        s = dict(_schema_for(args[0]))
        desc = next((a for a in args[1:] if isinstance(a, str)), None)
        if desc:
            s["description"] = desc
        return s
    if origin is Union or (hasattr(types, "UnionType") and origin is types.UnionType):
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _schema_for(non_none[0])
        return {"anyOf": [_schema_for(a) for a in non_none]}
    if origin is Literal:
        s: dict[str, Any] = {"enum": list(args)}
        if all(isinstance(a, str) for a in args):
            s["type"] = "string"
        return s
    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        return {"enum": [m.value for m in tp]}
    if tp is bool:
        return {"type": "boolean"}
    if tp is int:
        return {"type": "integer"}
    if tp is float:
        return {"type": "number"}
    if tp is str:
        return {"type": "string"}
    if tp in (list, tuple, set) or origin in (list, tuple, set):
        return {"type": "array", "items": _schema_for(args[0]) if args else {}}
    if tp is dict or origin is dict:
        return {"type": "object"}
    if hasattr(tp, "model_json_schema"):  # pydantic v2 models
        return tp.model_json_schema()
    return {}


def _parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Google-style docstring -> (description, {param: description})."""
    if not doc:
        return "", {}
    doc = inspect.cleandoc(doc)
    parts = re.split(r"\n\s*(?:Args|Arguments|Parameters):\s*\n", doc, maxsplit=1)
    desc = parts[0].strip()
    params: dict[str, str] = {}
    if len(parts) > 1:
        current = None
        for line in parts[1].splitlines():
            if re.match(r"^\S", line):  # next section (Returns:, Raises:)
                break
            m = re.match(r"^\s+(\*{0,2}\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$", line)
            if m:
                current = m.group(1).lstrip("*")
                params[current] = m.group(2).strip()
            elif current and line.strip():
                params[current] += " " + line.strip()
    return desc, params


_TYPE_CHECKS: dict[str, tuple[type, ...]] = {
    "string": (str,), "integer": (int,), "number": (int, float), "boolean": (bool,),
    "array": (list,), "object": (dict,),
}


def _coerce(name: str, value: Any, schema: dict[str, Any]) -> Any:
    """Light validation/coercion for the common mistakes models make (e.g. "3" for 3)."""
    t = schema.get("type")
    if "enum" in schema and value not in schema["enum"]:
        raise ToolError(f"argument {name!r} must be one of {schema['enum']}, got {value!r}")
    if t not in _TYPE_CHECKS:
        return value
    if t == "integer" and isinstance(value, bool) or t == "number" and isinstance(value, bool):
        raise ToolError(f"argument {name!r} must be {t}, got boolean")
    if isinstance(value, _TYPE_CHECKS[t]):
        return value
    if isinstance(value, str):
        try:
            if t == "integer":
                return int(value)
            if t == "number":
                return float(value)
            if t == "boolean" and value.lower() in ("true", "false"):
                return value.lower() == "true"
            if t in ("array", "object"):
                v = json.loads(value)
                if isinstance(v, _TYPE_CHECKS[t]):
                    return v
        except (ValueError, json.JSONDecodeError):
            pass
    if t == "integer" and isinstance(value, float) and value.is_integer():
        return int(value)
    raise ToolError(f"argument {name!r} must be {t}, got {type(value).__name__}")


class Tool:
    """A callable exposed to the model.

    If the function has a parameter annotated as ``RunContext`` (or named ``ctx``),
    the current run context is injected and hidden from the schema.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        name: str | None = None,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.fn = fn
        self.name = name or fn.__name__
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", self.name):
            raise ValueError(f"invalid tool name {self.name!r}: use [A-Za-z0-9_-], max 64 chars")
        self.metadata = dict(metadata or {})
        doc_desc, doc_params = _parse_docstring(fn.__doc__)
        self.description = description or doc_desc or self.name
        self.ctx_param: str | None = None
        sig = inspect.signature(fn)
        try:
            hints = get_type_hints(fn, include_extras=True)
        except Exception:
            hints = {}
        props: dict[str, Any] = {}
        required: list[str] = []
        for pname, p in sig.parameters.items():
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            ann = hints.get(pname, p.annotation)
            if ann is RunContext or (pname == "ctx" and ann in (inspect.Parameter.empty, Any)):
                self.ctx_param = pname
                continue
            s = _schema_for(ann)
            if pname in doc_params and "description" not in s:
                s["description"] = doc_params[pname]
            if p.default is inspect.Parameter.empty:
                required.append(pname)
            elif p.default is not None:
                s.setdefault("default", p.default)
            props[pname] = s
        self.parameters = parameters or {"type": "object", "properties": props, "required": required}

    def schema(self) -> dict[str, Any]:
        """OpenAI function-calling schema."""
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.parameters}}

    def run(self, ctx: RunContext, arguments: dict[str, Any]) -> str:
        if INVALID_ARGS_KEY in arguments:
            raise ToolError(f"tool arguments were not valid JSON: {arguments[INVALID_ARGS_KEY]!r}")
        props = self.parameters.get("properties", {})
        missing = [r for r in self.parameters.get("required", []) if r not in arguments]
        if missing:
            raise ToolError(f"missing required argument(s): {', '.join(missing)}")
        unknown = [k for k in arguments if k not in props]
        if unknown and not self.parameters.get("additionalProperties", False):
            raise ToolError(f"unknown argument(s): {', '.join(unknown)}; expected: {', '.join(props)}")
        kwargs = {k: _coerce(k, v, props.get(k, {})) for k, v in arguments.items()}
        if self.ctx_param:
            kwargs[self.ctx_param] = ctx
        result = self.fn(**kwargs)
        if inspect.isawaitable(result):
            result = asyncio.run(result)  # async tools run in their own event loop
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False, default=str)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)

    def __repr__(self) -> str:
        return f"Tool({self.name!r})"


def tool(fn: Callable[..., Any] | None = None, *, name: str | None = None, description: str | None = None,
         parameters: dict[str, Any] | None = None, **metadata: Any) -> Any:
    """Decorator: ``@tool`` or ``@tool(name=..., description=...)``."""

    def wrap(f: Callable[..., Any]) -> Tool:
        return Tool(f, name=name, description=description, parameters=parameters, metadata=metadata)

    return wrap(fn) if fn is not None else wrap
