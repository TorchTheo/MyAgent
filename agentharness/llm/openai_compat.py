"""OpenAI-compatible Chat Completions backend (OpenAI, DeepSeek, Qwen/DashScope, vLLM, SGLang, Ollama...).

Uses only the standard library.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Iterator

from agentharness.core.messages import LLMResponse, Message, ToolCall, Usage, new_id
from agentharness.llm.base import LLM, LLMError, StreamDelta
from agentharness.tools.base import INVALID_ARGS_KEY

_RETRYABLE = {408, 409, 429, 500, 502, 503, 504}


def to_openai_message(m: Message) -> dict[str, Any]:
    d: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.tool_calls:
        d["tool_calls"] = [
            {"id": c.id, "type": "function",
             "function": {"name": c.name, "arguments": json.dumps(c.arguments, ensure_ascii=False)}}
            for c in m.tool_calls
        ]
    if m.role == "tool":
        d["tool_call_id"] = m.tool_call_id
    return d


class OpenAICompatLLM(LLM):
    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        extra_headers: dict[str, str] | None = None,
        stream_usage: bool = True,
        **default_params: Any,
    ):
        self.model = model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_headers = dict(extra_headers or {})
        self.default_params = default_params
        # ask for a final usage chunk when streaming; set False for servers that reject stream_options
        self.stream_usage = stream_usage

    def build_payload(self, messages: list[Message], tools: list[dict[str, Any]] | None, params: dict[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [to_openai_message(m) for m in messages],
            **self.default_params,
            **params,
        }
        if tools:
            body["tools"] = tools
        return body

    def _post(self, payload: dict[str, Any], stream: bool = False) -> Any:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if stream:
            headers["Accept"] = "text/event-stream"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body, headers=headers, method="POST")
        for attempt in range(self.max_retries + 1):
            try:
                return urllib.request.urlopen(req, timeout=self.timeout)
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", errors="replace")[:2000]
                if e.code in _RETRYABLE and attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 30))
                    continue
                raise LLMError(f"HTTP {e.code} from {self.base_url}: {detail}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 30))
                    continue
                raise LLMError(f"request to {self.base_url} failed: {e}") from e
        raise LLMError("unreachable")

    def complete(self, messages: list[Message], tools: list[dict[str, Any]] | None = None, **params: Any) -> LLMResponse:
        with self._post(self.build_payload(messages, tools, params)) as r:
            data = json.loads(r.read().decode("utf-8"))
        return self.parse_response(data)

    def parse_response(self, data: dict[str, Any]) -> LLMResponse:
        try:
            choice = data["choices"][0]
            m = choice["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"unexpected response shape: {str(data)[:500]}") from e
        calls: list[ToolCall] = []
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            raw = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else raw
                if not isinstance(args, dict):
                    args = {INVALID_ARGS_KEY: raw}
            except json.JSONDecodeError:
                args = {INVALID_ARGS_KEY: raw}
            calls.append(ToolCall(name=fn.get("name", ""), arguments=args, id=tc.get("id") or new_id("call_")))
        u = data.get("usage") or {}
        return LLMResponse(
            message=Message(role="assistant", content=m.get("content"), tool_calls=calls,
                            reasoning=m.get("reasoning_content") or m.get("reasoning")),
            usage=Usage(int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)),
            model=data.get("model") or self.model,
            finish_reason=choice.get("finish_reason"),
            raw=data,
        )

    # ---- streaming (SSE) ---------------------------------------------
    def stream(self, messages: list[Message], tools: list[dict[str, Any]] | None = None, **params: Any) -> Iterator[StreamDelta]:
        payload = self.build_payload(messages, tools, params)
        payload["stream"] = True
        if self.stream_usage:
            payload.setdefault("stream_options", {"include_usage": True})
        resp = self._post(payload, stream=True)
        text: list[str] = []
        reasoning: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        usage = Usage()
        model, finish = self.model, None
        with resp:
            for data in _iter_sse(resp):
                if data.get("error"):
                    raise LLMError(f"stream error: {str(data['error'])[:500]}")
                model = data.get("model") or model
                if data.get("usage"):
                    u = data["usage"]
                    usage = Usage(int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0))
                for ch in data.get("choices") or []:
                    finish = ch.get("finish_reason") or finish
                    d = ch.get("delta") or {}
                    r = d.get("reasoning_content") or d.get("reasoning")
                    if r:
                        reasoning.append(r)
                        yield StreamDelta("reasoning", r)
                    if d.get("content"):
                        text.append(d["content"])
                        yield StreamDelta("text", d["content"])
                    for tc in d.get("tool_calls") or []:
                        i = tc.get("index", len(calls))
                        slot = calls.setdefault(i, {"id": None, "name": "", "args": []})
                        slot["id"] = tc.get("id") or slot["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["name"] += fn["name"]
                        frag = fn.get("arguments") or ""
                        if frag:
                            slot["args"].append(frag)
                        yield StreamDelta("tool_call", frag, index=i, id=slot["id"], name=slot["name"] or None)
        tool_calls = [ToolCall(name=s["name"], arguments=_parse_args("".join(s["args"])), id=s["id"] or new_id("call_"))
                      for _, s in sorted(calls.items())]
        msg = Message(role="assistant", content="".join(text) or None, tool_calls=tool_calls,
                      reasoning="".join(reasoning) or None)
        yield StreamDelta("done", response=LLMResponse(msg, usage, model, finish))


def _parse_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        args = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {INVALID_ARGS_KEY: raw}
    return args if isinstance(args, dict) else {INVALID_ARGS_KEY: raw}


def _iter_sse(resp: Any) -> Iterator[dict[str, Any]]:
    """Parse a text/event-stream body into JSON payloads (stops at ``[DONE]``)."""
    buf: list[str] = []
    for raw in resp:
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        if line.startswith(":"):
            continue  # comment / keep-alive
        if line.startswith("data:"):
            buf.append(line[5:].lstrip())
            continue
        if line == "" and buf:
            payload, buf = "\n".join(buf), []
            if payload.strip() == "[DONE]":
                return
            try:
                yield json.loads(payload)
            except json.JSONDecodeError:
                continue
    if buf and "\n".join(buf).strip() != "[DONE]":
        try:
            yield json.loads("\n".join(buf))
        except json.JSONDecodeError:
            pass
