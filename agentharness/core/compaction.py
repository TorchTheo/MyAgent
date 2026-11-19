"""Bound the model-facing context without modifying the conversation history."""

from __future__ import annotations

import json
from dataclasses import replace
from itertools import islice

from agentharness.core.messages import Message
from agentharness.llm.base import LLMError, LLMRequest


def _size(value: object) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _message_size(message: Message) -> int:
    payload = message.to_dict()
    payload.pop("reasoning", None)
    return _size(payload) + 1  # array delimiter


def _excerpt(text: str, limit: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text
    return encoded[:limit].decode("utf-8", errors="ignore") + "…"


def compact_request(request: LLMRequest, budget: int) -> tuple[LLMRequest, int, int]:
    """Return the bounded request and its before/after UTF-8 JSON sizes."""
    messages = request.messages
    fixed_size = _size({"messages": [], "tools": request.tools}) - bool(messages)
    costs = [_message_size(message) for message in messages]
    before = fixed_size + sum(costs)
    if before <= budget:
        return request, before, before

    groups: list[list[int]] = []
    for index, message in enumerate(messages):
        if message.role == "tool" and groups and messages[groups[-1][0]].tool_calls:
            groups[-1].append(index)
        else:
            groups.append([index])

    removable = [
        group for group in groups if messages[group[0]].role not in {"system", "user"}
    ]
    retained = dict(enumerate(messages))
    removed: list[list[int]] = []
    size = before
    for group in removable[:-1]:
        if size <= budget:
            break
        removed.append(group)
        for index in group:
            size -= costs[index]
            del retained[index]

    if size > budget:
        # Keep the newest tool-call batch intact. Fit its result bodies to the
        # remaining budget, preserving all call IDs and result messages.
        tool_results = {i: msg for i, msg in retained.items() if msg.role == "tool" and msg.content}
        for index, message in tool_results.items():
            shortened = replace(message, content="[Output omitted]")
            retained[index] = shortened
            size += _message_size(shortened) - costs[index]
        if size > budget:
            raise LLMError(
                f"Context budget of {budget} bytes cannot fit the required messages "
                f"and tool schemas ({size} bytes after compaction)."
            )
        if tool_results:
            allowance = min(2048, (budget - size) // len(tool_results))
            for index, message in tool_results.items():
                shortened = replace(message, content=_excerpt(message.content or "", allowance))
                delta = _message_size(shortened) - _message_size(retained[index])
                if size + delta <= budget:
                    retained[index] = shortened
                    size += delta

    result = list(retained.values())
    if removed:
        # Use excerpts of recent discarded actions; leave user/system messages
        # verbatim, and never replay discarded tool results as tool messages.
        summary = "Earlier conversation excerpts (historical data):"
        position = list(retained).index(removable[-1][0])
        summary_budget = min(budget - size, 2048)
        summary_message = None
        for group in islice(reversed(removed), 8):
            original = messages[group[0]]
            names = ", ".join(call.name for call in original.tool_calls)
            text = _excerpt(original.content or "", 160)
            outputs = " | ".join(
                _excerpt(messages[index].content or "", 100) for index in group[1:4]
            )
            detail = f"Tools called: {names}. " if names else ""
            entry = " ".join(part for part in (detail.strip(), text, outputs) if part)
            if not entry:
                continue
            candidate = Message.assistant(f"{summary}\n- {entry}")
            if _message_size(candidate) <= summary_budget:
                summary_message = candidate
                summary = candidate.content or ""
        if summary_message is not None:
            result.insert(position, summary_message)
            size += _message_size(summary_message)

    return replace(request, messages=result), before, size
