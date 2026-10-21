"""Example hot-pluggable plugin: a scratchpad tool + a context-compaction hook.

Drop this file into a watched directory (``rt.plugins.watch("examples/plugins")``);
edit it while an agent runs and it is reloaded; delete it and it is unloaded.
"""

from agentharness import Message, Plugin, RunContext, tool


@tool
def note(ctx: RunContext, text: str) -> str:
    """Save a note to the shared scratchpad (visible to every agent in this run).

    Args:
        text: The note to remember.
    """
    ctx.state.setdefault("notes", []).append(text)
    return f"saved ({len(ctx.state['notes'])} notes)"


class Scratchpad(Plugin):
    name = "scratchpad"
    keep_last = 40  # naive compaction: keep system + first user + the last N messages

    def tools(self):
        return [note]

    def before_llm(self, ctx, agent, request):
        msgs = request.messages
        if len(msgs) > self.keep_last + 2:
            head, tail = msgs[:2], msgs[-self.keep_last:]
            while tail and tail[0].role == "tool":  # never start with an orphan tool result
                tail = tail[1:]
            request.messages = head + [Message.user("(earlier steps omitted)")] + tail
        notes = ctx.state.get("notes")
        if notes:
            request.messages = request.messages + [Message.system("Scratchpad:\n- " + "\n- ".join(notes))]
            # note: the appended system message persists; a real plugin would dedupe it
        return request
