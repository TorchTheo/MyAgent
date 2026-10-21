"""Streaming tests."""

import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agentharness import (Agent, ConsolePlugin, LLMError, OpenAICompatLLM, Parallel, RunCancelled, Runtime,
                          ScriptedLLM, Sequential, tool)
from agentharness.core.events import STREAM

S = ScriptedLLM


@pytest.fixture
def rt(tmp_path):
    (tmp_path / "ws").mkdir()
    return Runtime(tmp_path / "ws", trace_dir=tmp_path / "traces")


@tool
def add(a: int, b: int = 1) -> int:
    """Add."""
    return a + b


def test_stream_text_and_result(rt):
    agent = Agent("a", S([S.tool_call("add", a=1), S.text("the answer is 2")], stream_chunk=3), tools=[add], runtime=rt)
    s = agent.stream("q")
    deltas = list(s.text())
    assert "".join(deltas) == "the answer is 2" and len(deltas) == 5
    r = s.result
    assert r.output == "the answer is 2" and r.steps == 2
    # identical message history to the non-streaming run
    r2 = Agent("a", S([S.tool_call("add", a=1), S.text("the answer is 2")]), tools=[add], runtime=rt).run("q")
    strip = lambda ms: [(m.role, m.content, [(c.name, c.arguments) for c in m.tool_calls]) for m in ms]
    assert strip(r.messages) == strip(r2.messages)


def test_stream_events_order_and_tool_deltas(rt):
    s = Agent("a", S([S.tool_call("add", "thinking", a=5), S.text("6")]), tools=[add], runtime=rt).stream("q")
    evs = list(s)
    kinds = [(e.type, e.kind) for e in evs if e.type != STREAM]
    assert kinds[0] == ("span_start", "agent") and kinds[-1] == ("span_end", "agent")
    tc = [e for e in evs if e.type == STREAM and e.data["type"] == "tool_call"]
    assert json.loads("".join(e.delta for e in tc)) == {"a": 5} and tc[0].data["name"] == "add"
    # deltas arrive between the llm span start and end
    llm_start = next(i for i, e in enumerate(evs) if e.type == "span_start" and e.kind == "llm")
    first_delta = next(i for i, e in enumerate(evs) if e.type == STREAM)
    llm_end = next(i for i, e in enumerate(evs) if e.type == "span_end" and e.kind == "llm")
    assert llm_start < first_delta < llm_end
    assert s.result.output == "6"


def test_stream_is_live_not_buffered(rt):
    agent = Agent("a", S([S.text("x" * 20)], stream_chunk=1, stream_delay=0.03), runtime=rt)
    t0 = time.time()
    s = agent.stream("q")
    first = next(iter(s.text()))
    assert first == "x" and time.time() - t0 < 0.3  # whole text takes ~0.6s


def test_stream_multi_agent_tags_agents(rt):
    worker = Agent("worker", S([S.text("from worker")]), runtime=rt)
    boss = Agent("boss", S([S.tool_call("worker", task="t"), S.text("from boss")]), tools=[worker.as_tool()], runtime=rt)
    par = Parallel([Agent("p1", S([S.text("one")]), runtime=rt), Agent("p2", S([S.text("two")]), runtime=rt)], runtime=rt)
    s = Sequential([boss, par], runtime=rt).stream("go")
    by_agent: dict[str, str] = {}
    for e in s:
        if e.type == STREAM and e.data["type"] == "text":
            by_agent[e.data["agent"]] = by_agent.get(e.data["agent"], "") + e.delta
    assert by_agent == {"worker": "from worker", "boss": "from boss", "p1": "one", "p2": "two"}
    assert s.result.output == {"p1": "one", "p2": "two"}


def test_concurrent_streams_are_isolated(rt):
    a = Agent("a", S([S.text("aaaa")], stream_chunk=1, stream_delay=0.01), runtime=rt)
    b = Agent("b", S([S.text("bbbb")], stream_chunk=1, stream_delay=0.01), runtime=rt)
    sa, sb = a.stream("x"), b.stream("y")
    out = {}
    th = threading.Thread(target=lambda: out.setdefault("b", "".join(sb.text())))
    th.start()
    out["a"] = "".join(sa.text())
    th.join()
    assert out == {"a": "aaaa", "b": "bbbb"}


def test_cancel_on_break(rt):
    llm = S([S.text("y" * 200)] + [S.text("never")], stream_chunk=1, stream_delay=0.005)
    s = Agent("a", llm, runtime=rt).stream("q")
    for i, _ in enumerate(s.text()):
        if i == 3:
            break
    assert s.wait(2)
    with pytest.raises(RunCancelled):
        s.result
    ev = [json.loads(l) for f in rt.trace_dir.glob("*.jsonl") for l in f.read_text().splitlines()]
    agent_end = [e for e in ev if e["type"] == "span_end" and e["kind"] == "agent"]
    assert agent_end and "RunCancelled" in agent_end[0]["data"]["error"]
    assert not any(e["type"] == STREAM for e in ev)  # deltas not persisted by default


def test_error_propagates(rt):
    s = Agent("a", S([]), runtime=rt).stream("q")
    with pytest.raises(LLMError, match="exhausted"):
        list(s)


def test_trace_stream_option(tmp_path):
    rt = Runtime(tmp_path, trace_dir=tmp_path / "tr", trace_stream=True)
    r = Agent("a", S([S.text("abcdefgh")]), runtime=rt).stream("q").result
    ev = [json.loads(l) for l in (rt.trace_dir / f"{r.trace_id}.jsonl").read_text().splitlines()]
    assert "".join(e["data"]["delta"] for e in ev if e["type"] == STREAM) == "abcdefgh"


def test_console_live(rt):
    buf = io.StringIO()
    rt.plugins.load(ConsolePlugin(stream=buf))
    Agent("a", S([S.text("hello world")]), runtime=rt).stream("q").result
    out = buf.getvalue()
    assert "hello world" in out and out.count("hello world") == 1


# ---------------- real SSE against a fake OpenAI server ----------------
def _sse_server(chunks_for):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b": keep-alive\n\n")
            for c in chunks_for(body):
                self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _chunk(delta=None, finish=None, usage=None):
    c = {"model": "fake", "choices": [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if usage:
        c["usage"] = usage
    return c


def test_openai_sse_stream(rt):
    def chunks(body):
        if any(m["role"] == "tool" for m in body["messages"]):
            return [_chunk({"role": "assistant", "reasoning_content": "hmm "}), _chunk({"content": "res"}),
                    _chunk({"content": "ult 3"}, "stop"), _chunk(usage={"prompt_tokens": 10, "completion_tokens": 4})]
        return [
            _chunk({"role": "assistant", "content": None, "tool_calls": [
                {"index": 0, "id": "c1", "type": "function", "function": {"name": "add", "arguments": ""}}]}),
            _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"a"'}}]}),
            _chunk({"tool_calls": [{"index": 0, "function": {"arguments": ": 2}"}}]}),
            _chunk({"tool_calls": [{"index": 1, "id": "c2", "function": {"name": "add", "arguments": '{"a": 7, "b": 1}'}}]}),
            _chunk({}, "tool_calls"), _chunk(usage={"prompt_tokens": 5, "completion_tokens": 6}),
        ]

    srv = _sse_server(chunks)
    try:
        llm = OpenAICompatLLM("fake", base_url=f"http://127.0.0.1:{srv.server_address[1]}/v1", api_key="k")
        s = Agent("a", llm, tools=[add], runtime=rt).stream("2+1")
        evs = list(s)
        r = s.result
        assert r.output == "result 3"
        assert [m.content for m in r.messages if m.role == "tool"] == ["3", "8"]
        assert [c.id for c in r.messages[2].tool_calls] == ["c1", "c2"]
        assert r.messages[-1].reasoning == "hmm "
        assert r.usage.prompt_tokens == 15 and r.usage.completion_tokens == 10
        texts = [e.delta for e in evs if e.type == STREAM and e.data["type"] == "text"]
        assert texts == ["res", "ult 3"]
    finally:
        srv.shutdown()


def test_openai_sse_error_chunk(rt):
    srv = _sse_server(lambda body: [{"error": {"message": "overloaded"}}])
    try:
        llm = OpenAICompatLLM("fake", base_url=f"http://127.0.0.1:{srv.server_address[1]}/v1", api_key="k")
        with pytest.raises(LLMError, match="overloaded"):
            Agent("a", llm, runtime=rt).stream("x").result
    finally:
        srv.shutdown()
