import json
import threading
import time
import urllib.request
from pathlib import Path

import pytest

from agentharness import (Agent, LLMRequest, Message, OpenAICompatLLM, Parallel, Plugin, Runtime, ScriptedLLM,
                          Sequential, ToolCall, ToolPolicyPlugin, builtin_tools, tool)
from agentharness.viewer.server import make_server

S = ScriptedLLM


def events(rt, trace_id):
    return [json.loads(l) for l in (rt.trace_dir / f"{trace_id}.jsonl").read_text().splitlines()]


@pytest.fixture
def rt(tmp_path):
    (tmp_path / "ws").mkdir()
    return Runtime(tmp_path / "ws", trace_dir=tmp_path / "traces")


@tool
def add(a: int, b: int = 1) -> int:
    """Add numbers.

    Args:
        a: first
        b: second
    """
    return a + b


# ---------------- tools ----------------
def test_schema_inference():
    fn = add.schema()["function"]
    assert fn["description"] == "Add numbers."
    assert fn["parameters"]["required"] == ["a"]
    assert fn["parameters"]["properties"]["a"] == {"type": "integer", "description": "first"}
    assert fn["parameters"]["properties"]["b"]["default"] == 1
    # ctx param hidden
    assert "ctx" not in builtin_tools("read_file")[0].parameters["properties"]


def test_loop_with_tools_and_errors(rt):
    llm = S([
        S.tool_calls(("add", {"a": "2", "b": 3}), ("nope", {}), ("add", {})),
        S.text("5"),
    ])
    r = Agent("a", llm, tools=[add], runtime=rt).run("2+3?")
    assert r.output == "5" and r.steps == 2 and r.stop_reason == "final"
    tool_msgs = [m for m in r.messages if m.role == "tool"]
    assert tool_msgs[0].content == "5"  # "2" coerced to int
    assert "unknown tool" in tool_msgs[1].content
    assert "missing required" in tool_msgs[2].content
    # tool results are in call order and paired with ids
    calls = r.messages[2].tool_calls
    assert [m.tool_call_id for m in tool_msgs] == [c.id for c in calls]
    # second request sent to the LLM contains the tool results
    assert [m.role for m in llm.requests[1]] == ["system", "user", "assistant", "tool", "tool", "tool"]


def test_max_steps(rt):
    r = Agent("a", S([S.tool_call("add", a=1)] * 3), tools=[add], runtime=rt, max_steps=3).run("x")
    assert r.stop_reason == "max_steps" and r.steps == 3


def test_builtin_file_tools_and_sandbox(rt):
    llm = S([
        S.tool_call("write_file", path="d/a.txt", content="hello\nworld\n"),
        S.tool_call("edit_file", path="d/a.txt", old="world", new="there"),
        S.tool_calls(("read_file", {"path": "d/a.txt"}), ("grep", {"pattern": "there"}),
                     ("find_files", {"pattern": "*.txt"}), ("read_file", {"path": "../../etc/passwd"}),
                     ("bash", {"command": "echo hi"})),
        S.text("ok"),
    ])
    r = Agent("a", llm, tools=builtin_tools(), runtime=rt).run("go")
    out = [m.content for m in r.messages if m.role == "tool"]
    assert (rt.workspace / "d/a.txt").read_text() == "hello\nthere\n"
    assert "2\tthere" in out[2]
    assert out[3].startswith("d/a.txt:2:")
    assert out[4] == "d/a.txt"
    assert "outside the workspace" in out[5]
    assert "exit_code: 0" in out[6] and "hi" in out[6]


# ---------------- tracing ----------------
def test_trace_records_llm_calls(rt):
    r = Agent("a", S([S.tool_call("add", a=1), S.text("2")]), tools=[add], runtime=rt).run("q")
    ev = events(rt, r.trace_id)
    ends = [e for e in ev if e["type"] == "span_end"]
    kinds = [e["kind"] for e in ends]
    assert kinds == ["llm", "tool", "llm", "agent"]
    llm_start = next(e for e in ev if e["type"] == "span_start" and e["kind"] == "llm")
    assert llm_start["data"]["input"]["messages"][1] == {"role": "user", "content": "q"}
    assert ends[0]["data"]["output"]["tool_calls"][0]["name"] == "add"
    assert ends[0]["data"]["attrs"]["usage"]["total_tokens"] > 0
    agent_id = ends[-1]["span_id"]
    assert all(e["parent_id"] == agent_id for e in ends[:-1])


# ---------------- plugins ----------------
def test_plugin_hooks_and_tools_hot_plug(rt):
    seen = []

    class Guard(Plugin):
        def tools(self):
            @tool
            def secret() -> str:
                """Return a secret."""
                return "42"
            return [secret]

        def before_llm(self, ctx, agent, request: LLMRequest):
            seen.append(sorted(t["function"]["name"] for t in request.tools))

        def before_tool(self, ctx, agent, call):
            if call.name == "add":
                return ToolCall("add", {"a": 100}, call.id)  # rewrite

        def after_tool(self, ctx, agent, call, result):
            return f"[{result}]"

    agent = Agent("a", S([S.tool_calls(("secret", {}), ("add", {"a": 1})), S.text("done"),
                          S.tool_call("secret"), S.text("done2")]), tools=[add], runtime=rt)
    rt.plugins.load(Guard())
    r = agent.run("x")
    assert [m.content for m in r.messages if m.role == "tool"] == ["[42]", "[101]"]
    assert seen[0] == ["add", "secret"]
    rt.plugins.unload("Guard")
    assert "secret" not in rt.tools
    r2 = agent.run("y")
    assert "unknown tool 'secret'" in r2.messages[3].content


def test_plugin_failure_is_isolated(rt):
    class Broken(Plugin):
        def before_llm(self, ctx, agent, request):
            raise RuntimeError("boom")

    rt.plugins.load(Broken())
    r = Agent("a", S([S.text("fine")]), runtime=rt).run("x")
    assert r.output == "fine"
    assert any(e["type"] == "log" and "Broken.before_llm failed" in e["name"] for e in events(rt, r.trace_id))


def test_tool_policy(rt):
    rt.plugins.load(ToolPolicyPlugin(deny=["bash"]))
    r = Agent("a", S([S.tool_call("bash", command="touch pwned"), S.text("x")]), tools=builtin_tools(), runtime=rt).run("x")
    assert "not permitted" in r.messages[3].content
    assert not (rt.workspace / "pwned").exists()


def test_hot_reload_from_directory(rt, tmp_path):
    pdir = tmp_path / "plugins"
    pdir.mkdir()
    src = '''
from agentharness import Plugin, tool

@tool
def greet() -> str:
    """Greet."""
    return "{msg}"

class Greeter(Plugin):
    name = "greeter"
    def tools(self):
        return [greet]
'''
    f = pdir / "greeter.py"
    f.write_text(src.format(msg="v1"))
    changes = []
    stop = rt.plugins.watch(pdir, interval=0.05, on_change=lambda a, p, e: changes.append(a))
    try:
        assert rt.tools.get("greet").run(None, {}) == "v1"
        time.sleep(0.05)
        f.write_text(src.format(msg="v2"))
        import os
        os.utime(f, (time.time() + 5, time.time() + 5))  # guarantee a new mtime
        _wait(lambda: "reload" in changes)
        assert rt.tools.get("greet").run(None, {}) == "v2"
        assert rt.plugins.names().count("greeter") == 1
        f.unlink()
        _wait(lambda: "unload" in changes)
        assert "greet" not in rt.tools and "greeter" not in rt.plugins.names()
    finally:
        stop.set()


def _wait(cond, timeout=3.0):
    t = time.time()
    while not cond():
        if time.time() - t > timeout:
            raise AssertionError("timeout")
        time.sleep(0.02)


# ---------------- multi-agent ----------------
def test_multi_agent_single_trace_tree(rt):
    worker = Agent("worker", S([S.tool_call("add", a=1), S.text("worker says 2")]), tools=[add], runtime=rt)
    boss = Agent("boss", S([S.tool_call("worker", task="compute"), S.text("boss done")]),
                 tools=[worker.as_tool()], runtime=rt)
    par = Parallel([Agent("p1", S([S.text("A")], latency=0.1), runtime=rt),
                    Agent("p2", S([S.text("B")], latency=0.1), runtime=rt)],
                   aggregate=lambda outs, ctx: outs["p1"] + outs["p2"], runtime=rt)
    t = time.time()
    r = Sequential([boss, par, lambda x, ctx: x.lower()], runtime=rt).run("start")
    assert r.output == "ab"
    assert time.time() - t < 0.5  # p1/p2 ran concurrently with no serial LLM latency doubling beyond budget
    assert r.usage.total_tokens > 0
    tool_msg = [m for m in boss_messages(rt, r.trace_id)]
    ev = [e for e in events(rt, r.trace_id) if e["type"] == "span_start"]
    assert len({e["trace_id"] for e in ev}) == 1
    by_id = {e["span_id"]: e for e in ev}
    worker_span = next(e for e in ev if e["kind"] == "agent" and e["name"] == "worker")
    parent = by_id[worker_span["parent_id"]]
    assert parent["kind"] == "tool" and parent["name"] == "worker"
    assert by_id[parent["parent_id"]]["name"] == "boss"
    p_spans = [e for e in ev if e["name"] in ("p1", "p2")]
    assert {by_id[e["parent_id"]]["name"] for e in p_spans} == {"parallel"}
    assert tool_msg


def boss_messages(rt, trace_id):
    return [e for e in events(rt, trace_id) if e["type"] == "span_end" and e["kind"] == "tool" and e["name"] == "worker"
            and e["data"]["output"] == "worker says 2"]


def test_shared_state_and_stop(rt):
    @tool
    def finish(ctx, answer: str) -> str:
        """Submit the final answer."""
        ctx.state["answer"] = answer
        ctx.request_stop("submitted")
        return "ok"

    a = Agent("a", S([S.tool_call("finish", answer="x"), S.text("never")]), tools=[finish], runtime=rt)
    b = Agent("b", S([S.text("never")]), runtime=rt)
    r = Sequential([a, b], runtime=rt).run("go")
    assert r.stop_reason == "submitted" and r.steps == 1


# ---------------- openai backend ----------------
def test_openai_payload_and_parse():
    llm = OpenAICompatLLM("m", base_url="http://x/v1", api_key="k", temperature=0)
    msgs = [Message.user("hi"), Message.assistant(None, [ToolCall("add", {"a": 1}, "c1")]), Message.tool("2", "c1", "add")]
    p = llm.build_payload(msgs, [add.schema()], {})
    assert p["temperature"] == 0 and p["tools"][0]["function"]["name"] == "add"
    assert p["messages"][1]["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'
    assert p["messages"][2]["tool_call_id"] == "c1"
    r = llm.parse_response({"model": "m", "choices": [{"finish_reason": "tool_calls", "message": {
        "content": None, "reasoning_content": "think",
        "tool_calls": [{"id": "c2", "function": {"name": "add", "arguments": "{bad"}}]}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4}})
    assert r.usage.total_tokens == 7 and r.message.reasoning == "think"
    assert "__invalid_arguments__" in r.message.tool_calls[0].arguments
    with pytest.raises(Exception, match="not valid JSON"):
        add.run(None, r.message.tool_calls[0].arguments)


def test_openai_http_roundtrip(rt):
    """Real HTTP against a local fake OpenAI server."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    got = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            got.append((self.path, self.headers.get("Authorization"), body))
            done = any(m["role"] == "tool" for m in body["messages"])
            msg = {"content": "3"} if done else {"content": None, "tool_calls": [
                {"id": "c", "type": "function", "function": {"name": "add", "arguments": '{"a": 2}'}}]}
            data = json.dumps({"choices": [{"message": msg, "finish_reason": "stop"}], "usage": {}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        llm = OpenAICompatLLM("fake", base_url=f"http://127.0.0.1:{srv.server_address[1]}/v1", api_key="sk-test")
        r = Agent("a", llm, tools=[add], runtime=rt).run("2+1")
        assert r.output == "3"
        assert got[0][0] == "/v1/chat/completions" and got[0][1] == "Bearer sk-test"
    finally:
        srv.shutdown()


# ---------------- viewer ----------------
def test_viewer_api(rt):
    r = Agent("a", S([S.text("<script>alert(1)</script>")]), runtime=rt).run("hello")
    srv = make_server(rt.trace_dir, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        lst = json.loads(urllib.request.urlopen(base + "/api/traces").read())
        assert lst[0]["id"] == r.trace_id and lst[0]["llm_calls"] == 1 and lst[0]["input"] == "hello"
        ev = json.loads(urllib.request.urlopen(f"{base}/api/traces/{r.trace_id}").read())
        assert len(ev) == 4
        html = urllib.request.urlopen(base + "/").read().decode()
        assert "Trace Viewer" in html and "innerHTML" not in html.replace("never innerHTML", "")
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "/api/traces/..%2F..%2Fetc%2Fpasswd")
    finally:
        srv.shutdown()
