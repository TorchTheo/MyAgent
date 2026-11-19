# agentharness

一个面向研究的可扩展 agent harness：零运行时依赖（纯标准库，Python ≥ 3.10）。

- **Agent loop**：LLM ↔ 工具循环，工具错误回传给模型，支持并行工具调用、`max_steps`、工具内主动停止
- **Tools**：`@tool` 从类型注解 + docstring 自动生成 JSON Schema；内置 `read_file / write_file / edit_file / list_dir / find_files / grep / bash`
- **可观测**：所有 agent / LLM / tool / workflow 都是嵌套 span，写入 `traces/<trace_id>.jsonl`；自带 Web 查看器
- **插件热插拔**：运行时 `load/unload`，或监听目录自动加载/重载/卸载 `.py` 插件
- **流式输出**：`agent.stream()` 实时产出 token 增量和 span 事件，OpenAI 兼容后端走真实 SSE，支持中途取消
- **多 agent**：`agent.as_tool()`（主管/子 agent）、`Sequential`（流水线）、`Parallel`（并发 + 聚合），可任意嵌套，共享同一条 trace

## 安装

```bash
pip install "git+https://github.com/TorchTheo/MyAgent.git@v0.0.4"
```

## 快速开始

```bash
cd agentharness
python -m agentharness.cli demo            # 离线多 agent demo，生成 trace
python -m agentharness.cli view            # 打开 http://127.0.0.1:8765
# 或 pip install -e . 后使用 `agentharness demo` / `agentharness view`
```

接真实模型（任何 OpenAI 兼容接口：OpenAI / DeepSeek / Qwen / vLLM / Ollama ...）：

```python
from agentharness import Agent, OpenAICompatLLM, Runtime, builtin_tools

rt = Runtime("./workspace")
llm = OpenAICompatLLM("deepseek-chat", base_url="https://api.deepseek.com/v1")  # 读取 OPENAI_API_KEY
agent = Agent("coder", llm, "You are a coding agent.", tools=builtin_tools(), runtime=rt)
print(agent.run("列出当前目录并总结项目结构").output)
```

编码 Agent 默认会在系统提示中优先推荐当前请求中可用的内置文件工具。
传入 `Agent(..., prefer_file_tools=False)` 可关闭这段指引。

从仓库根目录运行 `python -m examples.quickstart` 可体验真实模型。
示例读取仓库根目录 `.env` 中的 `url` 和 `key`，默认使用
`cn:deepseek-v4.1-flash`。环境变量 `OPENAI_BASE_URL`、`OPENAI_API_KEY`
和 `MODEL` 可以覆盖对应设置。

## 核心概念

| 模块 | 作用 |
| --- | --- |
| `Runtime` | 共享环境：workspace、插件管理器、全局工具表（插件提供的工具注册在这里） |
| `Agent` | agent loop；每一步重新解析工具表，所以热插拔的工具立即可见 |
| `RunContext` | 传给工具和插件钩子；`ctx.state` 是多 agent 共享黑板，`ctx.request_stop()` 结束循环 |
| `LLM` | 后端接口，只需实现 `complete(messages, tools, **params) -> LLMResponse` |
| `Plugin` | 生命周期 + loop 钩子 + 事件订阅 + 提供工具 |

### 自定义工具

```python
from agentharness import tool, RunContext, ToolError

@tool
def search_paper(ctx: RunContext, query: str, top_k: int = 5) -> list[dict]:
    """Search papers.

    Args:
        query: keywords
        top_k: number of results
    """
    if not query:
        raise ToolError("query is empty")  # 作为错误消息返回给模型
    return [...]
```

`RunContext` 类型（或名为 `ctx`）的参数会自动注入，不出现在 schema 里。

### 插件钩子

| 钩子 | 用途 |
| --- | --- |
| `tools()` | 插件加载时注册、卸载时移除的工具 |
| `on_event(event)` | 接收全部 span/log 事件（自定义观测、导出到 W&B / OTel） |
| `before_run / after_run / on_error` | 一次 agent 运行的前后 |
| `before_llm(ctx, agent, request)` | 改写 `LLMRequest`；替换 `request.messages` 会持久化（上下文压缩、记忆注入） |
| `after_llm` | 返回新的 `LLMResponse` 覆盖 |
| `before_tool` | 返回 `ToolCall` 改写调用，返回 `str` 拦截并直接作为结果（权限、缓存、人类审批） |
| `after_tool` | 返回 `str` 替换结果（截断、脱敏） |

插件异常默认被隔离并记为 log 事件，设 `strict = True` 则抛出。

热插拔：

```python
rt.plugins.load(MyPlugin())        # 同名插件会被替换
rt.plugins.unload("MyPlugin")      # 连同它注册的工具一起移除
rt.plugins.load_path("x.py")       # 从文件加载
stop = rt.plugins.watch("plugins/")  # 目录内新增/修改/删除文件即加载/重载/卸载；stop.set() 停止
```

参考 `examples/plugins/` 里的示例。

### 多 agent

```python
researcher = Agent("researcher", llm, "...", tools=builtin_tools("read_file", "grep"), runtime=rt,
                   description="Researches a question using workspace files.")
supervisor = Agent("supervisor", llm, "...", tools=[researcher.as_tool()], runtime=rt)

review = Parallel([critic, fact_checker], aggregate=judge, runtime=rt)
pipeline = Sequential([supervisor, review, lambda out, ctx: out.strip()], runtime=rt)
result = pipeline.run("...")
```

### 流式输出

任何 `Agent` / `Sequential` / `Parallel` 都可以 `.stream(input)`：后台线程运行，实时产出事件。

```python
s = agent.stream("分析这个仓库")
for delta in s.text():               # 只要助手文本；s.text(agent="writer") 可只看某个 agent
    print(delta, end="", flush=True)
result = s.result                    # RunResult，和 run() 完全一致；运行中的异常会在这里重新抛出

for e in agent.stream("..."):        # 完整事件流
    if e.type == "stream":           # e.data = {type: text|reasoning|tool_call, delta, agent, step, ...}
        ...
    elif e.type == "span_start" and e.kind == "tool":
        ...
```

- 多 agent 下整棵树（子 agent、Parallel 分支）都在流里，用 `e.data["agent"]` 区分。
- 工具调用参数也会以 `tool_call` 增量流出；工具本身在参数完整后才执行。
- `break` 出循环或 `s.cancel()` 会在下一个 delta / 下一步停止运行并关闭 HTTP 连接，`s.result` 抛 `RunCancelled`。
- `OpenAICompatLLM` 走真实 SSE（默认带 `stream_options.include_usage`，不支持的服务端传 `stream_usage=False`）。
  自定义后端只实现 `complete` 也能用：默认 `stream()` 会把完整结果一次性作为 delta 发出；要真流式就覆写 `stream()`，
  按顺序 yield `StreamDelta`，最后一个必须是 `StreamDelta("done", response=LLMResponse(...))`。
- `ConsolePlugin` 在流式运行时实时打印模型输出。
- token 增量默认**不写入 trace**（文件会变得很大，最终内容始终会记录）。`Runtime(trace_stream=True)` 开启后，
  查看器会对运行中的 trace 实时显示正在生成的内容。

## 上下文预算

给 `Agent` 传入 `context_budget_bytes=64_000` 可启用上下文压缩，默认关闭。
预算以框架序列化消息和工具定义后的 UTF-8 字节数计量。模型的 token 上限需要另行考虑，
实际 HTTP 请求还会包含模型名称和后端参数。

超过预算时，框架保留系统和用户消息、最近的交互以及成组的工具调用和结果。
它从最早的交互开始压缩，有空间时保留历史摘录，必要时缩短工具输出。
压缩只作用于发给模型的请求，`RunResult.messages` 仍保留原始消息。
trace 会记录压缩前后的字节数。必需内容仍超过预算时会在调用模型前抛出 `LLMError`。

## Trace 格式

每行一个事件：`{type: span_start|span_end|log|stream, trace_id, span_id, parent_id, kind, name, data, ts}`。
LLM span 的 `span_start.data.input.messages` 是完整 prompt，`span_end.data.output` 是模型回复，`attrs.usage` 是 token 用量，
便于用 pandas 做轨迹分析或导出 SFT/RL 数据。

## 安全说明

- `bash` 工具会在 workspace 中执行任意命令；对不可信模型用 `ToolPolicyPlugin(deny=["bash"])` 或在容器中运行。
- 文件工具默认禁止访问 workspace 之外的路径（`Runtime(allow_outside_workspace=True)` 可关闭）。
- Trace 查看器无鉴权，默认只监听 `127.0.0.1`；trace 包含完整 prompt 和工具输出，不要暴露到公网。

## 测试

```bash
uv run --with pytest==8.4.2 --no-project python -m pytest -q
```
