# agentharness

一个面向研究的可扩展 agent harness：零运行时依赖（纯标准库，Python ≥ 3.10）。

- **Agent loop**：LLM ↔ 工具循环，工具错误回传给模型，支持并行工具调用、`max_steps`、工具内主动停止
- **Tools**：`@tool` 从类型注解 + docstring 自动生成 JSON Schema；内置 `read_file / write_file / edit_file / list_dir / find_files / grep / bash`
- **可观测**：所有 agent / LLM / tool / workflow 都是嵌套 span，写入 `traces/<trace_id>.jsonl`；自带 Web 查看器
- **插件热插拔**：运行时 `load/unload`，或监听目录自动加载/重载/卸载 `.py` 插件
- **多 agent**：`agent.as_tool()`（主管/子 agent）、`Sequential`（流水线）、`Parallel`（并发 + 聚合），可任意嵌套，共享同一条 trace

## 安装

```bash
pip install "git+https://github.com/TorchTheo/MyAgent.git@v0.0.1"
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

## Trace 格式

每行一个事件：`{type: span_start|span_end|log, trace_id, span_id, parent_id, kind, name, data, ts}`。
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
