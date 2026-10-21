"""Run a real-model agent. Usage:

    OPENAI_API_KEY=... OPENAI_BASE_URL=https://api.deepseek.com/v1 MODEL=deepseek-chat \
        python examples/quickstart.py "统计当前目录下的 Python 文件行数"
"""

import os
import sys

from agentharness import Agent, ConsolePlugin, OpenAICompatLLM, Runtime, builtin_tools

rt = Runtime(".", plugins=[ConsolePlugin()])
stop = rt.plugins.watch("examples/plugins")  # edit/add/remove plugin files while the agent runs

llm = OpenAICompatLLM(os.environ.get("MODEL", "gpt-4o-mini"))
agent = Agent("coder", llm, "You are a careful coding agent. Use tools to inspect before answering.",
              tools=builtin_tools(), runtime=rt)
# stream: ConsolePlugin prints tokens live; use `for d in s.text(): ...` to handle them yourself
s = agent.stream(sys.argv[1] if len(sys.argv) > 1 else "List the files here and summarize the project.")
result = s.result
print("\n=== answer ===\n" + result.output)
print(f"steps={result.steps} tokens={result.usage.total_tokens} trace=traces/{result.trace_id}.jsonl")
