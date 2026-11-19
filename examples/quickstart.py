"""Run a real-model agent. Usage:

    python -m examples.quickstart "查看 README.md 并总结项目用途"

Set `url` and `key` in the repository's .env, or set OPENAI_BASE_URL and
OPENAI_API_KEY in the environment. Set MODEL to override the default model.
"""

import os
import sys
from pathlib import Path

from agentharness import Agent, ConsolePlugin, OpenAICompatLLM, Runtime, builtin_tools

settings = {}
env_file = Path(__file__).resolve().parents[1] / ".env"
if env_file.is_file():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        name, separator, value = line.partition("=")
        if separator and name.strip() in {"url", "key", "OPENAI_BASE_URL", "OPENAI_API_KEY", "MODEL"}:
            value = value.strip()
            if len(value) >= 2 and value[0] in {"'", '"'} and value[-1] == value[0]:
                value = value[1:-1]
            settings[name.strip()] = value

base_url = os.environ.get("OPENAI_BASE_URL") or settings.get("OPENAI_BASE_URL") or settings.get("url")
api_key = os.environ.get("OPENAI_API_KEY") or settings.get("OPENAI_API_KEY") or settings.get("key")
model = os.environ.get("MODEL") or settings.get("MODEL") or "cn:deepseek-v4.1-flash"
if not base_url or not api_key:
    raise SystemExit("Set url and key in .env or configure OPENAI_BASE_URL and OPENAI_API_KEY.")

rt = Runtime(".", plugins=[ConsolePlugin()])
stop = rt.plugins.watch("examples/plugins")  # edit/add/remove plugin files while the agent runs

llm = OpenAICompatLLM(model, base_url=base_url, api_key=api_key)
agent = Agent("coder", llm, "You are a careful coding agent. Use tools to inspect before answering.",
              tools=builtin_tools(), runtime=rt)
# stream: ConsolePlugin prints tokens live; use `for d in s.text(): ...` to handle them yourself
s = agent.stream(sys.argv[1] if len(sys.argv) > 1 else "List the files here and summarize the project.")
result = s.result
print("\n=== answer ===\n" + result.output)
print(f"steps={result.steps} tokens={result.usage.total_tokens} trace=traces/{result.trace_id}.jsonl")
