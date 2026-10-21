"""agentharness: an extensible, observable agent harness.

Quick map of the public API:

- ``Runtime``          shared environment: workspace, plugins, global tools, tracing
- ``Agent``            the agent loop (LLM <-> tools until a final answer)
- ``tool`` / ``Tool``  define tools from plain Python functions
- ``Plugin``           hook into the loop; hot-pluggable via ``runtime.plugins``
- ``Sequential`` / ``Parallel`` / ``Agent.as_tool``  multi-agent composition
- ``agent.stream(input)`` -> ``RunStream``  live token / span events of any agent or workflow
- ``OpenAICompatLLM`` / ``ScriptedLLM``  model backends
"""

from agentharness.core.agent import Agent, RunResult
from agentharness.core.context import RunContext
from agentharness.core.messages import LLMResponse, Message, ToolCall, Usage
from agentharness.core.multi import Parallel, Sequential, as_tool
from agentharness.core.events import Event
from agentharness.core.runtime import Runtime, default_runtime
from agentharness.core.streaming import RunCancelled, RunStream
from agentharness.core.tracing import log, span
from agentharness.llm import LLM, LLMError, LLMRequest, OpenAICompatLLM, ScriptedLLM, StreamDelta
from agentharness.plugins import ConsolePlugin, Plugin, ToolPolicyPlugin, TracingPlugin
from agentharness.tools import Tool, ToolError, ToolRegistry, builtin_tools, tool

__all__ = [
    "Agent", "RunResult", "RunContext", "Message", "ToolCall", "Usage", "LLMResponse",
    "Sequential", "Parallel", "as_tool", "Runtime", "default_runtime", "span", "log",
    "Event", "RunStream", "RunCancelled", "StreamDelta",
    "LLM", "LLMError", "LLMRequest", "OpenAICompatLLM", "ScriptedLLM",
    "Plugin", "TracingPlugin", "ConsolePlugin", "ToolPolicyPlugin",
    "Tool", "ToolError", "ToolRegistry", "builtin_tools", "tool",
]
__version__ = "0.0.2"
