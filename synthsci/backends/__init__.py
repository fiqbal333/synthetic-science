"""Model backends. The engine produces task packets; a backend executes them (mock, subagent, API) or defers them."""
from __future__ import annotations


def make_backend(name: str, cfg: dict, run_dir, store, mode: str = "mock"):
    if name == "mock":
        from .mock import MockBackend
        return MockBackend(cfg, run_dir, store)
    if name == "subagent":
        from .subagent import SubagentBackend
        return SubagentBackend(cfg, run_dir, store)
    if name == "anthropic":
        from .api_anthropic import AnthropicBackend
        return AnthropicBackend(cfg, run_dir, store)
    if name == "openai_compat":
        from .openai_compat import OpenAICompatBackend
        return OpenAICompatBackend(cfg, run_dir, store)
    raise ValueError(f"unknown backend {name!r}")
