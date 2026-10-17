"""
Which LLM each provider runs — shared by the agent kinds and the server's own
LLM client, so an agent bullet and a prompt bullet answer with the same model.
"""

from __future__ import annotations

import os

ENV_KEYS = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

# The model each provider runs, overridable per deployment.
MODEL_ENV = {
    "gemini": ("GEMINI_MODEL", "gemini-3.6-flash"),
    "openai": ("OPENAI_MODEL", "gpt-4o-mini"),
    "anthropic": ("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
    "openrouter": ("OPENROUTER_MODEL", "deepseek/deepseek-v4.1-flash"),
}


def model_name(provider: str) -> str:
    """The model id this server uses for *provider*."""
    var, default = MODEL_ENV[provider]
    return os.environ.get(var, default)
