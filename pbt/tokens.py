"""
Token usage reported by ``llm_call``.

``llm_call`` may return a :class:`LLMResult` instead of its bare output to tell
pbt how many tokens the call used::

    # client.py
    import pbt

    def llm_call(prompt):
        message = client.messages.create(...)
        return pbt.LLMResult(
            message.content[0].text,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
        )

pbt unwraps it straight away: the prompt cache, ``ref()`` and validators only
ever see ``output``.  The counts are stored with the model's result, and a
cache hit records the tokens the original call spent, so ``pbt docs`` can show
per run what was spent, what the cache saved, and what a cold run would cost.

Returning a plain value is still fine — the model's tokens are just unknown.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class LLMResult:
    """An ``llm_call`` return value carrying its token usage."""

    #: What the call produced — anything ``llm_call`` may otherwise return
    #: (a string, ``pbt.File``/``Dir``/``Output``, a dict holding them, …).
    output: Any
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass
class TokenUsage:
    """Input/output token counts, either of which may be unknown."""

    input_tokens: int | None = None
    output_tokens: int | None = None

    @property
    def known(self) -> bool:
        return self.input_tokens is not None or self.output_tokens is not None

    @property
    def total(self) -> int:
        return (self.input_tokens or 0) + (self.output_tokens or 0)

    def add(self, other: "TokenUsage") -> None:
        if other.input_tokens is not None:
            self.input_tokens = (self.input_tokens or 0) + other.input_tokens
        if other.output_tokens is not None:
            self.output_tokens = (self.output_tokens or 0) + other.output_tokens


def split_usage(result: Any) -> tuple[Any, TokenUsage]:
    """Return ``(output, usage)`` for an ``llm_call`` return value."""
    if isinstance(result, LLMResult):
        return result.output, TokenUsage(
            _count(result.input_tokens), _count(result.output_tokens)
        )
    return result, TokenUsage()


def _count(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
