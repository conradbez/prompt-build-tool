"""
Token usage reported by ``llm_call``.

``llm_call`` may return a :class:`LLMResult` instead of its bare output to tell
pbt how many tokens the call spent::

    # client.py
    import pbt

    def llm_call(prompt):
        message = client.messages.create(...)
        return pbt.LLMResult(
            message.content[0].text,
            spent_tokens=message.usage.input_tokens + message.usage.output_tokens,
        )

How ``spent_tokens`` is counted is up to client.py.  A sane default is
input + output + thinking, or the provider's own total when its API reports
one (Gemini's ``total_token_count``, OpenAI's ``total_tokens``).

pbt unwraps the result straight away: the prompt cache, ``ref()`` and
validators only ever see ``output``.  The count is stored with the model's
result, and a cache hit records the tokens the original call spent, so
``pbt docs`` can show per run what was spent, what the cache saved, and what a
cold run would cost.

Returning a plain value is still fine — the model's tokens are just unknown.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class LLMResult:
    """An ``llm_call`` return value carrying the tokens the call spent."""

    #: What the call produced — anything ``llm_call`` may otherwise return
    #: (a string, ``pbt.File``/``Dir``/``Output``, a dict holding them, …).
    output: Any
    spent_tokens: int | None = None


def split_usage(result: Any) -> tuple[Any, int | None]:
    """Return ``(output, spent_tokens)`` for an ``llm_call`` return value."""
    if isinstance(result, LLMResult):
        return result.output, _count(result.spent_tokens)
    return result, None


def add_tokens(a: int | None, b: int | None) -> int | None:
    """Sum two token counts, either of which may be unknown (None)."""
    return None if a is None and b is None else (a or 0) + (b or 0)


def _count(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
