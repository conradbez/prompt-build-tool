"""
Paths into a model's output, in the SQL/JSON spelling: ``parts.items[0]``, ``sections[*][*]``.

The same path works in ``ref('...')`` and in ``config(each='...')``:

``.key``   a dict key (or a list index, when the key is digits)
``[n]``    a list index
``[*]``    every element — unnest one level, like Postgres ``jsonb_array_elements``

Lax shorthand is limited to one case: ``each='chapters'`` iterates a list
without ``[*]``.  Everything else is strict, so a wrong path fails with the
type it found rather than silently yielding nothing.
"""

from __future__ import annotations

import re
from typing import Any

from pbt.files import Dir, Output

WILDCARD = "*"

_PATH = re.compile(r"^(\w+)((?:\.\w+|\[\d+\]|\[\*\])*)$")
_STEP = re.compile(r"\.(\w+)|\[(\d+|\*)\]")


def parse(path: str) -> tuple[str, list[str | int]]:
    """Split ``"parts.items[0]"`` into ``("parts", ["items", 0])``."""
    match = _PATH.match(path.strip())
    if match is None:
        raise ValueError(
            f"'{path}' is not a valid path. Use model.key, model[0] or model[*]."
        )
    steps: list[str | int] = []
    for key, index in _STEP.findall(match.group(2)):
        steps.append(key if key else (WILDCARD if index == WILDCARD else int(index)))
    return match.group(1), steps


def resolve(value: Any, steps: list[str | int], label: str) -> Any:
    """Follow *steps* into *value*.  Any ``[*]`` makes the result a flat list."""
    values, fanned = [value], False
    for step in steps:
        if step == WILDCARD:
            fanned = True
            values = [item for v in values for item in _elements(v, label)]
        else:
            values = [_step(v, step, label) for v in values]
    return values if fanned else values[0]


def as_items(value: Any) -> list | None:
    """The items a fan-out iterates over: a list, or the files of a Dir."""
    if isinstance(value, list):
        return value
    if isinstance(value, Dir):
        return value.files()
    return None


def describe(value: Any) -> str:
    """A short type description for error messages."""
    if isinstance(value, dict):
        return f"a dict with keys {sorted(value)}"
    if isinstance(value, list):
        return f"a list of {len(value)}"
    return f"a {type(value).__name__}"


def _elements(value: Any, label: str) -> list:
    items = as_items(value)
    if items is None:
        raise ValueError(f"{label}: [*] expects a list, got {describe(value)}.")
    return items


def _step(value: Any, step: str | int, label: str) -> Any:
    if isinstance(step, str) and step.isdigit() and isinstance(value, list):
        step = int(step)
    if isinstance(step, int) and isinstance(value, list):
        if step < len(value):
            return value[step]
    elif isinstance(value, dict) and step in value:
        return value[step]
    elif isinstance(value, Output) and step in value.files:
        return value.files[step]
    elif isinstance(value, Dir) and step in value.entries:
        return value.entries[step]
    shown = f"[{step}]" if isinstance(step, int) else f".{step}"
    raise ValueError(f"{label}: {shown} not found in {describe(value)}.")
