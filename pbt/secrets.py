"""
:class:`Secrets` — run-scoped credentials that never leak into what pbt keeps.

Pass them with ``pbt.run(secrets={...})`` or ``pbt run --secret NAME=VALUE``.  A
model kind reads them as ``call.secrets["NAME"]``; an ``llm_call`` that
declares a ``secrets`` parameter receives them too.

They are never rendered into a template, never part of a cache key, and never
written to storage or the run report — so rotating a key does not invalidate
the cache, and a stored run holds nothing worth stealing.
"""

from __future__ import annotations

import os
from typing import Iterable, Iterator, Mapping


class MissingSecretsError(ValueError):
    """A run's model kinds require secrets that were not passed to it."""


class Secrets(Mapping[str, str]):
    """Run-scoped credentials. Never rendered, cached, stored or printed."""

    def __init__(self, values: Mapping[str, str] | None = None):
        self._values = {k: v for k, v in (values or {}).items() if v}

    def __getitem__(self, k: str) -> str:
        try:
            return self._values[k]
        except KeyError:
            raise KeyError(
                f"secret {k!r} was not passed to this run "
                f"(secrets={{...}} / --secret {k})"
            ) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    # Names only, never values: a ModelCall or RunContext in a traceback, a
    # print() or a debug log would otherwise carry every credential with it.
    def __repr__(self) -> str:
        return f"Secrets({sorted(self._values)})"

    __str__ = __repr__

    def __reduce__(self):
        raise TypeError("Secrets cannot be pickled")


def parse_secret_flags(specs: Iterable[str], environ: Mapping[str, str] | None = None) -> Secrets:
    """Build :class:`Secrets` from ``--secret`` flags.

    ``NAME=VALUE`` sets the value directly, like ``--promptdata``.  A bare
    ``NAME`` reads environment variable ``NAME`` — handy for keys already in
    the environment or a ``.env`` file, and it keeps them out of shell history.
    Raises ValueError for a bare name whose variable is unset.  Errors never
    echo a value.
    """
    environ = os.environ if environ is None else environ
    values: dict[str, str] = {}
    for spec in specs:
        name, sep, value = spec.partition("=")
        name = name.strip()
        if not name:
            raise ValueError("--secret takes NAME=VALUE or NAME, got an empty name")
        if not sep:
            value = environ.get(name, "")
            if not value:
                raise ValueError(
                    f"--secret {name}: environment variable {name} is not set "
                    f"(pass --secret {name}=VALUE, or set it)"
                )
        values[name] = value
    return Secrets(values)
