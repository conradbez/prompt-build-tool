"""
:class:`Secrets` — run-scoped credentials that never leak into what pbt keeps.

Pass them with ``pbt.run(secrets={...})`` or ``pbt run --secret NAME``.  A
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


def secrets_from_env(specs: Iterable[str], environ: Mapping[str, str] | None = None) -> Secrets:
    """Build :class:`Secrets` from ``--secret`` flags, reading values from the env.

    ``NAME`` reads env var ``NAME``; ``NAME=VAR`` reads env var ``VAR``.  The
    flag only ever names a variable — a key typed on the command line would end
    up in shell history and ``ps``.  Raises ValueError for an unset variable.
    """
    environ = os.environ if environ is None else environ
    values: dict[str, str] = {}
    for spec in specs:
        name, _, var = spec.partition("=")
        name, var = name.strip(), (var or name).strip()
        if not name or not var.isidentifier():
            # Not echoed back: what follows '=' may be the key itself.
            raise ValueError(
                f"--secret {name or '?'}=...: takes NAME or NAME=ENV_VAR, where ENV_VAR "
                "names an environment variable. Never pass the value on the command line."
            )
        value = environ.get(var)
        if not value:
            raise ValueError(f"--secret {name}: environment variable {var} is not set")
        values[name] = value
    return Secrets(values)
