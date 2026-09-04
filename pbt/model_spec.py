"""
:class:`ModelSpec` — the parsed, inert description of one prompt model.

A spec is pure data: what the parser read out of a .prompt file, with no
execution logic attached.  The strategy that runs it is looked up separately by
``model_type`` (see :mod:`pbt.model_types`), and the run-wide inputs it needs
live on the :class:`~pbt.executor.run_context.RunContext`.

Keeping the three apart is what makes a new model type a single small class
instead of an edit in the parser, the graph, and the executor.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    """One node in the DAG, as parsed from a .prompt file (or an inline string)."""

    name: str
    source: str
    path: Path = Path("<inline>")

    #: The ``config(model_type=...)`` value; "" for a plain LLM call.  Always a
    #: registered type by the time the DAG is built — an unrecognised value is
    #: warned about and reset to "" at parse time.
    model_type: str = ""

    #: Raw ``config()`` block.  Values are strings, as captured from the template.
    config: dict = field(default_factory=dict)

    depends_on: list[str] = field(default_factory=list)
    promptdata_used: list[str] = field(default_factory=list)
    promptfiles_used: list[str] = field(default_factory=list)

    # -- convenience accessors over the stringly-typed config block ----------

    @property
    def output_format(self) -> str:
        return self.config.get("output_format", "text")

    @property
    def output_extension(self) -> str:
        return self.config.get("output_extension", "md")

    def config_int(self, key: str, default: int) -> int:
        """Read an integer config value, falling back to *default* if unusable."""
        try:
            return int(self.config[key])
        except (KeyError, TypeError, ValueError):
            return default

    def derive(self, **changes: Any) -> "ModelSpec":
        """Return a copy of this spec with *changes* applied.

        Used by :meth:`~pbt.model_types.BaseModelType.expand` to build the
        sub-nodes a single declared model expands into.
        """
        return replace(self, **changes)
