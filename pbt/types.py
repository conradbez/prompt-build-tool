"""Shared type aliases for the pbt public API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import IO, Union

# A promptfile value may be a string path, a pathlib.Path, or any binary
# file-like object (open(..., "rb"), io.BytesIO, etc.).
PromptFile = Union[str, "os.PathLike[str]", "IO[bytes]"]


@dataclass(slots=True)
class PromptModelsDict:
    """Mapping of model name → Jinja template source string.

    Example::

        PromptModelsDict(models={
            "topic":   "Write a topic about {{ promptdata('subject') }}.",
            "outline": "{{ config(output_format='json') }}\\nOutline: {{ ref('topic') }}",
        })
    """
    models: dict[str, str]
