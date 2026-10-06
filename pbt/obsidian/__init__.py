"""
Markdown notes as pbt projects — a conversion layer kept apart from the core.

Notes become ``.prompt`` models and links between them (``[[wikilinks]]`` or
``[text](other.md)``) become ``ref()``; the regular pbt commands then run the
result.  The source is an Obsidian vault folder, or a single ``.md`` note and
the notes it links to.  From the CLI use ``pbt obsidian``;
from Python::

    import pbt
    from pbt.obsidian import load_vault

    pbt.run(models_from_dict=load_vault("notes/plan.md"), llm_call=my_llm_call)
"""

from pbt.obsidian.converter import (
    ConvertedNote,
    ObsidianError,
    convert,
    convert_vault,
    load_vault,
    write_models,
)

__all__ = ["ConvertedNote", "ObsidianError", "convert", "convert_vault", "load_vault", "write_models"]
