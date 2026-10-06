"""
Obsidian vaults as pbt projects — a conversion layer kept apart from the core.

Notes become ``.prompt`` models and ``[[wikilinks]]`` become ``ref()``; the
regular pbt commands then run the result.  From the CLI use ``pbt obsidian``;
from Python::

    import pbt
    from pbt.obsidian import load_vault

    pbt.run(models_from_dict=load_vault("my_vault"), llm_call=my_llm_call)
"""

from pbt.obsidian.converter import (
    ConvertedNote,
    ObsidianError,
    convert_vault,
    load_vault,
    write_models,
)

__all__ = ["ConvertedNote", "ObsidianError", "convert_vault", "load_vault", "write_models"]
