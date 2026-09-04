"""
Compatibility shim for the pre-registry handler classes.

Model data and model behaviour used to live together in one ``BaseModelHandler``
dataclass per model type.  They are now separate:

===============================  ============================================
Old                              New
===============================  ============================================
``BaseModelHandler`` (the data)  :class:`pbt.model_spec.ModelSpec`
``BaseModelHandler`` (the code)  :class:`pbt.model_types.BaseModelType`
``LoopModelHandler``             :class:`pbt.executor.builtin_types.LoopModelType`
``ExecutePythonModelHandler``    :class:`pbt.executor.builtin_types.PythonModelType`
``QualityCheckModelHandler``     :class:`pbt.executor.builtin_types.QualityCheckModelType`
``execute_node``                 ``ModelType.execute(spec, ctx)``
``inject_extra_nodes``           ``ModelType.expand(spec, all_specs)``
``_MODEL_CLASS_MAP[name] = cls`` ``pbt.register_model_type(name, cls)``
===============================  ============================================

The names below keep old imports working.  New code should import from
:mod:`pbt.model_spec` and :mod:`pbt.model_types`.
"""

from __future__ import annotations

from pbt.executor.builtin_types import (
    LLMModelType,
    LoopModelType,
    PythonModelType,
    QualityCheckModelType,
    TemplateModelType,
)
from pbt.executor.run_context import _files_hash, parse_json_output
from pbt.model_spec import ModelSpec

#: Was the per-model dataclass; that role is now ModelSpec.
BaseModelHandler = ModelSpec

LoopModelHandler = LoopModelType
ExecutePythonModelHandler = PythonModelType
QualityCheckModelHandler = QualityCheckModelType

_parse_json_output = parse_json_output

__all__ = [
    "BaseModelHandler",
    "LoopModelHandler",
    "ExecutePythonModelHandler",
    "QualityCheckModelHandler",
    "LLMModelType",
    "TemplateModelType",
    "ModelSpec",
    "parse_json_output",
    "_parse_json_output",
    "_files_hash",
]
