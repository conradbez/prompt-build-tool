"""
Dependency graph for prompt models.

Loads every *.prompt file under the models/ directory, extracts ref()
dependencies, and validates the graph (cycle detection, unknown refs).
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from pbt.executor.model_constructs import BaseModelHandler, LoopModelHandler, ExecutePythonModelHandler, QualityCheckModelHandler
from pbt.executor.parser_initial import (
    extract_dependencies,
    parse_model_config,
    detect_used_promptdata,
)

# Supported file extensions, longest first so stripping is unambiguous.
_PROMPT_SUFFIXES = (".prompt.jinja", ".prompt")

# Map from model_type string to handler class.
_MODEL_CLASS_MAP: dict[str, type[BaseModelHandler]] = {
    "loop": LoopModelHandler,
    "execute_python": ExecutePythonModelHandler,
    "quality_check": QualityCheckModelHandler,
}


def _prompt_name(p: Path) -> str:
    """Return the model name for a prompt file, stripping any known suffix."""
    for suffix in _PROMPT_SUFFIXES:
        if p.name.endswith(suffix):
            return p.name[: -len(suffix)]
    return p.stem


def _apply_inject(models: dict[str, BaseModelHandler], name: str) -> None:
    """Call inject_extra_nodes on models[name] and apply the result in-place."""
    result = models[name].inject_extra_nodes(models)
    if result is None:
        return
    updated_self, extra_nodes = result
    models[name] = updated_self
    for node in extra_nodes:
        if node.name in models:
            raise ValueError(
                f"inject_extra_nodes for '{name}' produced node '{node.name}' "
                "which conflicts with an existing model name."
            )
        models[node.name] = node


class CyclicDependencyError(Exception):
    pass


class UnknownModelError(Exception):
    pass


def load_models(models_dir: str | Path = "models") -> dict[str, BaseModelHandler]:
    """
    Discover every *.prompt file in *models_dir* (recursing into subdirectories,
    like dbt) and return a mapping of model_name → BaseModelHandler subclass instance.

    The model name is the file stem (e.g. ``article`` for ``sub/article.prompt``).
    Names must be unique across all subdirectories — a clear error is raised
    if two files share the same stem.
    """
    models_dir = Path(models_dir)
    if not models_dir.exists():
        raise FileNotFoundError(
            f"Models directory '{models_dir}' not found. "
            "Create it and add *.prompt files."
        )

    models: dict[str, BaseModelHandler] = {}

    prompt_files = sorted(
        {*models_dir.rglob("*.prompt"), *models_dir.rglob("*.prompt.jinja")}
    )
    for prompt_file in prompt_files:
        name = _prompt_name(prompt_file)
        if name in models:
            raise ValueError(
                f"Duplicate model name '{name}': found in both "
                f"'{models[name].path}' and '{prompt_file.resolve()}'. "
                "Model names must be unique across all subdirectories."
            )
        source = prompt_file.read_text(encoding="utf-8")
        deps = extract_dependencies(source)
        config = parse_model_config(source)
        promptdata_used = detect_used_promptdata(source)
        _pf = config.get("promptfiles", "[]")
        _pf_parsed = json.loads(_pf) if _pf.startswith("[") else [_pf] if _pf else []
        promptfiles_used = _pf_parsed
        model_type = config.get("model_type", "")
        cls = _MODEL_CLASS_MAP.get(model_type, BaseModelHandler)
        models[name] = cls(
            name=name,
            path=prompt_file.resolve(),
            source=source,
            depends_on=deps,
            config=config,
            promptdata_used=promptdata_used,
            promptfiles_used=promptfiles_used,
        )
        _apply_inject(models, name)

    if not models:
        raise FileNotFoundError(
            f"No *.prompt / *.prompt.jinja files found in '{models_dir}'."
        )

    return models


def build_dag(models: dict[str, BaseModelHandler]) -> nx.DiGraph:
    """
    Build a directed acyclic graph where an edge A → B means
    "model A must run before model B" (B depends on A).

    Raises
    ------
    UnknownModelError
        If a ref() points to a model that doesn't exist.
    CyclicDependencyError
        If the graph contains a cycle.
    """
    dag: nx.DiGraph = nx.DiGraph()
    dag.add_nodes_from(sorted(models.keys()))  # sorted for determinism

    for name in sorted(models):               # sorted for determinism
        for dep in sorted(models[name].depends_on):
            if dep not in models:
                raise UnknownModelError(
                    f"Model '{name}' references ref('{dep}'), "
                    f"but '{dep}.prompt' / '{dep}.prompt.jinja' does not exist in the models directory."
                )
            # Edge: dep → model  (dep must execute first)
            dag.add_edge(dep, name)

    if not nx.is_directed_acyclic_graph(dag):
        cycles = list(nx.simple_cycles(dag))
        raise CyclicDependencyError(
            f"Circular dependency detected among prompt models: {cycles}"
        )

    return dag


def get_dag_promptdata(models: dict[str, BaseModelHandler]) -> list[str]:
    """
    Return a deduplicated list of all promptdata() keys used across every model
    in the DAG, in first-seen order.
    """
    seen: dict[str, None] = {}
    for model in models.values():
        for v in model.promptdata_used:
            seen[v] = None
    return list(seen)


def get_dag_promptfiles(models: dict[str, BaseModelHandler]) -> list[str]:
    """
    Return a deduplicated list of all promptfile names declared across every
    model in the DAG (via ``{{ config(promptfiles="...") }}``), in first-seen order.
    """
    seen: dict[str, None] = {}
    for model in models.values():
        for v in model.promptfiles_used:
            seen[v] = None
    return list(seen)


def build_models_from_dict(models: dict[str, str]) -> dict[str, BaseModelHandler]:
    """Build a models dict from {name: template_source} without the filesystem."""
    result: dict[str, BaseModelHandler] = {}
    for name, source in models.items():
        deps = extract_dependencies(source)
        config = parse_model_config(source)
        promptdata_used = detect_used_promptdata(source)
        _pf = config.get("promptfiles", "[]")
        _pf_parsed = json.loads(_pf) if _pf.startswith("[") else [_pf] if _pf else []
        promptfiles_used = _pf_parsed
        model_type = config.get("model_type", "")
        cls = _MODEL_CLASS_MAP.get(model_type, BaseModelHandler)
        result[name] = cls(
            name=name,
            path=Path("<inline>"),
            source=source,
            depends_on=deps,
            config=config,
            promptdata_used=promptdata_used,
            promptfiles_used=promptfiles_used,
        )
        _apply_inject(result, name)
    return result
