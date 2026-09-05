"""
Dependency graph for prompt models.

Loads every *.prompt file under the models/ directory, extracts ref()
dependencies, and validates the graph (cycle detection, unknown refs).

Parsing produces plain :class:`~pbt.model_spec.ModelSpec` data.  The kind that
will run a model is looked up by name from :mod:`pbt.model_types` — the graph
knows only that the name is registered, and whether it rewrites its own node.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import networkx as nx

from pbt.executor.parser_initial import (
    UnknownConfigKeyWarning,
    detect_used_promptdata,
    extract_dependencies,
    parse_model_config,
    warn_unknown_config_keys,
)
from pbt.model_spec import ModelSpec
from pbt.model_types import get_model_kind, known_model_kinds

# Supported file extensions, longest first so stripping is unambiguous.
_PROMPT_SUFFIXES = (".prompt.jinja", ".prompt")


def _prompt_name(p: Path) -> str:
    """Return the model name for a prompt file, stripping any known suffix."""
    for suffix in _PROMPT_SUFFIXES:
        if p.name.endswith(suffix):
            return p.name[: -len(suffix)]
    return p.stem


def _resolve_model_type(config: dict, name: str, path: object = None) -> str:
    """Return the model_type to use, warning and falling back on an unknown one.

    An unknown model_type would otherwise run as a plain LLM call with no
    indication that the requested behaviour never happened.
    """
    model_type = config.get("model_type", "")
    if model_type and get_model_kind(model_type) is None:
        where = f"'{name}'" + (f" ({path})" if path else "")
        warnings.warn(
            f"Model {where}: unknown model_type '{model_type}' — running it as a "
            f"normal LLM call. Known model types: "
            f"{', '.join(sorted(known_model_kinds()))}. Register your own with "
            f"pbt.register_model_kind(pbt.ModelKind('{model_type}', your_exec_fn)).",
            UnknownConfigKeyWarning,
            stacklevel=3,
        )
        return ""
    return model_type


def _parse_promptfiles(config: dict) -> list[str]:
    raw = config.get("promptfiles", "[]")
    if raw.startswith("["):
        return json.loads(raw)
    return [raw] if raw else []


def build_spec(name: str, source: str, path: Path | None = None) -> ModelSpec:
    """Parse one template into a :class:`ModelSpec`, warning about bad config."""
    config = parse_model_config(source)
    warn_unknown_config_keys(config, name, path)
    return ModelSpec(
        name=name,
        source=source,
        path=path.resolve() if path is not None else Path("<inline>"),
        model_type=_resolve_model_type(config, name, path),
        config=config,
        depends_on=extract_dependencies(source),
        promptdata_used=detect_used_promptdata(source),
        promptfiles_used=_parse_promptfiles(config),
    )


def _add_spec(specs: dict[str, ModelSpec], spec: ModelSpec) -> None:
    """Add *spec* to *specs*, letting its model kind expand it into more nodes.

    Expansion happens here, before the DAG is built, so the rest of pbt only
    ever sees ordinary nodes.  A kind that expands must return exactly one node
    keeping the declared name, so downstream ``ref()`` calls still resolve.
    """
    kind = get_model_kind(spec.model_type)
    expand_fn = kind.expand_fn if kind is not None else None
    replacements = expand_fn(spec, specs) if expand_fn is not None else None

    if replacements is None:
        specs[spec.name] = spec
        return

    if not any(node.name == spec.name for node in replacements):
        raise ValueError(
            f"Model kind '{spec.model_type}' expanded '{spec.name}' without "
            f"producing a node of that name, so downstream ref('{spec.name}') "
            "calls would break."
        )
    for node in replacements:
        if node.name != spec.name and node.name in specs:
            raise ValueError(
                f"Expanding '{spec.name}' produced node '{node.name}' which "
                "conflicts with an existing model name."
            )
        specs[node.name] = node


class CyclicDependencyError(Exception):
    pass


class UnknownModelError(Exception):
    pass


def load_models(models_dir: str | Path = "models") -> dict[str, ModelSpec]:
    """
    Discover every *.prompt file in *models_dir* (recursing into subdirectories,
    like dbt) and return a mapping of model_name → ModelSpec.

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

    specs: dict[str, ModelSpec] = {}

    prompt_files = sorted(
        {*models_dir.rglob("*.prompt"), *models_dir.rglob("*.prompt.jinja")}
    )
    for prompt_file in prompt_files:
        name = _prompt_name(prompt_file)
        if name in specs:
            raise ValueError(
                f"Duplicate model name '{name}': found in both "
                f"'{specs[name].path}' and '{prompt_file.resolve()}'. "
                "Model names must be unique across all subdirectories."
            )
        source = prompt_file.read_text(encoding="utf-8")
        _add_spec(specs, build_spec(name, source, prompt_file))

    if not specs:
        raise FileNotFoundError(
            f"No *.prompt / *.prompt.jinja files found in '{models_dir}'."
        )

    return specs


def build_models_from_dict(models: dict[str, str]) -> dict[str, ModelSpec]:
    """Build a models dict from {name: template_source} without the filesystem."""
    specs: dict[str, ModelSpec] = {}
    for name, source in models.items():
        _add_spec(specs, build_spec(name, source))
    return specs


def build_dag(models: dict[str, ModelSpec]) -> nx.DiGraph:
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


def get_dag_promptdata(models: dict[str, ModelSpec]) -> list[str]:
    """
    Return a deduplicated list of all promptdata() keys used across every model
    in the DAG, in first-seen order.
    """
    seen: dict[str, None] = {}
    for model in models.values():
        for v in model.promptdata_used:
            seen[v] = None
    return list(seen)


def get_dag_promptfiles(models: dict[str, ModelSpec]) -> list[str]:
    """
    Return a deduplicated list of all promptfile names declared across every
    model in the DAG (via ``{{ config(promptfiles="...") }}``), in first-seen order.
    """
    seen: dict[str, None] = {}
    for model in models.values():
        for v in model.promptfiles_used:
            seen[v] = None
    return list(seen)
