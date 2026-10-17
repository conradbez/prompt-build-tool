"""
Dependency graph for prompt models.

Loads every *.prompt file under the map/ directory, extracts ref()
dependencies, and validates the graph (cycle detection, unknown refs).

Parsing produces plain :class:`~promptmap.model_spec.ModelSpec` data.  The kind that
will run a model is looked up by name from :mod:`promptmap.model_types` — the graph
knows only that the name is registered.
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import networkx as nx

from promptmap.executor.parser_initial import (
    UnknownConfigKeyWarning,
    detect_used_promptdata,
    extract_dependencies,
    parse_model_config,
    warn_unknown_config_keys,
)
from promptmap import jsonpath
from promptmap.files import split_model_path
from promptmap.model_spec import ModelSpec
from promptmap.model_types import get_model_kind, known_model_kinds

# Supported file extensions, longest first so stripping is unambiguous.
_PROMPT_SUFFIXES = (".prompt.jinja", ".prompt")


#: A leading sequence prefix such as ``3_a_`` only orders files in a folder.
_SEQUENCE_PREFIX = re.compile(r"^\d+_[a-z]_(?=\w)")


def _prompt_name(p: Path) -> str:
    """Return the model name for a prompt file.

    Strips the suffix and any sequence prefix, so ``3_qa/3_a_bom_line.prompt``
    is the model ``bom_line``: files sort in reading order, and reordering
    them never breaks a ``ref()``.
    """
    name = p.stem
    for suffix in _PROMPT_SUFFIXES:
        if p.name.endswith(suffix):
            name = p.name[: -len(suffix)]
            break
    return _SEQUENCE_PREFIX.sub("", name)


# Kinds shipped by promptmap-extra-model-kinds, so the error can say what to install.
_EXTRAS_KINDS = {"python_modal", "agent_modal", "agent37"}


class UnknownModelTypeError(Exception):
    """A model asks for a model_type no installed kind provides."""


def _resolve_model_type(config: dict, name: str, path: object = None) -> str:
    """Return the model_type to use, raising on an unknown one.

    Running an unknown model_type as a plain LLM call would report success
    while the requested behaviour (an agent, a sandbox, files) never happened.
    """
    model_type = config.get("model_type", "")
    if model_type and get_model_kind(model_type) is None:
        where = f"'{name}'" + (f" ({path})" if path else "")
        hint = (
            f'Install it with: pip install "promptmap[extras]".'
            if model_type in _EXTRAS_KINDS
            else f"Register it with promptmap.register_model_kind(promptmap.ModelKind('{model_type}', your_exec_fn))."
        )
        raise UnknownModelTypeError(
            f"Model {where}: unknown model_type '{model_type}'. Known model types: "
            f"{', '.join(sorted(known_model_kinds()))}. {hint}"
        )
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
    depends_on = extract_dependencies(source)
    if config.get("each"):
        # The model a fan-out iterates is a dependency even if the template
        # never ref()s it.
        upstream, _ = jsonpath.parse(config["each"])
        if upstream not in depends_on:
            depends_on.append(upstream)
    return ModelSpec(
        name=name,
        source=source,
        path=path.resolve() if path is not None else Path("<inline>"),
        model_type=_resolve_model_type(config, name, path),
        config=config,
        depends_on=depends_on,
        promptdata_used=detect_used_promptdata(source),
        promptfiles_used=_parse_promptfiles(config),
    )


def _link_promptfile_deps(specs: dict[str, ModelSpec]) -> dict[str, ModelSpec]:
    """Make each model depend on the upstream models its promptfiles name.

    ``{{ config(promptfiles=["logo"]) }}`` attaches the files model ``logo``
    produced, so ``logo`` must run first — exactly as if the template had
    called ``ref('logo')``.  Names that are not models stay run-level
    promptfiles, supplied with ``--promptfile``.
    """
    for name, spec in list(specs.items()):
        extra = []
        for file_name in spec.promptfiles_used:
            upstream = split_model_path(file_name, specs)
            if upstream is None or upstream[0] == name:
                continue
            if upstream[0] not in spec.depends_on and upstream[0] not in extra:
                extra.append(upstream[0])
        if extra:
            specs[name] = spec.derive(depends_on=[*spec.depends_on, *extra])
    return specs


class CyclicDependencyError(Exception):
    pass


class UnknownModelError(Exception):
    pass


def load_models(map_dir: str | Path = "map") -> dict[str, ModelSpec]:
    """
    Discover every *.prompt file in *map_dir* (recursing into subdirectories,
    like dbt) and return a mapping of model_name → ModelSpec.

    The model name is the file stem (e.g. ``article`` for ``sub/article.prompt``).
    Names must be unique across all subdirectories — a clear error is raised
    if two files share the same stem.
    """
    map_dir = Path(map_dir)
    if not map_dir.exists():
        raise FileNotFoundError(
            f"Map directory '{map_dir}' not found. "
            "Create it and add *.prompt files."
        )

    specs: dict[str, ModelSpec] = {}

    prompt_files = sorted(
        {*map_dir.rglob("*.prompt"), *map_dir.rglob("*.prompt.jinja")}
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
        specs[name] = build_spec(name, source, prompt_file)

    if not specs:
        raise FileNotFoundError(
            f"No *.prompt / *.prompt.jinja files found in '{map_dir}'."
        )

    return _link_promptfile_deps(specs)


def build_models_from_dict(models: dict[str, str]) -> dict[str, ModelSpec]:
    """Build a models dict from {name: template_source} without the filesystem."""
    specs = {name: build_spec(name, source) for name, source in models.items()}
    return _link_promptfile_deps(specs)


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
                    f"but '{dep}.prompt' / '{dep}.prompt.jinja' does not exist in the map directory."
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

    Names that refer to an upstream model's files are left out: those are
    produced by the run, not supplied to it.
    """
    seen: dict[str, None] = {}
    for model in models.values():
        for v in model.promptfiles_used:
            upstream = split_model_path(v, models)
            if upstream is not None and upstream[0] != model.name:
                continue
            seen[v] = None
    return list(seen)
