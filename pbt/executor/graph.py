"""
Dependency graph for prompt models.

Loads every *.prompt file under the models/ directory, extracts ref()
dependencies, validates the graph, and returns a topologically-sorted
execution order (leaves first, dependents last) — identical to how dbt
resolves model DAGs.
"""

from __future__ import annotations

import hashlib
import heapq
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import networkx as nx

from pbt.executor.parser import extract_dependencies, parse_model_config, detect_used_promptdata

# Supported file extensions, longest first so stripping is unambiguous.
_PROMPT_SUFFIXES = (".prompt.jinja", ".prompt")


def _prompt_name(p: Path) -> str:
    """Return the model name for a prompt file, stripping any known suffix."""
    for suffix in _PROMPT_SUFFIXES:
        if p.name.endswith(suffix):
            return p.name[: -len(suffix)]
    return p.stem


@dataclass
class PromptModel:
    name: str          # stem of the .prompt file, e.g. "summary"
    path: Path         # absolute path to the .prompt file
    source: str        # raw file contents
    depends_on: list[str] = field(default_factory=list)
    config: dict = field(default_factory=dict)   # parsed {{ config(...) }} values
    promptdata_used: list[str] = field(default_factory=list)    # promptdata() keys used
    promptfiles_used: list[str] = field(default_factory=list)  # promptfiles names declared in config


# Sentinel stored as a recursive node's output when validation fails but more
# depths remain.  The next depth node detects this and retries the LLM call.
_RETRY_SENTINEL = "__pbt_needs_retry__"


class CyclicDependencyError(Exception):
    pass


class UnknownModelError(Exception):
    pass


def load_models(models_dir: str | Path = "models") -> dict[str, PromptModel]:
    """
    Discover every *.prompt file in *models_dir* (recursing into subdirectories,
    like dbt) and return a mapping of model_name → PromptModel.

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

    models: dict[str, PromptModel] = {}

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
        promptfiles_used = [
            f.strip()
            for f in config.get("promptfiles", "").split(",")
            if f.strip()
        ]
        models[name] = PromptModel(
            name=name,
            path=prompt_file.resolve(),
            source=source,
            depends_on=deps,
            config=config,
            promptdata_used=promptdata_used,
            promptfiles_used=promptfiles_used,
        )

    if not models:
        raise FileNotFoundError(
            f"No *.prompt / *.prompt.jinja files found in '{models_dir}'."
        )

    return expand_recursive_models(models)


def expand_recursive_models(models: dict[str, PromptModel]) -> dict[str, PromptModel]:
    """
    Unroll models with ``max_depth=N`` into N+1 explicit DAG nodes.

    A model named ``article`` with ``{{ config(max_depth=2) }}`` becomes::

        article_0   (depth 0 — original deps)
        article_1   (depth 1 — original deps + article_0)
        article_2   (depth 2 — original deps + article_1)

    At runtime each node follows this logic:

    * ``article_0``: call LLM + validator as normal.  If validation fails and
      this is not the final depth, store ``_RETRY_SENTINEL`` as output (so the
      model succeeds in the DAG but signals the next depth to retry).
    * ``article_k`` (k > 0): if the previous depth's output is **not** the
      sentinel the validator already passed — skip the LLM call and pass the
      output through unchanged.  Otherwise call LLM + validator normally.
    * ``article_{max_depth}``: the final node; validation failure raises as a
      normal model error (no further depths to fall back to).

    Other models that previously ``ref('article')`` are rewritten to
    ``ref('article_2')`` so downstream deps resolve correctly.
    """
    to_expand: dict[str, int] = {
        name: int(model.config["max_depth"])
        for name, model in models.items()
        if "max_depth" in model.config
    }

    if not to_expand:
        return models

    result: dict[str, PromptModel] = {}

    # Create the expanded depth-nodes for each recursive model.
    for orig_name, max_depth in to_expand.items():
        original = models[orig_name]
        for k in range(max_depth + 1):
            node_name = f"{orig_name}_{k}"
            deps = list(original.depends_on)
            if k > 0:
                deps.append(f"{orig_name}_{k - 1}")

            node_config = {
                **original.config,
                "_recursive_depth": str(k),
                "_recursive_max": str(max_depth),
                "_recursive_base": orig_name,
            }

            result[node_name] = PromptModel(
                name=node_name,
                path=original.path,
                source=original.source,
                depends_on=deps,
                config=node_config,
                promptdata_used=original.promptdata_used,
                promptfiles_used=original.promptfiles_used,
            )

    # Copy all other models, rewriting any ref('orig') → ref('orig_{N}').
    for name, model in models.items():
        if name in to_expand:
            continue  # originals are replaced by their expansions

        new_depends_on = list(model.depends_on)
        new_source = model.source

        for orig_name, max_depth in to_expand.items():
            final_node = f"{orig_name}_{max_depth}"
            # Rewrite depends_on list entry.
            if orig_name in new_depends_on:
                new_depends_on[new_depends_on.index(orig_name)] = final_node
            # Rewrite ref() calls in source (both quote styles).
            for q in ("'", '"'):
                new_source = new_source.replace(
                    f"ref({q}{orig_name}{q})",
                    f"ref({q}{final_node}{q})",
                )

        result[name] = PromptModel(
            name=name,
            path=model.path,
            source=new_source,
            depends_on=new_depends_on,
            config=model.config,
            promptdata_used=model.promptdata_used,
            promptfiles_used=model.promptfiles_used,
        )

    return result


def build_dag(models: dict[str, PromptModel]) -> nx.DiGraph:
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


def execution_order(models: dict[str, PromptModel]) -> list[PromptModel]:
    """
    Return models in topological order — upstream models first, so each
    model's dependencies are satisfied before it runs.

    The sort is deterministic: among models at the same depth, names are
    ordered lexicographically so the execution order never changes unless
    the DAG structure actually changes.
    """
    dag = build_dag(models)
    sorted_names = list(_lex_topo_sort(dag))
    return [models[name] for name in sorted_names]


def _lex_topo_sort(dag: nx.DiGraph) -> Iterator[str]:
    """
    Deterministic topological sort: at each step pick the lexicographically
    smallest ready node. Equivalent to nx.lexicographic_topological_sort
    (added in networkx 3.0) but works with older versions too.
    """
    in_degree = {n: dag.in_degree(n) for n in dag}
    heap: list[str] = [n for n, d in in_degree.items() if d == 0]
    heapq.heapify(heap)
    while heap:
        node = heapq.heappop(heap)
        yield node
        for successor in dag.successors(node):
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                heapq.heappush(heap, successor)


def get_dag_promptdata(models: dict[str, PromptModel]) -> list[str]:
    """
    Return a deduplicated list of all promptdata() keys used across every model
    in the DAG, in first-seen order.
    """
    seen: dict[str, None] = {}
    for model in models.values():
        for v in model.promptdata_used:
            seen[v] = None
    return list(seen)


def get_dag_promptfiles(models: dict[str, PromptModel]) -> list[str]:
    """
    Return a deduplicated list of all promptfile names declared across every
    model in the DAG (via ``{{ config(promptfiles="...") }}``), in first-seen order.
    """
    seen: dict[str, None] = {}
    for model in models.values():
        for v in model.promptfiles_used:
            seen[v] = None
    return list(seen)


def build_models_from_dict(models: dict[str, str]) -> dict[str, PromptModel]:
    """Build a models dict from {name: template_source} without the filesystem."""
    result: dict[str, PromptModel] = {}
    for name, source in models.items():
        deps = extract_dependencies(source)
        config = parse_model_config(source)
        promptdata_used = detect_used_promptdata(source)
        promptfiles_used = [
            f.strip() for f in config.get("promptfiles", "").split(",") if f.strip()
        ]
        result[name] = PromptModel(
            name=name,
            path=Path("<inline>"),
            source=source,
            depends_on=deps,
            config=config,
            promptdata_used=promptdata_used,
            promptfiles_used=promptfiles_used,
        )
    return expand_recursive_models(result)


def compute_dag_hash(models: dict[str, PromptModel]) -> str:
    """
    Return a short, deterministic hash of the DAG structure *and* content —
    i.e. model names, dependency edges, prompt source text, and config.

    The hash changes when:
      - a model is added or removed
      - any dependency edge is added or removed
      - any prompt file content changes
      - any model config block changes
    """
    structure = [
        (name, sorted(model.depends_on), model.source, json.dumps(model.config, sort_keys=True))
        for name, model in sorted(models.items())
    ]
    digest = hashlib.sha256(
        json.dumps(structure, separators=(",", ":")).encode()
    ).hexdigest()
    return digest[:16]


def models_to_json(models: dict[str, PromptModel]) -> str:
    """Serialise a models dict to a JSON string for storage in the dags table."""
    data = [
        {
            "name": m.name,
            "path": str(m.path),
            "source": m.source,
            "depends_on": m.depends_on,
            "config": m.config,
            "promptdata_used": m.promptdata_used,
            "promptfiles_used": m.promptfiles_used,
        }
        for m in sorted(models.values(), key=lambda m: m.name)
    ]
    return json.dumps(data)


def models_from_json(dag_json: str) -> dict[str, PromptModel]:
    """Deserialise a models dict from a JSON string produced by models_to_json()."""
    return {
        item["name"]: PromptModel(
            name=item["name"],
            path=Path(item["path"]),
            source=item["source"],
            depends_on=item["depends_on"],
            config=item["config"],
            promptdata_used=item["promptdata_used"],
            promptfiles_used=item["promptfiles_used"],
        )
        for item in json.loads(dag_json)
    }
