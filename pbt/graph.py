"""
Dependency graph for prompt models.

Loads every *.prompt file under the models/ directory, extracts ref()
dependencies, validates the graph, and returns a topologically-sorted
execution order (leaves first, dependents last) — identical to how dbt
resolves model DAGs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx

from pbt.parser import extract_dependencies


@dataclass
class PromptModel:
    name: str          # stem of the .prompt file, e.g. "summary"
    path: Path         # absolute path to the .prompt file
    source: str        # raw file contents
    depends_on: list[str] = field(default_factory=list)


class CyclicDependencyError(Exception):
    pass


class UnknownModelError(Exception):
    pass


def load_models(models_dir: str | Path = "models") -> dict[str, PromptModel]:
    """
    Discover every *.prompt file in *models_dir* and return a mapping of
    model_name → PromptModel.
    """
    models_dir = Path(models_dir)
    if not models_dir.exists():
        raise FileNotFoundError(
            f"Models directory '{models_dir}' not found. "
            "Create it and add *.prompt files."
        )

    models: dict[str, PromptModel] = {}

    for prompt_file in sorted(models_dir.glob("*.prompt")):
        name = prompt_file.stem
        source = prompt_file.read_text(encoding="utf-8")
        deps = extract_dependencies(source)
        models[name] = PromptModel(
            name=name,
            path=prompt_file.resolve(),
            source=source,
            depends_on=deps,
        )

    if not models:
        raise FileNotFoundError(
            f"No *.prompt files found in '{models_dir}'."
        )

    return models


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
    dag.add_nodes_from(models.keys())

    for model in models.values():
        for dep in model.depends_on:
            if dep not in models:
                raise UnknownModelError(
                    f"Model '{model.name}' references ref('{dep}'), "
                    f"but '{dep}.prompt' does not exist in the models directory."
                )
            # Edge: dep → model  (dep must execute first)
            dag.add_edge(dep, model.name)

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
    """
    dag = build_dag(models)
    sorted_names = list(nx.topological_sort(dag))
    return [models[name] for name in sorted_names]
