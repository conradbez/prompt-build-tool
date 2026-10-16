"""
Gate tests — tests that guard every ``pbt run``.

Every test in ``tests/`` is a gate unless it opts out with
``{{ config(gate=False) }}``.  ``pbt run`` splits the run in two:

  1. Run the models the gate tests read, and everything upstream of them.
  2. Judge the gate tests against those outputs.
  3. Run the rest of the pipeline only if every gate passed.

A gate that fails or errors fails the run, and nothing downstream of the gates
is called — no LLM spend on a pipeline already known to be broken.  Gate tests
are ordinary tests otherwise: ``pbt test`` runs them like any other.

With ``--select``, a gate applies when every model it reads is part of the
selected run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import networkx as nx

from pbt.executor.executor import ModelRunResult, execute_run
from pbt.executor.graph import UnknownModelError, _parse_promptfiles
from pbt.executor.parser_initial import extract_dependencies, parse_model_config
from pbt.files import split_model_path
from pbt.model_spec import ModelSpec
from pbt.storage.base import StorageBackend
from pbt.tester import TestResult, execute_tests, load_tests


def is_gate(source: str) -> bool:
    """True unless a test opts out with ``{{ config(gate=False) }}``."""
    return parse_model_config(source).get("gate", "true").strip().lower() not in ("false", "0", "no")


def load_gates(tests_dir: str | Path = "tests") -> dict[str, str]:
    """The gate tests in *tests_dir*, as test_name → source."""
    return {name: source for name, source in load_tests(tests_dir).items() if is_gate(source)}


def gate_reads(name: str, source: str, models: dict[str, ModelSpec]) -> list[str]:
    """The models a gate test reads: its ref()s and the models its promptfiles name."""
    reads = extract_dependencies(source)
    for file_name in _parse_promptfiles(parse_model_config(source)):
        upstream = split_model_path(file_name, models)
        if upstream is not None and upstream[0] not in reads:
            reads.append(upstream[0])
    unknown = [m for m in reads if m not in models]
    if unknown:
        raise UnknownModelError(
            f"Gate test '{name}' references {', '.join(unknown)}, which is not a model."
        )
    return reads


def gate_upstream(gates: dict[str, str], models: dict[str, ModelSpec], dag: nx.DiGraph) -> set[str]:
    """Every model the gates need: the ones they read and all of their ancestors."""
    needed: set[str] = set()
    for name, source in gates.items():
        for model in gate_reads(name, source, models):
            needed.add(model)
            needed |= nx.ancestors(dag, model)
    return needed


@dataclass
class GatedRunResult:
    model_results: list[ModelRunResult] = field(default_factory=list)
    gate_results: list[TestResult] = field(default_factory=list)
    #: Models not run because a gate failed.
    blocked: list[str] = field(default_factory=list)

    @property
    def gates_passed(self) -> bool:
        return all(r.status == "pass" for r in self.gate_results)


async def execute_gated_run(
    run_id: str,
    ordered_models: list[ModelSpec],
    gates: dict[str, str],
    gate_models: set[str],
    storage_backend: StorageBackend,
    llm_call: Callable,
    on_gates_start: Callable[[list[ModelSpec]], None] | None = None,
    on_test_start: Callable[[str], None] | None = None,
    on_test_done: Callable[[TestResult], None] | None = None,
    on_gates_done: Callable[[GatedRunResult], None] | None = None,
    judge: str = "llm",
    classify_call: Callable[[str, str], float] | None = None,
    test_promptfiles: dict[str, str] | None = None,
    **run_kwargs,
) -> GatedRunResult:
    """Run *gate_models*, judge *gates*, then the rest of *ordered_models* if all passed.

    *run_kwargs* go to both :func:`~pbt.executor.executor.execute_run` calls
    (callbacks, promptdata, validators, ...).  *test_promptfiles* are the
    run-level promptfiles as paths, which is what tests open.
    """
    first = [m for m in ordered_models if m.name in gate_models]
    rest = [m for m in ordered_models if m.name not in gate_models]
    skipped: set[str] = set()
    skip_downstream: set[str] = set()

    if on_gates_start:
        on_gates_start(first)
    first_results = await execute_run(
        run_id=run_id,
        ordered_models=first,
        storage_backend=storage_backend,
        llm_call=llm_call,
        skipped=skipped,
        skip_downstream=skip_downstream,
        **run_kwargs,
    )
    result = GatedRunResult(model_results=list(first_results))

    result.gate_results = execute_tests(
        run_id=run_id,
        tests=gates,
        model_outputs={
            r.model_name: r.llm_output
            for r in first_results
            if r.status == "success" and r.llm_output
        },
        storage_backend=storage_backend,
        on_test_start=on_test_start,
        on_test_done=on_test_done,
        llm_call=llm_call,
        promptdata=run_kwargs.get("promptdata"),
        promptfiles=test_promptfiles,
        judge=judge,
        classify_call=classify_call,
        secrets=run_kwargs.get("secrets"),
    )
    if not result.gates_passed or any(r.status == "error" for r in first_results):
        result.blocked = [m.name for m in rest]
    if on_gates_done:
        on_gates_done(result)
    if result.blocked:
        return result

    result.model_results += await execute_run(
        run_id=run_id,
        ordered_models=rest,
        storage_backend=storage_backend,
        llm_call=llm_call,
        preloaded_outputs={r.model_name: r.value for r in first_results if r.status == "success"},
        skipped=skipped,
        skip_downstream=skip_downstream,
        **run_kwargs,
    )
    return result
