"""prompt-build-tool (pbt) — dbt-inspired LLM prompt orchestration."""

from __future__ import annotations

from typing import Callable

__version__ = "0.1.0"


def run(
    models_dir: str = "models",
    select: list[str] | None = None,
    llm_call: Callable[[str], str] | None = None,
    rag_call: Callable[..., list] | None = None,
):
    """
    Execute prompt models as a Python library call.

    Parameters
    ----------
    models_dir:
        Path to the directory containing *.prompt files.
    select:
        Optional list of model names to run. Upstream outputs are loaded
        from the most recent matching run in the DB.
    llm_call:
        Optional function ``(prompt: str) -> str`` to use as the LLM backend.
        Falls back to ``models/client.py`` then the built-in Gemini client.
    rag_call:
        Optional function ``(*args) -> list | str`` to back
        ``return_list_RAG_results()`` in templates.
        Falls back to ``models/rag.py::do_RAG`` if present.

    Returns
    -------
    List of ``ModelRunResult`` (one per model executed).
    """
    import subprocess

    import networkx as nx

    from pbt import db
    from pbt.executor import execute_run, ModelRunResult
    from pbt.graph import (
        load_models,
        execution_order,
        build_dag,
        compute_dag_hash,
    )

    db.init_db()

    all_models = load_models(models_dir)
    ordered = execution_order(all_models)
    dag_hash = compute_dag_hash(all_models)

    preloaded_outputs: dict[str, str] = {}

    if select:
        selected_set = set(select)
        dag = build_dag(all_models)

        upstream_needed: set[str] = set()
        for name in selected_set:
            upstream_needed.update(nx.ancestors(dag, name))
        upstream_needed -= selected_set

        if upstream_needed:
            prev_run = db.get_latest_run_with_dag_hash(dag_hash)
            if prev_run is None:
                raise RuntimeError(
                    f"select= requires a previous run with DAG hash '{dag_hash}'. "
                    "Call pbt.run() without select first."
                )
            preloaded_outputs = db.get_model_outputs_from_run(
                prev_run["run_id"], list(upstream_needed)
            )

        ordered = [m for m in ordered if m.name in selected_set]

    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_sha = None

    run_id = db.create_run(
        model_count=len(ordered),
        dag_hash=dag_hash,
        git_sha=git_sha,
    )

    results = execute_run(
        run_id=run_id,
        ordered_models=ordered,
        models_dir=models_dir,
        preloaded_outputs=preloaded_outputs,
        llm_call=llm_call,
        rag_call=rag_call,
    )

    errors = sum(1 for r in results if r.status == "error")
    successes = sum(1 for r in results if r.status == "success")
    final_status = "success" if errors == 0 else ("partial" if successes > 0 else "error")
    db.finish_run(run_id, final_status)

    return results
