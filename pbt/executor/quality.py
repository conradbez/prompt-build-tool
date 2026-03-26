"""
Quality check DAG injection.

Finds models with ``model_type="quality_check"`` and expands them into
chains of retry attempts linked by skip logic.

Example
-------
Given: topic → article → article_quality → summary

With article_quality.prompt::

    {{ config(model_type="quality_check", completion_check_of="article", max_depth=3, output_format="json") }}
    Is this article high quality?
    {{ parent_model }}
    Return {"pass": true, "feedback": "..."}.

_quality_replace_node_callback expands the quality node into::

    topic → article
                  → article_quality_1      (intermediate quality check; outputs JSON)
                  → article_1              (skips if quality_1 passed, else reruns article)
                  → article_quality_2      (intermediate quality check; outputs JSON)
                  → article_2              (skips if quality_2 passed, else reruns article)
                  → article_quality        (terminal pass-through; outputs article_2 content)

    summary still references article_quality and receives article content —
    no rewriting of downstream nodes is required.

Cache propagation
-----------------
Once any quality check passes, the following retry model calls
``skip_and_set_to_value(ref('passing_article'))``.  Its output becomes
identical to the passing article, so every subsequent quality check renders
the same prompt and hits the cache — zero extra LLM calls.
"""

from __future__ import annotations

import re
from pathlib import Path

from pbt.executor.graph import PromptModel
from pbt.executor.model_type_registry import register_replace_node_callback

# Matches the bare `parent_model` identifier in templates.
_PARENT_MODEL_RE = re.compile(r"\bparent_model\b")


def _quality_replace_node_callback(
    qm: PromptModel,
    models: dict[str, PromptModel],
) -> list[PromptModel]:
    """Replace a quality_check node with a chain of intermediate nodes.

    Returns only new nodes owned by this quality model — no other existing
    node in the DAG is modified.  The terminal node keeps the original quality
    model's name and acts as a pass-through that exposes the best article
    content, so downstream models that ref() the quality model receive article
    text without any ref-rewriting of those downstream nodes.

    Parameters
    ----------
    qm:
        The quality_check model to expand.
    models:
        The full models dict (read-only reference for context lookups).

    Returns
    -------
    Ordered list of replacement PromptModel objects.

    Raises
    ------
    ValueError
        If a quality_check model cannot be resolved to exactly one source,
        or if the declared source does not exist in the project.
    """
    source_name = _resolve_source(qm, models)
    source_model = models[source_name]
    max_depth = int(qm.config.get("max_depth", "3"))

    if max_depth < 1:
        raise ValueError(
            f"Quality check model '{qm.name}': max_depth must be >= 1, "
            f"got {max_depth}."
        )

    result: list[PromptModel] = []

    # Build the chain iteratively.
    # For max_depth=1 the loop body never executes — only the terminal
    # pass-through node is created (config keys stripped, source normalised).
    prev_article = source_name

    for i in range(1, max_depth):
        quality_i_name = f"{qm.name}_{i}"
        retry_i_name = f"{source_name}_{i}"

        result.append(_make_quality_model(qm, quality_i_name, prev_article, source_name))
        result.append(_make_retry_model(source_model, retry_i_name, quality_i_name, prev_article))
        prev_article = retry_i_name

    # Terminal node: keeps the original quality model name and acts as a
    # pass-through that outputs the best article content.  Downstream models
    # that ref() this node receive article text, not quality-check JSON —
    # without any modification to those downstream nodes.
    result.append(_make_terminal_passthrough_node(qm, prev_article))

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_source(qm: PromptModel, models: dict[str, PromptModel]) -> str:
    """Return the single source model name for *qm*.

    Checks ``completion_check_of`` in config first, then falls back to the
    single entry in ``depends_on``.

    Raises
    ------
    ValueError
        If the source cannot be determined unambiguously, or does not exist.
    """
    explicit = qm.config.get("completion_check_of")
    if explicit is not None:
        if explicit not in models:
            raise ValueError(
                f"Quality check '{qm.name}': completion_check_of=\"{explicit}\" "
                "does not match any model in the project."
            )
        return explicit

    deps = qm.depends_on
    if len(deps) == 0:
        raise ValueError(
            f"Quality check model '{qm.name}' has no source. "
            "Set completion_check_of=\"model_name\" in {{ config() }}, "
            "or add a ref() call to the template."
        )
    if len(deps) > 1:
        raise ValueError(
            f"Quality check model '{qm.name}' depends on multiple models: "
            f"{deps}. Quality check models must have exactly one source. "
            "Set completion_check_of=\"model_name\" in {{ config() }} to "
            "declare it explicitly."
        )
    return deps[0]


def _make_quality_model(
    qm: PromptModel,
    new_name: str,
    parent_model_name: str,
    original_source_name: str,
) -> PromptModel:
    """Build an intermediate quality check model that checks *parent_model_name*.

    * ``parent_model`` occurrences and direct ``ref('original_source')``
      calls in the source are replaced with ``ref('parent_model_name')``.
    * ``model_type``, ``max_depth``, and ``completion_check_of`` are stripped
      from config — the generated model is a plain model from the executor's
      perspective.
    * ``output_format="json"`` is always set so the ``.pass`` field is
      machine-readable by downstream retry models.
    """
    new_source = _substitute_source_ref(
        qm.source, original_source_name, parent_model_name
    )

    # Preserve any auxiliary deps (e.g. ref('rubric')); replace the source dep.
    other_deps = [d for d in qm.depends_on if d != original_source_name]
    new_deps = sorted(set([parent_model_name] + other_deps))

    new_config = {
        k: v for k, v in qm.config.items()
        if k not in ("model_type", "max_depth", "completion_check_of")
    }
    new_config["output_format"] = "json"

    return PromptModel(
        name=new_name,
        path=qm.path,
        source=new_source,
        depends_on=new_deps,
        config=new_config,
        promptdata_used=qm.promptdata_used,
        promptfiles_used=qm.promptfiles_used,
    )


def _make_terminal_passthrough_node(
    qm: PromptModel,
    last_article_name: str,
) -> PromptModel:
    """Build the terminal node that keeps the original quality model's name.

    This node always skips its LLM call and returns the best article content
    via ``skip_and_set_to_value``.  Downstream models that ref() the original
    quality model name therefore receive article text without any modification
    to those downstream nodes — the owned quality chain is self-contained.
    """
    source = f"{{{{ skip_and_set_to_value(ref('{last_article_name}')) }}}}"
    config = {
        k: v for k, v in qm.config.items()
        if k not in ("model_type", "max_depth", "completion_check_of", "output_format")
    }
    return PromptModel(
        name=qm.name,
        path=qm.path,
        source=source,
        depends_on=[last_article_name],
        config=config,
        promptdata_used=[],
        promptfiles_used=[],
    )


def _make_retry_model(
    source_model: PromptModel,
    new_name: str,
    quality_check_name: str,
    passing_article_name: str,
) -> PromptModel:
    """Build a retry model that skips when the preceding quality check passed.

    If ``ref('quality_check_name')['pass']`` is truthy the model calls
    ``skip_and_set_to_value(ref('passing_article_name'))``, making its output
    identical to the already-passing article.  All subsequent quality checks
    therefore render the same prompt and hit the cache — no wasted LLM calls.

    If the quality check failed, a ``Quality feedback`` prefix is prepended to
    the rendered prompt.  This serves two purposes:

    1. **Cache busting** — the rendered prompt differs from the original
       model's prompt, so a real LLM call is made instead of returning the
       cached (failing) output.
    2. **Guided improvement** — the LLM receives the reviewer's feedback and
       can address specific shortcomings in its next attempt.
    """
    skip_prefix = (
        f"{{% if ref('{quality_check_name}')['pass'] %}}"
        f"{{{{ skip_and_set_to_value(ref('{passing_article_name}')) }}}}"
        f"{{% endif %}}\n"
    )
    # Feedback context: only rendered when quality failed AND feedback is non-empty.
    # Relies on Jinja2 attribute/item access: dict.get() is available on Python dicts.
    feedback_prefix = (
        f"{{% if not ref('{quality_check_name}')['pass']"
        f" and ref('{quality_check_name}').get('feedback') %}}"
        f"Quality feedback from previous attempt: "
        f"{{{{ ref('{quality_check_name}')['feedback'] }}}}\n"
        f"{{% endif %}}\n"
    )
    new_source = skip_prefix + feedback_prefix + source_model.source

    new_deps = sorted(
        set([quality_check_name, passing_article_name] + source_model.depends_on)
    )

    return PromptModel(
        name=new_name,
        path=source_model.path,
        source=new_source,
        depends_on=new_deps,
        config=dict(source_model.config),
        promptdata_used=source_model.promptdata_used,
        promptfiles_used=source_model.promptfiles_used,
    )


def _substitute_source_ref(
    source: str, original_source_name: str, new_name: str
) -> str:
    """Replace ``parent_model`` and ``ref('original_source_name')`` with ``ref('new_name')``.

    Handles both the ``parent_model`` convenience variable and direct
    ``ref('…')`` calls so that user templates can use either style.
    """
    # 1. Replace the parent_model convenience variable.
    result = _PARENT_MODEL_RE.sub(f"ref('{new_name}')", source)
    # 2. Replace any direct ref() call to the original source model.
    result = re.sub(
        r"""\bref\(\s*['"]""" + re.escape(original_source_name) + r"""['"]\s*\)""",
        f"ref('{new_name}')",
        result,
    )
    return result


# Register this module's callback with the global registry so that
# apply_replace_node_callbacks() expands quality_check nodes automatically.
register_replace_node_callback("quality_check", _quality_replace_node_callback)
