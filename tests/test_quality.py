"""Tests for inject_quality_intermediate_models (quality check DAG expansion)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import pbt
from pbt.executor.graph import build_models_from_dict, execution_order
from pbt.executor.quality import inject_quality_intermediate_models
from pbt.storage import MemoryStorageBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _models(raw: dict[str, str]):
    """Shorthand: build a models dict from template strings."""
    return build_models_from_dict(raw)


def _inject(raw: dict[str, str]):
    """Build + inject in one call."""
    return inject_quality_intermediate_models(_models(raw))


# ---------------------------------------------------------------------------
# Unit tests — transform shape
# ---------------------------------------------------------------------------

class TestBasicExpansion:
    """max_depth=3 should produce two intermediate pairs plus the terminal."""

    TEMPLATES = {
        "topic": "Write about {{ promptdata('subject') }}.",
        "article": "{{ config() }}\nWrite an article about {{ ref('topic') }}.",
        "article_quality": (
            '{{ config(model_type="quality_check", completion_check_of="article",'
            ' max_depth=3, output_format="json") }}\n'
            "Is this article good?\n{{ parent_model }}\n"
            'Return {"pass": true, "feedback": ""}.'
        ),
        "summary": "Summarise: {{ ref('article_quality') }}",
    }

    def test_expected_model_names(self):
        result = _inject(self.TEMPLATES)
        assert set(result) == {
            "topic",
            "article",
            "article_quality_1",
            "article_1",
            "article_quality_2",
            "article_2",
            "article_quality",   # terminal — same name
            "summary",
        }

    def test_intermediate_quality_deps(self):
        result = _inject(self.TEMPLATES)
        assert result["article_quality_1"].depends_on == ["article"]
        assert result["article_quality_2"].depends_on == ["article_1"]

    def test_terminal_quality_deps(self):
        result = _inject(self.TEMPLATES)
        assert result["article_quality"].depends_on == ["article_2"]

    def test_retry_deps_include_quality_and_passing_article(self):
        result = _inject(self.TEMPLATES)
        deps_1 = result["article_1"].depends_on
        assert "article_quality_1" in deps_1
        assert "article" in deps_1

        deps_2 = result["article_2"].depends_on
        assert "article_quality_2" in deps_2
        assert "article_1" in deps_2

    def test_intermediate_quality_forces_json_output(self):
        result = _inject(self.TEMPLATES)
        assert result["article_quality_1"].config["output_format"] == "json"
        assert result["article_quality_2"].config["output_format"] == "json"
        assert result["article_quality"].config["output_format"] == "json"

    def test_model_type_stripped_from_generated_models(self):
        result = _inject(self.TEMPLATES)
        for name in ("article_quality_1", "article_quality_2", "article_quality"):
            assert "model_type" not in result[name].config, name

    def test_max_depth_stripped_from_generated_models(self):
        result = _inject(self.TEMPLATES)
        for name in ("article_quality_1", "article_quality_2", "article_quality"):
            assert "max_depth" not in result[name].config, name

    def test_completion_check_of_stripped(self):
        result = _inject(self.TEMPLATES)
        for name in ("article_quality_1", "article_quality_2", "article_quality"):
            assert "completion_check_of" not in result[name].config, name

    def test_parent_model_substituted_in_quality_source(self):
        result = _inject(self.TEMPLATES)
        assert "parent_model" not in result["article_quality_1"].source
        assert "ref('article')" in result["article_quality_1"].source
        assert "ref('article_1')" in result["article_quality_2"].source
        assert "ref('article_2')" in result["article_quality"].source

    def test_retry_source_contains_skip_logic(self):
        result = _inject(self.TEMPLATES)
        src1 = result["article_1"].source
        assert "ref('article_quality_1')['pass']" in src1
        assert "skip_and_set_to_value" in src1
        assert "ref('article')" in src1

        src2 = result["article_2"].source
        assert "ref('article_quality_2')['pass']" in src2
        assert "ref('article_1')" in src2

    def test_downstream_ref_rewritten(self):
        result = _inject(self.TEMPLATES)
        # summary previously referenced article_quality; should now point at article_2
        assert "ref('article_2')" in result["summary"].source
        assert "article_quality" not in result["summary"].source
        assert "article_2" in result["summary"].depends_on

    def test_source_models_unchanged(self):
        result = _inject(self.TEMPLATES)
        original = _models(self.TEMPLATES)
        assert result["topic"].source == original["topic"].source
        assert result["article"].source == original["article"].source

    def test_dag_is_valid_and_ordered(self):
        result = _inject(self.TEMPLATES)
        order = [m.name for m in execution_order(result)]
        # Each model must appear after all its deps.
        pos = {name: i for i, name in enumerate(order)}
        for model in result.values():
            for dep in model.depends_on:
                assert pos[dep] < pos[model.name], (
                    f"{dep} should run before {model.name}"
                )


class TestMaxDepthOne:
    """max_depth=1 means no retries — only the terminal quality check is updated."""

    TEMPLATES = {
        "article": "{{ config() }}\nWrite something.",
        "article_quality": (
            '{{ config(model_type="quality_check", completion_check_of="article",'
            ' max_depth=1, output_format="json") }}\n'
            "Is this good? {{ parent_model }}\n"
            'Return {"pass": true}.'
        ),
    }

    def test_no_intermediates_injected(self):
        result = _inject(self.TEMPLATES)
        assert set(result) == {"article", "article_quality"}

    def test_terminal_deps_point_to_source(self):
        result = _inject(self.TEMPLATES)
        assert result["article_quality"].depends_on == ["article"]

    def test_model_type_stripped(self):
        result = _inject(self.TEMPLATES)
        assert "model_type" not in result["article_quality"].config


class TestMaxDepthTwo:
    """max_depth=2 should produce exactly one intermediate pair."""

    TEMPLATES = {
        "article": "{{ config() }}\nWrite something.",
        "article_quality": (
            '{{ config(model_type="quality_check", completion_check_of="article",'
            ' max_depth=2, output_format="json") }}\n'
            "Is this good? {{ parent_model }}\n"
            'Return {"pass": true}.'
        ),
    }

    def test_exactly_one_intermediate_pair(self):
        result = _inject(self.TEMPLATES)
        assert set(result) == {"article", "article_quality_1", "article_1", "article_quality"}

    def test_chain_deps(self):
        result = _inject(self.TEMPLATES)
        assert result["article_quality_1"].depends_on == ["article"]
        assert result["article_1"].depends_on == sorted({"article_quality_1", "article"})
        assert result["article_quality"].depends_on == ["article_1"]


class TestDirectRefStyle:
    """User writes ref('article') directly instead of parent_model."""

    TEMPLATES = {
        "article": "{{ config() }}\nWrite something.",
        "article_quality": (
            '{{ config(model_type="quality_check", max_depth=2, output_format="json") }}\n'
            "Check: {{ ref('article') }}\n"
            'Return {"pass": true}.'
        ),
    }

    def test_source_detected_from_depends_on(self):
        result = _inject(self.TEMPLATES)
        # intermediate quality check should reference article_1 via ref(), not article
        assert "ref('article_1')" in result["article_quality"].source
        # original ref('article') should be replaced in terminal
        assert "ref('article')" not in result["article_quality"].source

    def test_intermediate_chain_correct(self):
        result = _inject(self.TEMPLATES)
        assert set(result) == {"article", "article_quality_1", "article_1", "article_quality"}


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

class TestErrorCases:

    def test_multiple_deps_raises(self):
        templates = {
            "a": "{{ config() }}\nA.",
            "b": "{{ config() }}\nB.",
            "checker": (
                '{{ config(model_type="quality_check", output_format="json") }}\n'
                "Check {{ ref('a') }} and {{ ref('b') }}."
            ),
        }
        with pytest.raises(ValueError, match="exactly one source"):
            _inject(templates)

    def test_no_source_raises(self):
        templates = {
            "checker": (
                '{{ config(model_type="quality_check", output_format="json") }}\n'
                "Check this."
            ),
        }
        with pytest.raises(ValueError, match="no source"):
            _inject(templates)

    def test_completion_check_of_nonexistent_raises(self):
        templates = {
            "article": "{{ config() }}\nWrite.",
            "checker": (
                '{{ config(model_type="quality_check",'
                ' completion_check_of="nonexistent", output_format="json") }}\n'
                "Check {{ parent_model }}."
            ),
        }
        with pytest.raises(ValueError, match="nonexistent"):
            _inject(templates)

    def test_max_depth_zero_raises(self):
        templates = {
            "article": "{{ config() }}\nWrite.",
            "checker": (
                '{{ config(model_type="quality_check",'
                ' completion_check_of="article", max_depth=0, output_format="json") }}\n'
                "Check {{ parent_model }}."
            ),
        }
        with pytest.raises(ValueError, match="max_depth must be >= 1"):
            _inject(templates)


# ---------------------------------------------------------------------------
# Integration tests — full pbt.run() with a stub LLM
# ---------------------------------------------------------------------------

def _run_with_llm(templates: dict[str, str], llm_fn) -> dict:
    return pbt.run(
        models_from_dict=templates,
        llm_call=llm_fn,
        verbose=False,
        validation_dir=None,
        storage_backend=MemoryStorageBackend(),
    )


class TestIntegration:
    """End-to-end runs via pbt.run() with an in-memory LLM stub."""

    BASE_TEMPLATES = {
        "topic": "The topic is AI.",
        "article": "{{ config() }}\nWrite an article about {{ ref('topic') }}.",
        "article_quality": (
            '{{ config(model_type="quality_check", completion_check_of="article",'
            ' max_depth=3, output_format="json") }}\n'
            "Is this article good?\n{{ parent_model }}\n"
            'Return {"pass": true/false, "feedback": "..."}.'
        ),
        "summary": "Summarise: {{ ref('article_quality') }}",
    }

    def test_passes_on_first_attempt(self):
        """Quality passes immediately; no retries needed."""
        call_log: list[str] = []

        def llm(prompt: str) -> str:
            call_log.append(prompt)
            if "quality" in prompt.lower() or "good" in prompt.lower():
                return json.dumps({"pass": True, "feedback": ""})
            return "A great article about AI."

        result = _run_with_llm(self.BASE_TEMPLATES, llm)

        assert isinstance(result["summary"], str)
        # summary should contain the article text, not quality JSON
        assert "great article" in result["summary"]
        # article_quality intermediate models should be in result
        assert "article_quality_1" in result

    def test_passes_on_second_attempt(self):
        """Quality fails on first attempt, passes on second.

        The retry article must produce *different* content so that quality_2's
        rendered prompt is distinct from quality_1's — otherwise it would hit
        the cache and return the same fail result.  This mirrors real LLM
        non-determinism (or temperature > 0).
        """
        article_call: list[int] = []
        quality_call: list[int] = []

        def llm(prompt: str) -> str:
            if "Is this article good?" in prompt:
                quality_call.append(1)
                if len(quality_call) == 1:
                    return json.dumps({"pass": False, "feedback": "Too short."})
                return json.dumps({"pass": True, "feedback": ""})
            if "Write an article" in prompt:
                # Return unique content each call so the cache is not hit on retry.
                article_call.append(1)
                return f"Article draft {len(article_call)}: " + "AI content. " * 20
            return "Summary text."

        result = _run_with_llm(self.BASE_TEMPLATES, llm)

        assert isinstance(result["summary"], str)
        # quality was called at least twice — first fail, then pass
        assert len(quality_call) >= 2
        # article was generated more than once (the retry actually ran)
        assert len(article_call) >= 2

    def test_cache_propagation_after_pass(self):
        """Once quality passes, remaining checks should be cache hits (0 ms)."""
        llm_calls: list[str] = []

        def llm(prompt: str) -> str:
            llm_calls.append(prompt)
            if "Is this article good?" in prompt:
                return json.dumps({"pass": True, "feedback": ""})
            return "Article text."

        storage = MemoryStorageBackend()
        pbt.run(
            models_from_dict=self.BASE_TEMPLATES,
            llm_call=llm,
            verbose=False,
            validation_dir=None,
            storage_backend=storage,
        )

        # Count unique quality-check LLM calls (only article_quality_1 should
        # actually call the LLM; article_quality_2 and article_quality should
        # hit the cache with identical rendered prompts).
        quality_calls = [c for c in llm_calls if "Is this article good?" in c]
        # 3 quality checks total; after the first passes, article_1/article_2
        # skip so quality_2/quality_3 see the same content → cache hits.
        # We expect exactly 1 real LLM call for quality (the first one).
        assert len(quality_calls) == 1, (
            f"Expected 1 quality LLM call (cache propagation), got {len(quality_calls)}"
        )
