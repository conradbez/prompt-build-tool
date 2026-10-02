"""Models in stage folders, with sequence prefixes that only order the files."""

from __future__ import annotations

from pathlib import Path

import pytest

from pbt.executor.graph import load_models


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_sequence_prefix_is_not_part_of_the_name(tmp_path: Path) -> None:
    write(tmp_path / "1_design" / "1_a_brief.prompt", "Brief")
    write(tmp_path / "2_build" / "2_a_parts.prompt", "{{ ref('brief') }}")
    write(tmp_path / "3_qa" / "10_b_review.prompt.jinja", "{{ ref('parts') }}")
    write(tmp_path / "1_design" / "2024_report.prompt", "kept as is")

    models = load_models(tmp_path)

    assert set(models) == {"brief", "parts", "review", "2024_report"}
    assert models["parts"].depends_on == ["brief"]


def test_prefix_collision_is_a_duplicate(tmp_path: Path) -> None:
    write(tmp_path / "a" / "1_a_parts.prompt", "x")
    write(tmp_path / "b" / "parts.prompt", "y")
    with pytest.raises(ValueError, match="Duplicate model name 'parts'"):
        load_models(tmp_path)
