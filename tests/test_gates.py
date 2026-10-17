"""
Gate tests — every test guards ``promptmap run`` unless it sets ``gate=False``.

The stub client logs every call, so each test checks the order things ran in:
the gate's upstream models, then the gate, then the rest — or, when the gate
fails, nothing after it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import run_promptmap

CLIENT_PY = """\
import json
from pathlib import Path

def llm_call(prompt: str) -> str:
    first = prompt.strip().splitlines()[0]
    with open("calls.log", "a") as log:
        log.write(first + "\\n")
    if first.startswith("TEST"):
        return json.dumps({"results": Path("verdict.txt").read_text().strip()})
    return "out of " + first
"""

MODELS = {
    "topic": "MODEL topic",
    "outline": "MODEL outline\n{{ ref('topic') }}",
    "article": "MODEL article\n{{ ref('outline') }}",
    "other": "MODEL other",
}

GATE = "TEST outline_ok\n{{ ref('outline') }}"  # a gate by default


@pytest.fixture()
def proj(tmp_path: Path) -> Path:
    (tmp_path / "models").mkdir()
    (tmp_path / "tests").mkdir()
    for name, source in MODELS.items():
        (tmp_path / "models" / f"{name}.prompt").write_text(source, encoding="utf-8")
    (tmp_path / "tests" / "outline_ok.prompt").write_text(GATE, encoding="utf-8")
    (tmp_path / "client.py").write_text(CLIENT_PY, encoding="utf-8")
    (tmp_path / "verdict.txt").write_text("pass", encoding="utf-8")
    return tmp_path


def calls(proj: Path) -> list[str]:
    return (proj / "calls.log").read_text().splitlines()


def test_gate_runs_after_its_upstream_and_before_the_rest(proj: Path) -> None:
    result = run_promptmap("run", cwd=proj, check=False)

    assert result.returncode == 0, result.stdout + result.stderr
    log = calls(proj)
    assert log[:3] == ["MODEL topic", "MODEL outline", "TEST outline_ok"]
    assert sorted(log[3:]) == ["MODEL article", "MODEL other"]
    assert "Gates passed" in result.stdout


def test_failing_gate_stops_the_pipeline(proj: Path) -> None:
    (proj / "verdict.txt").write_text("fail", encoding="utf-8")

    result = run_promptmap("run", cwd=proj, check=False)

    assert result.returncode == 1
    assert calls(proj) == ["MODEL topic", "MODEL outline", "TEST outline_ok"]
    assert "gate failed: outline_ok" in result.stdout
    assert "Not run: article, other" in result.stdout
    # What did run is still written out, to see why the gate failed.
    assert (proj / "outputs" / "outline.md").exists()
    assert not (proj / "outputs" / "article.md").exists()


def test_gate_false_opts_a_test_out_of_promptmap_run(proj: Path) -> None:
    (proj / "tests" / "outline_ok.prompt").write_text("{{ config(gate=False) }}\n" + GATE, encoding="utf-8")

    result = run_promptmap("run", cwd=proj, check=False)

    assert result.returncode == 0, result.stdout + result.stderr
    assert not any(line.startswith("TEST") for line in calls(proj))


def test_gate_outside_the_selected_models_is_not_applied(proj: Path) -> None:
    (proj / "verdict.txt").write_text("fail", encoding="utf-8")

    result = run_promptmap("run", "--select", "other", cwd=proj, check=False)

    assert result.returncode == 0, result.stdout + result.stderr
    assert calls(proj) == ["MODEL other"]


def test_selecting_downstream_of_a_gate_runs_the_gate_first(proj: Path) -> None:
    result = run_promptmap("run", "--select", "article", cwd=proj, check=False)

    assert result.returncode == 0, result.stdout + result.stderr
    assert calls(proj) == ["MODEL topic", "MODEL outline", "TEST outline_ok", "MODEL article"]


def test_gate_referencing_unknown_model_is_a_dependency_error(proj: Path) -> None:
    (proj / "tests" / "outline_ok.prompt").write_text(GATE.replace("outline')", "nope')"), encoding="utf-8")

    result = run_promptmap("run", cwd=proj, check=False)

    assert result.returncode == 1
    assert "nope" in result.stderr


def test_skip_downstream_carries_past_the_gate(proj: Path) -> None:
    (proj / "models" / "topic.prompt").write_text(
        'MODEL topic\n{{ skip_this_and_downstream("no topic") }}', encoding="utf-8"
    )
    (proj / "tests" / "outline_ok.prompt").write_text(
        "TEST topic_ok\n{{ ref('topic') }}", encoding="utf-8"
    )

    result = run_promptmap("run", cwd=proj, check=False)

    assert result.returncode == 0, result.stdout + result.stderr
    assert calls(proj) == ["TEST topic_ok", "MODEL other"]
    assert "Unresolvable" not in result.stdout
