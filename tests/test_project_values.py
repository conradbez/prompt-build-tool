"""secrets.yml / promptdata.yml / promptfiles.yml defaults, and `promptmap init --secrets|--data|--files`."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from promptmap.cli import main
from promptmap.model_types import _REGISTRY
from promptmap.project_values import ProjectValuesError, load_values

CLIENT_PY = """\
import promptmap

@promptmap.model_kind("needs_yml_secret_test", secrets={"YML_KEY"})
async def needs_secret(rendered, call):
    return "secret=" + ("ok" if call.secrets["YML_KEY"] == "from-yml" else "flag")

def llm_call(prompt, files=None, secrets=None):
    attached = ",".join(f.read().decode() for f in files or [])
    return f"{prompt.strip()}|files={attached}"
"""


@pytest.fixture
def project(tmp_path, monkeypatch):
    (tmp_path / "models").mkdir()
    (tmp_path / "client.py").write_text(CLIENT_PY)
    (tmp_path / "models" / "greet.prompt").write_text(
        '{{ config(promptfiles=["doc"]) }}\n'
        'topic={{ promptdata("topic") }} tone={{ promptdata("tone") }}'
    )
    (tmp_path / "models" / "kind.prompt").write_text('{{ config(model_type="needs_yml_secret_test") }}\nhi')
    (tmp_path / "a.txt").write_text("A")
    (tmp_path / "b.txt").write_text("B")
    monkeypatch.chdir(tmp_path)
    yield tmp_path
    _REGISTRY.pop("needs_yml_secret_test", None)


def _run(*args):
    return CliRunner().invoke(main, ["run", *args])


def test_run_uses_yml_values_when_flags_absent(project):
    (project / "promptdata.yml").write_text("topic: cats\ntone: dry\n")
    (project / "promptfiles.yml").write_text("doc: a.txt\n")
    (project / "secrets.yml").write_text("YML_KEY: from-yml\n")
    result = _run()
    assert result.exit_code == 0, result.output
    assert (project / "outputs" / "greet.md").read_text() == "topic=cats tone=dry|files=A"
    assert (project / "outputs" / "kind.md").read_text() == "secret=ok"
    assert "from-yml" not in result.output


def test_cli_flags_override_yml_key_by_key(project):
    (project / "promptdata.yml").write_text("topic: cats\ntone: dry\n")
    (project / "promptfiles.yml").write_text("doc: a.txt\n")
    (project / "secrets.yml").write_text("YML_KEY: from-yml\n")
    result = _run(
        "--promptdata", "topic=dogs",
        "--promptfile", "doc=b.txt",
        "--secret", "YML_KEY=from-flag",
    )
    assert result.exit_code == 0, result.output
    assert (project / "outputs" / "greet.md").read_text() == "topic=dogs tone=dry|files=B"
    assert (project / "outputs" / "kind.md").read_text() == "secret=flag"


def test_missing_secret_still_fails_with_blank_template_values(project):
    (project / "secrets.yml").write_text('YML_KEY: ""\n')
    result = _run()
    assert result.exit_code == 1
    assert "YML_KEY" in result.output


def test_load_values_skips_blanks_and_rejects_nesting(tmp_path):
    assert load_values(tmp_path / "absent.yml") == {}
    path = tmp_path / "v.yml"
    path.write_text("a: 1\nb: ''\nc:\n")
    assert load_values(path) == {"a": "1"}
    path.write_text("a: [1, 2]\n")
    with pytest.raises(ProjectValuesError, match="single value"):
        load_values(path)
    path.write_text("- a\n")
    with pytest.raises(ProjectValuesError, match="mapping"):
        load_values(path)


def test_bad_secrets_yml_never_echoes_its_contents(project):
    (project / "secrets.yml").write_text("YML_KEY: sk-oops: [\n")
    result = _run()
    assert result.exit_code == 1
    assert "sk-oops" not in result.output


# ---------------------------------------------------------------------------
# promptmap init --secrets | --data | --files
# ---------------------------------------------------------------------------

def _init(kind):
    return CliRunner().invoke(main, ["init", f"--{kind}"])


def test_init_data_and_files_list_model_keys(project):
    assert _init("data").exit_code == 0
    assert yaml.safe_load(Path("promptdata.yml").read_text()) == {"topic": "", "tone": ""}
    assert _init("files").exit_code == 0
    assert yaml.safe_load(Path("promptfiles.yml").read_text()) == {"doc": ""}


def test_init_secrets_writes_template_and_gitignores(project):
    result = _init("secrets")
    assert result.exit_code == 0, result.output
    assert yaml.safe_load(Path("secrets.yml").read_text()) == {"YML_KEY": ""}
    assert yaml.safe_load(Path("secrets-template.yml").read_text()) == {"YML_KEY": ""}
    assert "secrets.yml" in Path(".gitignore").read_text().splitlines()
    _init("secrets")
    assert Path(".gitignore").read_text().count("secrets.yml") == 1


def test_init_keeps_existing_values_and_appends_new_keys(project):
    Path("promptdata.yml").write_text("# mine\ntopic: cats\n")
    result = _init("data")
    assert result.exit_code == 0, result.output
    text = Path("promptdata.yml").read_text()
    assert text.startswith("# mine\ntopic: cats\n")
    assert yaml.safe_load(text) == {"topic": "cats", "tone": ""}


def test_init_project_has_no_values_files_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert CliRunner().invoke(main, ["init", "proj"]).exit_code == 0
    proj = tmp_path / "proj"
    assert (proj / "models").is_dir()
    for name in ("secrets.yml", "secrets-template.yml", "promptdata.yml", "promptfiles.yml", ".gitignore"):
        assert not (proj / name).exists(), name


def test_init_project_with_flags_scaffolds_values_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(main, ["init", "proj", "--secrets", "--data", "--files"])
    assert result.exit_code == 0, result.output
    proj = tmp_path / "proj"
    assert yaml.safe_load((proj / "promptdata.yml").read_text()) == {"topic": ""}
    assert (proj / "promptfiles.yml").is_file()
    assert (proj / "secrets.yml").is_file() and (proj / "secrets-template.yml").is_file()
    assert "secrets.yml" in (proj / ".gitignore").read_text().splitlines()
    assert not (tmp_path / "promptdata.yml").exists()  # written into the project, not cwd
