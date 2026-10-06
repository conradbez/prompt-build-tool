"""
Tests for promptparams YAML cases — parameterised test runs.

Covers:
- find/load: defaults, combining files and directories, stable order
- baselines: implicit default, extends (string/list/chained), null removal,
  opting out, cross-file baselines, loop and unknown-baseline errors
- names: descriptive names, generated names, duplicates
- promptfile paths resolve relative to the YAML file
- save_case / write_example
- execute_tests cross-join with named cases
- CLI pbt test: per-case runs, --case, inline params with baselines,
  --save-case, the example file, and the CSV migration error
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from pbt.promptparams import (
    PromptParamsError,
    TestCase,
    find_promptparams_files,
    load_cases,
    save_case,
    write_example,
)
from pbt.tester import execute_tests
from pbt.storage.memory import MemoryStorageBackend
from tests.conftest import run_pbt


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture()
def in_tmp(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Finding and combining files
# ---------------------------------------------------------------------------

def test_no_files_means_no_cases(in_tmp: Path) -> None:
    assert load_cases() == []


def test_explicit_missing_path_is_an_error(in_tmp: Path) -> None:
    with pytest.raises(PromptParamsError, match="not found"):
        load_cases(["nope.yml"])


def test_tests_dir_files_are_combined_in_order(in_tmp: Path) -> None:
    write(in_tmp / "tests" / "cases.yml", "cases:\n  - name: root\n")
    write(in_tmp / "tests" / "b.yml", "cases:\n  - name: b\n")
    write(in_tmp / "tests" / "a" / "nested.yaml", "cases:\n  - name: nested\n")
    write(in_tmp / "tests" / "notes.txt", "ignored")
    write(in_tmp / "tests" / "check.prompt", "not yaml")

    assert [c.name for c in load_cases(models={"a"})] == ["nested", "b", "root"]


# ---------------------------------------------------------------------------
# A folder named after a model scopes given/expect to it
# ---------------------------------------------------------------------------

def test_model_folder_puts_given_and_expect_under_the_model(in_tmp: Path) -> None:
    write(in_tmp / "tests" / "fixtures" / "board" / "reversed.yml", """\
baselines:
  reversed_board:
    given: {pads: [1, 2]}
cases:
  - name: Reversed
    extends: reversed_board
    expect: {ok: false}
  - name: Removed
    extends: reversed_board
    given: null
""")
    cases = {c.name: c for c in load_cases(models={"board", "review"})}
    assert cases["Reversed"].given == {"board": {"pads": [1, 2]}}
    assert cases["Reversed"].expect == {"board": {"ok": False}}
    assert cases["Removed"].given == {}  # null drops the inherited board


def test_model_folder_value_can_be_any_yaml(in_tmp: Path) -> None:
    write(in_tmp / "tests" / "research" / "two.yml", "cases:\n  - name: X\n    given: [a, b]\n")
    assert load_cases(models={"research"})[0].given == {"research": ["a", "b"]}


def test_scoped_baseline_combines_with_an_unscoped_one(in_tmp: Path) -> None:
    write(in_tmp / "tests" / "base.yml", "baselines:\n  default:\n    given: {parts: [R1]}\n")
    write(in_tmp / "tests" / "board" / "b.yml", """\
baselines:
  bad_board:
    given: {pads: []}
cases:
  - name: Bad board
    extends: [default, bad_board]
""")
    case = load_cases(models={"parts", "board"})[0]
    assert case.given == {"parts": ["R1"], "board": {"pads": []}}


def test_folder_that_is_not_a_model_is_an_error(in_tmp: Path) -> None:
    write(in_tmp / "tests" / "fixtures" / "x.yml", "cases:\n  - name: X\n")
    with pytest.raises(PromptParamsError, match="folder 'fixtures' is not a model"):
        load_cases(models={"board"})


def test_explicit_file_in_a_subfolder_is_not_scoped(in_tmp: Path) -> None:
    write(in_tmp / "tests" / "fixtures" / "x.yml", "cases:\n  - name: X\n    given: {board: 1}\n")
    assert load_cases(["tests/fixtures/x.yml"])[0].given == {"board": 1}


def test_a_file_named_twice_loads_once(in_tmp: Path) -> None:
    write(in_tmp / "cases" / "one.yml", "cases:\n  - name: one\n")
    assert len(find_promptparams_files(["cases", "cases/one.yml"])) == 1
    assert [c.name for c in load_cases(["cases", "cases/one.yml"])] == ["one"]


def test_empty_file_is_fine(in_tmp: Path) -> None:
    write(in_tmp / "tests" / "cases.yml", "")
    assert load_cases() == []


# ---------------------------------------------------------------------------
# Baselines and inheritance
# ---------------------------------------------------------------------------

BASE = """\
baselines:
  default:
    promptdata:
      tone: formal
      audience: engineers
  casual:
    extends: default
    promptdata:
      tone: casual
  long:
    promptdata:
      length: long
"""


def cases_by_name(in_tmp: Path, cases_yaml: str) -> dict[str, TestCase]:
    write(in_tmp / "tests" / "base.yml", BASE)
    write(in_tmp / "tests" / "cases.yml", cases_yaml)
    return {c.name: c for c in load_cases()}


def test_case_without_extends_inherits_default(in_tmp: Path) -> None:
    cases = cases_by_name(in_tmp, "cases:\n  - name: Plain\n")
    assert cases["Plain"].promptdata == {"tone": "formal", "audience": "engineers"}


def test_case_values_override_the_baseline(in_tmp: Path) -> None:
    cases = cases_by_name(in_tmp, """\
cases:
  - name: Devs
    promptdata:
      audience: developers
""")
    assert cases["Devs"].promptdata == {"tone": "formal", "audience": "developers"}


def test_chained_and_listed_baselines(in_tmp: Path) -> None:
    cases = cases_by_name(in_tmp, """\
cases:
  - name: Casual
    extends: casual
  - name: Casual and long
    extends: [casual, long]
""")
    assert cases["Casual"].promptdata == {"tone": "casual", "audience": "engineers"}
    assert cases["Casual and long"].promptdata == {
        "tone": "casual", "audience": "engineers", "length": "long",
    }


def test_null_removes_an_inherited_key(in_tmp: Path) -> None:
    cases = cases_by_name(in_tmp, """\
cases:
  - name: No audience
    promptdata:
      audience: null
""")
    assert cases["No audience"].promptdata == {"tone": "formal"}


def test_empty_extends_opts_out_of_default(in_tmp: Path) -> None:
    cases = cases_by_name(in_tmp, """\
cases:
  - name: Alone
    extends: []
    promptdata:
      tone: terse
""")
    assert cases["Alone"].promptdata == {"tone": "terse"}


def test_extending_another_baseline_does_not_add_default_twice(in_tmp: Path) -> None:
    cases = cases_by_name(in_tmp, "cases:\n  - name: Long\n    extends: long\n")
    # `long` does not extend default, and naming it replaces the implicit default.
    assert cases["Long"].promptdata == {"length": "long"}


def test_native_yaml_values_are_kept(in_tmp: Path) -> None:
    cases = cases_by_name(in_tmp, """\
cases:
  - name: Rich
    promptdata:
      count: 3
      brief: |
        Two lines
        of brief.
""")
    assert cases["Rich"].promptdata["count"] == 3
    assert cases["Rich"].promptdata["brief"] == "Two lines\nof brief.\n"


def test_empty_baseline_is_allowed(in_tmp: Path) -> None:
    write(in_tmp / "tests" / "cases.yml", "baselines:\n  default:\ncases:\n  - name: X\n")
    assert load_cases()[0].promptdata == {}


def test_unknown_baseline_is_an_error(in_tmp: Path) -> None:
    with pytest.raises(PromptParamsError, match="unknown baseline 'nope'"):
        cases_by_name(in_tmp, "cases:\n  - name: X\n    extends: nope\n")


def test_baseline_loop_is_an_error(in_tmp: Path) -> None:
    write(in_tmp / "tests" / "cases.yml", """\
baselines:
  a: {extends: b}
  b: {extends: a}
cases:
  - {name: X, extends: a}
""")
    with pytest.raises(PromptParamsError, match="loop"):
        load_cases()


def test_baseline_defined_twice_is_an_error(in_tmp: Path) -> None:
    write(in_tmp / "tests" / "a.yml", "baselines:\n  default: {}\n")
    write(in_tmp / "tests" / "b.yml", "baselines:\n  default: {}\n")
    with pytest.raises(PromptParamsError, match="already defined"):
        load_cases()


def test_typos_in_keys_are_errors(in_tmp: Path) -> None:
    write(in_tmp / "tests" / "cases.yml", "cases:\n  - name: X\n    promtdata: {a: 1}\n")
    with pytest.raises(PromptParamsError, match="promtdata"):
        load_cases()


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------

def test_unnamed_cases_get_file_based_names(in_tmp: Path) -> None:
    write(in_tmp / "tests" / "edge.yml", "cases:\n  - promptdata: {a: 1}\n  - promptdata: {a: 2}\n")
    assert [c.name for c in load_cases()] == ["edge_1", "edge_2"]


def test_duplicate_case_names_across_files_are_an_error(in_tmp: Path) -> None:
    write(in_tmp / "tests" / "a.yml", "cases:\n  - name: Same\n")
    write(in_tmp / "tests" / "b.yml", "cases:\n  - name: Same\n")
    with pytest.raises(PromptParamsError, match="already used"):
        load_cases()


# ---------------------------------------------------------------------------
# Promptfile paths
# ---------------------------------------------------------------------------

def test_promptfiles_are_relative_to_the_declaring_file(in_tmp: Path) -> None:
    write(in_tmp / "shared" / "base.yml", """\
baselines:
  default:
    promptfiles:
      doc: docs/report.pdf
""")
    write(in_tmp / "tests" / "cases.yml", """\
cases:
  - name: Default doc
  - name: Two charts
    promptfiles:
      charts: [q1.png, q2.png]
""")
    cases = {c.name: c for c in load_cases(["shared", "tests"])}
    assert cases["Default doc"].promptfiles == {"doc": str(Path("shared/docs/report.pdf"))}
    assert cases["Two charts"].promptfiles["charts"] == [
        str(Path("tests/q1.png")), str(Path("tests/q2.png")),
    ]


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def test_save_case_round_trips_and_keeps_inheriting(in_tmp: Path) -> None:
    write(in_tmp / "tests" / "base.yml", BASE)
    write(in_tmp / "report.pdf", "x")
    path = save_case("tests", "Playful tone!", {"tone": "playful"}, {"doc": "report.pdf"})

    assert path == Path("tests/playful_tone.yml")
    saved = yaml.safe_load(path.read_text())
    assert saved == {"cases": [{
        "name": "Playful tone!",
        "promptdata": {"tone": "playful"},
        "promptfiles": {"doc": "../report.pdf"},
    }]}
    case = {c.name: c for c in load_cases()}["Playful tone!"]
    assert case.promptdata == {"tone": "playful", "audience": "engineers"}
    assert case.promptfiles == {"doc": "report.pdf"}


def test_save_case_refuses_to_overwrite(in_tmp: Path) -> None:
    save_case("tests", "One", {"a": "1"}, {})
    with pytest.raises(PromptParamsError, match="already exists"):
        save_case("tests", "one", {"a": "2"}, {})


def test_write_example_is_valid_yaml_with_a_default_baseline(tmp_path: Path) -> None:
    out = tmp_path / "promptparams.yml.example"
    write_example(out, ["tone", "audience"], ["document"])

    doc = yaml.safe_load(out.read_text())
    assert doc["baselines"]["default"] == {
        "promptdata": {"tone": "<tone>", "audience": "<audience>"},
        "promptfiles": {"document": "<path/to/document>"},
    }
    assert [c["name"] for c in doc["cases"]] == [
        "Baseline inputs", "<describe what this case checks>",
    ]


def test_write_example_no_inputs_does_nothing(tmp_path: Path) -> None:
    out = tmp_path / "promptparams.yml.example"
    write_example(out, [], [])
    assert not out.exists()


# ---------------------------------------------------------------------------
# execute_tests cross-join
# ---------------------------------------------------------------------------

def _stub_llm_pass(prompt: str) -> str:
    return json.dumps({"results": "pass"})


SIMPLE_TEST_SOURCE = """\
Topic is: {{ promptdata("topic") }}
Always respond with exactly this JSON: {"results": "pass"}
"""

NO_PARAMS_TEST_SOURCE = """\
Always respond with exactly this JSON: {"results": "pass"}
"""


def _run_tests(tests, llm_call=_stub_llm_pass, **kwargs):
    storage = MemoryStorageBackend()
    storage.init_db()
    run_id = storage.create_run(model_count=0)
    return execute_tests(
        run_id=run_id,
        tests=tests,
        model_outputs={},
        storage_backend=storage,
        llm_call=llm_call,
        **kwargs,
    )


def test_execute_tests_no_cases() -> None:
    """Without cases, each test runs once with its base name."""
    results = _run_tests({"smoke": NO_PARAMS_TEST_SOURCE})
    assert [(r.test_name, r.status, r.param_label) for r in results] == [("smoke", "pass", "")]


def test_execute_tests_cross_join_uses_case_names() -> None:
    cases = [
        TestCase("Formal tone", {"tone": "formal"}),
        TestCase("Casual [short]", {"tone": "casual"}),
    ]
    results = _run_tests({"alpha": NO_PARAMS_TEST_SOURCE, "beta": NO_PARAMS_TEST_SOURCE}, cases=cases)

    assert {r.test_name for r in results} == {
        "alpha[Formal tone]", "alpha[Casual [short]]",
        "beta[Formal tone]", "beta[Casual [short]]",
    }
    # The label is the case name exactly, brackets and all.
    assert {r.param_label for r in results} == {"Formal tone", "Casual [short]"}
    assert all(r.status == "pass" for r in results)


def test_execute_tests_promptdata_injected() -> None:
    captured: list[str] = []

    def recording_llm(prompt: str) -> str:
        captured.append(prompt)
        return json.dumps({"results": "pass"})

    _run_tests(
        {"pd_test": SIMPLE_TEST_SOURCE},
        llm_call=recording_llm,
        cases=[TestCase("Cats", {"topic": "cats"}), TestCase("Dogs", {"topic": "dogs"})],
    )
    assert any("cats" in p for p in captured)
    assert any("dogs" in p for p in captured)


def test_execute_tests_param_label_suffix() -> None:
    results = _run_tests({"t": NO_PARAMS_TEST_SOURCE}, param_label="Formal tone")
    assert results[0].test_name == "t[Formal tone]"
    assert results[0].param_label == "Formal tone"


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------

SIMPLE_PROMPTPARAMS_CLIENT_PY = """\
import json

def llm_call(prompt: str, config: dict | None = None) -> str:
    if "respond only with valid json" in prompt.lower():
        return json.dumps({"results": "pass"})
    return "Hello"
"""

SIMPLE_TEST_PROMPT_WITH_PROMPTDATA = """\
Tone {{ promptdata("tone") }} for {{ promptdata("audience") }}.
Respond ONLY with valid JSON: {"results": "pass"} if it does, {"results": "fail"} if not.
"""

PROJECT_CASES = """\
baselines:
  default:
    promptdata:
      tone: formal
      audience: engineers
  casual:
    promptdata:
      tone: casual
      audience: everyone

cases:
  - name: Formal for engineers
  - name: Formal for managers
    promptdata:
      audience: managers
"""


@pytest.fixture()
def promptparams_proj(tmp_path: Path) -> Path:
    proj = tmp_path / "pp_proj"
    (proj / "models").mkdir(parents=True)
    (proj / "tests").mkdir()

    (proj / "models" / "greet.prompt").write_text("Say hello in one word.", encoding="utf-8")
    (proj / "tests" / "tone_test.prompt").write_text(SIMPLE_TEST_PROMPT_WITH_PROMPTDATA, encoding="utf-8")
    (proj / "client.py").write_text(SIMPLE_PROMPTPARAMS_CLIENT_PY, encoding="utf-8")
    write(proj / "tests" / "tone.yml", PROJECT_CASES)
    write(proj / "tests" / "more.yml", "cases:\n  - name: Casual\n    extends: casual\n")

    yield proj
    shutil.rmtree(proj)


def output(result) -> str:
    return result.stdout + result.stderr


def test_cli_runs_every_named_case_from_combined_files(promptparams_proj: Path) -> None:
    result = run_pbt("test", cwd=promptparams_proj, check=False)
    out = output(result)
    assert result.returncode == 0, out
    assert "3 cases" in out
    for name in ("Formal for engineers", "Formal for managers", "Casual"):
        assert f"tone_test[{name}]" in out


def test_cli_case_filter(promptparams_proj: Path) -> None:
    result = run_pbt("test", "--case", "formal*", cwd=promptparams_proj, check=False)
    out = output(result)
    assert "2 cases" in out
    assert "tone_test[Casual]" not in out

    missing = run_pbt("test", "--case", "nothing", cwd=promptparams_proj, check=False)
    assert missing.returncode != 0
    assert "no promptparams case matches" in output(missing)


def test_cli_explicit_promptparams_path(promptparams_proj: Path) -> None:
    result = run_pbt("test", "--promptparams", "tests/tone.yml", cwd=promptparams_proj, check=False)
    out = output(result)
    assert "2 cases" in out
    assert "tone_test[Casual]" not in out


def test_cli_inline_params_inherit_default_and_save_case(promptparams_proj: Path) -> None:
    result = run_pbt(
        "test", "--promptdata", "tone=playful", "--save-case", "Playful tone",
        cwd=promptparams_proj, check=False,
    )
    out = output(result)
    assert result.returncode == 0, out
    assert "1 case" in out
    assert "tone=playful, audience=engineers" in out  # the default baseline filled the gap
    assert "tone_test[Playful tone]" in out

    saved = promptparams_proj / "tests" / "playful_tone.yml"
    assert yaml.safe_load(saved.read_text()) == {
        "cases": [{"name": "Playful tone", "promptdata": {"tone": "playful"}}]
    }
    # ...and it joins the next parameterised run.
    assert "4 cases" in output(run_pbt("test", cwd=promptparams_proj, check=False))


def test_cli_inline_extends(promptparams_proj: Path) -> None:
    result = run_pbt(
        "test", "--promptdata", "audience=kids", "--extends", "casual",
        cwd=promptparams_proj, check=False,
    )
    assert "tone=casual, audience=kids" in output(result)


def test_cli_save_case_requires_params(promptparams_proj: Path) -> None:
    result = run_pbt("test", "--save-case", "X", cwd=promptparams_proj, check=False)
    assert result.returncode != 0
    assert not (promptparams_proj / "tests" / "x.yml").exists()


def test_cli_bad_yaml_is_reported(promptparams_proj: Path) -> None:
    write(promptparams_proj / "tests" / "bad.yml", "cases:\n  - extends: nope\n")
    result = run_pbt("test", cwd=promptparams_proj, check=False)
    assert result.returncode != 0
    assert "'nope'" in output(result) and "unknown baseline" in output(result)


def test_cli_writes_yaml_example(promptparams_proj: Path) -> None:
    run_pbt("test", cwd=promptparams_proj, check=False)
    example = promptparams_proj / "promptparams.yml.example"
    doc = yaml.safe_load(example.read_text())
    assert set(doc["baselines"]["default"]["promptdata"]) == {"tone", "audience"}


# ---------------------------------------------------------------------------
# given: pin a model's output for a case
# ---------------------------------------------------------------------------

def test_given_merges_like_promptdata(in_tmp: Path) -> None:
    write(in_tmp / "tests" / "cases.yml", """\
baselines:
  default:
    given: {board: {pads: [12V, DRAIN]}, parts: [R1]}
cases:
  - name: Reversed
    given: {board: {pads: [DRAIN, 12V]}, parts: null}
""")
    (case,) = load_cases()
    assert case.given == {"board": {"pads": ["DRAIN", "12V"]}}


def test_models_for_case_skips_given_and_what_only_feeds_them() -> None:
    from pbt.cli.test import models_for_case
    from pbt.executor.graph import build_models_from_dict

    specs = list(build_models_from_dict({
        "brief": "b",
        "parts": "{{ ref('brief') }}",
        "board": "{{ ref('parts') }}",
        "review": "{{ ref('board') }} {{ ref('brief') }}",
    }).values())
    kept = [m.name for m in models_for_case(specs, {"board": {}})]
    assert kept == ["brief", "review"]  # parts only fed board


GIVEN_CLIENT_PY = '''\
import pathlib

def llm_call(prompt: str) -> str:
    with open(pathlib.Path(__file__).parent / "calls.log", "a") as f:
        f.write(prompt.replace("\\\\n", " ") + "\\\\n")
    return '{"results": "pass"}'
'''


def test_cli_given_pins_a_model_and_reaches_downstream_and_tests(tmp_path: Path) -> None:
    proj = tmp_path / "given_proj"
    write(proj / "models" / "board.prompt", "Design the board.")
    write(proj / "models" / "review.prompt",
          "{{ config(model_type='template') }}D1 pad 1: {{ ref('board').pads[0] }}")
    write(proj / "tests" / "review_flags.prompt",
          "Review: {{ ref('review') }} / board: {{ ref('board').pads | join(',') }}")
    write(proj / "client.py", GIVEN_CLIENT_PY)
    write(proj / "tests" / "cases.yml", """\
cases:
  - name: Reversed flyback diode
    given:
      board: {pads: [MOTOR_DRAIN, 12V]}
""")
    result = run_pbt("test", cwd=proj, check=False)
    out = output(result)
    assert result.returncode == 0, out
    assert "given: board" in out
    calls = (proj / "calls.log").read_text()
    assert "Design the board" not in calls  # the given model never ran
    assert "Review: D1 pad 1: MOTOR_DRAIN / board: MOTOR_DRAIN,12V" in calls


def test_cli_given_unknown_model_errors(tmp_path: Path) -> None:
    proj = tmp_path / "given_bad"
    write(proj / "models" / "a.prompt", "A")
    write(proj / "tests" / "t.prompt", "{{ ref('a') }}")
    write(proj / "client.py", GIVEN_CLIENT_PY)
    write(proj / "tests" / "cases.yml", "cases:\n  - name: X\n    given: {nope: 1}\n")
    result = run_pbt("test", cwd=proj, check=False)
    assert result.returncode != 0
    assert "gives nope, which is not a model" in output(result)


def test_expect_inherits_and_replaces_per_model(in_tmp):
    write(in_tmp / 'tests' / 'cases.yml', '''
baselines:
  default:
    expect: {a: {ok: true}, b: {ok: false}}
cases:
  - name: Override
    expect: {a: {ok: false}, b: null}
''')
    case = load_cases()[0]
    assert case.expect == {'a': {'ok': False}}
    assert not case.promptdata


def test_expect_without_judge_records_missing_output():
    results = _run_tests({}, llm_call=None, expect={'missing': {'ok': True}})
    assert results[0].status == 'fail'
    assert results[0].judge == 'assertion'
    assert 'missing model output' in results[0].error


@pytest.mark.parametrize('expected, exit_code', [('false', 0), ('true', 1)])
def test_cli_yaml_expect_without_prompt_tests(tmp_path, expected, exit_code):
    proj = tmp_path / 'expect_proj'
    write(proj / 'models' / 'board.prompt', 'Design the board.')
    write(proj / 'models' / 'polarity.prompt', '''{{ config(model_type='template') }}
[{"ref": "D1", "built_pads_agree": {{ ref('board').correct | tojson }}},
 {"ref": "Q2", "built_pads_agree": true}]
''')
    write(proj / 'client.py', GIVEN_CLIENT_PY)
    write(proj / 'tests' / 'cases.yml', f'''
cases:
  - name: Diode
    given:
      board: {{correct: false}}
    expect:
      polarity:
        - {{ref: Q2, built_pads_agree: true}}
        - {{ref: D1, built_pads_agree: {expected}}}
''')
    result = run_pbt('test', cwd=proj, check=False)
    assert result.returncode == exit_code, output(result)
    assert 'expect.polarity' in output(result)
    assert not (proj / 'calls.log').exists()
    if exit_code:
        assert 'polarity[ref=D1].built_pads_agree' in output(result)


def test_pcb_cases_have_opposite_expectations():
    cases = load_cases([Path(__file__).parents[1] / 'examples/pcb_design/tests'])
    assert len(cases) == 2
    assert cases[0].expect['polarity'][0]['built_pads_agree'] is False
    assert cases[1].expect['polarity'][0]['built_pads_agree'] is True
    assert cases[0].given['board']['output']['pad_nets'][0]['pads'][0]['net'] == 'MOTOR_DRAIN'
    assert cases[1].given['board']['output']['pad_nets'][0]['pads'][0]['net'] == '12V'


def test_expect_can_run_alongside_prompt_tests():
    results = _run_tests({'smoke': NO_PARAMS_TEST_SOURCE}, cases=[
        TestCase(name='Known', expect={'absent': True})
    ])
    assert [(r.test_name, r.status) for r in results] == [
        ('expect.absent[Known]', 'fail'), ('smoke[Known]', 'pass')
    ]


@pytest.mark.parametrize('expectation, fragment', [
    ('{missing: true}', 'expects unknown model'),
    ('[true]', "'expect' must be a mapping"),
])
def test_cli_rejects_invalid_expectations(tmp_path, expectation, fragment):
    proj = tmp_path / 'invalid_expect'
    write(proj / 'models' / 'a.prompt', "{{ config(model_type='template') }}ok")
    write(proj / 'client.py', GIVEN_CLIENT_PY)
    write(proj / 'tests' / 'cases.yml', f'cases:\n  - name: Invalid\n    expect: {expectation}\n')
    result = run_pbt('test', cwd=proj, check=False)
    assert result.returncode != 0
    assert fragment in " ".join(output(result).split())


def test_cli_model_failure_fails_even_when_expectations_pass(tmp_path):
    proj = tmp_path / 'failed_run'
    write(proj / 'models' / 'good.prompt', "{{ config(model_type='template') }}ok")
    write(proj / 'models' / 'bad.prompt', 'Fail this model')
    write(proj / 'client.py', 'def llm_call(prompt):\n    raise RuntimeError("model unavailable")\n')
    write(proj / 'tests' / 'cases.yml', 'cases:\n  - name: Failure\n    expect: {good: ok}\n')
    result = run_pbt('test', cwd=proj, check=False)
    assert result.returncode != 0
    assert 'expect.good' in output(result)
    assert 'PASS' in output(result)
