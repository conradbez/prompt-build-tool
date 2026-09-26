"""
Classifier-judged tests: the question above ``---``, the text to judge below,
pass when classify_call's P(yes) reaches the threshold.
"""

from __future__ import annotations

import json
import math
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

import pbt
from pbt.classifier import split_question
from pbt.cli import init_files
from pbt.storage.memory import MemoryStorageBackend
from pbt.tester import execute_tests
from tests.conftest import run_pbt

HAIKU = "old pond\nfrog leaps in\nwater's sound"

CLASSIFIER_TEST = """\
Does this haiku have exactly three lines?
---
{{ ref('haiku') }}
"""


def _run(tests, *, storage=None, **kwargs):
    storage = storage or MemoryStorageBackend()
    storage.init_db()
    run_id = storage.create_run(model_count=0)
    return execute_tests(
        run_id=run_id,
        tests=tests,
        model_outputs={"haiku": HAIKU},
        storage_backend=storage,
        **kwargs,
    )


def _classifier(p: float, calls: list | None = None):
    def classify_call(state: str, question: str) -> float:
        if calls is not None:
            calls.append((state, question))
        return p
    return classify_call


# ---------------------------------------------------------------------------
# split_question
# ---------------------------------------------------------------------------

def test_split_question_at_first_separator() -> None:
    question, state = split_question("Is it short?\n---\nline one\n---\nline two\n")
    assert question == "Is it short?"
    assert state == "line one\n---\nline two"


@pytest.mark.parametrize("text", ["no separator here", "---\nstate only", "question only\n---\n"])
def test_split_question_rejects_malformed(text: str) -> None:
    with pytest.raises(ValueError):
        split_question(text)


# ---------------------------------------------------------------------------
# execute_tests with judge="classifier"
# ---------------------------------------------------------------------------

def test_classifier_receives_question_and_state() -> None:
    calls: list = []
    results = _run({"lines": CLASSIFIER_TEST}, judge="classifier", classify_call=_classifier(0.9, calls))
    assert calls == [(HAIKU, "Does this haiku have exactly three lines?")]
    assert [(r.status, r.judge, r.score) for r in results] == [("pass", "classifier", 0.9)]


def test_classifier_below_threshold_fails() -> None:
    results = _run({"lines": CLASSIFIER_TEST}, judge="classifier", classify_call=_classifier(0.3))
    assert [(r.status, r.score) for r in results] == [("fail", 0.3)]


def test_classifier_threshold_from_config() -> None:
    source = "{{ config(threshold=0.95) }}\n" + CLASSIFIER_TEST
    results = _run({"lines": source}, judge="classifier", classify_call=_classifier(0.9))
    assert results[0].status == "fail"


def test_classifier_without_separator_errors() -> None:
    results = _run(
        {"lines": "Does this have three lines? {{ ref('haiku') }}"},
        judge="classifier",
        classify_call=_classifier(0.9),
    )
    assert results[0].status == "error"
    assert "---" in results[0].error


def test_classifier_without_classify_call_errors() -> None:
    results = _run({"lines": CLASSIFIER_TEST}, judge="classifier")
    assert results[0].status == "error"
    assert "classify_call" in results[0].error


def test_classifier_rejects_promptfiles() -> None:
    source = '{{ config(promptfiles=["haiku"]) }}\n' + CLASSIFIER_TEST
    results = _run({"lines": source}, judge="classifier", classify_call=_classifier(0.9))
    assert results[0].status == "error"
    assert "promptfiles" in results[0].error


def test_classifier_score_is_cached_and_threshold_reapplied() -> None:
    storage = MemoryStorageBackend()
    calls: list = []
    _run({"lines": CLASSIFIER_TEST}, storage=storage, judge="classifier", classify_call=_classifier(0.7, calls))

    stricter = "{{ config(threshold=0.8) }}\n" + CLASSIFIER_TEST
    results = _run({"lines": stricter}, storage=storage, judge="classifier", classify_call=_classifier(0.7, calls))
    assert len(calls) == 1
    assert [(r.status, r.score) for r in results] == [("fail", 0.7)]


# ---------------------------------------------------------------------------
# Choosing the judge
# ---------------------------------------------------------------------------

def test_llm_judge_gets_separator_verbatim() -> None:
    prompts: list[str] = []

    def llm(prompt: str) -> str:
        prompts.append(prompt)
        return json.dumps({"results": "pass"})

    results = _run({"lines": CLASSIFIER_TEST}, llm_call=llm)
    assert results[0].status == "pass"
    assert prompts == [f"Does this haiku have exactly three lines?\n---\n{HAIKU}\n"]


def test_test_config_overrides_run_judge() -> None:
    llm_test = '{{ config(judge="llm") }}\nRespond {"results": "pass"}'
    classifier_test = '{{ config(judge="classifier") }}\n' + CLASSIFIER_TEST
    llm = lambda prompt: json.dumps({"results": "pass"})  # noqa: E731

    results = _run(
        {"a_llm": llm_test, "b_classifier": classifier_test},
        judge="llm",
        llm_call=llm,
        classify_call=_classifier(0.9),
    )
    assert [(r.test_name, r.judge, r.status) for r in results] == [
        ("a_llm", "llm", "pass"),
        ("b_classifier", "classifier", "pass"),
    ]

    results = _run({"a_llm": llm_test}, judge="classifier", llm_call=llm, classify_call=_classifier(0.1))
    assert [(r.judge, r.status) for r in results] == [("llm", "pass")]


def test_unknown_judge_in_config_errors() -> None:
    results = _run({"t": '{{ config(judge="vibes") }}\nhi'}, llm_call=lambda p: "")
    assert results[0].status == "error"
    assert "judge must be one of" in results[0].error


# ---------------------------------------------------------------------------
# systemone_classifier — the /v1/systemone wire format
# ---------------------------------------------------------------------------

@pytest.fixture()
def systemone_server():
    seen: list = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, self.headers.get("Authorization"), body))
            payload = json.dumps({
                "model": body["model"],
                "answers": {"pass": {"type": "noul", "noul": 0.87}},
                "usage": {"input_tokens": 10, "output_tokens": 0},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", seen
    server.shutdown()


def test_systemone_classifier_request_shape(systemone_server) -> None:
    url, seen = systemone_server
    classify = pbt.systemone_classifier(model="laya:en", base_url=url, api_key="k")
    assert classify("the text", "Is it good?") == 0.87
    assert seen == [(
        "/v1/systemone",
        "Bearer k",
        {
            "model": "laya:en",
            "state": "the text",
            "questions": {"pass": {"type": "noul", "instructions": "Is it good?"}},
        },
    )]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

CLIENT_PY = """\
import json
import math

def llm_call(prompt: str, config: dict | None = None) -> str:
    if "---" in prompt:
        return json.dumps({"results": "fail"})
    return "old pond\\nfrog leaps in\\nwater's sound"

def classify_call(state: str, question: str) -> float:
    return 0.9 if state.count("\\n") == 2 else 0.1
"""


@pytest.fixture()
def classifier_proj(tmp_path: Path) -> Path:
    proj = tmp_path / "proj"
    (proj / "models").mkdir(parents=True)
    (proj / "tests").mkdir()
    (proj / "models" / "haiku.prompt").write_text("Write a haiku.", encoding="utf-8")
    (proj / "tests" / "three_lines.prompt").write_text(CLASSIFIER_TEST, encoding="utf-8")
    (proj / "client.py").write_text(CLIENT_PY, encoding="utf-8")
    run_pbt("run", cwd=proj)
    return proj


def test_cli_judge_flag(classifier_proj: Path) -> None:
    result = run_pbt("test", "--judge", "classifier", cwd=classifier_proj, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "P(yes)=0.90" in result.stdout

    # Same file, LLM judge: the stub LLM fails anything containing '---'.
    result = run_pbt("test", "--judge", "llm", cwd=classifier_proj, check=False)
    assert result.returncode == 1


def test_cli_test_judge_from_client(classifier_proj: Path) -> None:
    client = classifier_proj / "client.py"
    client.write_text(CLIENT_PY + 'test_judge = "classifier"\n', encoding="utf-8")
    result = run_pbt("test", cwd=classifier_proj, check=False)
    assert result.returncode == 0, result.stdout + result.stderr



# ---------------------------------------------------------------------------
# Scaffolded client.py — `pbt init --classifier`
# ---------------------------------------------------------------------------

def _fake_sdk_modules(completion):
    """Stub the provider SDKs so a scaffolded client.py runs without network."""
    import types

    create = lambda **kwargs: (seen.append(kwargs), completion)[1]  # noqa: E731
    seen: list = []
    chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
    openai = types.ModuleType("openai")
    openai.OpenAI = lambda **kwargs: types.SimpleNamespace(chat=chat)
    return {"openai": openai}, seen


@pytest.mark.parametrize("provider", init_files.PROVIDERS)
def test_scaffolded_clients_compile(provider) -> None:

    compile(init_files.CLIENT_PY[provider], "client.py", "exec")
    source = init_files.CLIENT_PY[provider] + init_files.CLASSIFIER_PY[provider]
    compile(source, "client.py", "exec")
    assert "classify_call = classifier_for_test_feedback" in source


def test_init_classifier_is_opt_in(tmp_path: Path) -> None:
    run_pbt("init", "plain", cwd=tmp_path)
    run_pbt("init", "judged", "--provider", "deepseek", "--classifier", cwd=tmp_path)
    assert "classify_call" not in (tmp_path / "plain" / "client.py").read_text()
    assert "classify_call" in (tmp_path / "judged" / "client.py").read_text()


@pytest.mark.parametrize(("provider", "completion", "expected"), [
    ("deepseek", {"logprobs": [("Yes", 0.6), (" no", 0.2), ("Maybe", 0.1)]}, 0.75),
    ("kimi", {"content": "No."}, 0.0),
    ("xiaomi", {"content": "yes, it does"}, 1.0),
])
def test_openai_compatible_classifier(provider, completion, expected, monkeypatch) -> None:
    import sys
    import types


    top = [types.SimpleNamespace(token=t, logprob=math.log(p)) for t, p in completion.get("logprobs", [])]
    choice = types.SimpleNamespace(
        message=types.SimpleNamespace(content=completion.get("content")),
        logprobs=types.SimpleNamespace(content=[types.SimpleNamespace(top_logprobs=top)]),
    )
    modules, seen = _fake_sdk_modules(types.SimpleNamespace(choices=[choice]))
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv(init_files.OPENAI_COMPATIBLE[provider]["key_env"], "k")

    namespace: dict = {}
    exec(init_files.CLIENT_PY[provider] + init_files.CLASSIFIER_PY[provider], namespace)
    assert namespace["classify_call"]("the text", "Is it good?") == pytest.approx(expected)
    assert seen[0]["model"] == init_files.OPENAI_COMPATIBLE[provider]["model"]
    assert ("logprobs" in seen[0]) == ("logprobs" in completion)
