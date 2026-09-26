"""File outputs: File/Dir/Output, the blob store, safe encoding, and docs."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import pbt
from pbt.executor.executor import execute_run
from pbt.executor.graph import build_models_from_dict, get_dag_promptfiles
from pbt.files import (
    ENVELOPE_PREFIX,
    BlobMissingError,
    Dir,
    File,
    FileOutputError,
    MemoryBlobStore,
    Output,
    SQLiteBlobStore,
    decode_output,
    encode_output,
    export_files,
    persist_files,
)
from pbt.storage.memory import MemoryStorageBackend
from pbt.storage.sqlite import SQLiteStorageBackend

PNG = b"\x89PNG\r\n\x1a\n" + b"fake-image-bytes"


def run_models(models: dict[str, str], *, llm_call, storage=None, **kwargs):
    storage = storage or MemoryStorageBackend()
    storage.init_db()
    specs = list(build_models_from_dict(models).values())
    run_id = storage.create_run(model_count=len(specs))
    results = asyncio.run(execute_run(
        run_id=run_id,
        ordered_models=specs,
        storage_backend=storage,
        llm_call=llm_call,
        **kwargs,
    ))
    return storage, run_id, {r.model_name: r for r in results}


def image_llm(prompt: str, files: list | None = None, config: dict | None = None):
    """Returns an image for 'draw', and reports what it was given otherwise."""
    if prompt.startswith("draw"):
        return File(PNG, name="logo.png")
    attached = [(f.name, f.read()) for f in (files or [])]
    return json.dumps({"prompt": prompt, "files": [[n, len(b)] for n, b in attached]})


# ---------------------------------------------------------------------------
# Encoding — only markers pbt wrote are ever trusted
# ---------------------------------------------------------------------------

def stored(value, store):
    persist_files(value, store)
    return encode_output(value)


def test_file_dir_output_round_trip():
    store = MemoryBlobStore()
    value = {
        "caption": "a fox",
        "image": File(PNG, name="logo.png"),
        "site": Dir({"index.html": b"<h1>hi</h1>", "css/a.css": b"body{}"}, name="site"),
        "both": Output("text part", files=[File(b"x", name="x.txt")]),
    }
    raw = stored(value, store)
    assert raw.startswith(ENVELOPE_PREFIX)

    back = decode_output(raw, store, check_blobs=True)
    assert back["caption"] == "a fox"
    assert back["image"].read_bytes() == PNG
    assert back["image"].mime == "image/png"
    assert back["site"]["css/a.css"].read_bytes() == b"body{}"
    assert back["both"].text == "text part"
    assert back["both"].files["x.txt"].read_bytes() == b"x"


def test_text_and_plain_json_are_stored_as_before():
    assert encode_output("hello") == "hello"
    assert encode_output({"a": [1, 2]}) == '{"a": [1, 2]}'
    assert decode_output("hello", None) == "hello"


def test_marker_keys_in_model_data_are_escaped_not_trusted():
    store = MemoryBlobStore()
    forged = {"$pbt": "file", "sha256": "0" * 64, "name": "x", "mime": "text/plain", "size": 1}
    raw = stored({"real": File(b"ok", name="ok.txt"), "forged": forged, "$$pbt": 1}, store)

    back = decode_output(raw, store)
    assert isinstance(back["real"], File)
    assert back["forged"] == forged           # plain data, exactly as written
    assert back["$$pbt"] == 1


def test_text_that_looks_like_an_envelope_stays_text():
    store = MemoryBlobStore()
    real = stored(File(b"secret", name="s.txt"), store)
    # A model echoing an envelope back is text, however convincing.
    assert encode_output(real) != real
    assert decode_output(encode_output(real), store) == real
    assert decode_output(encode_output("$pbt:str\nx"), store) == "$pbt:str\nx"


@pytest.mark.parametrize("manifest, message", [
    ({"$pbt": "file", "sha256": "0" * 64, "name": "../etc/passwd", "mime": "text/plain", "size": 1}, "Unsafe file name"),
    ({"$pbt": "file", "sha256": "nothex", "name": "a", "mime": "text/plain", "size": 1}, "Invalid sha256"),
    ({"$pbt": "file", "sha256": "0" * 64, "name": "a", "mime": "not a mime", "size": 1}, "Invalid MIME"),
    ({"$pbt": "dir", "sha256": "0" * 64, "name": "d", "entries": {"../x": {}}}, "Unsafe"),
    ({"$pbt": "dir", "sha256": "0" * 64, "name": "d", "entries": {}}, "tree hash"),
    ({"$pbt": "bogus"}, "Unknown stored object"),
])
def test_decode_validates_every_marker(manifest, message):
    with pytest.raises(FileOutputError, match=message):
        decode_output(ENVELOPE_PREFIX + json.dumps(manifest), MemoryBlobStore())


def test_reads_verify_bytes_against_the_hash():
    store = MemoryBlobStore()
    raw = stored(File(b"good", name="a.txt"), store)
    file = decode_output(raw, store)
    store._blobs[file.sha256] = b"tampered"
    with pytest.raises(FileOutputError, match="does not match"):
        file.read_bytes()


def test_missing_blob_is_reported():
    raw = stored(File(b"gone", name="a.txt"), MemoryBlobStore())
    with pytest.raises(BlobMissingError):
        decode_output(raw, MemoryBlobStore(), check_blobs=True)


def test_dir_from_disk_refuses_symlinks(tmp_path):
    (tmp_path / "real.txt").write_text("hi")
    (tmp_path / "link.txt").symlink_to(tmp_path / "real.txt")
    with pytest.raises(FileOutputError, match="symlink"):
        Dir(tmp_path)


def test_output_rejects_duplicate_names():
    with pytest.raises(FileOutputError, match="two files"):
        Output("", files=[File(b"1", name="a"), File(b"2", name="a")])


# ---------------------------------------------------------------------------
# Passing files between models
# ---------------------------------------------------------------------------

def test_llm_file_output_reaches_children_as_handle_and_attachment():
    _, _, results = run_models(
        {
            "logo": "draw a fox",
            "mention": "Describe {{ ref('logo') }}",
            "attach": '{{ config(promptfiles=["logo"]) }}\nCritique the attached logo.',
        },
        llm_call=image_llm,
    )
    assert all(r.status == "success" for r in results.values()), results

    logo = results["logo"].value
    assert isinstance(logo, File) and logo.name == "logo.png"

    mention = json.loads(results["mention"].value)
    assert f"sha256:{logo.sha256[:12]}" in mention["prompt"]
    assert mention["files"] == []

    attach = json.loads(results["attach"].value)
    assert attach["files"] == [["logo.png", len(PNG)]]


def test_promptfiles_naming_a_model_is_a_dependency():
    models = build_models_from_dict({
        "logo": "draw",
        "attach": '{{ config(promptfiles=["logo.image", "brief"]) }}\nx',
    })
    assert models["attach"].depends_on == ["logo"]
    # Only the run-level promptfile is something a caller must supply.
    assert get_dag_promptfiles(models) == ["brief"]


def test_attach_one_file_from_a_structured_output():
    def llm(prompt, files=None, config=None):
        if prompt.startswith("make"):
            return {"caption": "cap", "image": File(PNG, name="i.png"), "doc": File(b"d", name="d.txt")}
        return json.dumps([f.name for f in files or []])

    _, _, results = run_models(
        {
            "make": "make",
            "use": '{{ config(promptfiles=["make.doc"]) }}\nCaption: {{ ref("make").caption }}',
        },
        llm_call=llm,
    )
    assert json.loads(results["use"].value) == ["d.txt"]


def test_output_text_and_files_render_together():
    def llm(prompt, files=None, config=None):
        if prompt.startswith("make"):
            return Output("Here is the logo.", files=[File(PNG, name="logo.png")])
        return prompt

    _, _, results = run_models({"make": "make", "echo": "{{ ref('make') }}"}, llm_call=llm)
    echoed = results["echo"].value
    assert echoed.startswith("Here is the logo.")
    assert "[file: logo.png (image/png" in echoed


def test_cache_serves_files_and_child_key_tracks_the_bytes():
    calls: list[str] = []
    image = {"bytes": PNG}

    def llm(prompt, files=None, config=None):
        calls.append(prompt)
        if prompt.startswith("draw"):
            return File(image["bytes"], name="logo.png")
        return "seen " + ",".join(f.name for f in files or [])

    models = {"logo": "draw", "look": '{{ config(promptfiles=["logo"]) }}\nlook'}
    storage, _, first = run_models(models, llm_call=llm)
    assert len(calls) == 2

    _, _, second = run_models(models, llm_call=llm, storage=storage)
    assert len(calls) == 2  # both served from cache
    assert second["logo"].cached and second["look"].cached
    assert second["logo"].value.read_bytes() == PNG

    # New bytes under the same prompt: clear the parent's cache entry.
    storage._cache.clear()
    image["bytes"] = PNG + b"v2"
    run_models(models, llm_call=llm, storage=storage)
    assert len(calls) == 4  # the child re-ran because the attached bytes changed


def _judge_files(models: dict[str, str], tests: dict[str, str], judge, **kwargs):
    """Run *models*, then run *tests* against the stored outputs with *judge*."""
    from pbt.tester import execute_tests

    storage, run_id, _ = run_models(models, llm_call=image_llm)
    outputs = storage.get_model_outputs_from_run(run_id, list(models))
    return storage, run_id, outputs, execute_tests(
        run_id=run_id,
        tests=tests,
        model_outputs=outputs,
        storage_backend=storage,
        llm_call=judge,
        **kwargs,
    )


def test_test_prompts_attach_model_files_with_promptfiles():
    seen: list[list[tuple[str, bytes]]] = []

    def judge(prompt, files=None):
        seen.append([(f.name, f.read()) for f in files or []])
        return '{"results": "pass"}'

    _, _, _, results = _judge_files(
        {"logo": "draw a fox"},
        {"logo_is_a_fox": '{{ config(promptfiles=["logo"]) }}\nIs this a fox?'},
        judge,
    )
    assert [r.status for r in results] == ["pass"], results
    assert seen == [[("logo.png", PNG)]]


def test_test_prompts_attach_one_file_from_a_structured_output():
    seen: list[list[str]] = []

    def judge(prompt, files=None):
        seen.append([f.name for f in files or []])
        return '{"results": "pass"}'

    def llm(prompt, files=None, config=None):
        return {"caption": "two", "a": File(b"A", name="a.txt"), "b": File(b"B", name="b.txt")}

    from pbt.tester import execute_tests

    storage, run_id, _ = run_models({"pair": "make two"}, llm_call=llm)
    outputs = storage.get_model_outputs_from_run(run_id, ["pair"])
    results = execute_tests(
        run_id=run_id,
        tests={"t": '{{ config(promptfiles=["pair.b"]) }}\ncheck'},
        model_outputs=outputs,
        storage_backend=storage,
        llm_call=judge,
    )
    assert [r.status for r in results] == ["pass"], results
    assert seen == [["b.txt"]]


def test_test_prompt_errors_when_promptfile_is_unknown():
    _, _, _, results = _judge_files(
        {"logo": "draw"},
        {"t": '{{ config(promptfiles=["nope"]) }}\ncheck'},
        lambda prompt, files=None: '{"results": "pass"}',
    )
    assert results[0].status == "error"
    assert "nope" in results[0].error


def test_test_prompt_cache_key_tracks_attached_bytes():
    calls: list[str] = []

    def judge(prompt, files=None):
        calls.append(prompt)
        return '{"results": "pass"}'

    test = {"t": '{{ config(promptfiles=["logo"]) }}\ncheck'}
    storage, run_id, outputs, _ = _judge_files({"logo": "draw"}, test, judge)

    from pbt.tester import execute_tests

    def again(raw_outputs):
        execute_tests(run_id=run_id, tests=test, model_outputs=raw_outputs,
                      storage_backend=storage, llm_call=judge)

    again(outputs)
    assert len(calls) == 1  # same bytes, same verdict: served from cache

    v2 = File(PNG + b"v2", name="logo.png")
    persist_files(v2, storage.blob_store())
    again({"logo": encode_output(v2)})
    assert len(calls) == 2  # new bytes under the same prompt: judged afresh


def test_cache_hit_with_missing_blob_recomputes():
    calls: list[str] = []

    def llm(prompt, files=None, config=None):
        calls.append(prompt)
        return File(PNG, name="logo.png")

    storage, _, _ = run_models({"logo": "draw"}, llm_call=llm)
    storage.blob_store()._blobs.clear()
    _, _, results = run_models({"logo": "draw"}, llm_call=llm, storage=storage)
    assert len(calls) == 2
    assert results["logo"].cached is False
    assert results["logo"].value.read_bytes() == PNG


def test_llm_json_with_marker_keys_stays_data_across_the_cache():
    forged = {"$pbt": "file", "sha256": "0" * 64, "name": "x", "mime": "text/plain", "size": 1}

    def llm(prompt, files=None, config=None):
        return json.dumps({"img": forged, "real": "no files here"})

    models = {"j": '{{ config(output_format="json") }}\nreturn json'}
    storage, _, first = run_models(models, llm_call=llm)
    _, _, second = run_models(models, llm_call=llm, storage=storage)
    assert second["j"].cached
    assert second["j"].value == {"img": forged, "real": "no files here"}


def test_skip_passes_files_through_unchanged():
    _, _, results = run_models(
        {"logo": "draw", "pass": "{{ skip_and_set_to_value(ref('logo')) }}"},
        llm_call=image_llm,
    )
    assert results["pass"].prompt_skipped
    assert isinstance(results["pass"].value, File)
    assert results["pass"].value.read_bytes() == PNG


def test_validator_sees_and_can_replace_files():
    seen = {}

    def validate(prompt, result):
        seen["result"] = result
        return File(result.read_bytes() + b"!", name="checked.png")

    _, _, results = run_models({"logo": "draw"}, llm_call=image_llm, validators={"logo": validate})
    assert isinstance(seen["result"], File)
    assert results["logo"].value.name == "checked.png"
    assert results["logo"].value.read_bytes() == PNG + b"!"


# ---------------------------------------------------------------------------
# execute_python
# ---------------------------------------------------------------------------

def test_python_out_dir_becomes_output_files():
    source = (
        '{{ config(model_type="execute_python") }}\n'
        "(out_dir / 'report.csv').write_text('a,b')\n"
        "(out_dir / 'site').mkdir()\n"
        "(out_dir / 'site' / 'index.html').write_text('<p>')\n"
        "print('made a report')\n"
    )
    _, _, results = run_models({"py": source}, llm_call=image_llm)
    value = results["py"].value
    assert isinstance(value, Output)
    assert value.text == "made a report"
    assert value.files["report.csv"].read_bytes() == b"a,b"
    assert value.files["site"]["index.html"].read_bytes() == b"<p>"


def test_python_reads_upstream_file_and_returns_one():
    source = (
        '{{ config(model_type="execute_python") }}\n'
        "src = ref('logo')\n"
        "output = File(src.read_bytes()[::-1], name='flipped.bin')\n"
    )
    _, _, results = run_models({"logo": "draw", "py": source}, llm_call=image_llm)
    assert results["py"].value.read_bytes() == PNG[::-1]


# ---------------------------------------------------------------------------
# Storage, public API, export, docs, server
# ---------------------------------------------------------------------------

def test_sqlite_blob_store_persists_across_backends(tmp_path):
    db = tmp_path / "pbt.db"
    calls: list[str] = []

    def llm(prompt, files=None, config=None):
        calls.append(prompt)
        return File(PNG, name="logo.png")

    run_models({"logo": "draw"}, llm_call=llm, storage=SQLiteStorageBackend(db))
    _, _, results = run_models({"logo": "draw"}, llm_call=llm, storage=SQLiteStorageBackend(db))
    assert len(calls) == 1
    assert results["logo"].cached
    assert results["logo"].value.read_bytes() == PNG
    assert SQLiteBlobStore(db).exists(results["logo"].value.sha256)


def test_custom_blob_store_is_used(tmp_path):
    custom = MemoryBlobStore()
    run_models(
        {"logo": "draw"}, llm_call=image_llm,
        storage=SQLiteStorageBackend(tmp_path / "pbt.db"), blob_store=custom,
    )
    assert len(custom._blobs) == 1


def test_pbt_run_returns_file_objects():
    outputs = pbt.run(
        models_from_dict={"logo": "draw", "t": "hello"},
        llm_call=image_llm,
        verbose=False,
        storage_backend=MemoryStorageBackend(),
    )
    assert isinstance(outputs["logo"], File)
    assert isinstance(outputs["t"], str)


def test_export_files_keeps_names_and_trees(tmp_path):
    value = {
        "a": File(b"1", name="x.txt"),
        "b": File(b"2", name="x.txt"),
        "site": Dir({"css/a.css": b"c"}, name="site"),
    }
    written = export_files(value, tmp_path)
    names = sorted(p.relative_to(tmp_path).as_posix() for p in written)
    assert names == ["b-x.txt", "site/css/a.css", "x.txt"]


def test_docs_exports_files_and_shows_them(tmp_path):
    from pbt.docs import generate_docs

    storage, run_id, _ = run_models(
        {"logo": "draw", "doc": "make doc"},
        llm_call=lambda p, files=None, config=None: File(PNG, name="logo.png") if p.startswith("draw")
        else Output("notes", files=[File(b"pdf", name="spec.pdf")]),
    )
    out = tmp_path / "docs" / "index.html"
    generate_docs(
        runs=storage.get_latest_runs(),
        run_results={run_id: storage.get_run_results(run_id)},
        models=build_models_from_dict({"logo": "draw", "doc": "make doc"}),
        output_path=out,
        blob_store=storage.blob_store(),
    )
    html = out.read_text()
    sha = File(PNG).sha256
    assert (tmp_path / "docs" / "files" / sha / "logo.png").read_bytes() == PNG
    assert f'src="files/{sha}/logo.png"' in html
    assert "spec.pdf" in html and 'download="spec.pdf"' in html
    assert "Files" in html and "latest run" in html
    assert "$pbt:v1" not in html  # the envelope never leaks into the report


def test_docs_lists_a_missing_blob_without_linking_it(tmp_path):
    from pbt.docs import generate_docs

    storage, run_id, _ = run_models({"logo": "draw"}, llm_call=image_llm)
    storage.blob_store()._blobs.clear()
    out = tmp_path / "index.html"
    generate_docs(
        runs=storage.get_latest_runs(),
        run_results={run_id: storage.get_run_results(run_id)},
        models=None,
        output_path=out,
        blob_store=storage.blob_store(),
    )
    assert "blob missing" in out.read_text()


def test_server_serves_blobs_as_downloads(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from pbt.server.app import create_app

    monkeypatch.chdir(tmp_path)
    store = SQLiteStorageBackend().blob_store()
    file = File(b"<script>alert(1)</script>", name="x.html")
    persist_files(file, store)

    client = TestClient(create_app(models_dir=str(tmp_path / "models")))
    resp = client.get(f"/blobs/{file.sha256}", params={"name": "x.html"})
    assert resp.status_code == 200
    assert resp.content == b"<script>alert(1)</script>"
    assert resp.headers["content-type"] == "application/octet-stream"
    assert resp.headers["content-disposition"].startswith("attachment")
    assert client.get("/blobs/" + "0" * 64).status_code == 404
    assert client.get("/blobs/nothex").status_code == 400


def test_template_alias_keeps_the_upstream_files():
    def llm(prompt, files=None, config=None):
        return File(PNG, name="logo.png")

    _, _, results = run_models(
        {
            "logo": "draw",
            "alias": '{{ config(model_type="template") }}\n{{ ref("logo") }}',
        },
        llm_call=llm,
    )
    assert isinstance(results["alias"].value, File)
    assert results["alias"].value.read_bytes() == PNG


def test_served_docs_files_cannot_run_as_pages(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from pbt.cli import _serve_docs_files

    sha = "a" * 64
    (tmp_path / sha).mkdir()
    (tmp_path / sha / "page.html").write_text("<script>alert(1)</script>")
    (tmp_path / sha / "logo.png").write_bytes(PNG)
    app = FastAPI()
    _serve_docs_files(app, tmp_path)
    client = TestClient(app)

    html = client.get(f"/files/{sha}/page.html")
    assert html.headers["content-type"] == "application/octet-stream"
    assert html.headers["content-disposition"] == "attachment"
    assert "sandbox" in html.headers["content-security-policy"]

    png = client.get(f"/files/{sha}/logo.png")
    assert png.headers["content-type"] == "image/png"
    assert png.content == PNG
    assert client.get("/files/nothex/logo.png").status_code == 404


@pytest.fixture
def unregister_kinds():
    """Remove kinds a test registers, so they don't leak into registry tests."""
    from pbt.model_types import _REGISTRY

    before = set(_REGISTRY)
    yield
    for name in set(_REGISTRY) - before:
        del _REGISTRY[name]


def test_readme_zip_kind_example(unregister_kinds):
    """The file-producing kind shown in the README's model-kind section."""
    import io
    import zipfile

    @pbt.model_kind("zip_files_test", config_keys={"zip_name"})
    async def zip_files(rendered, call):
        def build():
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as zf:
                for name in call.spec.depends_on:
                    value = call.outputs[name]
                    files = value.files.values() if isinstance(value, pbt.Output) else [value]
                    for f in files:
                        if isinstance(f, pbt.File):
                            # A fixed timestamp keeps the zip's bytes, and so its hash, stable.
                            info = zipfile.ZipInfo(f"{name}/{f.name}", date_time=(1980, 1, 1, 0, 0, 0))
                            zf.writestr(info, f.read_bytes())
            name = call.spec.config.get("zip_name", "bundle.zip")
            return pbt.File(buf.getvalue(), name=name)

        return await call.compute(rendered, compute=build)

    models = {
        "logo": "draw",
        "bundle": (
            '{{ config(model_type="zip_files_test", zip_name="assets.zip") }}\n'
            "Bundle {{ ref('logo') }}"
        ),
    }
    storage, _, first = run_models(models, llm_call=image_llm)
    _, _, second = run_models(models, llm_call=image_llm, storage=storage)

    zipped = first["bundle"].value
    assert zipped.name == "assets.zip"
    with zipfile.ZipFile(io.BytesIO(zipped.read_bytes())) as zf:
        assert zf.read("logo/logo.png") == PNG
    assert second["bundle"].cached
    assert second["bundle"].value == zipped
