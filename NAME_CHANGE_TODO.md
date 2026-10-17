# Name change TODO (pbt → promptmap)

The code, docs, examples and tests have been renamed. Still PoC with no external
users, so no migration path or backwards compatibility is needed.

- [ ] Rename the GitHub repo `conradbez/prompt-build-tool` → `conradbez/promptmap`, then update the link in `promptmap/html_templates/_nav.html`.
- [ ] Publish `promptmap` to PyPI.
- [ ] Publish `promptmap-extra-model-kinds` to PyPI.
- [ ] Re-run `uv lock` in `promptmap-extra-model-kinds/` once `promptmap` is on PyPI (its lockfile still references `prompt-build-tool` / `pbt-extra-model-kinds`).
- [ ] Optional: rename the beads Dolt database in `.beads/metadata.json` (`prompt_build_tool_for_LLMs`) — only if the backing database is renamed too.
- [ ] Optional: yank or archive the old `prompt-build-tool` package on PyPI.
- [ ] Delete this file.
