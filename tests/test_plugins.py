"""Model kinds discovered from installed packages via the pbt.model_kinds entry point group."""

from __future__ import annotations

import importlib.metadata
import logging

import pytest

import pbt
from pbt import model_types
from pbt.model_types import ENTRY_POINT_GROUP, _REGISTRY, get_model_kind, known_model_kinds


class FakeEntryPoint:
    def __init__(self, name, obj):
        self.name = name
        self._obj = obj

    def load(self):
        if isinstance(self._obj, Exception):
            raise self._obj
        return self._obj


@pytest.fixture
def plugins(monkeypatch):
    """Install fake entry points and re-arm discovery; returns the list to fill."""
    eps: list[FakeEntryPoint] = []
    groups: list[str] = []

    def fake_entry_points(*, group):
        groups.append(group)
        return list(eps)

    monkeypatch.delenv("PBT_DISABLE_PLUGINS", raising=False)
    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)
    monkeypatch.setattr(model_types, "_plugins_loaded", False)
    before = set(_REGISTRY)
    yield eps, groups
    for name in set(_REGISTRY) - before:
        del _REGISTRY[name]


def test_register_function_entry_point_adds_kind(plugins):
    eps, groups = plugins

    def register():
        pbt.register_model_kind(pbt.ModelKind("plugin_fn_test", config_keys=frozenset({"plugin_key"})))

    eps.append(FakeEntryPoint("fn", register))

    assert get_model_kind("plugin_fn_test") is not None
    assert "plugin_key" in model_types.registered_config_keys()
    assert groups == [ENTRY_POINT_GROUP]


def test_module_entry_point_is_not_called(plugins):
    """A module value registers on import; pbt must not try to call it."""
    import types

    module = types.ModuleType("fake_plugin_module")
    pbt.register_model_kind(pbt.ModelKind("plugin_module_test"))
    eps, _ = plugins
    eps.append(FakeEntryPoint("mod", module))

    assert "plugin_module_test" in known_model_kinds()


def test_discovery_runs_once(plugins):
    eps, groups = plugins
    calls = []
    eps.append(FakeEntryPoint("count", lambda: calls.append(1)))

    get_model_kind("x")
    known_model_kinds()
    model_types.registered_config_keys()

    assert calls == [1]
    assert len(groups) == 1


def test_broken_plugin_is_logged_and_skipped(plugins, caplog):
    eps, _ = plugins
    eps.append(FakeEntryPoint("broken", ImportError("no such module")))
    eps.append(FakeEntryPoint("good", lambda: pbt.register_model_kind(pbt.ModelKind("plugin_good_test"))))

    with caplog.at_level(logging.ERROR, logger="pbt"):
        assert get_model_kind("plugin_good_test") is not None

    assert "broken" in caplog.text


def test_disable_env_var_skips_discovery(plugins, monkeypatch):
    eps, groups = plugins
    monkeypatch.setenv("PBT_DISABLE_PLUGINS", "1")
    eps.append(FakeEntryPoint("fn", lambda: pbt.register_model_kind(pbt.ModelKind("plugin_disabled_test"))))

    assert get_model_kind("plugin_disabled_test") is None
    assert groups == []
