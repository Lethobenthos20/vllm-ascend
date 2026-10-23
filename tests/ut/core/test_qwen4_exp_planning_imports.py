# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Qwen startup must retain access to upstream's moved planning helpers."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def load_planning(monkeypatch):
    def load(*, split=True, existing=False):
        for name in ("vllm", "vllm.v1", "vllm.v1.core"):
            package = ModuleType(name)
            package.__path__ = []
            monkeypatch.setitem(sys.modules, name, package)
        legacy_config = ModuleType("vllm.transformers_utils.configs.deepseek_v4")
        monkeypatch.setitem(sys.modules, legacy_config.__name__, legacy_config)
        core = sys.modules["vllm.v1.core"]
        legacy = ModuleType("vllm.v1.core.kv_cache_utils")
        planning = ModuleType("vllm.v1.core.kv_cache_planning") if split else legacy
        for name in ("_approximate_gcd", "may_override_num_blocks", "create_kv_cache_group_specs"):
            setattr(planning, name, lambda *args: args)
        planning.get_kv_cache_groups = lambda *args: ("groups", args)
        planning.unify_hybrid_kv_cache_specs = lambda *args: ("unified", args)
        original = lambda *args: ("original", args)
        if existing:
            legacy.may_override_num_blocks = original
        core.kv_cache_utils = legacy
        monkeypatch.setitem(sys.modules, legacy.__name__, legacy)
        if split:
            core.kv_cache_planning = planning
            monkeypatch.setitem(sys.modules, planning.__name__, planning)
        else:
            monkeypatch.delitem(sys.modules, "vllm.v1.core.kv_cache_planning", raising=False)
        path = Path(__file__).resolve().parents[3] / "vllm_ascend/core/kv_cache_planning.py"
        spec = importlib.util.spec_from_file_location("qwen_planning_under_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, legacy, planning, original

    return load


def test_qwen_startup_uses_real_moved_helpers(load_planning):
    module, legacy, planning, _ = load_planning()
    assert legacy.may_override_num_blocks is planning.may_override_num_blocks
    assert legacy.create_kv_cache_group_specs is planning.create_kv_cache_group_specs
    assert module.may_override_num_blocks is planning.may_override_num_blocks
    module.install_legacy_planning_exports()
    assert legacy.may_override_num_blocks is planning.may_override_num_blocks


def test_qwen_import_compatibility_preserves_existing_exports(load_planning):
    _, legacy, planning, original = load_planning(existing=True)
    assert legacy.may_override_num_blocks is original
    assert legacy.create_kv_cache_group_specs is planning.create_kv_cache_group_specs


def test_qwen_legacy_planning_requires_no_aliases(load_planning):
    module, legacy, planning, _ = load_planning(split=False)
    assert module.planning is legacy is planning
    assert module.may_override_num_blocks is legacy.may_override_num_blocks


def test_qwen_group_hooks_resolve_after_platform_installation(load_planning):
    module, _, planning, _ = load_planning()
    planning.get_kv_cache_groups = lambda *args: ("patched", args)
    assert module.get_kv_cache_groups("qwen") == ("patched", ("qwen",))
    assert module.unify_hybrid_kv_cache_specs("qwen") == ("unified", ("qwen",))


def test_qwen_startup_config_import_preserves_installed_module(load_planning):
    module, _, _, _ = load_planning()
    legacy = "vllm.transformers_utils.configs.deepseek_v4"
    original = sys.modules[legacy]
    module.install_qwen_startup_config_import()
    assert sys.modules[legacy] is original


def test_qwen_startup_config_import_reuses_real_provider(load_planning, monkeypatch):
    module, _, _, _ = load_planning()
    legacy = "vllm.transformers_utils.configs.deepseek_v4"
    monkeypatch.delitem(sys.modules, legacy)
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda name: None)
    provider = ModuleType("upstream_configuration_provider")
    imports = []

    def import_provider(name):
        imports.append(name)
        return provider

    monkeypatch.setattr(module.importlib, "import_module", import_provider)
    module.install_qwen_startup_config_import()
    assert sys.modules[legacy] is provider
    assert imports == ["transformers.models.deepseek_v4.configuration_deepseek_v4"]
    monkeypatch.delitem(sys.modules, legacy)
