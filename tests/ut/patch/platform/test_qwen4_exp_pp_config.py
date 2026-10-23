# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from vllm.model_executor.models import config as model_config

from vllm_ascend.patch.platform import patch_qwen4_exp_pp


@pytest.fixture
def pp_config(monkeypatch):
    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    base_verifier = MagicMock()
    monkeypatch.setattr(model_config.Qwen3_5ForConditionalGenerationConfig, "verify_and_update_config", base_verifier)
    monkeypatch.setattr("vllm.platforms.current_platform.is_cuda_alike", lambda: False)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(num_hidden_layers=48, ple_layer_ids=[2], hc_count=4),
            multimodal_config=None,
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=2, use_sequence_parallel_moe=False, enable_dbo=False, ubatch_size=1
        ),
        speculative_config=None,
    )
    return config, base_verifier


def test_registered_verifier_accepts_first_stage_ple(pp_config):
    config, base_verifier = pp_config
    model_config.Qwen4ExpForConditionalGenerationConfig.verify_and_update_config(config)
    base_verifier.assert_called_once_with(config)
    assert config.model_config.hf_text_config.ple_layer_ids == [2]


@pytest.mark.parametrize("attribute,value", [("enable_dbo", True), ("ubatch_size", 2)])
def test_pp_patch_preserves_microbatch_rejection(pp_config, attribute, value):
    config, _ = pp_config
    setattr(config.parallel_config, attribute, value)
    with pytest.raises(NotImplementedError, match="microbatching"):
        patch_qwen4_exp_pp.verify_and_update_config(config)


def test_pp_patch_preserves_hc_guard(pp_config):
    config, _ = pp_config
    config.model_config.hf_text_config.hc_count = 1
    with pytest.raises(ValueError, match="hc_count"):
        patch_qwen4_exp_pp.verify_and_update_config(config)


def test_pp_patch_preserves_speculative_method_guard(pp_config):
    config, _ = pp_config
    config.speculative_config = SimpleNamespace(method="eagle3")
    with pytest.raises(NotImplementedError, match="native MTP"):
        patch_qwen4_exp_pp.verify_and_update_config(config)


def test_pp_patch_preserves_cuda_moe_sp_guard(pp_config, monkeypatch):
    config, _ = pp_config
    config.parallel_config.use_sequence_parallel_moe = True
    monkeypatch.setattr("vllm.platforms.current_platform.is_cuda_alike", lambda: True)
    with pytest.raises(ValueError, match="MoE SP"):
        patch_qwen4_exp_pp.verify_and_update_config(config)


def test_registered_verifier_rejects_ple_on_later_stage(pp_config):
    config, _ = pp_config
    config.model_config.hf_text_config.ple_layer_ids = [25]
    with pytest.raises(NotImplementedError, match="first pipeline stage"):
        model_config.Qwen4ExpForConditionalGenerationConfig.verify_and_update_config(config)
