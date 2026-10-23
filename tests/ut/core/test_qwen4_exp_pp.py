# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
from types import SimpleNamespace

import pytest

from vllm_ascend.core.qwen4_exp_pp import is_qwen4_exp_model_config, validate_ple_pipeline_partition


@pytest.mark.parametrize(
    "model_type,expected",
    [
        ("qwen4_exp", True),
        ("qwen4_exp_text", True),
        ("qwen4_exp_mtp", True),
        ("qwen3_next", False),
        ("deepseek_v4", False),
        ("glm5next", False),
        (None, False),
    ],
)
def test_pp_adaptation_model_scope(model_type, expected):
    config = SimpleNamespace(hf_text_config=SimpleNamespace(model_type=model_type))
    assert is_qwen4_exp_model_config(config) is expected


def test_pp_scope_accepts_explicit_architecture_and_rejects_missing_config():
    config = SimpleNamespace(hf_config=SimpleNamespace(architectures=["Qwen4ExpForConditionalGeneration"]))
    assert is_qwen4_exp_model_config(config)
    assert not is_qwen4_exp_model_config(None)


@pytest.mark.parametrize("pp_size", [1, 2, 4, 8])
def test_real_flash_next_ple_is_first_stage(monkeypatch, pp_size):
    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    validate_ple_pipeline_partition(
        SimpleNamespace(num_hidden_layers=48, ple_layer_ids=[2]),
        SimpleNamespace(pipeline_parallel_size=pp_size),
    )


@pytest.mark.parametrize("partition,accepted", [("2,46", True), ("1,47", False), ("24,24", True)])
def test_explicit_partition_uses_one_based_ple_ids(monkeypatch, partition, accepted):
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", partition)
    args = (SimpleNamespace(num_hidden_layers=48, ple_layer_ids=[2]), SimpleNamespace(pipeline_parallel_size=2))
    if accepted:
        validate_ple_pipeline_partition(*args)
    else:
        with pytest.raises(NotImplementedError, match="first pipeline stage"):
            validate_ple_pipeline_partition(*args)


def test_later_ple_layer_is_not_silently_dropped(monkeypatch):
    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    with pytest.raises(NotImplementedError, match="25"):
        validate_ple_pipeline_partition(
            SimpleNamespace(num_hidden_layers=48, ple_layer_ids=[2, 25]),
            SimpleNamespace(pipeline_parallel_size=2),
        )
