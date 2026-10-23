# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""The PLE ownership contract supported by the Ascend PP runners."""


def is_qwen4_exp_model_config(model_config) -> bool:
    """Scope this adaptation to the Qwen3.8-Flash-Next architecture."""
    if model_config is None:
        return False
    for config in (getattr(model_config, "hf_config", None), getattr(model_config, "hf_text_config", None)):
        if getattr(config, "model_type", None) in {"qwen4_exp", "qwen4_exp_text", "qwen4_exp_mtp"}:
            return True
        if set(getattr(config, "architectures", None) or ()) & {
            "Qwen4ExpForCausalLM",
            "Qwen4ExpForConditionalGeneration",
            "Qwen4ExpMTP",
        }:
            return True
    return False


def validate_ple_pipeline_partition(text_config, parallel_config) -> None:
    """Require raw-token PLE work to remain on the first pipeline stage.

    PLE layer IDs in the checkpoint are one-based. Respect the same explicit
    VLLM_PP_LAYER_PARTITION as make_layers, including uneven partitions.
    """
    from vllm.distributed.utils import get_pp_indices

    if not text_config.ple_layer_ids or parallel_config.pipeline_parallel_size == 1:
        return
    _, first_stage_end = get_pp_indices(text_config.num_hidden_layers, 0, parallel_config.pipeline_parallel_size)
    outside = [layer for layer in text_config.ple_layer_ids if not 1 <= layer <= first_stage_end]
    if outside:
        raise NotImplementedError(
            "Ascend Qwen4Exp PP requires all PLE layers on the first pipeline stage; "
            f"one-based PLE layers {outside} fall outside its layers 1..{first_stage_end}. "
            "Choose a compatible VLLM_PP_LAYER_PARTITION or use PP=1."
        )
