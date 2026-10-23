# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Scoped PLE PP validation; keep upstream's other Qwen4Exp constraints.

Remove when upstream exposes a platform PLE ownership verifier. This mirrors
the verifier from vLLM 876d5da; it does not change the checkpoint configuration.
"""

from vllm.model_executor.models import config as model_config

from vllm_ascend.core.qwen4_exp_pp import validate_ple_pipeline_partition


def verify_and_update_config(vllm_config):
    from vllm.platforms import current_platform

    model_config.Qwen3_5ForConditionalGenerationConfig.verify_and_update_config(vllm_config)
    text_config = vllm_config.model_config.hf_text_config
    if text_config.hc_count <= 1:
        raise ValueError("Qwen4Exp requires hc_count > 1")
    parallel_config = vllm_config.parallel_config
    if (
        current_platform.is_cuda_alike()
        and parallel_config.use_sequence_parallel_moe
        and parallel_config.pipeline_parallel_size != 1
    ):
        raise ValueError("Qwen4Exp MoE SP requires PP=1")
    uses_ple_or_qsa = bool(text_config.ple_layer_ids) or getattr(text_config, "indexer_n_heads", None) is not None
    if uses_ple_or_qsa and (parallel_config.enable_dbo or parallel_config.ubatch_size > 1):
        raise NotImplementedError("Qwen4Exp PLE/QSA does not support dual-batch overlap or microbatching")
    validate_ple_pipeline_partition(text_config, parallel_config)
    multimodal_config = vllm_config.model_config.multimodal_config
    if multimodal_config is not None and multimodal_config.language_model_only:
        model_config._strip_qwen4_exp_mrope(vllm_config.model_config)
    spec_config = vllm_config.speculative_config
    if spec_config is not None and spec_config.method not in {"mtp", "ngram", "ngram_gpu"}:
        raise NotImplementedError(
            "Qwen4Exp speculative decoding supports only its native MTP checkpoint and linear n-gram proposers"
        )


model_config.Qwen4ExpForConditionalGenerationConfig.verify_and_update_config = staticmethod(verify_and_update_config)
