# SPDX-License-Identifier: Apache-2.0
"""Source-level ownership and integration guards for Qwen4Exp on Ascend."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_qwen4_exp_reuses_upstream_vllm_model_ownership() -> None:
    model = _source("vllm_ascend/models/qwen4_exp/model.py")
    qsa = _source("vllm_ascend/models/qwen4_exp/qsa.py")
    mtp = _source("vllm_ascend/models/qwen4_exp/mtp.py")

    assert "vllm.models.qwen4_exp.amd" in model
    assert "vllm.models.qwen4_exp.amd" in qsa
    assert "vllm.models.qwen4_exp.amd" in mtp
    assert "vllm.model_executor.models.qwen4_exp" not in model + qsa + mtp

    model_root = ROOT / "vllm_ascend/models/qwen4_exp"
    for platform_neutral_dir in ("amd", "common", "config", "nvidia"):
        assert not (model_root / platform_neutral_dir).exists()


def test_qwen4_exp_ascend_wrapper_owns_sequence_parallel_moe_region() -> None:
    model = _source("vllm_ascend/models/qwen4_exp/model.py")
    mtp = _source("vllm_ascend/models/qwen4_exp/mtp.py")
    pp_compat = _source("vllm_ascend/patch/platform/patch_qwen4_exp_pp.py")
    model_state = _source("vllm_ascend/models/qwen4_exp/model_state.py")
    runner = _source("vllm_ascend/worker/v1/model_runner.py")

    assert "class AscendQwen4ExpSparseMoeBlock(Qwen3NextSparseMoeBlock)" in model
    assert "class AscendQwen4ExpPLELayer(upstream_model.Qwen4ExpPLELayer)" in model
    assert "self.mlp = AscendQwen4ExpSparseMoeBlock(" in model
    assert (
        "upstream_model.Qwen4ExpDecoderLayer = AscendQwen4ExpDecoderLayer" in model
    )
    assert "reduce_results=not self.use_sequence_parallel" in model
    assert "already_sequence_parallel=self.use_sequence_parallel" in model
    assert "block_input = sp_all_gather(block_input)" in model
    assert "attn_out = sp_reduce_scatter(attn_out)" in model
    assert "gathered_ple = sp_all_gather(" in model
    assert "torch.cat((gated_output, conv_input), dim=-1)" in model
    assert "gated_output.add_(conv_output)" in model
    assert "ple_output_is_full_tokens = self.use_sequence_parallel" in model
    assert "injection = sp_shard(injection)" in model
    assert "local_ngram_ids = sp_shard(ngram_ids)" in model
    assert "qwen4_exp_amd_ple_ngram_embedding_pinned(" in model
    assert "return ple_metadata.num_prefills > 0" in model
    assert "embeddings = sp_shard(embeddings)" in model
    assert "forward_context.is_padding = sp_padding_mask(" in model
    assert "hidden_states = sp_shard(hidden_states)" in model
    assert "pp_group = get_pp_group()" in model
    assert "if pp_group.is_first_rank:" in model
    assert "if not pp_group.is_last_rank:" in model
    assert "and hidden_states.shape[0] == positions.shape[-1]" in model
    assert "hidden_states = sp_all_gather(hidden_states)" in model
    assert "pp_intermediate_tensors_are_full_tokens = True" in model
    assert "def _get_pp_intermediate_tensors" in runner
    assert "MRV1 transports full PP intermediates" in runner
    assert "pipeline_parallel_size=1" not in model
    assert "torch.cat((multi_hidden, sample_hidden_states), dim=-1)" in model
    assert "and hidden_states.shape[0] == positions.shape[-1]" in mtp
    assert "hidden_states = sp_all_gather(hidden_states)" not in mtp
    assert "torch.cat((multi_hidden, sample_hidden_states), dim=-1)" in mtp
    assert "text_config.ple_layer_ids = []" in pp_compat
    assert "Qwen4ExpModelState.__init__" not in pp_compat
    assert "class AscendQwen4ExpModelState" in model_state
    assert "get_pp_group().is_first_rank" in model_state


def test_qwen4_exp_registry_uses_ascend_wrappers() -> None:
    registry = _source("vllm_ascend/models/__init__.py")
    for architecture, wrapper in (
        ("Qwen4ExpForCausalLM", "AscendQwen4ExpForCausalLM"),
        (
            "Qwen4ExpForConditionalGeneration",
            "AscendQwen4ExpForConditionalGeneration",
        ),
        ("Qwen4ExpMTP", "AscendQwen4ExpMTP"),
    ):
        assert f'"{architecture}"' in registry
        assert f"vllm_ascend.models.qwen4_exp:{wrapper}" in registry


def test_runner_uses_spec_decode_capabilities_not_qwen_type_checks() -> None:
    runner = _source("vllm_ascend/worker/v1/model_runner.py")
    # The concrete class is part of the static drafter union so mypy can see
    # the inherited proposer methods. Runtime dispatch still uses capabilities.
    assert "AscendQwen4ExpMTPProposer" in runner
    tree = ast.parse(runner)
    concrete_qwen_checks = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "isinstance"
        and len(node.args) == 2
        and "AscendQwen4ExpMTPProposer" in ast.unparse(node.args[1])
    ]
    assert concrete_qwen_checks == []
    assert "SupportsPerGroupAttentionMetadata" in runner
    assert "SupportsPerGroupBlockTables" in runner
    assert "SupportsPerGroupKernelBlockSizes" in runner
    assert "SupportsAttentionBackendInitialization" in runner
    assert "SupportsCUDAGraphInitialization" in runner

    for proposer in (
        "vllm_ascend/worker/v1/spec_decode/qwen4_exp.py",
        "vllm_ascend/worker/v1/spec_decode/dspark_proposer.py",
        "vllm_ascend/worker/v1/spec_decode/gemma4_proposer.py",
    ):
        assert "def uses_per_group_kernel_block_sizes" in _source(proposer)


def test_qwen4_exp_proposer_is_owned_by_qwen_spec_decode_module() -> None:
    qwen = _source("vllm_ascend/worker/v1/spec_decode/qwen4_exp.py")
    eagle = _source("vllm_ascend/worker/v1/spec_decode/eagle_proposer.py")

    assert "class AscendQwen4ExpMTPProposer" in qwen
    assert "Qwen4ExpMTPProposer" not in eagle
    assert "class AscendEagleProposer" in eagle


def test_runner_customizes_compressed_attention_cache_specs() -> None:
    runner = _source("vllm_ascend/worker/v1/model_runner.py")
    branch_start = runner.index("elif self.use_compress:")
    branch_end = runner.index(
        "elif isinstance(attn_module, Attention):",
        branch_start,
    )
    branch = runner[branch_start:branch_end]

    assert "isinstance(spec, AttentionSpec)" in branch
    assert "attn_module.get_attn_backend().customize_spec(spec)" in branch


def test_qsa_hot_paths_do_not_add_host_tensor_syncs() -> None:
    qsa_sources = "\n".join(
        _source(path)
        for path in (
            "vllm_ascend/models/qwen4_exp/qsa.py",
            "vllm_ascend/models/qwen4_exp/ops.py",
            "vllm_ascend/models/qwen4_exp/lightning_indexer.py",
            "vllm_ascend/ops/triton/qwen4_exp/qsa.py",
        )
    )
    assert ".item(" not in qsa_sources
    assert "torch.equal(" not in qsa_sources
    assert qsa_sources.count(
        "positions.clamp(0, cache.shape[0] - 1).contiguous()"
    ) >= 3


def test_qsa_metadata_ignores_full_graph_dummy_request_boundaries() -> None:
    runner = _source("vllm_ascend/worker/v1/model_runner.py")
    branch_start = runner.index("isinstance(builder, QSAMetadataBuilder)")
    branch_end = runner.index("if is_gdn_noop:", branch_start)
    branch = runner[branch_start:branch_end]

    assert "num_reqs_padded > num_reqs" in branch
    assert "query_start_loc_cpu=common_attn_metadata.query_start_loc_cpu[" in branch
    assert ": num_reqs + 1" in branch


def test_qsa_uses_only_fused_runtime_backends() -> None:
    qsa = _source("vllm_ascend/models/qwen4_exp/qsa.py")
    assert "return qsa_select_paged_tokens_lightning(" in qsa
    assert "return qsa_gbsa_attention(" in qsa
    assert "torch.ops.vllm.triton_qsa_rmsnorm_mrope(" in qsa
    assert "torch.ops.vllm.triton_split_qkv_rmsnorm_mrope(" in qsa
    for fallback in (
        "qsa_select_paged_tokens_reference",
        "qsa_select_paged_tokens_triton",
        "qsa_sparse_paged_attention_reference",
        "qsa_sparse_paged_attention(",
        "VLLM_ASCEND_FORCE_QSA_REFERENCE",
        "VLLM_ASCEND_ENABLE_QSA_LIGHTNING_INDEXER",
        "VLLM_ASCEND_ENABLE_QSA_INDEXER_FUSED_NORM_ROPE",
        "VLLM_ASCEND_ENABLE_QSA_MAIN_FUSED_NORM_ROPE",
    ):
        assert fallback not in qsa
