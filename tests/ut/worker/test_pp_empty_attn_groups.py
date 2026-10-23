"""Exercise V1 metadata construction with global groups absent on this PP rank."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder

from vllm_ascend.worker.v1.model_runner import NPUModelRunner


@pytest.mark.parametrize("empty_gid", [0, 1, 2])
def test_empty_pp_group_preserves_group_ids_and_gdn_metadata(empty_gid):
    runner = MagicMock()
    runner.sparse_kv_offload_enabled = False
    runner.device_metadata_executor = None
    runner.use_dcp = False
    runner.use_async_spec_decode = False
    runner.is_mm_prefix_lm = False
    runner.use_compress = False
    runner.speculative_config = None
    runner._has_gdn = True
    runner._offload_req_ids_tensor = None
    runner._offload_token_to_req = None
    runner._offload_pool_slots = None
    runner._offload_pool_active = None
    runner._copy_sfa_need_eager_tail_restore = False
    runner.device_metadata_providers = None
    runner.optimistic_seq_lens_cpu = torch.tensor([2], dtype=torch.int32)
    runner.seq_lens = torch.tensor([2], dtype=torch.int32)
    runner.positions = torch.tensor([0, 1], dtype=torch.int64)
    runner.actual_seq_lengths_q = [2]
    runner.query_start_loc = SimpleNamespace(cpu=torch.tensor([0, 2]), gpu=torch.tensor([0, 2]))
    runner.gdn_query_start_loc = SimpleNamespace(cpu=torch.tensor([0, 2]), gpu=torch.tensor([0, 2]))
    runner.input_batch.num_computed_tokens_cpu_tensor = torch.tensor([0])
    runner.input_batch.num_prompt_tokens_cpu_tensor = torch.tensor([2])
    runner._get_encoder_seq_lens.return_value = (None, None)
    runner._attn_replay_start.return_value = None
    runner.kv_cache_config.kv_cache_groups = [
        SimpleNamespace(kv_cache_spec=object(), layer_names=[] if gid == empty_gid else [f"layer{gid}"])
        for gid in range(3)
    ]
    tables = [torch.full((1, 1), gid, dtype=torch.int32) for gid in range(3)]
    slots = [torch.full((2,), gid, dtype=torch.int64) for gid in range(3)]
    runner.input_batch.block_table = [
        SimpleNamespace(get_device_tensor=lambda gid=gid: tables[gid], slot_mapping=SimpleNamespace(gpu=slots[gid]))
        for gid in range(3)
    ]
    builders = []
    runner.attn_groups = []
    for gid in range(3):
        builder = MagicMock(spec=GDNAttentionMetadataBuilder)
        builder.build.return_value = object()
        builders.append(builder)
        group = SimpleNamespace(layer_names=[f"layer{gid}"], get_metadata_builder=lambda _, b=builder: b)
        runner.attn_groups.append([] if gid == empty_gid else [group])

    metadata, draft = NPUModelRunner._build_attention_metadata(runner, num_tokens=2, num_reqs=1, max_query_len=2)

    assert draft is None
    assert set(metadata) == {f"layer{gid}" for gid in range(3) if gid != empty_gid}
    for gid, builder in enumerate(builders):
        if gid == empty_gid:
            builder.build.assert_not_called()
            continue
        cm = builder.build.call_args.kwargs["common_attn_metadata"]
        assert cm.block_table_tensor.data_ptr() == tables[gid].data_ptr()
        assert cm.slot_mapping.data_ptr() == slots[gid].data_ptr()
        assert cm.query_start_loc.data_ptr() == runner.gdn_query_start_loc.gpu.data_ptr()
        assert builder.build.call_args.kwargs["num_actual_reqs"] == 1
