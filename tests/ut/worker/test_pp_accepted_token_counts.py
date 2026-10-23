"""Earlier PP stages must select the accepted GDN state, not draft slot zero."""

from dataclasses import make_dataclass
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from vllm_ascend.worker.v1.model_runner import NPUModelRunner


def make_runner():
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.use_async_scheduling = False
    runner.speculative_config = object()
    runner.model_config = SimpleNamespace(is_hybrid=True, hf_text_config=SimpleNamespace(model_type="qwen4_exp_text"))
    runner.requests = {
        "a": SimpleNamespace(output_token_ids=[10]),
        "b": SimpleNamespace(output_token_ids=[20, 21]),
    }
    runner.input_batch = SimpleNamespace(
        req_id_to_index={"a": 0, "b": 1},
        num_accepted_tokens_cpu=[1, 1],
        spec_token_ids=[[11], [99]],
        num_tokens_no_spec=[6, 7],
    )
    return runner


def make_output():
    return SimpleNamespace(
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["a", "b"],
            num_output_tokens=[3, 3],
            # Only the latest sampled input is sent; len() is not acceptance.
            new_token_ids=[[12], [22]],
            resumed_req_ids=set(),
        ),
        kv_cache_block_copies=[],
        total_num_scheduled_tokens=2,
    )


def test_pp_acceptance_uses_output_length_delta_including_bonus_token():
    runner = make_runner()
    with patch(
        "vllm_ascend.worker.v1.model_runner.get_pp_group",
        return_value=SimpleNamespace(world_size=2, is_last_rank=False),
    ):
        assert runner._get_pp_accepted_token_counts(make_output()) == {"a": 2, "b": 1}


def test_counts_are_captured_before_parent_and_written_in_new_row_order():
    runner = make_runner()
    runner._update_replay_requests = Mock()
    runner._track_tmp_encoder_cache_refs = Mock()

    def parent_update(self, output):
        assert self.requests["a"].output_token_ids == [10, 11]
        assert self.input_batch.num_tokens_no_spec == [7, 7]
        self.requests["a"].output_token_ids.append(12)
        self.requests["b"].output_token_ids.append(22)
        self.input_batch.req_id_to_index = {"b": 0, "a": 1}
        self.input_batch.num_accepted_tokens_cpu[:] = [1, 1]
        return "sampling_metadata"

    with (
        patch(
            "vllm_ascend.worker.v1.model_runner.get_pp_group",
            return_value=SimpleNamespace(world_size=2, is_last_rank=False),
        ),
        patch("vllm.v1.worker.gpu_model_runner.GPUModelRunner._update_states", parent_update),
    ):
        assert runner._update_states(make_output()) == "sampling_metadata"
    assert runner.input_batch.num_accepted_tokens_cpu == [1, 2]
    assert runner.requests["a"].output_token_ids == [10, 11, 12]


def test_only_accepted_draft_prefix_is_committed():
    runner = make_runner()
    runner.input_batch.spec_token_ids[0] = [11, 12, 13]
    runner._restore_pp_accepted_draft_tokens({"a": 3, "b": 1})
    assert runner.requests["a"].output_token_ids == [10, 11, 12]
    assert runner.requests["b"].output_token_ids == [20, 21]
    assert runner.input_batch.num_tokens_no_spec == [8, 7]


def test_recompute_resumed_and_missing_requests_do_not_reuse_old_acceptance():
    runner = make_runner()
    output = make_output()
    output.scheduled_cached_reqs.req_ids += ["missing"]
    output.scheduled_cached_reqs.num_output_tokens = [0, 5, 8]
    output.scheduled_cached_reqs.resumed_req_ids = {"b"}
    with patch(
        "vllm_ascend.worker.v1.model_runner.get_pp_group",
        return_value=SimpleNamespace(world_size=2, is_last_rank=False),
    ):
        assert runner._get_pp_accepted_token_counts(output) == {}


@pytest.mark.parametrize("is_last_rank", [False, True])
def test_idle_pp_bubble_keeps_drafts_and_counts_but_still_removes_finished_rows(is_last_rank):
    runner = make_runner()
    runner.input_batch.req_id_to_index = {"a": 0, "b": 1, "preempted": 2, "resumed": 3}
    runner.input_batch.num_accepted_tokens_cpu = [2, 1]
    output_type = make_dataclass(
        "IdleOutput",
        [
            "total_num_scheduled_tokens",
            "num_scheduled_tokens",
            "finished_req_ids",
            "preempted_req_ids",
            "scheduled_cached_reqs",
        ],
    )
    output = output_type(0, {}, {"b"}, {"preempted"}, SimpleNamespace(resumed_req_ids={"resumed"}))
    with patch(
        "vllm_ascend.worker.v1.model_runner.get_pp_group",
        return_value=SimpleNamespace(world_size=2, is_last_rank=is_last_rank),
    ):
        retained = runner._preserve_pp_mtp_idle_batch(output)
    assert retained.num_scheduled_tokens == {"a": 0}
    assert retained.total_num_scheduled_tokens == 0
    assert output.num_scheduled_tokens == {}
    assert retained.finished_req_ids == {"b"}
    assert retained.preempted_req_ids == {"preempted"}
    assert runner.input_batch.num_accepted_tokens_cpu == [2, 1]
    assert runner.input_batch.spec_token_ids[0] == [11]


@pytest.mark.parametrize("mode", ["async", "pp1", "last", "no_spec", "non_hybrid"])
def test_existing_sampler_and_non_spec_paths_are_untouched(mode):
    runner = make_runner()
    if mode == "async":
        runner.use_async_scheduling = True
    if mode == "no_spec":
        runner.speculative_config = None
    if mode == "non_hybrid":
        runner.model_config.is_hybrid = False
    with patch(
        "vllm_ascend.worker.v1.model_runner.get_pp_group",
        return_value=SimpleNamespace(world_size=1 if mode == "pp1" else 2, is_last_rank=mode == "last"),
    ):
        assert runner._get_pp_accepted_token_counts(make_output()) == {}


@pytest.mark.parametrize("model_type", ["deepseek_v4", "qwen3_next", "qwen3_5", "glm5next"])
def test_other_models_keep_original_update_path(model_type):
    runner = make_runner()
    runner.model_config.hf_text_config.model_type = model_type
    runner._update_replay_requests = Mock()
    runner._track_tmp_encoder_cache_refs = Mock()
    runner._get_pp_accepted_token_counts = Mock(side_effect=AssertionError("Qwen helper called for other model"))
    runner._restore_pp_accepted_draft_tokens = Mock(side_effect=AssertionError("Qwen helper called for other model"))
    runner._preserve_pp_mtp_idle_batch = Mock(side_effect=AssertionError("Qwen helper called for other model"))
    output = make_output()
    with patch("vllm.v1.worker.gpu_model_runner.GPUModelRunner._update_states", return_value="original") as parent:
        assert runner._update_states(output) == "original"
        parent.assert_called_once_with(output)
    runner._get_pp_accepted_token_counts.assert_not_called()
    runner._restore_pp_accepted_draft_tokens.assert_not_called()
    runner._preserve_pp_mtp_idle_batch.assert_not_called()
