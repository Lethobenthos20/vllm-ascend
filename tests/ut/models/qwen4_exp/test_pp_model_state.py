# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.qwen4_exp import model_state as state_module
from vllm_ascend.worker.v2.model_states.mamba_hybrid import AscendMambaHybridModelState


@pytest.fixture
def make_state(monkeypatch):
    def base_init(self, config, model, encoder_cache, device):
        self.model_config = config.model_config
        self.max_num_reqs = 3
        self.device = device

    monkeypatch.setattr(AscendMambaHybridModelState, "__init__", base_init)
    monkeypatch.setattr(AscendMambaHybridModelState, "prepare_inputs", lambda *args: {"base_marker": True})
    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)

    def create(first):
        monkeypatch.setattr(state_module, "get_pp_group", lambda: SimpleNamespace(is_first_rank=first))
        config = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(ple_layer_ids=[2], num_hidden_layers=48, ngram_size=3, eos_token_id=999)
            ),
            parallel_config=SimpleNamespace(pipeline_parallel_size=2),
        )
        return state_module.AscendQwen4ExpModelState(config, None, None, torch.device("cpu"))

    return create


def test_first_stage_history_reorder_rollback_and_padding(make_state):
    state = make_state(True)
    batch = SimpleNamespace(
        num_reqs=2,
        num_reqs_after_padding=3,
        idx_mapping=torch.tensor([1, 0]),
        query_start_loc=torch.tensor([0, 1, 4, 4], dtype=torch.int32),
    )
    requests = SimpleNamespace(
        num_computed_tokens=SimpleNamespace(gpu=torch.tensor([1, 4])),
        all_token_ids=SimpleNamespace(gpu=torch.tensor([[10, 11, 12, 13, 14], [20, 21, 22, 23, 24]])),
    )
    inputs = state.prepare_inputs(batch, requests)
    pointer = inputs["ngram_context"].data_ptr()
    assert inputs["base_marker"]
    assert inputs["ngram_context"].tolist() == [[22, 23], [999, 10], [999, 999]]
    # A rejected speculative step rolls history back rather than appending
    # rejected drafts. Request indices may also be reordered between steps.
    requests.num_computed_tokens.gpu[1] = 2
    inputs = state.prepare_inputs(batch, requests)
    assert inputs["ngram_context"].data_ptr() == pointer
    assert inputs["ngram_context"].tolist() == [[20, 21], [999, 10], [999, 999]]


def test_later_stage_does_not_access_token_history(make_state):
    state = make_state(False)
    assert not state.uses_ngram_embedding
    assert state.prepare_inputs(None, None) == {"base_marker": True}
    assert isinstance(state, AscendMambaHybridModelState)
