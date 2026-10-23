"""PP drafts must travel with the matching output batch and request order."""

from types import SimpleNamespace
from unittest.mock import Mock

from vllm_ascend.worker.v1.model_runner import NPUModelRunner


def test_pp_drafts_follow_output_order_and_allow_missing_request():
    runner = SimpleNamespace(
        take_draft_token_ids=Mock(
            return_value=SimpleNamespace(req_ids=["b", "a"], draft_token_ids=[[21, 22], [11, 12]])
        )
    )
    output = SimpleNamespace(req_ids=["a", "missing", "b"])
    NPUModelRunner._attach_pp_draft_tokens(runner, output)
    assert output.spec_token_ids == [[11, 12], [], [21, 22]]


def test_absent_pp_drafts_leave_output_untouched():
    runner = SimpleNamespace(take_draft_token_ids=Mock(return_value=None))
    output = SimpleNamespace(req_ids=["a"])
    NPUModelRunner._attach_pp_draft_tokens(runner, output)
    assert not hasattr(output, "spec_token_ids")


def test_pp_draft_batches_do_not_reuse_previous_ids():
    runner = SimpleNamespace(
        take_draft_token_ids=Mock(
            side_effect=[
                SimpleNamespace(req_ids=["old"], draft_token_ids=[[11]]),
                SimpleNamespace(req_ids=["new"], draft_token_ids=[[22]]),
            ]
        )
    )
    old = SimpleNamespace(req_ids=["old"])
    new = SimpleNamespace(req_ids=["new"])
    NPUModelRunner._attach_pp_draft_tokens(runner, old)
    NPUModelRunner._attach_pp_draft_tokens(runner, new)
    assert old.spec_token_ids == [[11]]
    assert new.spec_token_ids == [[22]]
