"""V0-V10 checkers reject missing fields and accept a good payload."""

import pytest

from verify.ncshare.checkers import CHECKERS, CheckError, check_v0


def test_v0_requires_bandwidth(tmp_path):
    with pytest.raises(CheckError):
        check_v0({"kvm_present": False})
    check_v0({"bus_bandwidth_gbps": 12.0, "kvm_present": True})


def test_all_checkers_have_positive():
    goods = {
        0: {"bus_bandwidth_gbps": 1.0, "kvm_present": False},
        1: {"logit_max_abs_err": 1e-6, "grad_check": True, "overfit_one_batch": True},
        2: {"relative_loss_err": 1e-8, "routing_identical": True},
        3: {"fp8_vs_bf16_rel": 0.001, "nvfp4_numerics_ok": True},
        4: {"resume_bitwise_equal": True, "sdc_caught_flip": True, "spike_skipped_shard": True},
        5: {"loss_matches_ladder": True, "ckpt_resume_across_jobs": True},
        6: {"no_collapse": True, "accuracy_rises_with_budget": True, "thoughts_decode": True},
        7: {"reward_rose": True, "drift_halted": True, "planted_write_flagged": True},
        8: {"logprob_max_abs_err": 1e-5, "tiered_restore_match": True},
        9: {"planted_positive_replicated": True, "planted_negative_recorded": True},
        10: {"lost_tasks": 0, "duplicated_outputs": 0, "dead_tokens": 0, "tiny_standin": True},
    }
    for i, payload in goods.items():
        CHECKERS[i](payload)
