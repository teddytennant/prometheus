"""V0-V10 checkers reject CPU stand-ins and accept an H200 payload."""

import pytest

from verify.ncshare.checkers import CHECKERS, CheckError, check_v0, check_v2, check_v5, check_v10

H200 = {
    "gpu_name": "NVIDIA H200",
    "slurm_job_id": "732000",
    "n_devices": 2,
    "nccl_backend": "nccl",
    "identity_mesh": False,
    "mesh_size": 2,
    "tiny_standin": False,
    "standin": False,
}


def test_v0_rejects_cpu_and_numpy_nccl():
    with pytest.raises(CheckError, match="H200"):
        check_v0({"bus_bandwidth_gbps": 12.0, "kvm_present": True})
    with pytest.raises(CheckError, match="numpy"):
        check_v0({**H200, "bus_bandwidth_gbps": 12.0, "kvm_present": True, "nccl_backend": "numpy"})
    with pytest.raises(CheckError, match=">=2"):
        check_v0(
            {
                **H200,
                "n_devices": 1,
                "bus_bandwidth_gbps": 12.0,
                "kvm_present": True,
                "nccl_intra_ok": True,
            }
        )


def test_v2_rejects_identity_mesh():
    base = {
        **H200,
        "relative_loss_err": 0.0,
        "routing_identical": True,
        "identity_mesh": True,
        "mesh_size": 1,
    }
    with pytest.raises(CheckError, match="identity mesh"):
        check_v2(base)


def test_v5_rejects_overfit_standin():
    with pytest.raises(CheckError, match="20B"):
        check_v5(
            {
                **H200,
                "loss_matches_ladder": True,
                "ckpt_resume_across_jobs": True,
                "tokens_seen": 20 * 4 * 8,
                "n_devices": 8,
                "ckpt_job_ids": ["1", "2"],
                "active_params": 1.5e8,
            }
        )
    with pytest.raises(CheckError, match="8 GPUs"):
        check_v5(
            {
                **H200,
                "n_devices": 2,
                "loss_matches_ladder": True,
                "ckpt_resume_across_jobs": True,
                "tokens_seen": 2e10,
                "ckpt_job_ids": ["1", "2"],
                "active_params": 1.5e8,
            }
        )
    with pytest.raises(CheckError, match="across jobs"):
        check_v5(
            {
                **H200,
                "n_devices": 8,
                "loss_matches_ladder": True,
                "ckpt_resume_across_jobs": True,
                "tokens_seen": 2e10,
                "ckpt_job_ids": ["1"],
                "active_params": 1.5e8,
            }
        )
    with pytest.raises(CheckError, match="two points"):
        check_v5(
            {
                **H200,
                "n_devices": 8,
                "loss_matches_ladder": True,
                "ckpt_resume_across_jobs": True,
                "tokens_seen": 2e10,
                "ckpt_job_ids": ["1", "2"],
                "active_params": 1.5e8,
                "losses": [3.46, 2.05],
                "n_loss_points": 2,
            }
        )


def test_v10_rejects_tiny_standin():
    with pytest.raises(CheckError, match="tiny_standin"):
        check_v10(
            {
                **H200,
                "lost_tasks": 0,
                "duplicated_outputs": 0,
                "dead_tokens": 0,
                "hours": 0.0,
                "tiny_standin": True,
            }
        )
    with pytest.raises(CheckError, match="72h"):
        check_v10(
            {
                **H200,
                "lost_tasks": 0,
                "duplicated_outputs": 0,
                "dead_tokens": 0,
                "hours": 0.5,
                "tiny_standin": False,
            }
        )


def test_all_checkers_have_positive():
    goods = {
        0: {
            **H200,
            "bus_bandwidth_gbps": 1.0,
            "kvm_present": False,
            "nccl_intra_ok": True,
        },
        1: {
            **H200,
            "logit_max_abs_err": 1e-6,
            "grad_check": True,
            "overfit_one_batch": True,
            "n_params": 1e7,
            "flagship_shape": True,
            "toy_embed": False,
        },
        2: {
            **H200,
            "relative_loss_err": 1e-8,
            "routing_identical": True,
            "identity_mesh": False,
            "mesh_size": 8,
            "n_devices": 8,
            "n_steps": 200,
            "ep": 8,
        },
        3: {
            **H200,
            "fp8_vs_bf16_rel": 0.001,
            "nvfp4_numerics_ok": True,
            "n_steps": 2000,
            "n_params": 1.5e8,
        },
        4: {
            **H200,
            "resume_bitwise_equal": True,
            "sdc_caught_flip": True,
            "spike_skipped_shard": True,
            "n_param_leaves": 8,
            "ckpt_bytes": 4096,
        },
        5: {
            **H200,
            "n_devices": 8,
            "loss_matches_ladder": True,
            "ckpt_resume_across_jobs": True,
            "tokens_seen": 2e10,
            "ckpt_job_ids": ["100", "101"],
            "active_params": 1.5e8,
            "tiny": False,
            "losses": [4.0, 3.8, 3.6, 3.4, 3.2, 3.0, 2.8, 2.6],
            "n_loss_points": 8,
        },
        6: {
            **H200,
            "no_collapse": True,
            "accuracy_rises_with_budget": True,
            "thoughts_decode": True,
            "adapter_trained": True,
            "decode_loss_start": 1.0,
            "decode_loss_end": 0.1,
        },
        7: {
            **H200,
            "reward_rose": True,
            "drift_halted": True,
            "planted_write_flagged": True,
            "used_gspo": True,
        },
        8: {
            **H200,
            "logprob_max_abs_err": 1e-5,
            "tiered_restore_match": True,
            "independent_ref": True,
        },
        9: {
            **H200,
            "planted_positive_replicated": True,
            "planted_negative_recorded": True,
            "ledger_rows": 2,
        },
        10: {
            **H200,
            "lost_tasks": 0,
            "duplicated_outputs": 0,
            "dead_tokens": 0,
            "hours": 72.0,
            "tiny_standin": False,
            "faults_injected": 1,
            "tasks": 10,
        },
    }
    for i, payload in goods.items():
        CHECKERS[i](payload)
