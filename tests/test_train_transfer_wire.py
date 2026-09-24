"""Failing tests for wiring spec 5.4 Muon transfer into train_step.

``muon_transfer_step`` is already tested. These tests lock the opt-in path
inside ``train_step``. ``muon_transfer=True`` updates each ``MUON_2D`` leaf
with that primitive, at the scheduled Muon rate, using the returned param
and the caller-supplied weight decay. AdamW stays the Jordan path. ``False``
ignores ``muon_weight_decay``.

``model.forward`` is patched with a one-logit stand-in so the test does not
need a full parameter tree. The patch still produces a real grad, so the
update the step applies is the code under test. The stub raises
``NotImplementedError`` before that update, so every test fails today.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import model
import train
from train import Batch, ParamKind, classify_param, wsd_lr


def _cfg() -> model.ModelConfig:
    return model.ModelConfig(
        d_model=4,
        n_layers=4,
        n_dense=1,
        n_moe=3,
        n_linear_attn=3,
        n_mla=1,
        n_routed_experts=2,
        n_shared_experts=0,
        top_k=1,
        expert_hidden=4,
        mtp_heads=0,
        core_block_layers=2,
        recurrence_train_mean=1.0,
        recurrence_max=1,
        vocab_size=4,
        max_context=4,
        prelude_layers=1,
        coda_layers=1,
        adapter_hidden=4,
    )


def _batch() -> Batch:
    tokens = np.array([[1, 2, 0, 1]], dtype=np.int32)
    return Batch(
        tokens=jnp.asarray(tokens),
        loss_mask=jnp.ones((1, 4), dtype=jnp.float32),
        positions=jnp.arange(4, dtype=jnp.float32)[None, :],
    )


def _params() -> dict:
    rng = np.random.default_rng(0)
    # "w" is MUON_2D (2-D, no AdamW name token). "final_norm" is AdamW.
    return {
        "w": jnp.asarray(rng.normal(size=(4, 4)).astype(np.float32)),
        "final_norm": jnp.ones((4,), dtype=jnp.float32),
    }


def _fake_forward(tokens, params, config, *, r=None, thoughts=None, truncated_recurrence=4):
    del r, thoughts, truncated_recurrence, config
    h = params["w"]
    logits = jnp.broadcast_to(h[None, :, :], (tokens.shape[0], tokens.shape[1], h.shape[0]))
    empty_p = jnp.zeros((tokens.shape[0], tokens.shape[1], 1), dtype=jnp.float32)
    empty_i = jnp.zeros((tokens.shape[0], tokens.shape[1], 1), dtype=jnp.int32)
    return model.ForwardOutput(
        logits=logits,
        mtp_logits=(),
        z_loss=jnp.float32(0.0),
        router_probs=empty_p,
        expert_ids=empty_i,
        hidden=jnp.zeros((tokens.shape[0], tokens.shape[1], 4), dtype=jnp.float32),
        r_used=1,
    )


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(model, "forward", _fake_forward)
    monkeypatch.setattr(train.model, "forward", _fake_forward)


def _muon_lr(step: int, cfg: train.TrainConfig) -> float:
    peak = float(cfg.peak_lr)
    if peak == 0.0:
        return 0.0
    return float(cfg.muon_lr) * wsd_lr(step, cfg) / peak


def test_transfer_rejects_non_bool(patched) -> None:
    """An int is not a bool. The check must run before the True branch."""
    del patched
    params = _params()
    tcfg = train.tiny_train_config()
    with pytest.raises(ValueError, match="muon_transfer") as exc:
        train.train_step(
            params,
            train.init_opt_state(params, tcfg),
            _batch(),
            _cfg(),
            tcfg,
            step=1,
            muon_transfer=1,  # type: ignore[arg-type]
        )
    assert type(exc.value) is ValueError


def test_transfer_rejects_bad_weight_decay(patched) -> None:
    del patched
    params = _params()
    opt = train.init_opt_state(params, train.tiny_train_config())
    with pytest.raises(ValueError, match="muon_weight_decay"):
        train.train_step(
            params,
            opt,
            _batch(),
            _cfg(),
            train.tiny_train_config(),
            step=1,
            muon_transfer=True,
            muon_weight_decay=float("nan"),
        )
    with pytest.raises(ValueError, match="muon_weight_decay"):
        train.train_step(
            params,
            opt,
            _batch(),
            _cfg(),
            train.tiny_train_config(),
            step=1,
            muon_transfer=True,
            muon_weight_decay=-0.1,
        )


def test_transfer_matches_muon_transfer_step(patched) -> None:
    """MUON_2D leaf equals the primitive's returned param, not a subtracted delta."""
    del patched
    params = _params()
    tcfg = train.tiny_train_config()
    opt = train.init_opt_state(params, tcfg)
    step = 1
    wd = 0.05
    out = train.train_step(
        params,
        opt,
        _batch(),
        _cfg(),
        tcfg,
        step=step,
        muon_transfer=True,
        muon_weight_decay=wd,
    )
    assert classify_param("w", params["w"]) is ParamKind.MUON_2D
    # Reconstruct the grad the step saw by rerunning the patched loss.
    tokens = _batch().tokens
    mask = _batch().loss_mask

    def packed(p):
        logits = _fake_forward(tokens, p, _cfg()).logits
        capped = train._soft_cap_j(logits[:, :-1, :], float(tcfg.softcap))
        return train._cross_entropy_j(capped, tokens[:, 1:], mask[:, 1:])

    grad = jax.grad(packed)(params)["w"]
    lr = _muon_lr(step, tcfg)
    new_p, new_m = train.muon_transfer_step(
        params["w"],
        grad,
        opt["w"]["momentum"],
        lr=lr,
        momentum_coeff=tcfg.muon_momentum,
        ns_steps=tcfg.muon_ns_steps,
        weight_decay=wd,
    )
    assert np.allclose(np.asarray(out.params["w"]), np.asarray(new_p), atol=1e-5)
    assert np.allclose(
        np.asarray(out.opt_state["w"]["momentum"]), np.asarray(new_m), atol=1e-5
    )
    # AdamW leaf is not on the transfer path.
    assert "m" in out.opt_state["final_norm"]


def test_transfer_weight_decay_differs_from_jordan(patched) -> None:
    del patched
    params = _params()
    tcfg = train.tiny_train_config()
    opt = train.init_opt_state(params, tcfg)
    true_out = train.train_step(
        params,
        opt,
        _batch(),
        _cfg(),
        tcfg,
        step=1,
        muon_transfer=True,
        muon_weight_decay=0.2,
    )
    false_out = train.train_step(
        params,
        opt,
        _batch(),
        _cfg(),
        tcfg,
        step=1,
        muon_transfer=False,
        muon_weight_decay=0.2,
    )
    assert not np.allclose(
        np.asarray(true_out.params["w"]),
        np.asarray(false_out.params["w"]),
        atol=1e-6,
    )
    assert np.allclose(
        np.asarray(true_out.params["final_norm"]),
        np.asarray(false_out.params["final_norm"]),
        atol=1e-5,
    )


def test_transfer_jit_matches_eager(patched) -> None:
    del patched
    params = _params()
    tcfg = train.tiny_train_config()
    opt = train.init_opt_state(params, tcfg)
    batch = _batch()
    cfg = _cfg()
    eager = train.train_step(
        params, opt, batch, cfg, tcfg, step=1, muon_transfer=True, muon_weight_decay=0.01
    )

    def run(params, opt, batch):
        return train.train_step(
            params,
            opt,
            batch,
            cfg,
            tcfg,
            step=1,
            muon_transfer=True,
            muon_weight_decay=0.01,
        )

    traced = jax.jit(run)(params, opt, batch)
    assert np.allclose(
        np.asarray(eager.params["w"]),
        np.asarray(traced.params["w"]),
        atol=1e-5,
    )
