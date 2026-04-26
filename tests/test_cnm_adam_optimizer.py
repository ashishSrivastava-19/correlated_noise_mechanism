"""
Direct unit tests for the CNMAdamOptimizer math.

These tests construct CNMAdamOptimizer manually (not through CNMEngine) and
exercise its `adam_step` / `_get_phi_t` methods in isolation. The full
`step()` -> `pre_step()` -> `add_noise()` flow is covered by
``test_engine_blt_adam.py``.
"""
from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from correlated_noise_mechanism.optimizers.adam_optimizer import CNMAdamOptimizer


def _make_optimizer(
    *,
    n_params: int = 1,
    phi_schedule: torch.Tensor | None = None,
    mode: str = "BLT-Adam",
    beta1: float = 0.9,
    beta2: float = 0.999,
    lr: float = 1e-2,
    gamma_prime: float = 1e-4,
    noise_multiplier: float = 0.0,
    max_grad_norm: float = 1.0,
    steps: int = 100,
    expected_batch_size: int = 1,
    loss_reduction: str = "sum",  # avoid 1/B^2 scaling for math-checking tests
) -> tuple[CNMAdamOptimizer, list[torch.nn.Parameter]]:
    params = [
        nn.Parameter(torch.zeros(3, 3, dtype=torch.float64))
        for _ in range(n_params)
    ]
    inner = torch.optim.SGD(params, lr=lr)  # acts only as a parameter holder
    a = torch.tensor([0.1, 0.1], dtype=torch.float64)
    lamda = torch.tensor([0.5, 0.3], dtype=torch.float64)
    opt = CNMAdamOptimizer(
        inner,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        mode=mode,
        a=a,
        lamda=lamda,
        steps=steps,
        expected_batch_size=expected_batch_size,
        loss_reduction=loss_reduction,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        gamma_prime=gamma_prime,
        phi_schedule=phi_schedule,
    )
    return opt, params


def test_psi_ema_matches_closed_form():
    """Psi_t = sum_{s<=t} (1-beta2) * beta2^{t-s} * Phi_s — the EMA invariant."""
    beta2 = 0.999
    n = 50
    phi_schedule = torch.linspace(0.5, 2.0, n, dtype=torch.float64)

    opt, (p,) = _make_optimizer(phi_schedule=phi_schedule, beta2=beta2)

    # Drive the optimizer with zero gradients so only psi_state evolves.
    for t in range(1, n + 1):
        p.grad = torch.zeros_like(p)
        opt.adam_step()
        # Closed form: Psi_t = sum_{s=1..t} (1-beta2) * beta2^{t-s} * Phi_{s-1}
        # (we index phi_schedule by s-1 since adam_step_count is 1-indexed and
        # _get_phi_t uses idx = (step-1) % steps).
        expected = sum(
            (1 - beta2) * (beta2 ** (t - s)) * phi_schedule[s - 1].item()
            for s in range(1, t + 1)
        )
        actual = opt.psi_state[0].flatten()[0].item()
        assert actual == pytest.approx(expected, rel=1e-9), (
            f"step {t}: psi {actual} != expected {expected}"
        )


def test_correction_constant_phi_reduces_to_phi():
    """Constant Phi_t == Phi  =>  correction_t == Phi  for all t  (DP-AdamBC limit)."""
    beta2 = 0.999
    n = 60
    phi_const = 0.7
    phi_schedule = torch.full((n,), phi_const, dtype=torch.float64)

    opt, (p,) = _make_optimizer(phi_schedule=phi_schedule, beta2=beta2)

    for t in range(1, 41):
        p.grad = torch.zeros_like(p)
        opt.adam_step()
        bias_corr2 = 1.0 - beta2 ** t
        correction_t = opt.psi_state[0].flatten()[0].item() / bias_corr2
        assert correction_t == pytest.approx(phi_const, rel=1e-9, abs=1e-12)


def test_gamma_prime_clamps_negative_denominator():
    """When v_hat < correction the denominator clamps to gamma_prime; updates stay finite.

    Without the clamp, ``sqrt(v_hat - correction)`` would yield ``NaN`` (negative argument
    to sqrt) and corrupt every subsequent step.
    """
    n = 20
    # Phi >> any plausible v_hat -> v_hat - correction is very negative each step.
    phi_schedule = torch.full((n,), 1e6, dtype=torch.float64)
    gamma_prime = 1e-2  # the floor sqrt(gamma_prime) is what's actually used

    opt, (p,) = _make_optimizer(
        phi_schedule=phi_schedule, gamma_prime=gamma_prime, beta2=0.999, lr=1e-3
    )
    p_initial = p.detach().clone()

    for _ in range(5):
        p.grad = torch.full_like(p, 0.01)  # tiny "real" signal
        opt.adam_step()

    # Most important property: nothing went NaN/Inf.
    assert torch.all(torch.isfinite(p.data))
    # Some update occurred (m_hat nonzero, denom finite).
    assert (p.data - p_initial).abs().max().item() > 0.0


def test_zero_noise_recovers_plain_adam():
    """With noise_multiplier=0 and phi_schedule of zeros, behavior matches torch.optim.Adam."""
    n = 30
    phi_schedule = torch.zeros(n, dtype=torch.float64)
    lr, beta1, beta2 = 1e-2, 0.9, 0.999

    cnm_opt, (p_cnm,) = _make_optimizer(
        phi_schedule=phi_schedule,
        noise_multiplier=0.0,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        gamma_prime=1e-12,  # avoid clamp interference
    )
    p_ref = nn.Parameter(p_cnm.detach().clone())
    ref_opt = torch.optim.Adam([p_ref], lr=lr, betas=(beta1, beta2), eps=1e-12)

    torch.manual_seed(0)
    for _ in range(15):
        g = torch.randn_like(p_cnm)
        p_cnm.grad = g
        p_ref.grad = g.clone()
        cnm_opt.adam_step()
        ref_opt.step()

    # We use a relaxed tolerance because torch.Adam's denom is sqrt(v_hat) + eps,
    # whereas CNMAdamOptimizer's is sqrt(max(v_hat - 0, gamma_prime)) =
    # sqrt(v_hat) when v_hat > gamma_prime. With gamma_prime=1e-12 they match
    # closely; small float drift is expected.
    assert torch.allclose(p_cnm.data, p_ref.data, rtol=1e-4, atol=1e-6)


def test_iid_fallback_when_phi_schedule_none():
    """phi_schedule=None falls back to constant Phi = (sigma * max_grad_norm)^2."""
    sigma = 0.5
    C = 2.0
    opt, _ = _make_optimizer(
        phi_schedule=None, noise_multiplier=sigma, max_grad_norm=C, loss_reduction="sum"
    )
    opt.adam_step_count = 7  # pretend we're mid-training
    phi = opt._get_phi_t()
    assert phi == pytest.approx((sigma * C) ** 2, rel=1e-12)


def test_phi_t_scales_by_batch_size_when_loss_reduction_is_mean():
    """loss_reduction='mean' must divide Phi by B^2 because pre_step rescales p.grad."""
    sigma_zeta = 1.0
    n = 5
    phi_raw = torch.full((n,), sigma_zeta ** 2, dtype=torch.float64)
    B = 8

    opt_sum, _ = _make_optimizer(
        phi_schedule=phi_raw, loss_reduction="sum", expected_batch_size=B
    )
    opt_mean, _ = _make_optimizer(
        phi_schedule=phi_raw, loss_reduction="mean", expected_batch_size=B
    )
    opt_sum.adam_step_count = 1
    opt_mean.adam_step_count = 1
    assert opt_sum._get_phi_t() == pytest.approx(sigma_zeta ** 2)
    assert opt_mean._get_phi_t() == pytest.approx(sigma_zeta ** 2 / B ** 2)


def test_per_epoch_reset_uses_modulo_indexing():
    """In BLT-Adam mode, schedule index resets every `steps` steps."""
    n_per_epoch = 10
    phi_schedule = torch.arange(n_per_epoch, dtype=torch.float64)  # 0,1,...,9
    opt, _ = _make_optimizer(
        phi_schedule=phi_schedule,
        mode="BLT-Adam",
        steps=n_per_epoch,
        loss_reduction="sum",
    )

    # Step 1 -> idx 0; step 11 -> idx 0 (wraps)
    opt.adam_step_count = 1
    assert opt._get_phi_t() == 0.0
    opt.adam_step_count = 11
    assert opt._get_phi_t() == 0.0
    opt.adam_step_count = 5
    assert opt._get_phi_t() == 4.0


def test_multi_epoch_mode_uses_global_step():
    """In Multi-Epoch-BLT-Adam, schedule does NOT reset; clamp at last index."""
    n = 5
    phi_schedule = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0], dtype=torch.float64)
    opt, _ = _make_optimizer(
        phi_schedule=phi_schedule,
        mode="Multi-Epoch-BLT-Adam",
        steps=n,
        loss_reduction="sum",
    )
    opt.adam_step_count = 1
    assert opt._get_phi_t() == 10.0
    opt.adam_step_count = 5
    assert opt._get_phi_t() == 50.0
    # Past the schedule end -> clamped to last entry
    opt.adam_step_count = 999
    assert opt._get_phi_t() == 50.0


def test_lr_inferred_from_inner_optimizer():
    """If lr is None, it should be read from the inner optimizer's param_group."""
    params = [nn.Parameter(torch.zeros(2, dtype=torch.float64))]
    inner = torch.optim.SGD(params, lr=0.1234)
    a = torch.tensor([0.1], dtype=torch.float64)
    lamda = torch.tensor([0.5], dtype=torch.float64)
    opt = CNMAdamOptimizer(
        inner,
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        mode="BLT-Adam",
        a=a,
        lamda=lamda,
        steps=10,
        expected_batch_size=1,
        loss_reduction="sum",
        lr=None,  # should pick up 0.1234
    )
    assert opt.lr == pytest.approx(0.1234)


def test_rejects_non_blt_mode():
    params = [nn.Parameter(torch.zeros(2))]
    inner = torch.optim.SGD(params, lr=0.1)
    with pytest.raises(ValueError):
        CNMAdamOptimizer(
            inner,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            mode="DP-SGD-BASE",  # not allowed
            steps=10,
            expected_batch_size=1,
        )
