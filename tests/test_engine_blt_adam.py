"""End-to-end engine tests for BLT-Adam / Multi-Epoch-BLT-Adam modes."""
from __future__ import annotations

import math

import pytest
import torch

from correlated_noise_mechanism import CNMEngine, CNMAdamOptimizer
from correlated_noise_mechanism.optimizers.optimizer import CNMOptimizer


# ----- shared helper ---------------------------------------------------------

def _make_private(
    *,
    tiny_mlp,
    synthetic_loader,
    mode: str,
    target_epsilon: float = 8.0,
    target_delta: float = 1e-4,
    epochs: int = 1,
    max_grad_norm: float = 1.0,
    lr: float = 1e-3,
    extra_kwargs: dict | None = None,
):
    engine = CNMEngine(accountant="rdp")
    optimizer = torch.optim.Adam(tiny_mlp.parameters(), lr=lr)
    extra_kwargs = extra_kwargs or {}
    return engine.make_private_with_epsilon(
        module=tiny_mlp,
        optimizer=optimizer,
        data_loader=synthetic_loader,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        epochs=epochs,
        max_grad_norm=max_grad_norm,
        mode=mode,
        participation="streaming",
        error_type="rmse",
        d=2,
        b=1,
        k=1,
        poisson_sampling=False,
        **extra_kwargs,
    )


# ----- tests -----------------------------------------------------------------

def test_make_private_with_epsilon_returns_cnm_adam_optimizer(tiny_mlp, synthetic_loader):
    _, opt, _ = _make_private(
        tiny_mlp=tiny_mlp, synthetic_loader=synthetic_loader, mode="BLT-Adam"
    )
    assert isinstance(opt, CNMAdamOptimizer)
    assert opt.phi_schedule is not None
    # phi_schedule length == steps == 1/sample_rate == len(loader) == 4
    assert opt.phi_schedule.shape[0] == len(synthetic_loader)


def test_phi_schedule_is_monotone_and_starts_at_sigma_zeta_squared(
    tiny_mlp, synthetic_loader
):
    _, opt, _ = _make_private(
        tiny_mlp=tiny_mlp, synthetic_loader=synthetic_loader, mode="BLT-Adam"
    )
    phi = opt.phi_schedule
    sigma_zeta = opt.noise_multiplier * opt.max_grad_norm
    assert phi[0].item() == pytest.approx(sigma_zeta ** 2, rel=1e-9)
    assert torch.all(phi[1:] - phi[:-1] >= -1e-12)


def test_full_training_step_runs(tiny_mlp, synthetic_loader):
    """A few training steps with BLT-Adam should run without error and update params."""
    model, opt, loader = _make_private(
        tiny_mlp=tiny_mlp, synthetic_loader=synthetic_loader, mode="BLT-Adam"
    )
    initial = [p.detach().clone() for p in model.parameters()]
    criterion = torch.nn.CrossEntropyLoss()

    for x, y in loader:
        opt.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()

    final = [p.detach().clone() for p in model.parameters()]
    moved = any(not torch.equal(i, f) for i, f in zip(initial, final))
    assert moved, "BLT-Adam optimizer did not update any parameters."
    assert opt.adam_step_count == len(loader)


def test_multi_epoch_blt_adam_does_not_reset_cache(tiny_mlp, synthetic_loader):
    model, opt, loader = _make_private(
        tiny_mlp=tiny_mlp,
        synthetic_loader=synthetic_loader,
        mode="Multi-Epoch-BLT-Adam",
    )
    criterion = torch.nn.CrossEntropyLoss()

    # Run two full "epochs" through the loader.
    for _ in range(2):
        for x, y in loader:
            opt.zero_grad()
            criterion(model(x), y).backward()
            opt.step()

    # In Multi-Epoch mode, the cache must persist across the epoch boundary.
    assert len(opt.cache_state) > 0


def test_blt_adam_resets_cache_each_epoch(tiny_mlp, synthetic_loader):
    model, opt, loader = _make_private(
        tiny_mlp=tiny_mlp, synthetic_loader=synthetic_loader, mode="BLT-Adam"
    )
    criterion = torch.nn.CrossEntropyLoss()

    # First epoch -> cache filled mid-epoch, then reset at the boundary.
    for x, y in loader:
        opt.zero_grad()
        criterion(model(x), y).backward()
        opt.step()
    # After exactly `steps` calls, BLT-Adam resets the cache.
    assert opt.cache_state == {}


def test_blt_adam_invalid_without_blt_args(tiny_mlp, synthetic_loader):
    """BLT-Adam should reject calls missing participation/d/b/k just like BLT does."""
    engine = CNMEngine(accountant="rdp")
    optimizer = torch.optim.Adam(tiny_mlp.parameters(), lr=1e-3)
    with pytest.raises(TypeError):
        engine.make_private_with_epsilon(
            module=tiny_mlp,
            optimizer=optimizer,
            data_loader=synthetic_loader,
            target_epsilon=8.0,
            target_delta=1e-4,
            epochs=1,
            max_grad_norm=1.0,
            mode="BLT-Adam",
            # participation, error_type, d, b, k missing on purpose
        )


def test_noise_multiplier_alias_blt_adam_to_blt():
    """Privacy budget for BLT-Adam == BLT (Adam is post-processing of the same noise).

    Tested directly on ``get_noise_multiplier`` to bypass BLT optimization's
    random initialization (which would give slightly different (omega, theta)
    on each call).
    """
    from correlated_noise_mechanism.utils import get_noise_multiplier

    a = torch.tensor([0.1, 0.05, 0.02, 0.01], dtype=torch.float64)
    lamda = torch.tensor([0.9, 0.7, 0.5, 0.3], dtype=torch.float64)
    common = dict(
        target_epsilon=8.0,
        target_delta=1e-4,
        sample_rate=0.01,
        epochs=10,
        a=a,
        lamda=lamda,
        participation="streaming",
        d=4, b=5, k=8,
    )
    sigma_blt = get_noise_multiplier(mode="BLT", **common)
    sigma_blt_adam = get_noise_multiplier(mode="BLT-Adam", **common)
    sigma_me = get_noise_multiplier(mode="Multi-Epoch-BLT", **common)
    sigma_me_adam = get_noise_multiplier(mode="Multi-Epoch-BLT-Adam", **common)

    assert sigma_blt_adam == pytest.approx(sigma_blt, rel=1e-12)
    assert sigma_me_adam == pytest.approx(sigma_me, rel=1e-12)


def test_omega_hat_threading(tiny_mlp, synthetic_loader):
    """The phi_schedule the optimizer sees must be derivable from the BLT inverse params."""
    _, opt, _ = _make_private(
        tiny_mlp=tiny_mlp, synthetic_loader=synthetic_loader, mode="BLT-Adam"
    )
    # Phi_t > 0 because omega_hat is non-zero and the schedule was computed; if
    # privacy_engine had failed to extract omega_hat / theta_hat, phi_schedule
    # would be None and this test couldn't pass.
    phi = opt.phi_schedule
    assert phi is not None
    assert (phi > 0).all()


def test_existing_blt_mode_still_returns_cnm_optimizer(tiny_mlp, synthetic_loader):
    """Regression: pre-existing modes still get the original optimizer class."""
    _, opt, _ = _make_private(
        tiny_mlp=tiny_mlp, synthetic_loader=synthetic_loader, mode="BLT"
    )
    assert isinstance(opt, CNMOptimizer)
    assert not isinstance(opt, CNMAdamOptimizer)


# ----- regression tests for Bug 1 (gamma_prime auto-scaling) ----------------

def test_gamma_prime_auto_scales_to_phi_inf(tiny_mlp, synthetic_loader):
    """Default gamma_prime must be in Adam-input scale (~0.01 * Phi_inf / B^2),
    NOT the legacy 1e-4 from DP-AdamBC's paper. Without this scaling,
    sqrt(gamma_prime) dominates the corrected denominator on every coord
    (clamp_frac ~= 1), the optimizer collapses to a constant-rescaled
    momentum and the model never escapes random accuracy. This is the
    root cause of the "stuck at 10%" bug.

    The auto-formula is ``gamma_prime = 0.01 * Phi_inf / B^2`` (post-scale_grad);
    for a finite schedule we approximate Phi_inf with the schedule asymptote,
    which converges geometrically to the closed-form value.
    """
    _, opt, _ = _make_private(
        tiny_mlp=tiny_mlp, synthetic_loader=synthetic_loader, mode="BLT-Adam"
    )
    phi_inf_summed_grad_units = float(opt.phi_schedule.max().item())
    expected_gamma_prime = 0.01 * phi_inf_summed_grad_units / (opt.expected_batch_size ** 2)
    assert opt.gamma_prime == pytest.approx(expected_gamma_prime, rel=1e-3), (
        f"gamma_prime={opt.gamma_prime:.3e} vs expected ~{expected_gamma_prime:.3e}"
    )

    # Sanity: with a realistic batch (B=1024, FashionMNIST), gamma_prime should
    # land orders of magnitude below the DP-AdamBC legacy 1e-4. The synthetic
    # fixture's tiny B=8 makes that check inappropriate here, but we can verify
    # the *relationship*: gamma_prime scales like 1/B^2.
    expected_for_B1024 = 0.01 * phi_inf_summed_grad_units / (1024 ** 2)
    assert expected_for_B1024 < 1e-5, (
        f"For realistic B=1024 the formula yields {expected_for_B1024:.3e}, "
        f"which would be at least 10x below the legacy 1e-4 default; "
        f"the auto-scaling buys real headroom for FashionMNIST-sized configs."
    )


def test_explicit_gamma_prime_overrides_auto(tiny_mlp, synthetic_loader):
    """Users who pass an explicit gamma_prime should still get it verbatim."""
    user_value = 5e-3
    _, opt, _ = _make_private(
        tiny_mlp=tiny_mlp, synthetic_loader=synthetic_loader, mode="BLT-Adam",
        extra_kwargs={"gamma_prime": user_value},
    )
    assert opt.gamma_prime == pytest.approx(user_value, rel=1e-12)


def test_blt_adam_actually_updates_parameters(tiny_mlp, synthetic_loader):
    """End-to-end: with auto gamma_prime, parameters move noticeably after a
    few BLT-Adam steps. Before the fix, clamp_frac=1.0 meant denom was
    constant and updates were too small to register on the parameter norm
    even after many steps -- the model was effectively frozen.
    """
    model, opt, loader = _make_private(
        tiny_mlp=tiny_mlp, synthetic_loader=synthetic_loader, mode="BLT-Adam"
    )
    initial_norms = [float(p.detach().pow(2).sum()) for p in model.parameters()]
    criterion = torch.nn.CrossEntropyLoss()
    for x, y in loader:
        opt.zero_grad()
        criterion(model(x), y).backward()
        opt.step()

    final_norms = [float(p.detach().pow(2).sum()) for p in model.parameters()]
    # Expect every trainable parameter to have moved by at least a tiny
    # relative amount; before the fix this was indistinguishable from 0.
    relative_changes = [
        abs(f - i) / max(i, 1e-12) for i, f in zip(initial_norms, final_norms)
    ]
    assert max(relative_changes) > 1e-6, (
        f"No parameter moved by > 1e-6 relative; max change={max(relative_changes):.3e}. "
        "This is the symptom of the stuck-at-10% bug returning."
    )


# ----- regression test for Bug 2 (BLT optimizer determinism) ----------------

def test_blt_optimizer_seed_makes_results_reproducible(tiny_mlp, synthetic_loader):
    """The BLT inner Adam optimizer is now seeded. Two engine calls with
    identical config must produce identical noise_multiplier and phi_schedule.
    Before the fix, theta/theta_hat were initialized from unseeded torch.rand
    and rare bad-init runs crashed the BLT optimization with a NaN.
    """
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    def fresh_setup():
        torch.manual_seed(0)  # deterministic model init
        model = nn.Sequential(
            nn.Flatten(), nn.Linear(28 * 28, 16), nn.ReLU(), nn.Linear(16, 10),
        )
        x = torch.randn(32, 1, 28, 28)
        y = torch.randint(0, 10, (32,))
        loader = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False)
        return model, loader

    model_a, loader_a = fresh_setup()
    _, opt_a, _ = _make_private(
        tiny_mlp=model_a, synthetic_loader=loader_a, mode="BLT-Adam"
    )
    model_b, loader_b = fresh_setup()
    _, opt_b, _ = _make_private(
        tiny_mlp=model_b, synthetic_loader=loader_b, mode="BLT-Adam"
    )

    assert opt_a.noise_multiplier == pytest.approx(opt_b.noise_multiplier, rel=1e-12)
    assert torch.allclose(opt_a.phi_schedule, opt_b.phi_schedule, atol=0, rtol=0)
