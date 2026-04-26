"""Math unit tests for the per-step noise variance schedule Phi_t."""
from __future__ import annotations

import math

import pytest
import torch

from correlated_noise_mechanism.utils import (
    compute_steady_state_phi,
    precompute_noise_variance_schedule,
)


def test_phi_0_equals_sigma_zeta_squared(blt_inverse_params):
    """c_hat[0] == 1, so Phi_0 == (sigma * zeta) ** 2 for any (omega_hat, theta_hat)."""
    a_hat, lamda_hat = blt_inverse_params
    sigma_zeta = 1.7
    phi = precompute_noise_variance_schedule(a_hat, lamda_hat, n=50, sigma_zeta=sigma_zeta)
    assert phi[0].item() == pytest.approx(sigma_zeta ** 2, rel=1e-12)


def test_monotonically_non_decreasing(blt_inverse_params):
    """Phi_{t+1} = Phi_t + (sigma*zeta)^2 * c_hat_{t+1}^2 >= Phi_t."""
    a_hat, lamda_hat = blt_inverse_params
    phi = precompute_noise_variance_schedule(a_hat, lamda_hat, n=200, sigma_zeta=2.0)
    diffs = phi[1:] - phi[:-1]
    assert torch.all(diffs >= -1e-12), f"min diff = {diffs.min().item()}"


def test_steady_state_matches_long_truncation(blt_inverse_params):
    """compute_steady_state_phi should agree with phi_schedule[-1] at large n."""
    a_hat, lamda_hat = blt_inverse_params
    sigma_zeta = 1.3
    phi_long = precompute_noise_variance_schedule(
        a_hat, lamda_hat, n=20000, sigma_zeta=sigma_zeta
    )
    phi_inf_closed = compute_steady_state_phi(a_hat, lamda_hat, sigma_zeta)
    assert phi_long[-1].item() == pytest.approx(phi_inf_closed, rel=1e-6)


def test_iid_reduction_when_omega_hat_is_zero():
    """With omega_hat == 0 there is no correlation; Phi_t == const for all t."""
    a_hat = torch.zeros(4, dtype=torch.float64)
    lamda_hat = torch.tensor([0.9, 0.8, 0.5, 0.2], dtype=torch.float64)
    sigma_zeta = 0.7
    phi = precompute_noise_variance_schedule(a_hat, lamda_hat, n=100, sigma_zeta=sigma_zeta)
    expected = sigma_zeta ** 2
    assert torch.allclose(phi, torch.full_like(phi, expected), atol=1e-12)


def test_steady_state_iid_case():
    """With omega_hat == 0, Phi_inf collapses to (sigma*zeta)^2."""
    a_hat = torch.zeros(3, dtype=torch.float64)
    lamda_hat = torch.tensor([0.9, 0.5, 0.1], dtype=torch.float64)
    sigma_zeta = 2.0
    assert compute_steady_state_phi(a_hat, lamda_hat, sigma_zeta) == pytest.approx(
        sigma_zeta ** 2, rel=1e-12
    )


def test_recursive_matches_cumsum(blt_inverse_params):
    """Recursive Appendix-C state update should yield the same Phi_t as the cumsum form."""
    a_hat, lamda_hat = blt_inverse_params
    sigma_zeta = 1.0
    n = 200

    phi_vec = precompute_noise_variance_schedule(a_hat, lamda_hat, n=n, sigma_zeta=sigma_zeta)

    # Recursive form per Appendix C of CNM_Adam.md.
    s = a_hat.clone().to(dtype=torch.float64)  # s_{1,k} = omega_hat_k
    S = 1.0  # S_0 = c_hat_0^2 = 1
    phi_rec = [sigma_zeta ** 2 * S]
    for t in range(1, n):
        if t == 1:
            c_hat_t = s.sum().item()
        else:
            s = s * lamda_hat
            c_hat_t = s.sum().item()
        S += c_hat_t ** 2
        phi_rec.append(sigma_zeta ** 2 * S)
    phi_rec = torch.tensor(phi_rec, dtype=torch.float64)

    assert torch.allclose(phi_vec, phi_rec, atol=1e-10, rtol=1e-10)


def test_n_equals_one():
    """Schedule of length 1 returns just Phi_0."""
    a_hat = torch.tensor([-0.1], dtype=torch.float64)
    lamda_hat = torch.tensor([0.5], dtype=torch.float64)
    phi = precompute_noise_variance_schedule(a_hat, lamda_hat, n=1, sigma_zeta=3.0)
    assert phi.shape == (1,)
    assert phi[0].item() == pytest.approx(9.0, rel=1e-12)


def test_steady_state_rejects_unstable_theta():
    """Theta_hat values that produce 1 - theta_k * theta_l <= 0 should raise."""
    a_hat = torch.tensor([0.1, 0.1], dtype=torch.float64)
    lamda_hat = torch.tensor([1.0, 1.0], dtype=torch.float64)  # 1 - 1*1 = 0
    with pytest.raises(ValueError):
        compute_steady_state_phi(a_hat, lamda_hat, 1.0)


def test_accepts_numpy_input(blt_inverse_params):
    """API should accept numpy arrays (BLT optimizer historically returned numpy)."""
    import numpy as np

    a_hat, lamda_hat = blt_inverse_params
    phi_torch = precompute_noise_variance_schedule(
        a_hat, lamda_hat, n=20, sigma_zeta=1.0
    )
    phi_numpy = precompute_noise_variance_schedule(
        a_hat.numpy(), lamda_hat.numpy(), n=20, sigma_zeta=1.0
    )
    assert torch.allclose(phi_torch, phi_numpy, atol=1e-12)
