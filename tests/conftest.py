"""Shared pytest fixtures for the correlated_noise_mechanism test suite."""
from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@pytest.fixture(autouse=True)
def _seed_everything():
    """Reset all RNGs before every test for determinism."""
    seed = 0xC0FFEE
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    yield


@pytest.fixture
def tiny_mlp() -> nn.Module:
    """A 2-layer MLP that takes 28*28 inputs and produces 10 logits."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 16),
        nn.ReLU(),
        nn.Linear(16, 10),
    )


@pytest.fixture
def synthetic_loader() -> DataLoader:
    """Tiny FashionMNIST-shaped synthetic loader: 32 examples, 4 batches of 8."""
    n = 32
    x = torch.randn(n, 1, 28, 28)
    y = torch.randint(0, 10, (n,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=8, shuffle=False)


@pytest.fixture
def blt_inverse_params():
    """A small valid set of (omega_hat, theta_hat) for use in math tests.

    Values were chosen so that |theta_hat_k * theta_hat_l| < 1 (steady state
    converges) and omega_hat is non-zero. They are NOT the output of an actual
    BLT optimization -- the math tests don't require optimality, just validity.
    """
    a_hat = torch.tensor([-0.1, -0.05, -0.02, -0.01], dtype=torch.float64)
    lamda_hat = torch.tensor([0.95, 0.85, 0.7, 0.5], dtype=torch.float64)
    return a_hat, lamda_hat
