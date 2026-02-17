import logging
from math import sqrt
from typing import List, Optional, Union

import numpy as np
import torch
from dp_accounting import dp_event as event
from dp_accounting.pld import pld_privacy_accountant as pld
from dp_accounting.rdp import compute_epsilon
from dp_accounting.rdp import rdp_privacy_accountant as rdp
from scipy import optimize as opt

logger = logging.getLogger(__name__)

MAX_SIGMA = 1e6
RDP_ORDERS = (
    [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 4.5]
    + list(range(5, 64))
    + [128, 256, 512]
)


def calculate_multiplier(target_epsilon, target_delta, sampling, step, mode):

    def objective(noise_multiplier):
        accountant = rdp.RdpAccountant(RDP_ORDERS)
        if mode == "DP-SGD-AMPLIFIED":
            dpevent = event.SelfComposedDpEvent(
                event.PoissonSampledDpEvent(
                    sampling, event.GaussianDpEvent(noise_multiplier)
                ),
                step,
            )
        elif mode == "DP-SGD-BASE":
            dpevent = event.SelfComposedDpEvent(
                event.GaussianDpEvent(noise_multiplier),
                step,
            )
        accountant.compose(dpevent)
        eps = accountant.get_epsilon(target_delta)
        return eps - target_epsilon

    optimal_noise = opt.brentq(objective, 1e-6, 1000)
    return optimal_noise


def get_column_norm(a, lamda, n):
    first_column = np.zeros(n)
    first_column[0] = 1
    for k in range(1, n):
        first_column[k] = np.sum(a * lamda ** (k - 1))

    norm = np.linalg.norm(first_column)
    return norm


def calculate_sensitivity_squared(alpha, lambda_val, d, k, b, n):
    """
    Calculate sensitivity squared based on participation schema
    """
    first_col = torch.zeros(n)
    first_col[0] = 1

    for i in range(1, n):
        entry = 0
        for j in range(d):
            entry += alpha[j] * (lambda_val[j] ** (i - 1))
        first_col[i] = entry

    sens_squared = 0
    for i in range(min(k, (n - 1) // b + 1)):
        col_idx = i * b
        col = first_col[0 : n - col_idx]
        sens_squared += torch.sum(col**2)

    return sens_squared


def epsilon_for_rho(rho, delta):
    alpha = RDP_ORDERS
    rdp_value = [alpha_i * rho for alpha_i in alpha]

    epsilon_computed = compute_epsilon(orders=alpha, rdp=rdp_value, delta=delta)
    return epsilon_computed[0]


def calculate_rho(target_epsilon, target_delta):

    def objective(rho):
        return epsilon_for_rho(rho, target_delta) - target_epsilon

    rho_solution = opt.brentq(objective, 1e-4, 100)
    return rho_solution


def get_noise_multiplier(
    *,
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    mode: str,
    a: Optional[Union[float, torch.Tensor]] = None,
    lamda: Optional[Union[float, torch.Tensor]] = None,
    gamma: Optional[float] = None,
    participation: Optional[str] = None,
    d: Optional[int] = None,
    k: Optional[int] = None,
    b: Optional[int] = None,
    epochs: Optional[int] = None,
    steps: Optional[int] = None,
    epsilon_tolerance: float = 0.01,
    **kwargs,
) -> float:
    r"""
    Computes the noise level sigma to reach a total budget of (target_epsilon, target_delta)
    at the end of epochs, with a given sample_rate for a correlated noise mechanism

    Args:
        target_epsilon: the privacy budget's epsilon
        target_delta: the privacy budget's delta
        sample_rate: the sampling rate (usually batch_size / n_data)
        epochs: the number of epochs to run
        steps: number of steps to run
        accountant: accounting mechanism used to estimate epsilon
        epsilon_tolerance: precision for the binary search
    Returns:
        The noise level sigma to ensure privacy budget of (target_epsilon, target_delta) in the correlated noise setting
    """
    if epochs is None:
        if mode == "DP-SGD-BASE" or mode == "DP-SGD-AMPLIFIED":
            raise ValueError(
                "get_noise_multiplier needs number of epochs as input to calculate noise multiplier accurately"
            )
        else:
            epochs = 1

    if (steps is None) == (epochs is None):
        raise ValueError(
            "get_noise_multiplier takes as input EITHER a number of steps or a number of epochs"
        )

    if steps is None:
        steps = int(1 / sample_rate)

    logger.info("Mode : %s", mode)
    logger.info("Participation : %s", participation)

    if mode == "DP-SGD-BASE":
        noise_multiplier = calculate_multiplier(
            target_epsilon, target_delta, sample_rate, steps * epochs, mode
        )
    elif mode == "DP-SGD-AMPLIFIED":
        noise_multiplier = calculate_multiplier(
            target_epsilon, target_delta, sample_rate, steps * epochs, mode
        )
    elif mode == "Multi-Epoch-BLT":
        if participation == "streaming":
            col_norm = get_column_norm(a.cpu().numpy(), lamda.cpu().numpy(), steps)
            noise_multiplier = sqrt((epochs * col_norm) / (2 * target_epsilon))
            logger.info("Noise Multiplier : %s", noise_multiplier)
            return noise_multiplier
        elif participation == "cyclic" or participation == "minSep":
            col_norm = calculate_sensitivity_squared(
                a.cpu().numpy(), lamda.cpu().numpy(), d, k, b, steps
            )
        else:
            raise ValueError("Participation type not recognized")
        optimal_rho = calculate_rho(target_epsilon, target_delta)
        noise_multiplier = sqrt(col_norm / (2 * optimal_rho))
    else:
        if mode == "BLT":
            col_norm = get_column_norm(a.cpu().numpy(), lamda.cpu().numpy(), steps)
        elif mode == "Single Parameter":
            col_norm = (1 - gamma ** (steps)) / (1 - gamma)
        else:
            raise ValueError(f"Unknown mode '{mode}'")

        optimal_rho = calculate_rho(target_epsilon, target_delta)
        noise_multiplier = sqrt((epochs * col_norm) / (2 * optimal_rho))
    logger.info("Noise Multiplier : %s", noise_multiplier)
    return noise_multiplier
