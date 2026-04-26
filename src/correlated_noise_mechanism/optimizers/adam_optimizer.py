"""
CNMAdamOptimizer: Adam with time-varying bias correction for BLT correlated noise.

This implements **Approach 1 (CNM-AdamBC)** from ``CNM_Adam.md`` -- the
generalization of DP-AdamBC (Tang et al., AAAI 2024) to the BLT correlated-noise
setting where the per-step effective noise variance Phi_t is time-varying.

Privacy: identical guarantees to ``CNMOptimizer`` with the same ``noise_multiplier``.
The Adam moment updates and bias correction are *post-processing* of the
already-privatized gradient ``g_bar_t + z_tilde_t`` produced by
``CNMOptimizer.add_noise``.
"""
from __future__ import annotations

from typing import Optional, Union

import torch
from torch.optim import Optimizer

from .optimizer import CNMOptimizer, _ALL_BLT_MODES, _MULTI_EPOCH_BLT_LIKE_MODES


class CNMAdamOptimizer(CNMOptimizer):
    """
    Adam optimizer with time-varying bias correction for BLT correlated DP noise.

    Inherits noise generation, BLT correlation (z_tilde_t = z_t - M_t * alpha),
    and cache update from :class:`CNMOptimizer`. Replaces the inner optimizer's
    ``step()`` with a custom Adam moment update that subtracts the
    time-varying noise-variance EMA ``Psi_t`` from Adam's second moment
    ``v_hat_t``.

    Update rule per step (with t = adam_step_count after increment):

    .. code-block:: text

        m_t   = beta1 * m_{t-1} + (1-beta1) * g_tilde_t
        v_t   = beta2 * v_{t-1} + (1-beta2) * g_tilde_t**2
        Psi_t = beta2 * Psi_{t-1} + (1-beta2) * Phi_{t_eff}
        m_hat = m_t / (1 - beta1**t);  v_hat = v_t / (1 - beta2**t)
        corr  = Psi_t / (1 - beta2**t)
        denom = sqrt(clamp(v_hat - corr, min=gamma_prime))
        theta_t -= lr * m_hat / denom
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        noise_multiplier: float,
        max_grad_norm: float,
        mode: str,
        a: Optional[Union[float, torch.Tensor]] = None,
        lamda: Optional[Union[float, torch.Tensor]] = None,
        gamma: Optional[float] = None,
        steps: float,
        expected_batch_size: Optional[int],
        loss_reduction: str = "mean",
        generator=None,
        secure_mode: bool = False,
        # Adam-specific:
        beta1: float = 0.9,
        beta2: float = 0.999,
        lr: Optional[float] = None,
        gamma_prime: float = 1e-4,
        phi_schedule: Optional[torch.Tensor] = None,
    ):
        """
        Args (in addition to those of :class:`CNMOptimizer`):
            beta1: Adam first-moment decay (default 0.9).
            beta2: Adam second-moment decay (default 0.999).
            lr: Learning rate. If None, read from the inner optimizer's first
                param group (allowing users to pass ``torch.optim.Adam(p, lr=...)``).
            gamma_prime: Lower clamp for the corrected denominator
                ``v_hat - correction``; should be much larger than Adam's standard
                eps (1e-8) because the corrected second moment can occasionally go
                negative due to estimation noise. Default 1e-4 (matches DP-AdamBC).
            phi_schedule: Tensor of length ``steps`` holding the per-step noise
                variance schedule Phi_t. If None, falls back to constant
                Phi = (sigma * max_grad_norm) ** 2 (which makes this reduce to
                DP-AdamBC).
        """
        if mode not in _ALL_BLT_MODES and mode is not None:
            # Adam variants are only meaningful for BLT-style modes; in principle
            # we could support DP-SGD-BASE here but that's "DP-AdamBC" proper and
            # not part of this class's contract.
            raise ValueError(
                f"CNMAdamOptimizer requires a BLT-family mode, got mode='{mode}'. "
                f"Use one of: {sorted(_ALL_BLT_MODES)}."
            )

        super().__init__(
            optimizer=optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            mode=mode,
            a=a,
            lamda=lamda,
            gamma=gamma,
            steps=steps,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
        )

        # Resolve lr: prefer explicit arg, else read from inner optimizer.
        if lr is None:
            try:
                lr = float(self.original_optimizer.param_groups[0]["lr"])
            except (AttributeError, IndexError, KeyError) as e:
                raise ValueError(
                    "CNMAdamOptimizer: could not infer 'lr' from the inner "
                    "optimizer; pass lr=... explicitly."
                ) from e

        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.lr = float(lr)
        self.gamma_prime = float(gamma_prime)
        # Store schedule on CPU as float64 for numerical stability; values are
        # read scalar-by-scalar each step.
        if phi_schedule is not None:
            phi_schedule = phi_schedule.detach().to(dtype=torch.float64).cpu()
        self.phi_schedule = phi_schedule

        # Per-parameter Adam state (lazy init in adam_step).
        self.m_state: dict = {}
        self.v_state: dict = {}
        self.psi_state: dict = {}
        self.adam_step_count: int = 0

    # ------------------------------------------------------------------
    # Phi_t lookup
    # ------------------------------------------------------------------
    def _get_phi_t(self) -> float:
        """Return Phi_t (in Adam's input space) for the current step.

        ``phi_schedule[idx]`` is the variance of the correlated noise z_tilde_t
        added to ``summed_grad``. opacus's ``pre_step`` calls ``scale_grad``
        after ``add_noise``, dividing ``p.grad`` by ``expected_batch_size`` when
        ``loss_reduction == 'mean'``. So the variance of the noise *seen by Adam*
        (and hence the bias in v_t) is Phi_t / B^2 in that case.
        """
        if self.phi_schedule is None:
            # i.i.d. fallback: reduces to standard DP-AdamBC.
            raw = float((self.noise_multiplier * self.max_grad_norm) ** 2)
        else:
            n = self.phi_schedule.shape[0]
            t = self.adam_step_count  # already incremented in adam_step()
            if self.mode in _MULTI_EPOCH_BLT_LIKE_MODES:
                idx = min(t - 1, n - 1)
            else:
                # Per-epoch reset: schedule indexes restart each epoch.
                idx = (t - 1) % int(self.steps)
                idx = min(idx, n - 1)
            raw = float(self.phi_schedule[idx].item())

        if self.loss_reduction == "mean" and self.expected_batch_size:
            return raw / (float(self.expected_batch_size) ** 2)
        return raw

    # ------------------------------------------------------------------
    # The Adam moment update (replaces original_optimizer.step)
    # ------------------------------------------------------------------
    def adam_step(self) -> None:
        """Apply one Adam update with time-varying bias correction.

        Reads ``p.grad`` (already containing g_bar_t + z_tilde_t after
        :meth:`add_noise`) and updates ``p.data`` in place. Does not touch
        the inner optimizer.
        """
        self.adam_step_count += 1
        t = self.adam_step_count
        phi_t = self._get_phi_t()

        bias_correction1 = 1.0 - self.beta1 ** t
        bias_correction2 = 1.0 - self.beta2 ** t

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad.detach()

            if i not in self.m_state:
                self.m_state[i] = torch.zeros_like(p.data)
                self.v_state[i] = torch.zeros_like(p.data)
                self.psi_state[i] = torch.zeros_like(p.data)

            m = self.m_state[i]
            v = self.v_state[i]
            psi = self.psi_state[i]

            # m_t = beta1 * m_{t-1} + (1 - beta1) * grad
            m.mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
            # v_t = beta2 * v_{t-1} + (1 - beta2) * grad**2
            v.mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)
            # Psi_t = beta2 * Psi_{t-1} + (1 - beta2) * Phi_t
            psi.mul_(self.beta2).add_(phi_t * (1.0 - self.beta2))

            m_hat = m / bias_correction1
            v_hat = v / bias_correction2
            correction = psi / bias_correction2

            denom = torch.sqrt(torch.clamp(v_hat - correction, min=self.gamma_prime))
            p.data.add_(m_hat / denom, alpha=-self.lr)

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------
    def step(self, closure=None):
        """Execute one optimizer step.

        Mirrors the structure of opacus's ``DPOptimizer.step``: per-sample
        clipping + accumulation + correlated noise injection happen via
        :meth:`pre_step`, then we apply the custom Adam update instead of the
        inner optimizer's update. The privacy step-hook is fired exactly once,
        consistent with how privacy accounting works for ``CNMOptimizer``.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # pre_step (inherited from opacus.DPOptimizer) does:
        #   - skip-step bookkeeping
        #   - clip_and_accumulate (sets p.summed_grad)
        #   - add_noise (sets p.grad <- g_bar + z_tilde, updates BLT cache)
        #   - fires self.step_hook(self) for privacy accounting
        # Returns False if the step was skipped (e.g. due to virtual-batch
        # accumulation logic), True otherwise.
        if self.pre_step():
            self.adam_step()

        return loss
