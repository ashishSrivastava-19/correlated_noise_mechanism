from __future__ import annotations

from typing import List, Optional, Union

import torch
from opacus.optimizers import DPOptimizer
from torch.optim import Optimizer


def _mark_as_processed(obj: Union[torch.Tensor, List[torch.Tensor]]):
    """
    Marks parameters that have already been used in the optimizer step.

    DP-SGD puts certain restrictions on how gradients can be accumulated. In particular,
    no gradient can be used twice - client must call .zero_grad() between
    optimizer steps, otherwise privacy guarantees are compromised.
    This method marks tensors that have already been used in optimizer steps to then
    check if zero_grad has been duly called.

    Notes:
          This is used to only mark ``p.grad_sample`` and ``p.summed_grad``

    Args:
        obj: tensor or a list of tensors to be marked
    """

    if isinstance(obj, torch.Tensor):
        obj._processed = True
    elif isinstance(obj, list):
        for x in obj:
            x._processed = True


def _check_processed_flag_tensor(x: torch.Tensor):
    """
    Checks if this gradient tensor has been previously used in optimization step.

    See Also:
        :meth:`~opacus.optimizers.optimizer._mark_as_processed`

    Args:
        x: gradient tensor

    Raises:
        ValueError
            If tensor has attribute ``._processed`` previously set by
            ``_mark_as_processed`` method
    """

    if hasattr(x, "_processed"):
        raise ValueError(
            "Gradients haven't been cleared since the last optimizer step. "
            "In order to obtain privacy guarantees you must call optimizer.zero_grad()"
            "on each step"
        )


def _check_processed_flag(obj: Union[torch.Tensor, List[torch.Tensor]]):
    """
    Checks if this gradient tensor (or a list of tensors) has been previously
    used in optimization step.

    See Also:
        :meth:`~opacus.optimizers.optimizer._mark_as_processed`

    Args:
        x: gradient tensor or a list of tensors

    Raises:
        ValueError
            If tensor (or at least one tensor from the list) has attribute
            ``._processed`` previously set by ``_mark_as_processed`` method
    """

    if isinstance(obj, torch.Tensor):
        _check_processed_flag_tensor(obj)
    elif isinstance(obj, list):
        for x in obj:
            _check_processed_flag_tensor(x)


def _generate_noise(
    std: float,
    reference: torch.Tensor,
    generator=None,
    secure_mode: bool = False,
) -> torch.Tensor:
    """
    Generates noise according to a Gaussian distribution with mean 0

    Args:
        std: Standard deviation of the noise
        reference: The reference Tensor to get the appropriate shape and device
            for generating the noise
        generator: The PyTorch noise generator
        secure_mode: boolean showing if "secure" noise need to be generated
            (see the notes)

    Notes:
        If `secure_mode` is enabled, the generated noise is also secure
        against the floating point representation attacks, such as the ones
        in https://arxiv.org/abs/2107.10138 and https://arxiv.org/abs/2112.05307.
        The attack for Opacus first appeared in https://arxiv.org/abs/2112.05307.
        The implemented fix is based on https://arxiv.org/abs/2107.10138 and is
        achieved through calling the Gaussian noise function 2*n times, when n=2
        (see section 5.1 in https://arxiv.org/abs/2107.10138).

        Reason for choosing n=2: n can be any number > 1. The bigger, the more
        computation needs to be done (`2n` Gaussian samples will be generated).
        The reason we chose `n=2` is that, `n=1` could be easy to break and `n>2`
        is not really necessary. The complexity of the attack is `2^p(2n-1)`.
        In PyTorch, `p=53` and so complexity is `2^53(2n-1)`. With `n=1`, we get
        `2^53` (easy to break) but with `n=2`, we get `2^159`, which is hard
        enough for an attacker to break.
    """
    zeros = torch.zeros(reference.shape, device=reference.device)
    if std == 0:
        return zeros
    # TODO: handle device transfers: generator and reference tensor
    # could be on different devices
    if secure_mode:
        torch.normal(
            mean=0,
            std=std,
            size=(1, 1),
            device=reference.device,
            generator=generator,
        )  # generate, but throw away first generated Gaussian sample
        sum = zeros
        for _ in range(4):
            sum += torch.normal(
                mean=0,
                std=std,
                size=reference.shape,
                device=reference.device,
                generator=generator,
            )
        return sum / 2
    else:
        return torch.normal(
            mean=0,
            std=std,
            size=reference.shape,
            device=reference.device,
            generator=generator,
        )


_VALID_MODES = {
    "DP-SGD-BASE",
    "DP-SGD-AMPLIFIED",
    "BLT",
    "Multi-Epoch-BLT",
    "BLT-Adam",
    "Multi-Epoch-BLT-Adam",
    "Single Parameter",
}

# Modes whose noise-correlation pipeline is structurally identical to plain
# "BLT" (per-step cache reset at epoch boundary).  Used in CNMOptimizer.add_noise
# and .cache so that the Adam variants share the BLT noise machinery and only
# differ in the post-noise update rule (handled by CNMAdamOptimizer.adam_step).
_BLT_LIKE_MODES = {"BLT", "BLT-Adam"}
_MULTI_EPOCH_BLT_LIKE_MODES = {"Multi-Epoch-BLT", "Multi-Epoch-BLT-Adam"}
_ALL_BLT_MODES = _BLT_LIKE_MODES | _MULTI_EPOCH_BLT_LIKE_MODES


class CNMOptimizer(DPOptimizer):
    """
    This class extends the DPOptimizer class from Opacus to provide
    the functionality of performing Correlated Noise Mechanism for
    differentially private training
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
    ):
        """
        Args:
            mode: specify the mode of operation of the optimizer.
                  Must be one of 'DP-SGD-BASE', 'DP-SGD-AMPLIFIED', 'BLT',
                  'Multi-Epoch-BLT', or 'Single Parameter'.
            steps: number of steps after which the cache is reset
            a: parameters for BLT mode
            lamda: parameters for BLT mode
            gamma: a scalar for Single Parameter mode
        """
        if mode not in _VALID_MODES:
            raise ValueError(
                f"Unknown mode '{mode}'. Must be one of {_VALID_MODES}"
            )
        if mode in _ALL_BLT_MODES:
            if a is None or lamda is None:
                raise ValueError(
                    f"Mode '{mode}' requires both 'a' and 'lamda' to be provided"
                )
        if mode == "Single Parameter":
            if gamma is None:
                raise ValueError(
                    "Mode 'Single Parameter' requires 'gamma' to be provided"
                )

        self.original_optimizer = optimizer
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.loss_reduction = loss_reduction
        self.expected_batch_size = expected_batch_size
        self.step_hook = None
        self.generator = generator
        self.secure_mode = secure_mode
        self._step_skip_queue = []
        self._is_last_step_skipped = False
        self.mode = mode
        self.a = a
        self.lamda = lamda
        self.gamma = gamma
        self.cache_state = self.reset_cache()
        self.step_counter = 0
        self.steps = steps

        for p in self.params:
            p.summed_grad = None

    def cache(self, noise, param_id=None):
        if self.mode in _ALL_BLT_MODES:
            d1 = torch.ones(self.a.shape[0]).T.view(1, -1).to(self.a.device)
            update = noise.unsqueeze(-1) * d1
            diag = torch.diag(self.lamda).to(self.lamda.device)
            self.cache_state[param_id] = (
                torch.tensordot(self.cache_state[param_id], diag, dims=([-1], [0]))
                + update
            )
        if self.mode == "Single Parameter":
            self.cache_state[param_id] = noise

    def reset_cache(self):
        return {}

    def add_noise(self):
        """
        Adds noise to clipped gradients. Stores clipped and noised result in ``p.grad``
        """

        for i, p in enumerate(self.params):
            _check_processed_flag(p.summed_grad)

            noise = _generate_noise(
                std=self.noise_multiplier * self.max_grad_norm,
                reference=p.summed_grad,
                generator=self.generator,
                secure_mode=self.secure_mode,
            )
            if self.mode == "DP-SGD-BASE" or self.mode == "DP-SGD-AMPLIFIED":
                p.grad = (p.summed_grad + noise).view_as(p)

            if self.mode in _ALL_BLT_MODES:
                noise_shape = list(noise.shape)
                noise_shape.append(self.a.shape[0])
                if i not in self.cache_state:
                    self.cache_state[i] = torch.zeros(tuple(noise_shape)).to(self.a.device)
                noise = noise - torch.tensordot(
                    self.cache_state[i], self.a, dims=([-1], [0])
                )
                p.grad = (p.summed_grad + noise).view_as(p)
                self.cache(noise, i)
            if self.mode == "Single Parameter":
                if i not in self.cache_state.keys():
                    p.grad = (p.summed_grad + noise).view_as(p)
                else:
                    p.grad = (
                        p.summed_grad + noise - (self.gamma) * self.cache_state[i]
                    ).view_as(p)
                self.cache(noise, i)
            _mark_as_processed(p.summed_grad)

        self.step_counter += 1
        if self.step_counter % self.steps == 0:
            if self.mode in _MULTI_EPOCH_BLT_LIKE_MODES:
                pass
            else:
                self.cache_state = self.reset_cache()
