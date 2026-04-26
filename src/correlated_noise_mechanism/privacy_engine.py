import logging
import warnings
from itertools import chain
from typing import List, Optional, Tuple, Union

import torch
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from opacus.utils.fast_gradient_clipping_utils import DPOptimizerFastGradientClipping

from .utils import (
    compute_steady_state_phi,
    get_noise_multiplier,
    precompute_noise_variance_schedule,
)
from .optimizers import get_optimizer_class
from .optimizers.optimizer import CNMOptimizer
from .blt_optimizer import BLTOptimizer
from .blt_optimizer_diffloss import BLTDifferentiableLossOptimizer

_ADAM_MODES = {"BLT-Adam", "Multi-Epoch-BLT-Adam"}
_ANY_BLT_MODE = {"BLT", "Multi-Epoch-BLT", "BLT-Adam", "Multi-Epoch-BLT-Adam"}
# kwargs consumed by CNMAdamOptimizer that should NOT be forwarded to plain
# CNMOptimizer (which would reject them).
_ADAM_ONLY_KWARGS = {"beta1", "beta2", "lr", "gamma_prime", "phi_schedule"}

logger = logging.getLogger(__name__)
from opacus.grad_sample import (
    AbstractGradSampleModule,
    GradSampleModule,
    wrap_model,
    get_gsm_class,
)


class CNMEngine(PrivacyEngine):
    def _prepare_optimizer(
        self,
        *,
        optimizer: optim.Optimizer,
        noise_multiplier: float,
        max_grad_norm: Union[float, List[float]],
        mode: str,
        a: Optional[Union[float, torch.Tensor]] = None,
        lamda: Optional[Union[float, torch.Tensor]] = None,
        gamma: Optional[float] = None,
        steps: float,
        expected_batch_size: int,
        loss_reduction: str = "mean",
        distributed: bool = False,
        clipping: str = "flat",
        noise_generator=None,
        grad_sample_mode="hooks",
        **kwargs,
    ) -> CNMOptimizer:
        if isinstance(optimizer, CNMOptimizer):
            optimizer = optimizer.original_optimizer

        generator = None
        if self.secure_mode:
            generator = self.secure_rng
        elif noise_generator is not None:
            generator = noise_generator

        optim_class = get_optimizer_class(
            clipping=clipping,
            distributed=distributed,
            grad_sample_mode=grad_sample_mode,
            mode=mode,
        )

        # BLT params come back from BLTDifferentiableLossOptimizer on whatever
        # device that optimizer chose (CUDA when available); move them to the
        # model's device so noise + cache_state + a all live together inside
        # CNMOptimizer.add_noise.
        try:
            target_device = next(iter(optimizer.param_groups[0]["params"])).device
        except (StopIteration, IndexError, KeyError):
            target_device = None
        if target_device is not None:
            if isinstance(a, torch.Tensor):
                a = a.to(target_device)
            if isinstance(lamda, torch.Tensor):
                lamda = lamda.to(target_device)

        # Filter Adam-only kwargs out for non-Adam optimizers so that plain
        # CNMOptimizer doesn't reject them.
        if mode not in _ADAM_MODES:
            kwargs = {k: v for k, v in kwargs.items() if k not in _ADAM_ONLY_KWARGS}

        return optim_class(
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
            secure_mode=self.secure_mode,
            **kwargs,
        )

    def _prepare_model(
        self,
        module: nn.Module,
        *,
        batch_first: bool = True,
        max_grad_norm: Union[float, List[float]] = 1.0,
        loss_reduction: str = "mean",
        grad_sample_mode: str = "hooks",
    ) -> AbstractGradSampleModule:
        # Ideally, validation should have been taken care of by calling
        # `get_compatible_module()`
        self.validate(module=module, optimizer=None, data_loader=None)

        # wrap
        if isinstance(module, AbstractGradSampleModule):
            if (
                module.batch_first != batch_first
                or module.loss_reduction != loss_reduction
                or type(module) is not get_gsm_class(grad_sample_mode)
            ):
                raise ValueError(
                    f"Pre-existing GradSampleModule doesn't match new arguments."
                    f"Got: module.batch_first: {module.batch_first}, module.loss_reduction: {module.loss_reduction}, type(module): {type(module)}"
                    f"Requested: batch_first:{batch_first}, loss_reduction: {loss_reduction}, grad_sample_mode: {grad_sample_mode} "
                    f"Please pass vanilla nn.Module instead"
                )

            return module
        else:
            if grad_sample_mode == "ghost":
                return wrap_model(
                    module,
                    grad_sample_mode=grad_sample_mode,
                    batch_first=batch_first,
                    loss_reduction=loss_reduction,
                    max_grad_norm=max_grad_norm,
                )
            else:
                return wrap_model(
                    module,
                    grad_sample_mode=grad_sample_mode,
                    batch_first=batch_first,
                    loss_reduction=loss_reduction,
                )

    def make_private(
        self,
        *,
        module: nn.Module,
        optimizer: optim.Optimizer,
        criterion=nn.CrossEntropyLoss(),  # Added deafult for backward compatibility
        data_loader: DataLoader,
        noise_multiplier: float,
        max_grad_norm: Union[float, List[float]],
        mode: str,
        a: Optional[Union[float, torch.Tensor]] = None,
        lamda: Optional[Union[float, torch.Tensor]] = None,
        gamma: Optional[float] = None,
        batch_first: bool = True,
        loss_reduction: str = "mean",
        poisson_sampling: bool = True,
        clipping: str = "flat",
        noise_generator=None,
        grad_sample_mode: str = "hooks",
        **kwargs,
    ) -> Tuple[GradSampleModule, CNMOptimizer, DataLoader]:
        """
        Add privacy-related responsibilities to the main PyTorch training objects:
        model, optimizer, and the data loader.

        For most use cases prefer :meth:`make_private_with_epsilon`, which
        derives the noise multiplier automatically from a target ``epsilon`` /
        ``delta`` budget. Use this method only when you have already computed
        the noise multiplier yourself.

        All of the returned objects act just like their non-private counterparts
        passed as arguments, but with added DP tasks.

        - Model is wrapped to also compute per sample gradients.
        - Optimizer is now responsible for gradient clipping and adding noise to the gradients.
        - DataLoader is updated to perform Poisson sampling.

        Notes:
            Using any other models, optimizers, or data sources during training
            will invalidate stated privacy guarantees.

        Parameters
        ----------
        module : torch.nn.Module
            PyTorch module to be used for training
        optimizer : torch.optim.Optimizer
            Optimizer to be used for training
        criterion : torch.nn.Module, default=nn.CrossEntropyLoss()
            Loss function to be used for training
        data_loader : torch.utils.data.DataLoader
            DataLoader to be used for training
        noise_multiplier : float
            The ratio of the standard deviation of the Gaussian noise to the L2-sensitivity
            of the function to which the noise is added (How much noise to add)
        max_grad_norm : Union[float, List[float]]
            The maximum norm of the per-sample gradients. Any gradient with norm higher than
            this will be clipped to this value.
        mode : str
            One of ``'DP-SGD-BASE'``, ``'DP-SGD-AMPLIFIED'``, ``'BLT'``,
            ``'Multi-Epoch-BLT'``, ``'Single Parameter'``, ``'BLT-Adam'``, or
            ``'Multi-Epoch-BLT-Adam'``. The two ``*-Adam`` variants route to
            :class:`CNMAdamOptimizer` (Adam with time-varying bias correction
            for BLT correlated noise — CNM-AdamBC).
        a : Optional[Union[float, torch.Tensor]], default=None
            Parameters for BLT mode
        lamda : Optional[Union[float, torch.Tensor]], default=None
            Parameters for BLT mode
        gamma : Optional[float], default=None
            A scalar for Single Parameter mode
        batch_first : bool, default=True
            Flag to indicate if the input tensor has the first dimension representing the batch
        loss_reduction : str, default='mean'
            Indicates if the loss reduction is a sum or mean operation ('sum' or 'mean')
        poisson_sampling : bool, default=True
            Whether to use standard sampling required for DP guarantees
        clipping : str, default='flat'
            Per sample gradient clipping mechanism ('flat', 'per_layer', or 'adaptive')
        noise_generator : Optional[torch.Generator], default=None
            Generator for noise
        grad_sample_mode : str, default='hooks'
            Mode for computing per sample gradients

        Returns
        -------
        Tuple[GradSampleModule, CNMOptimizer, DataLoader]
            Tuple of (model, optimizer, data_loader) with added privacy guarantees

        Model is a wrapper around the original model that also computes per sample
            gradients
        Optimizer is a wrapper around the original optimizer that also does
            gradient clipping and noise addition to the gradients
        DataLoader is a brand new DataLoader object, constructed to behave as
            equivalent to the original data loader, possibly with updated
            sampling mechanism. Points to the same dataset object.
        """
        if noise_generator and self.secure_mode:
            raise ValueError("Passing seed is prohibited in secure mode")

        # compare module parameter with optimizer parameters
        model_parameters = set(module.parameters())
        for p in chain.from_iterable(
            [param_group["params"] for param_group in optimizer.param_groups]
        ):
            if p not in model_parameters:
                raise ValueError(
                    "Module parameters are different than optimizer Parameters"
                )

        distributed = isinstance(module, (DPDDP, DDP))

        module = self._prepare_model(
            module,
            batch_first=batch_first,
            max_grad_norm=max_grad_norm,
            loss_reduction=loss_reduction,
            grad_sample_mode=grad_sample_mode,
        )
        if poisson_sampling:
            module.forbid_grad_accumulation()

        data_loader = self._prepare_data_loader(
            data_loader, distributed=distributed, poisson_sampling=poisson_sampling
        )

        sample_rate = 1 / len(data_loader)
        expected_batch_size = int(len(data_loader.dataset) * sample_rate)

        # expected_batch_size is the *per worker* batch size
        if distributed:
            world_size = torch.distributed.get_world_size()
            expected_batch_size /= world_size

        optimizer = self._prepare_optimizer(
            optimizer=optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            mode=mode,
            a=a,
            lamda=lamda,
            gamma=gamma,
            steps=1 / sample_rate,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            noise_generator=noise_generator,
            distributed=distributed,
            clipping=clipping,
            grad_sample_mode=grad_sample_mode,
            **kwargs,
        )

        optimizer.attach_step_hook(
            self.accountant.get_optimizer_hook_fn(sample_rate=sample_rate)
        )
        if grad_sample_mode == "ghost":
            criterion = DPOptimizerFastGradientClipping(
                module, optimizer, criterion, loss_reduction
            )
            return module, optimizer, criterion, data_loader

        return module, optimizer, data_loader

    def make_private_with_epsilon(
        self,
        *,
        module: nn.Module,
        optimizer: optim.Optimizer,
        criterion=nn.CrossEntropyLoss(),  # Added deafult for backward compatibility
        data_loader: DataLoader,
        target_epsilon: float,
        target_delta: float,
        epochs: int,
        max_grad_norm: Union[float, List[float]],
        mode: str,
        participation: str,
        error_type: str,
        d: int,
        b: int,
        k: int,
        gamma: Optional[float] = None,
        batch_first: bool = True,
        loss_reduction: str = "mean",
        poisson_sampling: bool = True,
        clipping: str = "flat",
        noise_generator=None,
        grad_sample_mode: str = "hooks",
        # Adam-only knobs (forwarded to CNMAdamOptimizer for BLT-Adam modes)
        beta1: float = 0.9,
        beta2: float = 0.999,
        lr: Optional[float] = None,
        gamma_prime: Optional[float] = None,
        **kwargs,
    ):
        """
        Version of make_private that calculates privacy parameters based on a given privacy budget.
        This is the recommended method for most use cases as it automatically handles noise
        multiplier calculations.

        Parameters
        ----------
        module : torch.nn.Module
            PyTorch module to be used for training.
        optimizer : torch.optim.Optimizer
            Inner optimizer. For ``BLT-Adam`` / ``Multi-Epoch-BLT-Adam`` modes
            pass an Adam-family optimizer (its ``lr`` is read by default);
            otherwise SGD-family is typical.
        criterion : torch.nn.Module, default=nn.CrossEntropyLoss()
            Loss function to be used for training.
        data_loader : torch.utils.data.DataLoader
            DataLoader to be used for training.
        target_epsilon : float
            Target epsilon to be achieved, a metric of privacy loss at differential changes in data.
        target_delta : float
            Target delta to be achieved. Probability of information being leaked.
        epochs : int
            Number of training epochs you intend to perform; noise_multiplier relies on this
            to calculate an appropriate sigma to ensure privacy budget of (target_epsilon,
            target_delta) at the end of epochs.
        max_grad_norm : Union[float, List[float]]
            The maximum norm of the per-sample gradients. Any gradient with norm higher than
            this will be clipped to this value.
        mode : str
            One of ``'DP-SGD-BASE'``, ``'DP-SGD-AMPLIFIED'``, ``'BLT'``,
            ``'Multi-Epoch-BLT'``, ``'Single Parameter'``, ``'BLT-Adam'``, or
            ``'Multi-Epoch-BLT-Adam'``. The two ``*-Adam`` variants route to
            :class:`CNMAdamOptimizer` (Adam with time-varying bias correction
            for BLT correlated noise — CNM-AdamBC; generalizes the DP-AdamBC
            of Tang et al., AAAI 2024). Privacy guarantees are identical to
            the underlying ``BLT`` / ``Multi-Epoch-BLT`` mode (Adam acts as
            post-processing of an already-privatized gradient).
        participation : str
            ``'streaming'``, ``'cyclic'``, or ``'minSep'``. Controls how the
            BLT participation pattern is modeled when sizing the noise.
        error_type : str
            ``'rmse'`` or ``'max'``; the BLT inner optimizer's objective.
        d : int
            Number of BLT buffers (poles in the Toeplitz parameterization).
        b : int
            Minimum participation separation (BLT/Multi-Epoch-BLT only).
        k : int
            Maximum participations per data point (BLT/Multi-Epoch-BLT only).
        gamma : Optional[float], default=None
            A scalar for Single Parameter mode.
        batch_first : bool, default=True
            Flag to indicate if the input tensor has the first dimension representing the batch.
        loss_reduction : str, default='mean'
            Indicates if the loss reduction is a sum or mean operation (``'sum'`` or ``'mean'``).
        poisson_sampling : bool, default=True
            Whether to use standard sampling required for DP guarantees.
        clipping : str, default='flat'
            Per sample gradient clipping mechanism (``'flat'``, ``'per_layer'``, or ``'adaptive'``).
        noise_generator : Optional[torch.Generator], default=None
            Generator for noise.
        grad_sample_mode : str, default='hooks'
            Mode for computing per sample gradients.

        Other Parameters
        ----------------
        beta1 : float, default=0.9
            Adam first-moment decay. *Used only for* ``BLT-Adam`` /
            ``Multi-Epoch-BLT-Adam``.
        beta2 : float, default=0.999
            Adam second-moment decay. *Used only for* ``BLT-Adam`` /
            ``Multi-Epoch-BLT-Adam``.
        lr : Optional[float], default=None
            Adam learning rate. If ``None``, read from the inner optimizer's
            first param-group, allowing ``optim.Adam(p, lr=...)`` to flow
            through. *Used only for* ``BLT-Adam`` / ``Multi-Epoch-BLT-Adam``.
        gamma_prime : Optional[float], default=None
            Stability constant for the corrected Adam denominator
            ``sqrt(max(v_hat - correction, gamma_prime))``. If ``None``, it
            is auto-tuned to ``0.01 * Phi_inf / expected_batch_size**2``
            (per CNM_Adam.md §5.5) where ``Phi_inf`` is the closed-form
            steady-state noise variance of the BLT mechanism. Pass an
            explicit value to override. *Used only for* ``BLT-Adam`` /
            ``Multi-Epoch-BLT-Adam``.

        Returns
        -------
        Tuple[GradSampleModule, CNMOptimizer, DataLoader]
            Tuple of (model, optimizer, data_loader) with added privacy guarantees.
            For ``*-Adam`` modes the second element is a :class:`CNMAdamOptimizer`.

        Notes
        -----
        For ``*-Adam`` modes the BLT optimizer's *inverse* parameters
        (``omega_hat``, ``theta_hat``) — already computed by
        :class:`BLTDifferentiableLossOptimizer` — are extracted and used to
        precompute the time-varying noise variance schedule ``Phi_t`` consumed
        by :class:`CNMAdamOptimizer`. Both schedule precomputation and Adam
        moment updates are post-processing of the privatized gradient, so
        the privacy guarantee is unchanged.
        """
        sample_rate = 1 / len(data_loader)
        n = 1 / sample_rate

        if len(self.accountant) > 0:
            warnings.warn(
                "You're calling make_private_with_epsilon with non-zero privacy budget "
                "already spent. Returned noise_multiplier assumes zero starting point, "
                "so your overall privacy budget will be higher."
            )
        a_hat = None
        lamda_hat = None
        if mode in _ANY_BLT_MODE:
            blt_optimizer = BLTDifferentiableLossOptimizer(
                n=int(n),
                d=d,
                b=b,
                k=k,
                participation_pattern=participation,
                error_type=error_type,
                lambda_penalty=1e-7,
            )
            results = blt_optimizer.optimize(num_iterations=50, lr=0.01, verbose=False)
            best_loss = results["loss"]
            logger.info("=" * 50)
            logger.info("Optimization Results:")
            logger.info("=" * 50)
            logger.info("Final objective value: %.6e", best_loss)
            logger.info("=" * 50)
            a, lamda = results["omega"], results["theta"]
            # Inverse parameters (parameters of C^{-1}) -- needed for Adam modes.
            a_hat = results["omega_hat"]
            lamda_hat = results["theta_hat"]
        else:
            a, lamda = 0, 0

        # Compute the noise multiplier ONCE -- the Adam variants share the
        # privacy mechanism with their underlying BLT mode (Adam is post-processing).
        noise_multiplier = get_noise_multiplier(
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            sample_rate=sample_rate,
            epochs=epochs,
            mode=mode,
            a=a,
            lamda=lamda,
            gamma=gamma,
            participation=participation,
            d=d,
            b=b,
            k=k,
            **kwargs,
        )

        # Adam-only: precompute the per-step noise variance schedule Phi_t from
        # the BLT inverse parameters (omega_hat, theta_hat). The schedule is in
        # "summed_grad" units; CNMAdamOptimizer applies the 1/B^2 scaling
        # internally based on its loss_reduction.
        adam_kwargs = {}
        if mode in _ADAM_MODES:
            sigma_zeta = float(noise_multiplier) * float(max_grad_norm)
            phi_schedule = precompute_noise_variance_schedule(
                a_hat, lamda_hat, int(n), sigma_zeta
            )

            # Auto-tune gamma_prime when the user didn't pass one. Per
            # CNM_Adam.md §5.5, ``gamma_prime ~= 0.01 * Phi_inf`` is the
            # principled default. Phi_inf here is in Adam-input space:
            # Phi_inf_summed_grad / expected_batch_size^2 (when
            # loss_reduction='mean', which is opacus's default and the only
            # mode that triggers ``scale_grad``).
            if gamma_prime is None:
                phi_inf = compute_steady_state_phi(a_hat, lamda_hat, sigma_zeta)
                expected_batch_size_for_scale = float(
                    int(len(data_loader.dataset) * sample_rate)
                )
                if loss_reduction == "mean" and expected_batch_size_for_scale > 0:
                    phi_inf_in_adam_space = phi_inf / (expected_batch_size_for_scale ** 2)
                else:
                    phi_inf_in_adam_space = phi_inf
                gamma_prime = max(0.01 * phi_inf_in_adam_space, 1e-12)

            adam_kwargs = dict(
                beta1=beta1,
                beta2=beta2,
                lr=lr,
                gamma_prime=gamma_prime,
                phi_schedule=phi_schedule,
            )

        return self.make_private(
            module=module,
            optimizer=optimizer,
            data_loader=data_loader,
            criterion=criterion,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            mode=mode,
            a=a,
            lamda=lamda,
            gamma=gamma,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
            noise_generator=noise_generator,
            grad_sample_mode=grad_sample_mode,
            poisson_sampling=poisson_sampling,
            clipping=clipping,
            **adam_kwargs,
            **kwargs,
        )
