from opacus.optimizers.adaclipoptimizer import AdaClipDPOptimizer
from opacus.optimizers.ddp_perlayeroptimizer import (
    DistributedPerLayerOptimizer,
    SimpleDistributedPerLayerOptimizer,
)
from opacus.optimizers.ddpoptimizer import DistributedDPOptimizer
from opacus.optimizers.ddpoptimizer_fast_gradient_clipping import (
    DistributedDPOptimizerFastGradientClipping,
)
from .optimizer import CNMOptimizer
from .adam_optimizer import CNMAdamOptimizer
from opacus.optimizers.optimizer_fast_gradient_clipping import (
    DPOptimizerFastGradientClipping,
)
from opacus.optimizers.perlayeroptimizer import DPPerLayerOptimizer


__all__ = [
    "AdaClipDPOptimizer",
    "CNMOptimizer",
    "CNMAdamOptimizer",
    "DistributedPerLayerOptimizer",
    "DistributedDPOptimizer",
    "DPOptimizer",
    "DPOptimizerFastGradientClipping",
    "DistributedDPOptimizerFastGradientlipping",
    "DPPerLayerOptimizer",
    "SimpleDistributedPerLayerOptimizer",
]


_ADAM_MODES = {"BLT-Adam", "Multi-Epoch-BLT-Adam"}


def get_optimizer_class(
    clipping: str,
    distributed: bool,
    grad_sample_mode: str = None,
    mode: str = None,
):
    """Return the optimizer class for the given configuration.

    ``mode`` is consulted only for the ``flat`` + non-distributed branch: when it
    is ``"BLT-Adam"`` or ``"Multi-Epoch-BLT-Adam"``, :class:`CNMAdamOptimizer`
    is returned; otherwise :class:`CNMOptimizer` is returned (preserving existing
    behavior for all callers that don't pass ``mode``).
    """
    if grad_sample_mode == "ghost":
        if clipping == "flat" and distributed is False:
            return DPOptimizerFastGradientClipping
        elif clipping == "flat" and distributed is True:
            return DistributedDPOptimizerFastGradientClipping
        else:
            raise ValueError(
                f"Unsupported combination of parameters. Clipping: {clipping} and grad_sample_mode: {grad_sample_mode}"
            )
    elif clipping == "flat" and distributed is False:
        if mode in _ADAM_MODES:
            return CNMAdamOptimizer
        return CNMOptimizer
    elif clipping == "flat" and distributed is True:
        return DistributedDPOptimizer
    elif clipping == "per_layer" and distributed is False:
        return DPPerLayerOptimizer
    elif clipping == "per_layer" and distributed is True:
        if grad_sample_mode == "hooks":
            return DistributedPerLayerOptimizer
        elif grad_sample_mode == "ew":
            return SimpleDistributedPerLayerOptimizer
        else:
            raise ValueError(f"Unexpected grad_sample_mode: {grad_sample_mode}")
    elif clipping == "adaptive" and distributed is False:
        return AdaClipDPOptimizer
    raise ValueError(
        f"Unexpected optimizer parameters. Clipping: {clipping}, distributed: {distributed}"
    )
