# Correlated Noise Mechanism

<!-- Replace with your actual badges -->
[![Build Status](https://img.shields.io/github/workflow/status/grim-hitman0XX/correlated_noise_mechanism)](https://github.com/grim-hitman0XX/correlated_noise_mechanism)
[![PyPI version](https://badge.fury.io/py/correlated-noise-mechanism.svg)](https://pypi.org/project/correlated-noise-mechanism/0.3.0/)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Documentation Status](https://readthedocs.org/projects/correlated-noise-mechanism/badge/?version=latest)](https://correlated-noise-mechanism.readthedocs.io/en/latest/?badge=latest)

<!-- Optional: Add project logo here -->
<!-- <div align="center">
  <img src="docs/images/logo.png" alt="Correlated Noise Mechanism Logo" width="400">
</div> -->

## Overview

**Correlated Noise Mechanism** is an open source library for enabling differentially private training of deep learning models. This library provides streaming and multi-epoch setting support with the Opacus privacy engine.

### Key Features

- 🚀 **High Performance**: Enables comparable performance to benchmarks while preserving privacy
- 🔧 **Easy Integration**: Needs minimal modification to the PyTorch training codes
- 📊 **Multiple Algorithms**: Incorporates streaming, multi-epoch correlated noise mechanism, and DP-SGD from Opacus with a better accountant
- 🔬 **Research-Grade**: Can be used to benchmark differential privacy algorithms
- 🐍 **PyTorch/NumPy Compatible**: Compatible with PyTorch

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Examples](#examples)
- [Performance](#performance)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Installation

### Prerequisites

- Python 3.7 or higher
- numpy >= 1.26.4
- torch >= 2.3.0
- torchvision >= 0.18.0
- opacus >= 1.5.2

### Install from PyPI

```bash
pip install correlated-noise-mechanism
```

### Install from Source

```bash
git clone https://github.com/yourusername/correlated_noise_mechanism.git
cd correlated_noise_mechanism
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/yourusername/correlated_noise_mechanism.git
cd correlated_noise_mechanism
pip install -e ".[dev]"
```

## Quick Start

Here's a minimal example to get you started:

```python
from correlated_noise_mechanism.privacy_engine import CNMEngine

privacy_engine = CNMEngine()
model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    epochs=EPOCHS,
    target_epsilon=epsilon,
    target_delta=delta,
    max_grad_norm=grad_norm,
    mode = "BLT",
    participation = "streaming",
    error_type="rmse",
    d = 4,
    b = 5,
    k = 8,
)
```

## Documentation

- **[Full Documentation](https://correlated-noise-mechanism.readthedocs.io/)** - Complete API reference and guides [yet to go up]
- **[Tutorials](docs/tutorials/)** - Step-by-step tutorials [will be up soon]
- **[Examples](examples/)** - Example notebooks and scripts
- **[API Reference](docs/api/)** - Detailed API documentation [will be up soon]

## Examples

Explore our example gallery:

- **[Basic Usage](examples/basic_usage.py)** - Introduction to the library
- **[Jupyter Notebooks](notebooks/)** - Interactive examples [to be included soon]
- **[Benchmarks](benchmarks/)** - Performance comparisons [to be included soon]

## Performance

### Benchmarks

| Method | $\epsilon$ | $\delta$ | Accuracy |
|--------|-------------|-------------|----------|
| BLT (Multi Epoch) | 8 | $N^{-1.1}$ | 82.4758 |
| BLT (Streaming) | 8 | $N^{-1.1}$ | 76.3740 |
| DP-SGD | 8 | $N^{-1.1}$ | 82.0090 |
| Non Private | $\infty$ | $0$ | 87.8557 |


### Core Classes

#### `CNMEngine`

```python
class CNMEngine:
    """
    A privacy engine that extends Opacus's PrivacyEngine to provide correlated noise mechanism
    for differentially private training of deep learning models. This engine supports both
    streaming and multi-epoch settings with various noise correlation patterns.

    Parameters
    ----------
    module : torch.nn.Module
        PyTorch module to be used for training
    optimizer : torch.optim.Optimizer
        Optimizer to be used for training
    data_loader : torch.utils.data.DataLoader
        DataLoader to be used for training
    target_epsilon : float
        Target epsilon to be achieved, a metric of privacy loss at differential changes in data
    target_delta : float
        Target delta to be achieved. Probability of information being leaked
    epochs : int
        Number of training epochs
    max_grad_norm : Union[float, List[float]]
        The maximum norm of the per-sample gradients. Any gradient with norm higher than this will be clipped
    mode : str
        Mode of operation: 'DP-SGD', 'BLT', 'Single Parameter', or 'Multi-Epoch-BLT'
    participation : str
        Participation pattern: 'streaming', 'cyclic', or 'minSep'
    error_type : str
        Type of error to minimize: 'rmse' or 'max'
    d : int
        Number of parameters/buffers for BLT mode
    b : int
        Minimum separation parameter for BLT mode
    k : int
        Number of columns to consider in sensitivity for BLT mode
    gamma : Optional[float]
        A scalar for Single Parameter mode
    batch_first : bool, default=True
        Flag to indicate if the input tensor has the first dimension representing the batch
    loss_reduction : str, default='mean'
        Indicates if the loss reduction is a sum or mean operation ('sum' or 'mean')
    poisson_sampling : bool, default=True
        Whether to use standard sampling required for DP guarantees
    clipping : str, default='flat'
        Per sample gradient clipping mechanism ('flat', 'per_layer', or 'adaptive')
    noise_generator : Optional[torch.Generator]
        Generator for noise
    grad_sample_mode : str, default='hooks'
        Mode for computing per sample gradients
    """
```
#### `BLTDifferentiableLossOptimizer`

```python
class BLTDifferentiableLossOptimizer:
    """
    An optimizer that implements the Banded Linear Transformation (BLT) mechanism with differentiable loss
    for optimizing noise correlation parameters. This optimizer is used internally by CNMEngine to
    find optimal parameters for the BLT mechanism that minimize error while maintaining privacy guarantees.

    Parameters
    ----------
    n : int
        Number of rounds (size of the matrix)
    d : int
        Number of buffers/parameters
    b : int, default=5
        Minimum separation parameter
    k : int, default=10
        Maximum participations
    participation_pattern : str, default='minSep'
        Pattern of participation: 'minSep', 'cyclic', or 'streaming'
    error_type : str, default='rmse'
        Type of error to minimize: 'rmse' or 'max'
    lambda_penalty : float, default=1e-7
        Penalty strength for log-barrier optimization
    device : str, default='cuda' if available else 'cpu'
        Computation device
    """
```
#### `BLTOptimizer`

```python
class BLTOptimizer:
    """
    An optimizer that implements the Banded Linear Transformation (BLT) mechanism for differentially
    private training. This optimizer provides an alternative implementation of the BLT mechanism
    that focuses on optimizing the noise correlation parameters using closed-form expressions
    and gradient-based optimization.

    Parameters
    ----------
    n : int
        Size of the matrix (number of steps)
    d : int
        Number of parameters
    b : int, default=5
        Minimum separation parameter
    k : int, default=10
        Number of columns to consider in sensitivity
    error_type : str, default='rmse'
        Type of error to minimize: 'rmse' or 'max'
    participation : str, default='minSep'
        Participation pattern: 'minSep', 'cyclic', or 'single'
    device : str, default='cuda' if available else 'cpu'
        Computation device
    """
```
### Key Methods

#### `make_private()`

```python
def make_private(
    self,
    *,
    module: nn.Module,
    optimizer: optim.Optimizer,
    criterion=nn.CrossEntropyLoss(),
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
        Mode of operation: 'DP-SGD', 'BLT', 'Single Parameter', or 'Multi-Epoch-BLT'
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
    """
```

#### `make_private_with_epsilon()`

```python
def make_private_with_epsilon(
    self,
    *,
    module: nn.Module,
    optimizer: optim.Optimizer,
    criterion=nn.CrossEntropyLoss(),
    data_loader: DataLoader,
    target_epsilon: float,
    target_delta: float,
    epochs: int,
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
    Version of make_private that calculates privacy parameters based on a given privacy budget.
    This is the recommended method for most use cases as it automatically handles noise
    multiplier calculations.

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
    target_epsilon : float
        Target epsilon to be achieved, a metric of privacy loss at differential changes in data
    target_delta : float
        Target delta to be achieved. Probability of information being leaked
    epochs : int
        Number of training epochs you intend to perform; noise_multiplier relies on this
        to calculate an appropriate sigma to ensure privacy budget of (target_epsilon,
        target_delta) at the end of epochs
    max_grad_norm : Union[float, List[float]]
        The maximum norm of the per-sample gradients. Any gradient with norm higher than
        this will be clipped to this value
    mode : str
        Mode of operation: 'DP-SGD', 'BLT', 'Single Parameter', or 'Multi-Epoch-BLT'
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
    """
```

For complete API documentation, see [docs/api.md](docs/api.md) [not yet up].

## Contributing

Contributions are welcome! Please send an email to the email id listed below to collaborate.


## Citation

If you use this library in your research, please cite:

```bibtex
@software{correlated_noise_mechanism,
  author = {Ashish Srivastava},
  title = {Correlated Noise Mechanism: Extending opacus to enable BLT Mechanisms},
  year = {2025},
  url = {https://github.com/grim-hitman0XX/correlated_noise_mechanism},
  version = {v0.2.0}
}
```

See the [open issues](https://github.com/yourusername/correlated_noise_mechanism/issues) for a full list of proposed features and known issues.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project was developed as part of a course project at IIT Madras under the course DA7450 Topics in Privacy
- A large part of this work is inspired by the tutorials and collection of Prof Krishna Pillutla
- I'd like to thank my advisor Prof Krishna Pillutla for enabling this project and helping me throughout

## Related Projects

- **[Google's Differential Privacy](https://github.com/google/differential-privacy)** - Google's open-source differential privacy library that provides tools for building differentially private applications.
- **[Opacus](https://github.com/pytorch/opacus)** - PyTorch library for training deep learning models with differential privacy, which this library extends to support correlated noise mechanisms.

## Contact

- **Author**: Ashish Srivastava
- **Email**: ashish.srivastava1919@gmail.com
- **GitHub**: [@grim-hitman0XX](https://github.com/grim-hitman0XX)
- **Project Issues**: [GitHub Issues](https://github.com/grim-hitman0XX/correlated_noise_mechanism/issues)

---

<div align="center">
  <p>Made with ❤️ by Ashish Srivastava</p>
</div>
