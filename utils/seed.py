"""Deterministic seeding helpers used throughout the project."""
import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Seed Python's random, numpy global RNG, torch CPU RNG, and
    torch CUDA RNG (if CUDA is available).

    Note: this also sets torch.backends.cudnn.deterministic = True and
    torch.backends.cudnn.benchmark = False for maximal reproducibility.
    cuDNN's deterministic mode disables auto-tuned kernel selection and
    forbids non-deterministic algorithms; expect a modest slowdown on
    convolution-heavy workloads. The cost is acceptable here because
    bit-level reproducibility across runs is required for paper-grade
    experiments.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_rng(seed: int) -> np.random.Generator:
    """Return a numpy Generator (np.random.default_rng(seed)) for local
    use. Prefer passing this around explicitly rather than relying on
    the global numpy RNG, especially in functions that may be called
    in parallel or in tests.
    """
    return np.random.default_rng(seed)
