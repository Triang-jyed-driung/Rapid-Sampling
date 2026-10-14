"""CUDA histogram sampler. Import Torch before loading its native extension."""
import torch as _torch
from _rapid_sampling import (
    create_rng,
    sample,
    sample_with_penalties,
)

__version__ = "0.1.0"
__all__ = [
    "create_rng",
    "sample",
    "sample_with_penalties",
]
