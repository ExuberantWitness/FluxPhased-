"""PettingZoo ParallelEnv wrapper for GPU-vectorized MFAR environment."""

from .core import FluxPhasedPZEnv


def raw_env(**kwargs):
    """Factory function for PettingZoo compatibility."""
    return FluxPhasedPZEnv(**kwargs)
