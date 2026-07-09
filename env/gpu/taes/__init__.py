"""TAES asymmetric engagement testbed.

2 phased-array radars + commander + laser vs adaptive jammer + N targets.
Purpose-built for TAES WP0-WP3 regime validation; does NOT reuse IQ-level
MFARVecEnv (overkill for kill-chain coupling experiments).
"""

from .taes_env import TAESVecEnv

__all__ = ["TAESVecEnv"]
