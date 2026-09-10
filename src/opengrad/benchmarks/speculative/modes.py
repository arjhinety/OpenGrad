"""Speculative decoding evaluation modes."""

from __future__ import annotations

from enum import Enum


class SpeculativeMode(str, Enum):
    MODE_A_AR = "MODE_A_AR"  # Standard Autoregressive decoding baseline
    MODE_B_MTP = "MODE_B_MTP"  # Native Multi-Token Prediction heads
    MODE_C_DRAFT = "MODE_C_DRAFT"  # Separate draft model speculative decoding
    MODE_D_DSPARK = "MODE_D_DSPARK"  # DSpark speculative inference engine
    MODE_E_EXPERIMENTAL = "MODE_E_EXPERIMENTAL"  # Future custom verifier/draft algorithms
