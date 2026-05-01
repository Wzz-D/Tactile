from __future__ import annotations

from isaaclab.utils import configclass


@configclass
class FootTactileNoiseCfg:
    """Measurement noise configuration for foot tactile forces."""

    enable: bool = False
    seed: int | None = None

    preserve_total_force_after_noise: bool = False

    # Main default behavior: independent per-taxel relative error in [-max, +max].
    force_relative_error_max: float = 0.08

    # Optional Gaussian interfaces.
    additive_std: float = 0.0
    multiplicative_std: float = 0.0

    per_taxel_dropout_prob: float = 0.0
    per_foot_dropout_prob: float = 0.0

    burst_dropout_prob: float = 0.0
    burst_dropout_min_frames: int = 1
    burst_dropout_max_frames: int = 2

    delay_prob: float = 0.0
    max_delay_frames: int = 2

    clip_min: float = 0.0
    clip_max: float | None = None
    quantization_step: float | None = None
