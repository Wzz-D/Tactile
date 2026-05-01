from __future__ import annotations

from isaaclab.utils import configclass


@configclass
class PerFootGainNoiseCfg:
    enable: bool = False
    gain_range: tuple[float, float] = (1.0, 1.0)


@configclass
class PerFootBiasNoiseCfg:
    enable: bool = False
    bias_std: float = 0.0


@configclass
class RegionalGainNoiseCfg:
    enable: bool = False
    forefoot_split: float = 0.0
    forefoot_gain_range: tuple[float, float] = (1.0, 1.0)
    heel_gain_range: tuple[float, float] = (1.0, 1.0)
    medial_gain_range: tuple[float, float] = (1.0, 1.0)
    lateral_gain_range: tuple[float, float] = (1.0, 1.0)
    edge_gain_range: tuple[float, float] = (1.0, 1.0)


@configclass
class ResidualGaussianNoiseCfg:
    enable: bool = False
    multiplicative_std: float = 0.0
    additive_std: float = 0.0


@configclass
class DeadTaxelNoiseCfg:
    enable: bool = False
    dead_taxel_prob: float = 0.0


@configclass
class WeakPatchNoiseCfg:
    enable: bool = False
    patch_prob: float = 0.0
    patch_radius: float = 0.02
    attenuation_range: tuple[float, float] = (0.6, 0.9)


@configclass
class TaxelGeometryPerturbNoiseCfg:
    enable: bool = False
    per_foot_xy_offset_std: float = 0.0
    per_taxel_xy_jitter_std: float = 0.0


@configclass
class DeadzoneNoiseCfg:
    enable: bool = False
    force_threshold: float = 0.0


@configclass
class SoftSaturationNoiseCfg:
    enable: bool = False
    saturation_force: float = 0.0
    compression_alpha: float = 0.5


@configclass
class RangeBasedForceNoiseCfg:
    enable: bool = False
    low_force_threshold: float = 5.0
    high_force_threshold: float = 20.0
    low_force_multiplicative_std: float = 0.0
    mid_force_multiplicative_std: float = 0.0
    high_force_multiplicative_std: float = 0.0


@configclass
class QuantizationNoiseCfg:
    enable: bool = False
    quantization_step: float | None = None


@configclass
class TactileMeasurementProfileCfg:
    per_foot_gain: PerFootGainNoiseCfg = PerFootGainNoiseCfg()
    per_foot_bias: PerFootBiasNoiseCfg = PerFootBiasNoiseCfg()
    regional_gain: RegionalGainNoiseCfg = RegionalGainNoiseCfg()
    residual_gaussian: ResidualGaussianNoiseCfg = ResidualGaussianNoiseCfg()
    dead_taxel: DeadTaxelNoiseCfg = DeadTaxelNoiseCfg()
    weak_patch: WeakPatchNoiseCfg = WeakPatchNoiseCfg()
    taxel_geometry_perturb: TaxelGeometryPerturbNoiseCfg = TaxelGeometryPerturbNoiseCfg()
    deadzone: DeadzoneNoiseCfg = DeadzoneNoiseCfg()
    soft_saturation: SoftSaturationNoiseCfg = SoftSaturationNoiseCfg()
    range_based_force: RangeBasedForceNoiseCfg = RangeBasedForceNoiseCfg()
    quantization: QuantizationNoiseCfg = QuantizationNoiseCfg()


@configclass
class TactileDelayNoiseCfg:
    enable: bool = False
    delay_prob: float = 0.0
    max_delay_frames: int = 0


@configclass
class FrameHoldNoiseCfg:
    enable: bool = False
    hold_prob: float = 0.0


@configclass
class HysteresisNoiseCfg:
    enable: bool = False
    alpha: float = 1.0


@configclass
class DriftRandomWalkNoiseCfg:
    enable: bool = False
    per_foot_step_std: float = 0.0
    max_abs_drift: float | None = None


@configclass
class SparseTransportDropoutNoiseCfg:
    enable: bool = False
    per_foot_dropout_prob: float = 0.0


@configclass
class TactileTransportProfileCfg:
    delay: TactileDelayNoiseCfg = TactileDelayNoiseCfg()
    frame_hold: FrameHoldNoiseCfg = FrameHoldNoiseCfg()
    hysteresis: HysteresisNoiseCfg = HysteresisNoiseCfg()
    drift: DriftRandomWalkNoiseCfg = DriftRandomWalkNoiseCfg()
    sparse_dropout: SparseTransportDropoutNoiseCfg = SparseTransportDropoutNoiseCfg()


@configclass
class FootTactileNoiseCfg:
    """Measurement noise configuration for foot tactile forces."""

    enable: bool = False
    seed: int | None = None

    preserve_total_force_after_noise: bool = False
    use_structured_profiles: bool = False

    measurement_profile_cfg: TactileMeasurementProfileCfg = TactileMeasurementProfileCfg()
    transport_profile_cfg: TactileTransportProfileCfg = TactileTransportProfileCfg()

    # Legacy behavior: kept for compatibility and exact replay of the existing tactile pipeline.
    force_relative_error_max: float = 0.08
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


def make_tactile_noise_none_cfg() -> FootTactileNoiseCfg:
    return FootTactileNoiseCfg(enable=False)


def make_tactile_noise_A_calibration_cfg() -> FootTactileNoiseCfg:
    cfg = FootTactileNoiseCfg(enable=True, use_structured_profiles=True)
    cfg.measurement_profile_cfg.per_foot_gain.enable = True
    cfg.measurement_profile_cfg.per_foot_gain.gain_range = (0.94, 1.06)
    cfg.measurement_profile_cfg.per_foot_bias.enable = True
    cfg.measurement_profile_cfg.per_foot_bias.bias_std = 2.0
    return cfg


def make_tactile_noise_B_temporal_cfg() -> FootTactileNoiseCfg:
    cfg = FootTactileNoiseCfg(enable=True, use_structured_profiles=True)
    cfg.transport_profile_cfg.delay.enable = True
    cfg.transport_profile_cfg.delay.delay_prob = 0.05
    cfg.transport_profile_cfg.delay.max_delay_frames = 1
    cfg.transport_profile_cfg.frame_hold.enable = True
    cfg.transport_profile_cfg.frame_hold.hold_prob = 0.03
    cfg.transport_profile_cfg.hysteresis.enable = True
    cfg.transport_profile_cfg.hysteresis.alpha = 0.8
    return cfg


def make_tactile_noise_C_spatial_cfg() -> FootTactileNoiseCfg:
    cfg = FootTactileNoiseCfg(enable=True, use_structured_profiles=True)
    cfg.measurement_profile_cfg.regional_gain.enable = True
    cfg.measurement_profile_cfg.regional_gain.forefoot_gain_range = (0.95, 1.05)
    cfg.measurement_profile_cfg.regional_gain.heel_gain_range = (0.95, 1.05)
    cfg.measurement_profile_cfg.regional_gain.medial_gain_range = (0.95, 1.05)
    cfg.measurement_profile_cfg.regional_gain.lateral_gain_range = (0.95, 1.05)
    cfg.measurement_profile_cfg.regional_gain.edge_gain_range = (0.9, 1.0)
    cfg.measurement_profile_cfg.dead_taxel.enable = True
    cfg.measurement_profile_cfg.dead_taxel.dead_taxel_prob = 0.01
    cfg.measurement_profile_cfg.weak_patch.enable = True
    cfg.measurement_profile_cfg.weak_patch.patch_prob = 0.08
    return cfg


def make_tactile_noise_D_nonlinear_cfg() -> FootTactileNoiseCfg:
    cfg = FootTactileNoiseCfg(enable=True, use_structured_profiles=True)
    cfg.measurement_profile_cfg.deadzone.enable = True
    cfg.measurement_profile_cfg.deadzone.force_threshold = 1.0
    cfg.measurement_profile_cfg.soft_saturation.enable = True
    cfg.measurement_profile_cfg.soft_saturation.saturation_force = 35.0
    cfg.measurement_profile_cfg.soft_saturation.compression_alpha = 0.35
    cfg.measurement_profile_cfg.range_based_force.enable = True
    cfg.measurement_profile_cfg.range_based_force.low_force_multiplicative_std = 0.04
    cfg.measurement_profile_cfg.range_based_force.mid_force_multiplicative_std = 0.02
    cfg.measurement_profile_cfg.range_based_force.high_force_multiplicative_std = 0.01
    return cfg


def make_tactile_measurement_profile_cfg() -> FootTactileNoiseCfg:
    cfg = make_tactile_noise_A_calibration_cfg()
    spatial_cfg = make_tactile_noise_C_spatial_cfg().measurement_profile_cfg
    nonlinear_cfg = make_tactile_noise_D_nonlinear_cfg().measurement_profile_cfg
    cfg.measurement_profile_cfg.regional_gain = spatial_cfg.regional_gain
    cfg.measurement_profile_cfg.dead_taxel = spatial_cfg.dead_taxel
    cfg.measurement_profile_cfg.weak_patch = spatial_cfg.weak_patch
    cfg.measurement_profile_cfg.deadzone = nonlinear_cfg.deadzone
    cfg.measurement_profile_cfg.soft_saturation = nonlinear_cfg.soft_saturation
    cfg.measurement_profile_cfg.range_based_force = nonlinear_cfg.range_based_force
    cfg.measurement_profile_cfg.residual_gaussian.enable = True
    cfg.measurement_profile_cfg.residual_gaussian.multiplicative_std = 0.02
    return cfg


def make_tactile_transport_profile_cfg() -> FootTactileNoiseCfg:
    return make_tactile_noise_B_temporal_cfg()


def make_tactile_safe_sim2real_v1_cfg() -> FootTactileNoiseCfg:
    cfg = make_tactile_measurement_profile_cfg()
    cfg.transport_profile_cfg = make_tactile_transport_profile_cfg().transport_profile_cfg
    cfg.measurement_profile_cfg.quantization.enable = True
    cfg.measurement_profile_cfg.quantization.quantization_step = 0.25
    return cfg
