from __future__ import annotations

import math

from isaaclab.sensors import SensorBaseCfg
from isaaclab.utils import configclass

from .contact_stage_filter import ContactStageFilter


@configclass
class ContactStageCfg(SensorBaseCfg):
    """Configuration for the per-foot 5-stage contact filter."""

    class_type: type = ContactStageFilter

    enable: bool = True

    filter_prim_paths_expr: list[str] = list()

    # Geometry and kinematics helpers
    foot_local_normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    gravity_dir_w: tuple[float, float, float] = (0.0, 0.0, -1.0)
    h_eff_raycast_offset: float = 0.005
    h_eff_max_dist: float = 0.35
    cop_ap_axis: int = 0
    forefoot_split: float = 0.0

    eps: float = 1e-6

    # Contact hysteresis
    contact_force_on: float = 25.0
    contact_force_off: float = 12.0
    contact_area_on: float = 0.12
    contact_area_off: float = 0.06
    contact_on_frames: int = 2
    contact_off_frames: int = 2
    landing_window_frames: int = 3

    # stage score thresholds
    F_sw: float = 20.0
    F_pre: float = 20.0
    F_land: float = 35.0
    F_st: float = 80.0
    F_push: float = 30.0

    A_sw: float = 0.05
    A_pre: float = 0.05
    A_land: float = 0.10
    A_land_maintain: float = 0.07
    A_st: float = 0.25
    A_push: float = 0.15

    dF_land: float = 350.0
    dF_st: float = 220.0
    dF_push: float = 180.0
    dA_push: float = 0.35

    # Current-version focus: h_pre is intentionally wide to avoid missed pre-landing.
    v_pre: float = 0.08
    h_sw: float = 0.03
    h_pre: float = 0.18
    h_land: float = 0.08
    F_land_maintain: float = 24.0
    h_st: float = 0.10
    h_push: float = 0.10
    rho_fore0: float = 0.50

    # Swing -> PreLanding yielding behavior:
    # - "hard": swing is fully disabled when pre_core is true.
    # - "soft": swing stays eligible but quality is scaled by swing_prelanding_soft_scale.
    swing_prelanding_yield_mode: str = "soft"
    swing_prelanding_soft_scale: float = 0.5
    # During short landing window, stance quality is scaled to avoid immediate swallow of landing semantics.
    stance_during_landing_scale: float = 0.7

    # score smooth params
    beta_score: float = 4.0
    s_F: float = 10.0
    s_A: float = 0.05
    s_dF: float = 120.0
    s_dA: float = 0.20
    s_h: float = 0.01
    s_rho: float = 0.08

    # Lightweight derivative smoothing:
    # x_filt(t) = alpha * x_raw(t) + (1-alpha) * x_filt(t-1)
    derivative_filter_alpha: float = 0.8

    # weak-order prior
    eps_prior: float = 1e-4

    lambda_self_sw: float = 1.2
    lambda_self_pre: float = 0.30
    lambda_self_land: float = 0.30
    lambda_self_st: float = 1.2
    lambda_self_push: float = 0.7

    lambda_prev_sw: float = 0.9
    lambda_prev_pre: float = 0.9
    lambda_prev_land: float = 0.9
    lambda_prev_st: float = 0.9
    lambda_prev_push: float = 0.9

    # optional minimum stage duration
    enable_min_stage_duration: bool = False
    lambda_dwell: float = 0.7
    min_stage_duration_sw: float = 0.08
    min_stage_duration_pre: float = 0.03
    min_stage_duration_land: float = 0.03
    min_stage_duration_st: float = 0.06
    min_stage_duration_push: float = 0.04

    # belief update: higher value to better follow current frame evidence.
    ema_stage: float = 0.70

    # debug/self-check
    enable_self_check: bool = True
    self_check_tol: float = 1e-4

    def validate(self):
        super().validate()
        if self.contact_force_on <= self.contact_force_off:
            raise ValueError("contact_force_on must be > contact_force_off")
        if self.contact_area_on <= self.contact_area_off:
            raise ValueError("contact_area_on must be > contact_area_off")
        if self.contact_on_frames <= 0 or self.contact_off_frames <= 0:
            raise ValueError("contact_on_frames and contact_off_frames must be positive")
        if self.landing_window_frames <= 0:
            raise ValueError("landing_window_frames must be positive")
        if self.h_eff_max_dist <= 0.0:
            raise ValueError("h_eff_max_dist must be positive")
        if self.beta_score <= 0.0:
            raise ValueError("beta_score must be positive")
        if not (0.0 <= self.ema_stage <= 1.0):
            raise ValueError("ema_stage must be in [0, 1]")
        if self.s_F <= 0.0 or self.s_A <= 0.0 or self.s_dF <= 0.0 or self.s_dA <= 0.0:
            raise ValueError("all score smoothing scales must be positive")
        if self.s_h <= 0.0 or self.s_rho <= 0.0:
            raise ValueError("all score smoothing scales must be positive")
        if not (0.0 <= self.derivative_filter_alpha <= 1.0):
            raise ValueError("derivative_filter_alpha must be in [0, 1]")
        if self.eps_prior <= 0.0:
            raise ValueError("eps_prior must be positive")
        if self.lambda_dwell < 0.0:
            raise ValueError("lambda_dwell must be non-negative")
        if self.cop_ap_axis not in (0, 1):
            raise ValueError("cop_ap_axis must be 0 or 1")
        if self.h_pre <= 0.0 or self.h_land <= 0.0 or self.h_st <= 0.0 or self.h_push <= 0.0:
            raise ValueError("height thresholds must be positive")
        if self.F_land_maintain <= 0.0 or self.A_land_maintain <= 0.0:
            raise ValueError("landing maintain thresholds must be positive")
        if self.F_land_maintain > self.F_land:
            raise ValueError("F_land_maintain must be <= F_land")
        if self.A_land_maintain > self.A_land:
            raise ValueError("A_land_maintain must be <= A_land")
        if self.v_pre <= 0.0:
            raise ValueError("v_pre must be positive")
        if self.swing_prelanding_yield_mode not in ("hard", "soft"):
            raise ValueError("swing_prelanding_yield_mode must be 'hard' or 'soft'")
        if not (0.0 <= self.swing_prelanding_soft_scale <= 1.0):
            raise ValueError("swing_prelanding_soft_scale must be in [0, 1]")
        if not (0.0 < self.stance_during_landing_scale <= 1.0):
            raise ValueError("stance_during_landing_scale must be in (0, 1]")
        if any(duration < 0.0 for duration in [
            self.min_stage_duration_sw,
            self.min_stage_duration_pre,
            self.min_stage_duration_land,
            self.min_stage_duration_st,
            self.min_stage_duration_push,
        ]):
            raise ValueError("min stage durations must be non-negative")
        g_norm = math.sqrt(sum(component * component for component in self.gravity_dir_w))
        if g_norm < 1e-6:
            raise ValueError("gravity_dir_w must be non-zero")
