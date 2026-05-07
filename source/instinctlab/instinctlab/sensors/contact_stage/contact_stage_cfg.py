from __future__ import annotations

import math

from isaaclab.sensors import SensorBaseCfg
from isaaclab.utils import configclass

from .contact_stage_filter import ContactStageFilter


@configclass
class ContactStageCfg(SensorBaseCfg):
    """Configuration for the per-foot 4-stage contact filter."""

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

    # Hard 4-stage partition parameters.
    v_pre: float = 0.01
    h_zone: float = 0.14

    # Lightweight derivative smoothing:
    # x_filt(t) = alpha * x_raw(t) + (1-alpha) * x_filt(t-1)
    derivative_filter_alpha: float = 0.8

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
        if not (0.0 <= self.derivative_filter_alpha <= 1.0):
            raise ValueError("derivative_filter_alpha must be in [0, 1]")
        if self.cop_ap_axis not in (0, 1):
            raise ValueError("cop_ap_axis must be 0 or 1")
        if self.h_zone <= 0.0:
            raise ValueError("h_zone must be positive")
        if self.v_pre <= 0.0:
            raise ValueError("v_pre must be positive")
        g_norm = math.sqrt(sum(component * component for component in self.gravity_dir_w))
        if g_norm < 1e-6:
            raise ValueError("gravity_dir_w must be non-zero")
