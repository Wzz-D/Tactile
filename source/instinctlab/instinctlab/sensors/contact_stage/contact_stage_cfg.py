from __future__ import annotations

import math

from isaaclab.sensors import SensorBaseCfg
from isaaclab.utils import configclass

from .contact_stage_filter import ContactStageFilter


@configclass
class ContactStageDecisionRandomizationCfg:
    """Episode-level threshold randomization for tactile-to-stage interpretation."""

    enable: bool = False
    contact_force_on_range: tuple[float, float] | None = None
    contact_force_off_range: tuple[float, float] | None = None
    contact_area_on_range: tuple[float, float] | None = None
    contact_area_off_range: tuple[float, float] | None = None
    derivative_filter_alpha_range: tuple[float, float] | None = None


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
    decision_randomization_cfg: ContactStageDecisionRandomizationCfg = ContactStageDecisionRandomizationCfg()

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
        rand_cfg = self.decision_randomization_cfg
        for name, value_range in (
            ("contact_force_on_range", rand_cfg.contact_force_on_range),
            ("contact_force_off_range", rand_cfg.contact_force_off_range),
            ("contact_area_on_range", rand_cfg.contact_area_on_range),
            ("contact_area_off_range", rand_cfg.contact_area_off_range),
            ("derivative_filter_alpha_range", rand_cfg.derivative_filter_alpha_range),
        ):
            if value_range is None:
                continue
            if value_range[0] > value_range[1]:
                raise ValueError(f"{name} must satisfy low <= high")
        if rand_cfg.contact_force_on_range is not None and rand_cfg.contact_force_off_range is not None:
            if rand_cfg.contact_force_on_range[0] <= rand_cfg.contact_force_off_range[1]:
                raise ValueError("contact_force_on_range must stay strictly above contact_force_off_range")
        if rand_cfg.contact_area_on_range is not None and rand_cfg.contact_area_off_range is not None:
            if rand_cfg.contact_area_on_range[0] <= rand_cfg.contact_area_off_range[1]:
                raise ValueError("contact_area_on_range must stay strictly above contact_area_off_range")
        if rand_cfg.derivative_filter_alpha_range is not None:
            low, high = rand_cfg.derivative_filter_alpha_range
            if low < 0.0 or high > 1.0:
                raise ValueError("derivative_filter_alpha_range must lie in [0, 1]")


def make_contact_stage_noise_E_threshold_cfg() -> ContactStageDecisionRandomizationCfg:
    return ContactStageDecisionRandomizationCfg(
        enable=True,
        contact_force_on_range=(22.0, 28.0),
        contact_force_off_range=(10.0, 14.0),
        contact_area_on_range=(0.10, 0.14),
        contact_area_off_range=(0.05, 0.07),
        derivative_filter_alpha_range=(0.75, 0.85),
    )
