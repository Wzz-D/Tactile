from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Optional, Sequence

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_rotate_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg


def _zero_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)


def _get_tactile_data(env: "ManagerBasedRLEnv", sensor_cfg: SceneEntityCfg):
    """Best-effort tactile lookup. Returns None when the sensor is absent in this env variant."""
    try:
        tactile = env.scene.sensors[sensor_cfg.name]
    except KeyError:
        return None, None
    return tactile, tactile.data


def _infer_body_sides(body_names: list[str]) -> list[str]:
    """Infer left/right side tags from body names."""
    sides: list[str] = []
    for name in body_names:
        lower = name.lower()
        if "left" in lower or lower.startswith("l_") or "_l_" in lower:
            sides.append("left")
        elif "right" in lower or lower.startswith("r_") or "_r_" in lower:
            sides.append("right")
        else:
            # Fallback to left to keep the term numerically stable instead of crashing.
            sides.append("left")
    return sides


def _point_in_polygon_even_odd(points_xy: torch.Tensor, polygon_xy: torch.Tensor) -> torch.Tensor:
    """Even-odd rule point-in-polygon test.

    Args:
        points_xy: (N, 2)
        polygon_xy: (P, 2), not necessarily closed.
    Returns:
        inside: (N,) boolean
    """
    x = points_xy[:, 0:1]  # (N,1)
    y = points_xy[:, 1:2]  # (N,1)
    x1 = polygon_xy[:, 0].unsqueeze(0)  # (1,P)
    y1 = polygon_xy[:, 1].unsqueeze(0)  # (1,P)
    x2 = torch.roll(x1, shifts=-1, dims=1)  # (1,P)
    y2 = torch.roll(y1, shifts=-1, dims=1)  # (1,P)

    # Edge crosses the horizontal ray on +x side.
    cond = (y1 > y) != (y2 > y)
    x_inter = (x2 - x1) * (y - y1) / ((y2 - y1) + 1e-12) + x1
    hits = cond & (x < x_inter)
    return (hits.sum(dim=1) % 2) == 1


def _distance_point_to_polygon_boundary(points_xy: torch.Tensor, polygon_xy: torch.Tensor) -> torch.Tensor:
    """Shortest Euclidean distance from points to polygon boundary.

    Args:
        points_xy: (N, 2)
        polygon_xy: (P, 2), not necessarily closed.
    Returns:
        dist: (N,)
    """
    seg_start = polygon_xy  # (P,2)
    seg_end = torch.roll(polygon_xy, shifts=-1, dims=0)  # (P,2)
    seg_vec = seg_end - seg_start  # (P,2)

    rel = points_xy.unsqueeze(1) - seg_start.unsqueeze(0)  # (N,P,2)
    denom = (seg_vec * seg_vec).sum(dim=-1).clamp_min(1e-12).unsqueeze(0)  # (1,P)
    t = ((rel * seg_vec.unsqueeze(0)).sum(dim=-1) / denom).clamp(0.0, 1.0)  # (N,P)
    proj = seg_start.unsqueeze(0) + t.unsqueeze(-1) * seg_vec.unsqueeze(0)  # (N,P,2)
    dist = torch.norm(points_xy.unsqueeze(1) - proj, dim=-1)  # (N,P)
    return dist.min(dim=1).values


def _signed_distance_to_polygon(points_xy: torch.Tensor, polygon_xy: torch.Tensor) -> torch.Tensor:
    """Signed distance to polygon boundary: positive inside, negative outside."""
    unsigned = _distance_point_to_polygon_boundary(points_xy, polygon_xy)
    inside = _point_in_polygon_even_odd(points_xy, polygon_xy)
    return torch.where(inside, unsigned, -unsigned)


class foot_cop_margin(ManagerTermBase):
    """Reward COP staying away from the support boundary defined by tactile outline polygons."""

    def __init__(self, cfg: "RewardTermCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._sensor_cfg = cfg.params.get("sensor_cfg", SceneEntityCfg("foot_tactile"))
        self._tactile = None
        self._sensor_available = False
        self._num_bodies = int(cfg.params.get("num_feet", 2))
        self._body_has_polygon = torch.zeros(self._num_bodies, device=env.device, dtype=torch.bool)
        self._body_polygons: list[torch.Tensor | None] = [None for _ in range(self._num_bodies)]

        try:
            self._tactile = env.scene.sensors[self._sensor_cfg.name]
        except KeyError:
            self._tactile = None
            return

        self._build_outline_cache()

    def _build_outline_cache(self) -> None:
        """Cache per-body outline polygon tensors once to avoid per-step reconstruction."""
        tactile = self._tactile
        if tactile is None:
            self._sensor_available = False
            return

        template_cfg = tactile.cfg.template_cfg
        left_outline = getattr(template_cfg, "left_outline_xy", None)
        right_outline = getattr(template_cfg, "right_outline_xy", None)
        if left_outline is None or right_outline is None:
            self._sensor_available = False
            return

        left_outline_t = torch.tensor(left_outline, device=self._env.device, dtype=torch.float32)
        right_outline_t = torch.tensor(right_outline, device=self._env.device, dtype=torch.float32)
        if (
            left_outline_t.ndim != 2
            or right_outline_t.ndim != 2
            or left_outline_t.shape[-1] != 2
            or right_outline_t.shape[-1] != 2
            or left_outline_t.shape[0] < 3
            or right_outline_t.shape[0] < 3
        ):
            self._sensor_available = False
            return

        body_names = tactile.body_names
        body_sides = getattr(tactile, "_body_sides", None)
        if body_sides is None or len(body_sides) != len(body_names):
            body_sides = _infer_body_sides(body_names)

        self._num_bodies = tactile.num_bodies
        self._body_has_polygon = torch.zeros(self._num_bodies, device=self._env.device, dtype=torch.bool)
        self._body_polygons = [None for _ in range(self._num_bodies)]

        for body_id in range(self._num_bodies):
            side = body_sides[body_id] if body_id < len(body_sides) else "left"
            polygon = left_outline_t if side == "left" else right_outline_t
            self._body_polygons[body_id] = polygon
            self._body_has_polygon[body_id] = True

        self._sensor_available = True

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("foot_tactile"),
        min_contact_force: float = 20.0,
        target_margin: float = 0.01,
    ) -> torch.Tensor:
        if not self._sensor_available:
            return _zero_reward(env)

        tactile, data = _get_tactile_data(env, sensor_cfg)
        if data is None:
            return _zero_reward(env)

        cop_b = data.cop_b  # (N, B, 2)
        total_normal_force = data.total_normal_force  # (N, B)
        num_envs, num_bodies, _ = cop_b.shape

        if num_bodies != self._num_bodies:
            # A rare reinitialization path (e.g., if body mapping changes).
            self._tactile = tactile
            self._build_outline_cache()
            if not self._sensor_available:
                return _zero_reward(env)

        signed_margin = torch.full((num_envs, self._num_bodies), -1.0, device=cop_b.device, dtype=cop_b.dtype)

        for body_id in range(self._num_bodies):
            if not bool(self._body_has_polygon[body_id]):
                continue
            polygon = self._body_polygons[body_id]
            if polygon is None:
                continue
            signed_margin[:, body_id] = _signed_distance_to_polygon(cop_b[:, body_id, :], polygon.to(cop_b.dtype))

        margin_score = torch.clamp(signed_margin / max(target_margin, 1e-6), min=0.0, max=1.0)
        in_contact = total_normal_force > min_contact_force
        valid_feet = in_contact & self._body_has_polygon.unsqueeze(0)

        score = margin_score * valid_feet.float()
        denom = valid_feet.float().sum(dim=-1).clamp_min(1.0)
        return score.sum(dim=-1) / denom


def foot_contact_area(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("foot_tactile"),
    min_contact_force: float = 20.0,
    target_area_ratio: float = 0.35,
) -> torch.Tensor:
    """Reward sufficient contact patch usage; saturates after reaching target area ratio."""
    _, data = _get_tactile_data(env, sensor_cfg)
    if data is None:
        return _zero_reward(env)

    area_ratio = data.contact_area_ratio  # (N, B)
    in_contact = data.total_normal_force > min_contact_force  # (N, B)
    area_score = torch.clamp(area_ratio / max(target_area_ratio, 1e-6), min=0.0, max=1.0)

    score = area_score * in_contact.float()
    denom = in_contact.float().sum(dim=-1).clamp_min(1.0)
    return score.sum(dim=-1) / denom


class foot_cop_smoothness(ManagerTermBase):
    """Reward temporal smoothness of COP movement while a foot remains in contact."""

    def __init__(self, cfg: "RewardTermCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._sensor_cfg = cfg.params.get("sensor_cfg", SceneEntityCfg("foot_tactile"))

        num_feet = int(cfg.params.get("num_feet", 2))
        try:
            tactile = env.scene.sensors[self._sensor_cfg.name]
            num_feet = int(getattr(tactile, "num_bodies", num_feet))
        except KeyError:
            pass

        self._num_feet = num_feet
        self._prev_cop = torch.zeros((env.num_envs, self._num_feet, 2), device=env.device, dtype=torch.float32)
        self._prev_contact = torch.zeros((env.num_envs, self._num_feet), device=env.device, dtype=torch.bool)
        self._initialized = torch.zeros((env.num_envs, self._num_feet), device=env.device, dtype=torch.bool)

    def _resize_buffers(self, num_feet: int) -> None:
        if num_feet == self._num_feet:
            return
        self._num_feet = num_feet
        self._prev_cop = torch.zeros((self._env.num_envs, num_feet, 2), device=self._env.device, dtype=torch.float32)
        self._prev_contact = torch.zeros((self._env.num_envs, num_feet), device=self._env.device, dtype=torch.bool)
        self._initialized = torch.zeros((self._env.num_envs, num_feet), device=self._env.device, dtype=torch.bool)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._prev_cop[env_ids] = 0.0
        self._prev_contact[env_ids] = False
        self._initialized[env_ids] = False

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("foot_tactile"),
        min_contact_force: float = 20.0,
        sigma: float = 0.01,
    ) -> torch.Tensor:
        _, data = _get_tactile_data(env, sensor_cfg)
        if data is None:
            return _zero_reward(env)

        if int(data.cop_b.shape[1]) != self._num_feet:
            self._resize_buffers(int(data.cop_b.shape[1]))

        reset_ids = torch.nonzero(env.episode_length_buf <= 1, as_tuple=False).squeeze(-1)
        if reset_ids.numel() > 0:
            self.reset(reset_ids.tolist())

        cop = data.cop_b  # (N, B, 2)
        contact = data.total_normal_force > min_contact_force  # (N, B)
        valid_delta = self._initialized & self._prev_contact & contact

        cop_delta = torch.norm(cop - self._prev_cop, dim=-1)  # (N, B)
        cop_smooth = torch.exp(-torch.square(cop_delta) / (max(sigma, 1e-6) ** 2))

        score = cop_smooth * valid_delta.float()
        denom = valid_delta.float().sum(dim=-1).clamp_min(1.0)
        reward = score.sum(dim=-1) / denom

        self._initialized |= contact
        self._prev_cop = torch.where(contact.unsqueeze(-1), cop, self._prev_cop)
        self._prev_contact[:] = contact

        return reward


class contact_stage_reward_v1(ManagerTermBase):
    """Stage-aware reward V1: PreLanding + contact-phase dense terms + Landing event penalty."""

    SWING_STAGE_ID = 0
    PRE_STAGE_ID = 1
    LANDING_STAGE_ID = 2
    STANCE_STAGE_ID = 3

    def __init__(self, cfg: "RewardTermCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._stage_sensor_cfg = cfg.params.get("stage_sensor_cfg", SceneEntityCfg("contact_stage_filter"))
        self._tactile_sensor_cfg = cfg.params.get("tactile_sensor_cfg", SceneEntityCfg("foot_tactile"))
        self._asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))

        self._stage_sensor = None
        self._tactile_sensor = None

        self._num_feet = int(cfg.params.get("num_feet", 2))
        self._tactile_body_ids_for_stage = list(range(self._num_feet))

        self._body_weight_newton = None

        self._sensor_available = False
        self._body_has_polygon = torch.zeros(self._num_feet, device=env.device, dtype=torch.bool)
        self._body_polygons: list[Optional[torch.Tensor]] = [None for _ in range(self._num_feet)]

        self._prev_vz = torch.zeros((env.num_envs, self._num_feet), device=env.device, dtype=torch.float32)
        self._az_initialized = torch.zeros((env.num_envs, self._num_feet), device=env.device, dtype=torch.bool)
        self._az_filt = torch.zeros((env.num_envs, self._num_feet), device=env.device, dtype=torch.float32)
        self._prev_cop = torch.zeros((env.num_envs, self._num_feet, 2), device=env.device, dtype=torch.float32)
        self._cop_initialized = torch.zeros((env.num_envs, self._num_feet), device=env.device, dtype=torch.bool)

        self._landing_active_prev = torch.zeros((env.num_envs, self._num_feet), device=env.device, dtype=torch.bool)
        self._landing_F_peak = torch.zeros((env.num_envs, self._num_feet), device=env.device, dtype=torch.float32)
        self._landing_dF_peak = torch.zeros((env.num_envs, self._num_feet), device=env.device, dtype=torch.float32)
        self._landing_rho_peak_max = torch.zeros((env.num_envs, self._num_feet), device=env.device, dtype=torch.float32)

        self._debug_alpha_sw = torch.zeros((env.num_envs, self._num_feet), device=env.device, dtype=torch.float32)
        self._debug_alpha_pre = torch.zeros((env.num_envs, self._num_feet), device=env.device, dtype=torch.float32)
        self._debug_alpha_land = torch.zeros((env.num_envs, self._num_feet), device=env.device, dtype=torch.float32)
        self._debug_alpha_st = torch.zeros((env.num_envs, self._num_feet), device=env.device, dtype=torch.float32)
        self._debug_r_sw_h = torch.zeros((env.num_envs, self._num_feet), device=env.device, dtype=torch.float32)
        self._debug_r_pre_v = torch.zeros((env.num_envs, self._num_feet), device=env.device, dtype=torch.float32)
        self._debug_r_pre_a = torch.zeros((env.num_envs, self._num_feet), device=env.device, dtype=torch.float32)
        self._debug_r_st_cop = torch.zeros((env.num_envs, self._num_feet), device=env.device, dtype=torch.float32)
        self._debug_r_st_area = torch.zeros((env.num_envs, self._num_feet), device=env.device, dtype=torch.float32)
        self._debug_r_st_delta_cop = torch.zeros((env.num_envs, self._num_feet), device=env.device, dtype=torch.float32)
        self._debug_delta_cop = torch.zeros((env.num_envs, self._num_feet), device=env.device, dtype=torch.float32)
        self._debug_r_contact_base = torch.zeros((env.num_envs, self._num_feet), device=env.device, dtype=torch.float32)
        self._debug_contact_phase_weight = torch.zeros((env.num_envs, self._num_feet), device=env.device, dtype=torch.float32)
        self._debug_r_contact = torch.zeros((env.num_envs, self._num_feet), device=env.device, dtype=torch.float32)
        self._debug_landing_event_penalty = torch.zeros((env.num_envs, self._num_feet), device=env.device, dtype=torch.float32)
        self._debug_vz = torch.zeros((env.num_envs, self._num_feet), device=env.device, dtype=torch.float32)
        self._debug_az = torch.zeros((env.num_envs, self._num_feet), device=env.device, dtype=torch.float32)
        self._debug_cop_margin = torch.zeros((env.num_envs, self._num_feet), device=env.device, dtype=torch.float32)
        self._debug_contact_area_ratio = torch.zeros((env.num_envs, self._num_feet), device=env.device, dtype=torch.float32)
        self._debug_rho_peak = torch.zeros((env.num_envs, self._num_feet), device=env.device, dtype=torch.float32)
        self._debug_landing_window = torch.zeros((env.num_envs, self._num_feet), device=env.device, dtype=torch.float32)
        self._debug_enable_land = 0.0
        self._debug_enable_st = 0.0
        self._debug_gamma_land = 0.0
        self._debug_gamma_st = 0.0

        self._resolve_sensor_handles(env)
        self._sync_num_feet_from_stage_sensor()
        self._build_stage_to_tactile_index()
        self._build_outline_cache()
        self._resize_buffers(self._num_feet)

    def _resolve_sensor_handles(self, env: "ManagerBasedRLEnv") -> None:
        try:
            self._stage_sensor = env.scene.sensors[self._stage_sensor_cfg.name]
        except KeyError:
            self._stage_sensor = None
        try:
            self._tactile_sensor = env.scene.sensors[self._tactile_sensor_cfg.name]
        except KeyError:
            self._tactile_sensor = None

    def _sync_num_feet_from_stage_sensor(self) -> None:
        if self._stage_sensor is None:
            return
        num_feet = int(getattr(self._stage_sensor, "num_bodies", self._num_feet))
        if num_feet > 0:
            self._num_feet = num_feet

    def _resize_buffers(self, num_feet: int) -> None:
        if num_feet == self._num_feet and self._prev_vz.shape[1] == num_feet:
            return
        self._num_feet = num_feet
        self._prev_vz = torch.zeros((self._env.num_envs, num_feet), device=self._env.device, dtype=torch.float32)
        self._az_initialized = torch.zeros((self._env.num_envs, num_feet), device=self._env.device, dtype=torch.bool)
        self._az_filt = torch.zeros((self._env.num_envs, num_feet), device=self._env.device, dtype=torch.float32)
        self._prev_cop = torch.zeros((self._env.num_envs, num_feet, 2), device=self._env.device, dtype=torch.float32)
        self._cop_initialized = torch.zeros((self._env.num_envs, num_feet), device=self._env.device, dtype=torch.bool)
        self._landing_active_prev = torch.zeros((self._env.num_envs, num_feet), device=self._env.device, dtype=torch.bool)
        self._landing_F_peak = torch.zeros((self._env.num_envs, num_feet), device=self._env.device, dtype=torch.float32)
        self._landing_dF_peak = torch.zeros((self._env.num_envs, num_feet), device=self._env.device, dtype=torch.float32)
        self._landing_rho_peak_max = torch.zeros((self._env.num_envs, num_feet), device=self._env.device, dtype=torch.float32)

        self._debug_alpha_sw = torch.zeros((self._env.num_envs, num_feet), device=self._env.device, dtype=torch.float32)
        self._debug_alpha_pre = torch.zeros((self._env.num_envs, num_feet), device=self._env.device, dtype=torch.float32)
        self._debug_alpha_land = torch.zeros((self._env.num_envs, num_feet), device=self._env.device, dtype=torch.float32)
        self._debug_alpha_st = torch.zeros((self._env.num_envs, num_feet), device=self._env.device, dtype=torch.float32)
        self._debug_r_sw_h = torch.zeros((self._env.num_envs, num_feet), device=self._env.device, dtype=torch.float32)
        self._debug_r_pre_v = torch.zeros((self._env.num_envs, num_feet), device=self._env.device, dtype=torch.float32)
        self._debug_r_pre_a = torch.zeros((self._env.num_envs, num_feet), device=self._env.device, dtype=torch.float32)
        self._debug_r_st_cop = torch.zeros((self._env.num_envs, num_feet), device=self._env.device, dtype=torch.float32)
        self._debug_r_st_area = torch.zeros((self._env.num_envs, num_feet), device=self._env.device, dtype=torch.float32)
        self._debug_r_st_delta_cop = torch.zeros((self._env.num_envs, num_feet), device=self._env.device, dtype=torch.float32)
        self._debug_delta_cop = torch.zeros((self._env.num_envs, num_feet), device=self._env.device, dtype=torch.float32)
        self._debug_r_contact_base = torch.zeros((self._env.num_envs, num_feet), device=self._env.device, dtype=torch.float32)
        self._debug_contact_phase_weight = torch.zeros((self._env.num_envs, num_feet), device=self._env.device, dtype=torch.float32)
        self._debug_r_contact = torch.zeros((self._env.num_envs, num_feet), device=self._env.device, dtype=torch.float32)
        self._debug_landing_event_penalty = torch.zeros((self._env.num_envs, num_feet), device=self._env.device, dtype=torch.float32)
        self._debug_vz = torch.zeros((self._env.num_envs, num_feet), device=self._env.device, dtype=torch.float32)
        self._debug_az = torch.zeros((self._env.num_envs, num_feet), device=self._env.device, dtype=torch.float32)
        self._debug_cop_margin = torch.zeros((self._env.num_envs, num_feet), device=self._env.device, dtype=torch.float32)
        self._debug_contact_area_ratio = torch.zeros((self._env.num_envs, num_feet), device=self._env.device, dtype=torch.float32)
        self._debug_rho_peak = torch.zeros((self._env.num_envs, num_feet), device=self._env.device, dtype=torch.float32)
        self._debug_landing_window = torch.zeros((self._env.num_envs, num_feet), device=self._env.device, dtype=torch.float32)
        self._body_has_polygon = torch.zeros(num_feet, device=self._env.device, dtype=torch.bool)
        self._body_polygons = [None for _ in range(num_feet)]
        self._tactile_body_ids_for_stage = list(range(num_feet))

    def _build_stage_to_tactile_index(self) -> None:
        self._tactile_body_ids_for_stage = list(range(self._num_feet))
        if self._stage_sensor is None or self._tactile_sensor is None:
            return
        stage_names = list(getattr(self._stage_sensor, "body_names", []))
        tactile_names = list(getattr(self._tactile_sensor, "body_names", []))
        if not stage_names or not tactile_names:
            return
        tactile_name_to_id = {name: idx for idx, name in enumerate(tactile_names)}
        ids = []
        for name in stage_names[: self._num_feet]:
            if name not in tactile_name_to_id:
                return
            ids.append(int(tactile_name_to_id[name]))
        if len(ids) == self._num_feet:
            self._tactile_body_ids_for_stage = ids

    def _build_outline_cache(self) -> None:
        self._sensor_available = False
        if self._tactile_sensor is None:
            return

        template_cfg = self._tactile_sensor.cfg.template_cfg
        left_outline = getattr(template_cfg, "left_outline_xy", None)
        right_outline = getattr(template_cfg, "right_outline_xy", None)
        if left_outline is None or right_outline is None:
            return

        left_outline_t = torch.tensor(left_outline, device=self._env.device, dtype=torch.float32)
        right_outline_t = torch.tensor(right_outline, device=self._env.device, dtype=torch.float32)
        if (
            left_outline_t.ndim != 2
            or right_outline_t.ndim != 2
            or left_outline_t.shape[-1] != 2
            or right_outline_t.shape[-1] != 2
            or left_outline_t.shape[0] < 3
            or right_outline_t.shape[0] < 3
        ):
            return

        body_names = list(getattr(self._tactile_sensor, "body_names", []))
        body_sides = _infer_body_sides(body_names) if body_names else ["left"] * self._num_feet
        num_tactile_bodies = int(getattr(self._tactile_sensor, "num_bodies", self._num_feet))
        tactile_polygons: list[Optional[torch.Tensor]] = [None for _ in range(num_tactile_bodies)]
        tactile_has_polygon = torch.zeros(num_tactile_bodies, device=self._env.device, dtype=torch.bool)

        for body_id in range(num_tactile_bodies):
            side = body_sides[body_id] if body_id < len(body_sides) else "left"
            tactile_polygons[body_id] = left_outline_t if side == "left" else right_outline_t
            tactile_has_polygon[body_id] = True

        has_polygon = torch.zeros(self._num_feet, device=self._env.device, dtype=torch.bool)
        polygons: list[Optional[torch.Tensor]] = [None for _ in range(self._num_feet)]
        for stage_body_id, tactile_body_id in enumerate(self._tactile_body_ids_for_stage):
            if tactile_body_id < 0 or tactile_body_id >= num_tactile_bodies:
                continue
            has_polygon[stage_body_id] = tactile_has_polygon[tactile_body_id]
            polygons[stage_body_id] = tactile_polygons[tactile_body_id]

        self._body_has_polygon = has_polygon
        self._body_polygons = polygons
        self._sensor_available = bool(torch.any(has_polygon))

    def _compute_cop_margin(self, cop_b: torch.Tensor) -> torch.Tensor:
        num_envs = cop_b.shape[0]
        margin = torch.zeros((num_envs, self._num_feet), device=cop_b.device, dtype=cop_b.dtype)
        if not self._sensor_available:
            return margin
        for body_id in range(self._num_feet):
            if not bool(self._body_has_polygon[body_id]):
                continue
            polygon = self._body_polygons[body_id]
            if polygon is None:
                continue
            margin[:, body_id] = _signed_distance_to_polygon(cop_b[:, body_id, :], polygon.to(cop_b.dtype))
        return margin

    def _extract_tactile_aligned(self, tactile_data) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_tactile_bodies = int(tactile_data.cop_b.shape[1])
        if num_tactile_bodies <= 0:
            zeros_cop = torch.zeros((self._env.num_envs, self._num_feet, 2), device=self._env.device, dtype=torch.float32)
            zeros_area = torch.zeros((self._env.num_envs, self._num_feet), device=self._env.device, dtype=torch.float32)
            return zeros_cop, zeros_area, zeros_area

        ids = self._tactile_body_ids_for_stage
        if len(ids) < self._num_feet:
            ids = list(range(self._num_feet))
        ids = [int(min(max(body_id, 0), num_tactile_bodies - 1)) for body_id in ids[: self._num_feet]]
        while len(ids) < self._num_feet:
            ids.append(ids[-1] if ids else 0)
        cop_b = torch.nan_to_num(
            tactile_data.cop_b[:, ids, :],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        area_ratio = torch.nan_to_num(
            tactile_data.contact_area_ratio[:, ids],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        peak_force = torch.nan_to_num(
            tactile_data.peak_force[:, ids],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)
        return cop_b, area_ratio, peak_force

    def _infer_body_weight_newton(self, env: "ManagerBasedRLEnv", fallback_body_weight: float) -> float:
        if self._body_weight_newton is not None and self._body_weight_newton > 1e-6:
            return self._body_weight_newton

        body_weight = float(fallback_body_weight)
        try:
            asset = env.scene[self._asset_cfg.name]
            masses = asset.root_physx_view.get_masses()
            masses_t = torch.as_tensor(masses, device=env.device, dtype=torch.float32)
            if masses_t.ndim >= 2:
                total_mass = masses_t[0].sum()
            else:
                total_mass = masses_t.sum()
            total_mass = torch.nan_to_num(total_mass, nan=0.0, posinf=0.0, neginf=0.0)
            if float(total_mass.item()) > 1e-6:
                body_weight = float(total_mass.item()) * 9.81
        except Exception:
            body_weight = float(fallback_body_weight)

        self._body_weight_newton = max(body_weight, 1e-6)
        return self._body_weight_newton

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._prev_vz[env_ids] = 0.0
        self._az_initialized[env_ids] = False
        self._az_filt[env_ids] = 0.0
        self._prev_cop[env_ids] = 0.0
        self._cop_initialized[env_ids] = False
        self._landing_active_prev[env_ids] = False
        self._landing_F_peak[env_ids] = 0.0
        self._landing_dF_peak[env_ids] = 0.0
        self._landing_rho_peak_max[env_ids] = 0.0
        self._debug_alpha_sw[env_ids] = 0.0
        self._debug_alpha_pre[env_ids] = 0.0
        self._debug_alpha_land[env_ids] = 0.0
        self._debug_alpha_st[env_ids] = 0.0
        self._debug_r_sw_h[env_ids] = 0.0
        self._debug_r_pre_v[env_ids] = 0.0
        self._debug_r_pre_a[env_ids] = 0.0
        self._debug_r_st_cop[env_ids] = 0.0
        self._debug_r_st_area[env_ids] = 0.0
        self._debug_r_st_delta_cop[env_ids] = 0.0
        self._debug_delta_cop[env_ids] = 0.0
        self._debug_r_contact_base[env_ids] = 0.0
        self._debug_contact_phase_weight[env_ids] = 0.0
        self._debug_r_contact[env_ids] = 0.0
        self._debug_landing_event_penalty[env_ids] = 0.0
        self._debug_vz[env_ids] = 0.0
        self._debug_az[env_ids] = 0.0
        self._debug_cop_margin[env_ids] = 0.0
        self._debug_contact_area_ratio[env_ids] = 0.0
        self._debug_rho_peak[env_ids] = 0.0
        self._debug_landing_window[env_ids] = 0.0

    def get_debug_dict(self, env_id: int) -> dict[str, torch.Tensor]:
        scalar_shape = (self._num_feet,)
        return {
            "alpha_sw": self._debug_alpha_sw[env_id].detach().clone(),
            "alpha_pre": self._debug_alpha_pre[env_id].detach().clone(),
            "alpha_land": self._debug_alpha_land[env_id].detach().clone(),
            "alpha_st": self._debug_alpha_st[env_id].detach().clone(),
            "r_sw_h": self._debug_r_sw_h[env_id].detach().clone(),
            "r_pre_v": self._debug_r_pre_v[env_id].detach().clone(),
            "r_pre_a": self._debug_r_pre_a[env_id].detach().clone(),
            "r_st_cop": self._debug_r_st_cop[env_id].detach().clone(),
            "r_st_area": self._debug_r_st_area[env_id].detach().clone(),
            "r_st_delta_cop": self._debug_r_st_delta_cop[env_id].detach().clone(),
            "delta_cop": self._debug_delta_cop[env_id].detach().clone(),
            "r_cop": self._debug_r_st_cop[env_id].detach().clone(),
            "r_area": self._debug_r_st_area[env_id].detach().clone(),
            "r_contact_base": self._debug_r_contact_base[env_id].detach().clone(),
            "contact_phase_weight": self._debug_contact_phase_weight[env_id].detach().clone(),
            "r_contact": self._debug_r_contact[env_id].detach().clone(),
            "landing_F_peak": self._landing_F_peak[env_id].detach().clone(),
            "landing_dF_peak": self._landing_dF_peak[env_id].detach().clone(),
            "landing_rho_peak_max": self._landing_rho_peak_max[env_id].detach().clone(),
            "landing_event_penalty": self._debug_landing_event_penalty[env_id].detach().clone(),
            "vz": self._debug_vz[env_id].detach().clone(),
            "az": self._debug_az[env_id].detach().clone(),
            "cop_margin": self._debug_cop_margin[env_id].detach().clone(),
            "contact_area_ratio": self._debug_contact_area_ratio[env_id].detach().clone(),
            "rho_peak": self._debug_rho_peak[env_id].detach().clone(),
            "landing_window": self._debug_landing_window[env_id].detach().clone(),
            "enable_land": torch.full(scalar_shape, self._debug_enable_land, device=self._env.device, dtype=torch.float32),
            "enable_st": torch.full(scalar_shape, self._debug_enable_st, device=self._env.device, dtype=torch.float32),
            "gamma_land": torch.full(scalar_shape, self._debug_gamma_land, device=self._env.device, dtype=torch.float32),
            "gamma_st": torch.full(scalar_shape, self._debug_gamma_st, device=self._env.device, dtype=torch.float32),
        }

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        stage_sensor_cfg: Optional[SceneEntityCfg] = None,
        tactile_sensor_cfg: Optional[SceneEntityCfg] = None,
        asset_cfg: Optional[SceneEntityCfg] = None,
        enable_stage_reward_v1: bool = True,
        enable_swing_clearance_reward: bool = False,
        w_swing_h: float = 0.08,
        swing_h_ref: float = 0.07,
        swing_command_name: str = "base_velocity",
        swing_vel_threshold: float = 0.15,
        enable_prelanding_reward: bool = True,
        pre_vz_ref: float = 0.24,
        pre_az_ref: float = 7.0,
        pre_az_filter_alpha: float = 0.7,
        w_pre_v: float = 0.22,
        w_pre_a: float = 0.035,
        enable_landing_event_penalty: bool = True,
        w_land_F: float = 0.22,
        w_land_dF: float = 0.10,
        w_land_rho: float = 0.10,
        land_dF_ref: float = 3500.0,
        body_weight: float = 420.0,
        enable_contact_quality_reward: bool = True,
        enable_contact_quality_on_landing: bool = True,
        enable_contact_quality_on_stance: bool = True,
        gamma_land: float = 0.6,
        gamma_st: float = 1.0,
        cop_margin_max: float = 0.038,
        w_cop: float = 0.24,
        w_area: float = 0.18,
        enable_stance_delta_cop_reward: bool = True,
        w_st_delta_cop: float = 0.03,
        delta_cop_ref: float = 0.01,
        contact_quality_eps: float = 1e-6,
        enable_stage_reward_warmup: bool = True,
        stage_reward_warmup_steps: int = 10000,
        enable_self_check: bool = True,
    ) -> torch.Tensor:
        if stage_sensor_cfg is not None:
            self._stage_sensor_cfg = stage_sensor_cfg
        if tactile_sensor_cfg is not None:
            self._tactile_sensor_cfg = tactile_sensor_cfg
        if asset_cfg is not None:
            self._asset_cfg = asset_cfg

        if not enable_stage_reward_v1:
            return _zero_reward(env)

        self._resolve_sensor_handles(env)
        self._sync_num_feet_from_stage_sensor()
        self._resize_buffers(self._num_feet)
        self._build_stage_to_tactile_index()
        self._build_outline_cache()

        if self._stage_sensor is None or self._tactile_sensor is None:
            return _zero_reward(env)

        stage_data = self._stage_sensor.data
        tactile_data = self._tactile_sensor.data

        if int(stage_data.stage_eligibility.shape[1]) != self._num_feet:
            self._resize_buffers(int(stage_data.stage_eligibility.shape[1]))
            self._build_stage_to_tactile_index()
            self._build_outline_cache()

        reset_ids = torch.nonzero(env.episode_length_buf <= 1, as_tuple=False).squeeze(-1)
        if reset_ids.numel() > 0:
            self.reset(reset_ids.tolist())

        eps = 1e-6
        step_dt = max(float(env.step_dt), eps)

        alpha = torch.nan_to_num(stage_data.stage_eligibility, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
        if alpha.shape[-1] <= self.STANCE_STAGE_ID:
            raise RuntimeError(
                "contact_stage_reward_v1 requires at least 4 stage channels "
                "(Swing/PreLanding/Landing/Stance)."
            )
        alpha_sw = alpha[..., self.SWING_STAGE_ID]
        alpha_pre = alpha[..., self.PRE_STAGE_ID]
        alpha_land = alpha[..., self.LANDING_STAGE_ID]
        alpha_st = alpha[..., self.STANCE_STAGE_ID]

        h_eff = torch.nan_to_num(stage_data.h_eff, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        swing_h_ref = max(float(swing_h_ref), 1e-6)
        r_sw_h = torch.clamp(h_eff / swing_h_ref, min=0.0, max=1.0)
        command = env.command_manager.get_command(swing_command_name)
        command_active = torch.logical_or(
            torch.norm(command[:, :2], dim=1) > float(swing_vel_threshold),
            torch.abs(command[:, 2]) > float(swing_vel_threshold),
        ).unsqueeze(-1)
        r_sw_h = torch.where(command_active, r_sw_h, torch.zeros_like(r_sw_h))
        r_sw_h = torch.nan_to_num(r_sw_h, nan=0.0, posinf=0.0, neginf=0.0)

        vz = torch.nan_to_num(stage_data.foot_vz, nan=0.0, posinf=0.0, neginf=0.0)
        prev_vz = self._prev_vz
        az_raw = (vz - prev_vz) / step_dt
        az_raw = torch.where(self._az_initialized, az_raw, torch.zeros_like(az_raw))
        az_raw = torch.nan_to_num(az_raw, nan=0.0, posinf=0.0, neginf=0.0)
        az_alpha = float(min(max(pre_az_filter_alpha, 0.0), 1.0))
        az_filt_candidate = az_alpha * az_raw + (1.0 - az_alpha) * self._az_filt
        az = torch.where(self._az_initialized, az_filt_candidate, torch.zeros_like(az_raw))
        az = torch.nan_to_num(az, nan=0.0, posinf=0.0, neginf=0.0)
        self._prev_vz = vz
        self._az_filt = az
        self._az_initialized[:] = True

        delta_v_down = torch.clamp(-vz - float(pre_vz_ref), min=0.0)
        delta_a_down = torch.clamp(-az - float(pre_az_ref), min=0.0)
        r_pre_v = -torch.square(delta_v_down)
        r_pre_a = -torch.square(delta_a_down)
        r_pre = float(w_pre_v) * r_pre_v + float(w_pre_a) * r_pre_a
        r_pre = torch.nan_to_num(r_pre, nan=0.0, posinf=0.0, neginf=0.0)

        cop_b, contact_area_ratio, _ = self._extract_tactile_aligned(tactile_data)
        cop_margin = self._compute_cop_margin(cop_b)
        contact_quality_eps = max(float(contact_quality_eps), 1e-12)
        cop_margin_max_value = max(float(cop_margin_max), contact_quality_eps)
        r_cop = torch.clamp(cop_margin / cop_margin_max_value, min=0.0, max=1.0)
        r_area = torch.clamp(contact_area_ratio, min=0.0, max=1.0)
        r_contact_base = float(w_cop) * r_cop + float(w_area) * r_area
        r_contact_base = torch.nan_to_num(r_contact_base, nan=0.0, posinf=0.0, neginf=0.0)

        enable_land = 1.0 if bool(enable_contact_quality_on_landing) else 0.0
        enable_st = 1.0 if bool(enable_contact_quality_on_stance) else 0.0
        gamma_land = max(float(gamma_land), 0.0)
        gamma_st = max(float(gamma_st), 0.0)
        contact_phase_weight = (
            enable_land * gamma_land * alpha_land
            + enable_st * gamma_st * alpha_st
        )
        contact_phase_weight = torch.nan_to_num(contact_phase_weight, nan=0.0, posinf=0.0, neginf=0.0)
        r_contact = contact_phase_weight * r_contact_base
        r_contact = torch.nan_to_num(r_contact, nan=0.0, posinf=0.0, neginf=0.0)
        if not enable_contact_quality_reward:
            r_contact = torch.zeros_like(r_contact)

        contact_active = stage_data.contact_active
        delta_cop = torch.linalg.vector_norm(cop_b - self._prev_cop, dim=-1)
        delta_cop = torch.nan_to_num(delta_cop, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        delta_cop_ref = max(float(delta_cop_ref), 1e-6)
        r_st_delta_cop = torch.exp(-torch.square(delta_cop / delta_cop_ref))
        valid_delta_cop = self._cop_initialized & contact_active
        r_st_delta_cop = torch.where(valid_delta_cop, r_st_delta_cop, torch.zeros_like(r_st_delta_cop))
        r_st_delta_cop = torch.nan_to_num(r_st_delta_cop, nan=0.0, posinf=0.0, neginf=0.0)
        self._prev_cop = torch.where(contact_active.unsqueeze(-1), cop_b, self._prev_cop)
        self._cop_initialized = contact_active

        R_stage = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
        if enable_swing_clearance_reward:
            R_stage += (alpha_sw * float(w_swing_h) * r_sw_h).sum(dim=-1)
        if enable_prelanding_reward:
            R_stage += (alpha_pre * r_pre).sum(dim=-1)
        R_stage += r_contact.sum(dim=-1)
        if enable_stance_delta_cop_reward:
            R_stage += (alpha_st * float(w_st_delta_cop) * r_st_delta_cop).sum(dim=-1)

        landing_event_penalty_per_foot = torch.zeros_like(alpha_pre)
        R_landing_event = torch.zeros_like(R_stage)
        if enable_landing_event_penalty:
            total_force = torch.nan_to_num(stage_data.total_force, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
            d_force = torch.nan_to_num(stage_data.dF, nan=0.0, posinf=0.0, neginf=0.0)
            rho_peak = torch.nan_to_num(stage_data.rho_peak, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
            landing_window_active = stage_data.landing_window > 0

            landing_start = landing_window_active & (~self._landing_active_prev)
            d_force_pos = torch.clamp(d_force, min=0.0)

            self._landing_F_peak = torch.where(landing_start, total_force, self._landing_F_peak)
            self._landing_dF_peak = torch.where(landing_start, d_force_pos, self._landing_dF_peak)
            self._landing_rho_peak_max = torch.where(landing_start, rho_peak, self._landing_rho_peak_max)

            self._landing_F_peak = torch.where(
                landing_window_active,
                torch.maximum(self._landing_F_peak, total_force),
                self._landing_F_peak,
            )
            self._landing_dF_peak = torch.where(
                landing_window_active,
                torch.maximum(self._landing_dF_peak, d_force_pos),
                self._landing_dF_peak,
            )
            self._landing_rho_peak_max = torch.where(
                landing_window_active,
                torch.maximum(self._landing_rho_peak_max, rho_peak),
                self._landing_rho_peak_max,
            )

            landing_end = (~landing_window_active) & self._landing_active_prev
            body_weight_n = self._infer_body_weight_newton(env, fallback_body_weight=float(body_weight))
            F_peak_norm = self._landing_F_peak / max(body_weight_n, eps)
            dF_peak_norm = self._landing_dF_peak / max(float(land_dF_ref), eps)
            landing_penalty_raw = (
                -float(w_land_F) * F_peak_norm
                - float(w_land_dF) * dF_peak_norm
                - float(w_land_rho) * self._landing_rho_peak_max
            )
            landing_event_penalty_per_foot = torch.where(
                landing_end,
                landing_penalty_raw,
                torch.zeros_like(landing_penalty_raw),
            )
            landing_event_penalty_per_foot = torch.nan_to_num(
                landing_event_penalty_per_foot,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            R_landing_event = landing_event_penalty_per_foot.sum(dim=-1)

            clear_mask = ~landing_window_active
            self._landing_F_peak = torch.where(clear_mask, torch.zeros_like(self._landing_F_peak), self._landing_F_peak)
            self._landing_dF_peak = torch.where(clear_mask, torch.zeros_like(self._landing_dF_peak), self._landing_dF_peak)
            self._landing_rho_peak_max = torch.where(
                clear_mask,
                torch.zeros_like(self._landing_rho_peak_max),
                self._landing_rho_peak_max,
            )
            self._landing_active_prev = landing_window_active
        else:
            self._landing_active_prev[:] = False
            self._landing_F_peak[:] = 0.0
            self._landing_dF_peak[:] = 0.0
            self._landing_rho_peak_max[:] = 0.0

        warmup_scale = 1.0
        if enable_stage_reward_warmup:
            warmup_steps = max(int(stage_reward_warmup_steps), 0)
            if warmup_steps > 0:
                warmup_iteration_getter = getattr(env, "get_stage_reward_warmup_iteration", None)
                if callable(warmup_iteration_getter):
                    warmup_progress = float(max(int(warmup_iteration_getter()), 0))
                else:
                    # Fallback for env variants that do not expose PPO iteration context.
                    common_steps_raw = getattr(env, "common_step_counter", None)
                    warmup_progress = float(max(int(common_steps_raw), 0)) if common_steps_raw is not None else 0.0
                warmup_scale = min(warmup_progress / float(warmup_steps), 1.0)
        reward = torch.nan_to_num((R_stage + R_landing_event) * float(warmup_scale), nan=0.0, posinf=0.0, neginf=0.0)

        self._debug_alpha_sw[:] = torch.nan_to_num(alpha_sw, nan=0.0, posinf=0.0, neginf=0.0)
        self._debug_alpha_pre[:] = torch.nan_to_num(alpha_pre, nan=0.0, posinf=0.0, neginf=0.0)
        self._debug_alpha_land[:] = torch.nan_to_num(alpha_land, nan=0.0, posinf=0.0, neginf=0.0)
        self._debug_alpha_st[:] = torch.nan_to_num(alpha_st, nan=0.0, posinf=0.0, neginf=0.0)
        self._debug_r_sw_h[:] = torch.nan_to_num(r_sw_h, nan=0.0, posinf=0.0, neginf=0.0)
        self._debug_r_pre_v[:] = torch.nan_to_num(r_pre_v, nan=0.0, posinf=0.0, neginf=0.0)
        self._debug_r_pre_a[:] = torch.nan_to_num(r_pre_a, nan=0.0, posinf=0.0, neginf=0.0)
        self._debug_r_st_cop[:] = torch.nan_to_num(r_cop, nan=0.0, posinf=0.0, neginf=0.0)
        self._debug_r_st_area[:] = torch.nan_to_num(r_area, nan=0.0, posinf=0.0, neginf=0.0)
        self._debug_r_st_delta_cop[:] = torch.nan_to_num(r_st_delta_cop, nan=0.0, posinf=0.0, neginf=0.0)
        self._debug_delta_cop[:] = torch.nan_to_num(delta_cop, nan=0.0, posinf=0.0, neginf=0.0)
        self._debug_r_contact_base[:] = torch.nan_to_num(r_contact_base, nan=0.0, posinf=0.0, neginf=0.0)
        self._debug_contact_phase_weight[:] = torch.nan_to_num(contact_phase_weight, nan=0.0, posinf=0.0, neginf=0.0)
        self._debug_r_contact[:] = torch.nan_to_num(r_contact, nan=0.0, posinf=0.0, neginf=0.0)
        self._debug_landing_event_penalty[:] = landing_event_penalty_per_foot
        self._debug_vz[:] = torch.nan_to_num(vz, nan=0.0, posinf=0.0, neginf=0.0)
        self._debug_az[:] = torch.nan_to_num(az, nan=0.0, posinf=0.0, neginf=0.0)
        self._debug_cop_margin[:] = torch.nan_to_num(cop_margin, nan=0.0, posinf=0.0, neginf=0.0)
        self._debug_contact_area_ratio[:] = torch.nan_to_num(contact_area_ratio, nan=0.0, posinf=0.0, neginf=0.0)
        self._debug_rho_peak[:] = torch.nan_to_num(stage_data.rho_peak, nan=0.0, posinf=0.0, neginf=0.0)
        self._debug_landing_window[:] = stage_data.landing_window.to(dtype=torch.float32)
        self._debug_enable_land = enable_land
        self._debug_enable_st = enable_st
        self._debug_gamma_land = gamma_land
        self._debug_gamma_st = gamma_st

        if enable_self_check:
            if not torch.isfinite(az).all():
                raise RuntimeError("contact_stage_reward_v1 produced invalid az.")
            if not torch.isfinite(reward).all():
                raise RuntimeError("contact_stage_reward_v1 produced NaN/inf reward.")
            if enable_landing_event_penalty and not torch.isfinite(self._landing_F_peak).all():
                raise RuntimeError("contact_stage_reward_v1 landing cache contains NaN/inf.")

        return reward

#----------------------------------foot tactile rewards----------------------------------#

#----------------------------------other rewards----------------------------------#


def feet_air_time(env, command_name: str, vel_threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    # no reward for zero command
    reward *= torch.logical_or(
        torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > vel_threshold,
        torch.abs(env.command_manager.get_command(command_name)[:, 2]) > vel_threshold,
    )
    return reward


def stand_still(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.15,
    offset: float = 1.0,
) -> torch.Tensor:
    """Penalize moving when there is no velocity command."""
    asset = env.scene[asset_cfg.name]
    dof_error = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return (
        (dof_error - offset)
        * (torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) < threshold)
        * (torch.abs(env.command_manager.get_command(command_name)[:, 2]) < threshold)
    )


def target_reached(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_dist_threshold: Optional[float] = None,
) -> torch.Tensor:
    """Reward reaching the sampled target position in xy using the same thresholding logic as the command term."""
    command_term = env.command_manager.get_term(command_name)
    if not hasattr(command_term, "pos_command_w") or not hasattr(command_term, "robot"):
        return _zero_reward(env)

    target_vec = command_term.pos_command_w - command_term.robot.data.root_pos_w[:, :3]
    target_dist = torch.norm(target_vec[:, :2], dim=1)

    if target_dist_threshold is None:
        target_dist_threshold = float(getattr(command_term.cfg, "target_dis_threshold", 0.2))

    return (target_dist <= target_dist_threshold).float()


def feet_close_xy_gauss(
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), std: float = 0.1
) -> torch.Tensor:
    """Penalize when feet are too close together in the y distance."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]

    # Get feet positions (assuming first two body_ids are left and right feet)
    left_foot_xy = asset.data.body_pos_w[:, asset_cfg.body_ids[0], :2]
    right_foot_xy = asset.data.body_pos_w[:, asset_cfg.body_ids[1], :2]
    heading_w = asset.data.heading_w

    # Transform feet positions to robot frame
    cos_heading = torch.cos(heading_w)
    sin_heading = torch.sin(heading_w)

    # Rotate to robot frame
    left_foot_robot_frame = torch.stack(
        [
            cos_heading * left_foot_xy[:, 0] + sin_heading * left_foot_xy[:, 1],
            -sin_heading * left_foot_xy[:, 0] + cos_heading * left_foot_xy[:, 1],
        ],
        dim=1,
    )

    right_foot_robot_frame = torch.stack(
        [
            cos_heading * right_foot_xy[:, 0] + sin_heading * right_foot_xy[:, 1],
            -sin_heading * right_foot_xy[:, 0] + cos_heading * right_foot_xy[:, 1],
        ],
        dim=1,
    )

    feet_distance_y = torch.abs(left_foot_robot_frame[:, 1] - right_foot_robot_frame[:, 1])

    # Return continuous penalty using exponential decay
    return torch.exp(-torch.clamp(threshold - feet_distance_y, min=0.0) / std**2) - 1


def heading_error(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Compute the heading error between the robot's current heading and the goal heading."""
    # compute the error
    ang_vel_cmd = torch.abs(env.command_manager.get_command(command_name)[:, 2])
    return ang_vel_cmd


def dont_wait(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize standing still when there is a forward velocity command."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_cmd_x = env.command_manager.get_command(command_name)[:, 0]
    lin_vel_x = asset.data.root_lin_vel_b[:, 0]
    return (lin_vel_cmd_x > 0.3) * ((lin_vel_x < 0.15).float() + (lin_vel_x < 0).float() + (lin_vel_x < -0.15).float())


def feet_orientation_contact(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward feet being oriented vertically when in contact with the ground."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    left_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]
    left_projected_gravity = quat_rotate_inverse(left_quat, asset.data.GRAVITY_VEC_W)
    right_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[1], :]
    right_projected_gravity = quat_rotate_inverse(right_quat, asset.data.GRAVITY_VEC_W)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > 1

    return (
        torch.sum(torch.square(left_projected_gravity[:, :2]), dim=-1) ** 0.5 * is_contact[:, 0]
        + torch.sum(torch.square(right_projected_gravity[:, :2]), dim=-1) ** 0.5 * is_contact[:, 1]
    )


def feet_at_plane(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg,
    left_height_scanner_cfg: SceneEntityCfg,
    right_height_scanner_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_offset=0.035,
) -> torch.Tensor:
    """Reward feet being at certain height above the ground plane."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, contact_sensor_cfg.body_ids], dim=-1), dim=1)[0] > 1
    left_sensor = env.scene[left_height_scanner_cfg.name]
    left_sensor_data = left_sensor.data.ray_hits_w[..., 2]
    left_sensor_data = torch.where(torch.isinf(left_sensor_data), 0.0, left_sensor_data)
    right_sensor = env.scene[right_height_scanner_cfg.name]
    right_sensor_data = right_sensor.data.ray_hits_w[..., 2]
    right_sensor_data = torch.where(torch.isinf(right_sensor_data), 0.0, right_sensor_data)
    left_height = asset.data.body_pos_w[:, asset_cfg.body_ids[0], 2]
    right_height = asset.data.body_pos_w[:, asset_cfg.body_ids[1], 2]

    left_reward = (
        torch.clamp(left_height.unsqueeze(-1) - left_sensor_data - height_offset, min=0.0, max=0.3) * is_contact[:, 0:1]
    )
    right_reward = (
        torch.clamp(right_height.unsqueeze(-1) - right_sensor_data - height_offset, min=0.0, max=0.3)
        * is_contact[:, 1:2]
    )
    return torch.sum(left_reward, dim=-1) + torch.sum(right_reward, dim=-1)


def link_orientation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat link orientation using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    link_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]
    link_projected_gravity = quat_rotate_inverse(link_quat, asset.data.GRAVITY_VEC_W)

    return torch.sum(torch.square(link_projected_gravity[:, :2]), dim=1)

    
