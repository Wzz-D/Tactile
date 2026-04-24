from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Any, Optional, Sequence

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


def _normalize_signature_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(_normalize_signature_value(item) for item in value)
    return value


def _scene_entity_signature(cfg: Optional[SceneEntityCfg]) -> Any:
    if cfg is None:
        return None
    return (
        getattr(cfg, "name", None),
        _normalize_signature_value(getattr(cfg, "body_names", None)),
        _normalize_signature_value(getattr(cfg, "joint_names", None)),
        bool(getattr(cfg, "preserve_order", False)),
    )


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


class _ContactStageRewardV1SharedCore:
    SWING_STAGE_ID = 0
    PRE_STAGE_ID = 1
    LANDING_STAGE_ID = 2
    STANCE_STAGE_ID = 3

    def __init__(
        self,
        env: "ManagerBasedRLEnv",
        stage_sensor_cfg: SceneEntityCfg,
        tactile_sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
        num_feet: int = 2,
    ):
        self._env = env
        self._stage_sensor_cfg = stage_sensor_cfg
        self._tactile_sensor_cfg = tactile_sensor_cfg
        self._asset_cfg = asset_cfg

        self._stage_sensor = None
        self._tactile_sensor = None

        self._num_feet = int(num_feet)
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

        self._debug_buffers_enabled = False
        self._debug_alpha_sw: Optional[torch.Tensor] = None
        self._debug_alpha_pre: Optional[torch.Tensor] = None
        self._debug_alpha_land: Optional[torch.Tensor] = None
        self._debug_alpha_st: Optional[torch.Tensor] = None
        self._debug_r_sw_h: Optional[torch.Tensor] = None
        self._debug_r_pre_v: Optional[torch.Tensor] = None
        self._debug_r_pre_a: Optional[torch.Tensor] = None
        self._debug_r_st_cop: Optional[torch.Tensor] = None
        self._debug_r_st_area: Optional[torch.Tensor] = None
        self._debug_r_st_delta_cop: Optional[torch.Tensor] = None
        self._debug_delta_cop: Optional[torch.Tensor] = None
        self._debug_vz: Optional[torch.Tensor] = None
        self._debug_az: Optional[torch.Tensor] = None
        self._debug_cop_margin: Optional[torch.Tensor] = None
        self._debug_contact_area_ratio: Optional[torch.Tensor] = None
        self._debug_rho_peak: Optional[torch.Tensor] = None
        self._debug_landing_window: Optional[torch.Tensor] = None

        self._last_metrics: Optional[dict[str, torch.Tensor]] = None
        self._last_compute_step: Optional[int] = None

        self._resolve_sensor_handles(env)
        self._sync_num_feet_from_stage_sensor()
        self._resize_buffers(self._num_feet)
        self._build_stage_to_tactile_index()
        self._build_outline_cache()

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
        self._body_has_polygon = torch.zeros(num_feet, device=self._env.device, dtype=torch.bool)
        self._body_polygons = [None for _ in range(num_feet)]
        self._tactile_body_ids_for_stage = list(range(num_feet))
        if self._debug_buffers_enabled:
            self._allocate_debug_buffers(num_feet)

    def _allocate_debug_buffers(self, num_feet: int) -> None:
        device = self._env.device
        num_envs = self._env.num_envs
        self._debug_alpha_sw = torch.zeros((num_envs, num_feet), device=device, dtype=torch.float32)
        self._debug_alpha_pre = torch.zeros((num_envs, num_feet), device=device, dtype=torch.float32)
        self._debug_alpha_land = torch.zeros((num_envs, num_feet), device=device, dtype=torch.float32)
        self._debug_alpha_st = torch.zeros((num_envs, num_feet), device=device, dtype=torch.float32)
        self._debug_r_sw_h = torch.zeros((num_envs, num_feet), device=device, dtype=torch.float32)
        self._debug_r_pre_v = torch.zeros((num_envs, num_feet), device=device, dtype=torch.float32)
        self._debug_r_pre_a = torch.zeros((num_envs, num_feet), device=device, dtype=torch.float32)
        self._debug_r_st_cop = torch.zeros((num_envs, num_feet), device=device, dtype=torch.float32)
        self._debug_r_st_area = torch.zeros((num_envs, num_feet), device=device, dtype=torch.float32)
        self._debug_r_st_delta_cop = torch.zeros((num_envs, num_feet), device=device, dtype=torch.float32)
        self._debug_delta_cop = torch.zeros((num_envs, num_feet), device=device, dtype=torch.float32)
        self._debug_vz = torch.zeros((num_envs, num_feet), device=device, dtype=torch.float32)
        self._debug_az = torch.zeros((num_envs, num_feet), device=device, dtype=torch.float32)
        self._debug_cop_margin = torch.zeros((num_envs, num_feet), device=device, dtype=torch.float32)
        self._debug_contact_area_ratio = torch.zeros((num_envs, num_feet), device=device, dtype=torch.float32)
        self._debug_rho_peak = torch.zeros((num_envs, num_feet), device=device, dtype=torch.float32)
        self._debug_landing_window = torch.zeros((num_envs, num_feet), device=device, dtype=torch.float32)

    def _enable_debug_buffers(self) -> None:
        if self._debug_buffers_enabled:
            return
        self._debug_buffers_enabled = True
        self._allocate_debug_buffers(self._num_feet)

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

    def _extract_tactile_aligned(self, tactile_data) -> tuple[torch.Tensor, torch.Tensor]:
        num_tactile_bodies = int(tactile_data.cop_b.shape[1])
        if num_tactile_bodies <= 0:
            zeros_cop = torch.zeros((self._env.num_envs, self._num_feet, 2), device=self._env.device, dtype=torch.float32)
            zeros_area = torch.zeros((self._env.num_envs, self._num_feet), device=self._env.device, dtype=torch.float32)
            return zeros_cop, zeros_area

        ids = self._tactile_body_ids_for_stage
        if len(ids) < self._num_feet:
            ids = list(range(self._num_feet))
        ids = [int(min(max(body_id, 0), num_tactile_bodies - 1)) for body_id in ids[: self._num_feet]]
        while len(ids) < self._num_feet:
            ids.append(ids[-1] if ids else 0)
        cop_b = torch.nan_to_num(tactile_data.cop_b[:, ids, :], nan=0.0, posinf=0.0, neginf=0.0)
        area_ratio = torch.nan_to_num(
            tactile_data.contact_area_ratio[:, ids],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        return cop_b, area_ratio

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

    def _store_last_metrics(self, metrics: dict[str, torch.Tensor], enable_debug_buffers: bool) -> None:
        self._last_metrics = metrics
        if enable_debug_buffers:
            self._enable_debug_buffers()
        if not self._debug_buffers_enabled:
            return
        assert self._debug_alpha_sw is not None
        self._debug_alpha_sw.copy_(metrics["alpha_sw"])
        self._debug_alpha_pre.copy_(metrics["alpha_pre"])
        self._debug_alpha_land.copy_(metrics["alpha_land"])
        self._debug_alpha_st.copy_(metrics["alpha_st"])
        self._debug_r_sw_h.copy_(metrics["r_sw_h"])
        self._debug_r_pre_v.copy_(metrics["r_pre_v"])
        self._debug_r_pre_a.copy_(metrics["r_pre_a"])
        self._debug_r_st_cop.copy_(metrics["r_cop"])
        self._debug_r_st_area.copy_(metrics["r_area"])
        self._debug_r_st_delta_cop.copy_(metrics["r_st_delta_cop"])
        self._debug_delta_cop.copy_(metrics["delta_cop"])
        self._debug_vz.copy_(metrics["vz"])
        self._debug_az.copy_(metrics["az"])
        self._debug_cop_margin.copy_(metrics["cop_margin"])
        self._debug_contact_area_ratio.copy_(metrics["contact_area_ratio"])
        self._debug_rho_peak.copy_(metrics["rho_peak"])
        self._debug_landing_window.copy_(metrics["landing_window"])

    def compute(
        self,
        env: "ManagerBasedRLEnv",
        swing_command_name: str,
        swing_vel_threshold: float,
        swing_h_ref: float,
        pre_vz_ref: float,
        pre_az_ref: float,
        pre_az_filter_alpha: float,
        land_dF_ref: float,
        body_weight: float,
        cop_margin_max: float,
        delta_cop_ref: float,
        contact_quality_eps: float,
        enable_debug_buffers: bool = False,
    ) -> Optional[dict[str, torch.Tensor]]:
        current_step = int(getattr(env, "common_step_counter", 0))
        if self._last_compute_step == current_step:
            if self._last_metrics is not None and enable_debug_buffers:
                self._store_last_metrics(self._last_metrics, enable_debug_buffers=True)
            return self._last_metrics

        self._resolve_sensor_handles(env)
        self._sync_num_feet_from_stage_sensor()
        self._resize_buffers(self._num_feet)
        self._build_stage_to_tactile_index()
        self._build_outline_cache()

        self._last_compute_step = current_step
        if self._stage_sensor is None or self._tactile_sensor is None:
            self._last_metrics = None
            return None

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
        az_raw = (vz - self._prev_vz) / step_dt
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

        cop_b, contact_area_ratio = self._extract_tactile_aligned(tactile_data)
        cop_margin = self._compute_cop_margin(cop_b)
        contact_quality_eps = max(float(contact_quality_eps), 1e-12)
        cop_margin_max_value = max(float(cop_margin_max), contact_quality_eps)
        r_cop = torch.clamp(cop_margin / cop_margin_max_value, min=0.0, max=1.0)
        r_area = torch.clamp(contact_area_ratio, min=0.0, max=1.0)

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
        landing_F_peak_event = self._landing_F_peak
        landing_dF_peak_event = self._landing_dF_peak
        landing_rho_peak_event = self._landing_rho_peak_max
        body_weight_n = self._infer_body_weight_newton(env, fallback_body_weight=float(body_weight))
        landing_F_peak_norm = landing_F_peak_event / max(body_weight_n, eps)
        landing_dF_peak_norm = landing_dF_peak_event / max(float(land_dF_ref), eps)
        landing_force_norm_step = total_force / max(body_weight_n, eps)
        landing_dF_norm_step = d_force_pos / max(float(land_dF_ref), eps)

        clear_mask = ~landing_window_active
        self._landing_F_peak = torch.where(clear_mask, torch.zeros_like(self._landing_F_peak), self._landing_F_peak)
        self._landing_dF_peak = torch.where(clear_mask, torch.zeros_like(self._landing_dF_peak), self._landing_dF_peak)
        self._landing_rho_peak_max = torch.where(
            clear_mask,
            torch.zeros_like(self._landing_rho_peak_max),
            self._landing_rho_peak_max,
        )
        self._landing_active_prev = landing_window_active

        metrics = {
            "alpha_sw": torch.nan_to_num(alpha_sw, nan=0.0, posinf=0.0, neginf=0.0),
            "alpha_pre": torch.nan_to_num(alpha_pre, nan=0.0, posinf=0.0, neginf=0.0),
            "alpha_land": torch.nan_to_num(alpha_land, nan=0.0, posinf=0.0, neginf=0.0),
            "alpha_st": torch.nan_to_num(alpha_st, nan=0.0, posinf=0.0, neginf=0.0),
            "r_sw_h": torch.nan_to_num(r_sw_h, nan=0.0, posinf=0.0, neginf=0.0),
            "r_pre_v": torch.nan_to_num(r_pre_v, nan=0.0, posinf=0.0, neginf=0.0),
            "r_pre_a": torch.nan_to_num(r_pre_a, nan=0.0, posinf=0.0, neginf=0.0),
            "r_cop": torch.nan_to_num(r_cop, nan=0.0, posinf=0.0, neginf=0.0),
            "r_area": torch.nan_to_num(r_area, nan=0.0, posinf=0.0, neginf=0.0),
            "r_st_delta_cop": torch.nan_to_num(r_st_delta_cop, nan=0.0, posinf=0.0, neginf=0.0),
            "delta_cop": torch.nan_to_num(delta_cop, nan=0.0, posinf=0.0, neginf=0.0),
            "vz": torch.nan_to_num(vz, nan=0.0, posinf=0.0, neginf=0.0),
            "az": torch.nan_to_num(az, nan=0.0, posinf=0.0, neginf=0.0),
            "cop_margin": torch.nan_to_num(cop_margin, nan=0.0, posinf=0.0, neginf=0.0),
            "contact_area_ratio": torch.nan_to_num(contact_area_ratio, nan=0.0, posinf=0.0, neginf=0.0),
            "rho_peak": rho_peak,
            "landing_window": stage_data.landing_window.to(dtype=torch.float32),
            "landing_event_mask": landing_end,
            "landing_F_peak_norm": torch.nan_to_num(landing_F_peak_norm, nan=0.0, posinf=0.0, neginf=0.0),
            "landing_dF_peak_norm": torch.nan_to_num(landing_dF_peak_norm, nan=0.0, posinf=0.0, neginf=0.0),
            "landing_rho_peak_event": torch.nan_to_num(landing_rho_peak_event, nan=0.0, posinf=0.0, neginf=0.0),
            "landing_force_norm_step": torch.nan_to_num(landing_force_norm_step, nan=0.0, posinf=0.0, neginf=0.0),
            "landing_dF_norm_step": torch.nan_to_num(landing_dF_norm_step, nan=0.0, posinf=0.0, neginf=0.0),
        }
        self._store_last_metrics(metrics, enable_debug_buffers=enable_debug_buffers)
        return metrics

    def _get_metric_slice(
        self,
        name: str,
        env_id: int,
        default_shape: tuple[int, ...],
        default_dtype: torch.dtype = torch.float32,
        debug_name: Optional[str] = None,
    ) -> torch.Tensor:
        if self._debug_buffers_enabled:
            debug_tensor = getattr(self, f"_debug_{debug_name or name}", None)
            if torch.is_tensor(debug_tensor):
                return debug_tensor[env_id].detach().clone()
        if self._last_metrics is not None:
            metric = self._last_metrics.get(name)
            if torch.is_tensor(metric):
                return metric[env_id].detach().clone()
        return torch.zeros(default_shape, device=self._env.device, dtype=default_dtype)

    def get_debug_dict(self, env_id: int, debug_context: dict[str, Any]) -> dict[str, torch.Tensor]:
        env_id = int(max(0, min(env_id, self._env.num_envs - 1)))
        scalar_shape = (self._num_feet,)

        alpha_sw = self._get_metric_slice("alpha_sw", env_id, scalar_shape)
        alpha_pre = self._get_metric_slice("alpha_pre", env_id, scalar_shape)
        alpha_land = self._get_metric_slice("alpha_land", env_id, scalar_shape)
        alpha_st = self._get_metric_slice("alpha_st", env_id, scalar_shape)
        r_sw_h = self._get_metric_slice("r_sw_h", env_id, scalar_shape)
        r_pre_v = self._get_metric_slice("r_pre_v", env_id, scalar_shape)
        r_pre_a = self._get_metric_slice("r_pre_a", env_id, scalar_shape)
        r_cop = self._get_metric_slice("r_cop", env_id, scalar_shape, debug_name="r_st_cop")
        r_area = self._get_metric_slice("r_area", env_id, scalar_shape, debug_name="r_st_area")
        r_st_delta_cop = self._get_metric_slice("r_st_delta_cop", env_id, scalar_shape)
        delta_cop = self._get_metric_slice("delta_cop", env_id, scalar_shape)
        vz = self._get_metric_slice("vz", env_id, scalar_shape)
        az = self._get_metric_slice("az", env_id, scalar_shape)
        cop_margin = self._get_metric_slice("cop_margin", env_id, scalar_shape)
        contact_area_ratio = self._get_metric_slice("contact_area_ratio", env_id, scalar_shape)
        rho_peak = self._get_metric_slice("rho_peak", env_id, scalar_shape)
        landing_window = self._get_metric_slice("landing_window", env_id, scalar_shape)

        landing_event_mask = torch.zeros(scalar_shape, device=self._env.device, dtype=torch.bool)
        landing_F_peak_norm = torch.zeros(scalar_shape, device=self._env.device, dtype=torch.float32)
        landing_dF_peak_norm = torch.zeros(scalar_shape, device=self._env.device, dtype=torch.float32)
        landing_rho_peak_event = torch.zeros(scalar_shape, device=self._env.device, dtype=torch.float32)
        landing_force_norm_step = torch.zeros(scalar_shape, device=self._env.device, dtype=torch.float32)
        landing_dF_norm_step = torch.zeros(scalar_shape, device=self._env.device, dtype=torch.float32)
        if self._last_metrics is not None:
            if "landing_event_mask" in self._last_metrics:
                landing_event_mask = self._last_metrics["landing_event_mask"][env_id].detach().clone()
            if "landing_F_peak_norm" in self._last_metrics:
                landing_F_peak_norm = self._last_metrics["landing_F_peak_norm"][env_id].detach().clone()
            if "landing_dF_peak_norm" in self._last_metrics:
                landing_dF_peak_norm = self._last_metrics["landing_dF_peak_norm"][env_id].detach().clone()
            if "landing_rho_peak_event" in self._last_metrics:
                landing_rho_peak_event = self._last_metrics["landing_rho_peak_event"][env_id].detach().clone()
            if "landing_force_norm_step" in self._last_metrics:
                landing_force_norm_step = self._last_metrics["landing_force_norm_step"][env_id].detach().clone()
            if "landing_dF_norm_step" in self._last_metrics:
                landing_dF_norm_step = self._last_metrics["landing_dF_norm_step"][env_id].detach().clone()

        w_cop = float(debug_context.get("w_cop", 0.0))
        w_area = float(debug_context.get("w_area", 0.0))
        r_contact_base = torch.nan_to_num(w_cop * r_cop + w_area * r_area, nan=0.0, posinf=0.0, neginf=0.0)

        enable_land = 1.0 if bool(debug_context.get("enable_contact_quality_on_landing", True)) else 0.0
        enable_st = 1.0 if bool(debug_context.get("enable_contact_quality_on_stance", True)) else 0.0
        gamma_land = max(float(debug_context.get("gamma_land", 0.0)), 0.0)
        gamma_st = max(float(debug_context.get("gamma_st", 0.0)), 0.0)
        contact_phase_weight = torch.nan_to_num(
            enable_land * gamma_land * alpha_land + enable_st * gamma_st * alpha_st,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if bool(debug_context.get("enable_contact_quality_reward", False)):
            r_contact = torch.nan_to_num(contact_phase_weight * r_contact_base, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            r_contact = torch.zeros_like(r_contact_base)

        if bool(debug_context.get("enable_landing_event_penalty", False)):
            landing_event_penalty = torch.where(
                landing_event_mask,
                -float(debug_context.get("w_land_F", 0.0)) * landing_F_peak_norm
                - float(debug_context.get("w_land_dF", 0.0)) * landing_dF_peak_norm
                - float(debug_context.get("w_land_rho", 0.0)) * landing_rho_peak_event,
                torch.zeros_like(landing_F_peak_norm),
            )
        else:
            landing_event_penalty = torch.zeros_like(landing_F_peak_norm)
        landing_event_penalty = torch.nan_to_num(landing_event_penalty, nan=0.0, posinf=0.0, neginf=0.0)

        landing_dense_F = torch.zeros_like(landing_force_norm_step)
        landing_dense_dF = torch.zeros_like(landing_dF_norm_step)
        landing_dense_reward = torch.zeros_like(landing_force_norm_step)
        if bool(debug_context.get("enable_landing_dense_reward", False)):
            land_dense_F_ref = max(float(debug_context.get("land_dense_F_ref", 1.0)), 1e-6)
            F_excess = torch.clamp(landing_force_norm_step - land_dense_F_ref, min=0.0)
            dF_excess = torch.clamp(landing_dF_norm_step - 1.0, min=0.0)
            landing_dense_F = -torch.square(F_excess)
            landing_dense_dF = -torch.square(dF_excess)
            landing_dense_reward = torch.nan_to_num(
                alpha_land
                * (
                    float(debug_context.get("w_land_dense_F", 0.0)) * landing_dense_F
                    + float(debug_context.get("w_land_dense_dF", 0.0)) * landing_dense_dF
                ),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

        return {
            "alpha_sw": alpha_sw,
            "alpha_pre": alpha_pre,
            "alpha_land": alpha_land,
            "alpha_st": alpha_st,
            "r_sw_h": r_sw_h,
            "r_pre_v": r_pre_v,
            "r_pre_a": r_pre_a,
            "r_st_cop": r_cop,
            "r_st_area": r_area,
            "r_st_delta_cop": r_st_delta_cop,
            "delta_cop": delta_cop,
            "r_cop": r_cop,
            "r_area": r_area,
            "r_contact_base": r_contact_base,
            "contact_phase_weight": contact_phase_weight,
            "r_contact": r_contact,
            "landing_F_peak": self._landing_F_peak[env_id].detach().clone(),
            "landing_dF_peak": self._landing_dF_peak[env_id].detach().clone(),
            "landing_rho_peak_max": self._landing_rho_peak_max[env_id].detach().clone(),
            "landing_event_penalty": landing_event_penalty,
            "r_land_dense_f": landing_dense_F,
            "r_land_dense_df": landing_dense_dF,
            "r_land_dense": landing_dense_reward,
            "vz": vz,
            "az": az,
            "cop_margin": cop_margin,
            "contact_area_ratio": contact_area_ratio,
            "rho_peak": rho_peak,
            "landing_window": landing_window,
            "enable_land": torch.full(scalar_shape, enable_land, device=self._env.device, dtype=torch.float32),
            "enable_st": torch.full(scalar_shape, enable_st, device=self._env.device, dtype=torch.float32),
            "gamma_land": torch.full(scalar_shape, gamma_land, device=self._env.device, dtype=torch.float32),
            "gamma_st": torch.full(scalar_shape, gamma_st, device=self._env.device, dtype=torch.float32),
        }


class contact_stage_reward_v1(ManagerTermBase):
    """Stage-aware reward V1: PreLanding + contact-phase dense terms + Landing event penalty."""

    SWING_STAGE_ID = _ContactStageRewardV1SharedCore.SWING_STAGE_ID
    PRE_STAGE_ID = _ContactStageRewardV1SharedCore.PRE_STAGE_ID
    LANDING_STAGE_ID = _ContactStageRewardV1SharedCore.LANDING_STAGE_ID
    STANCE_STAGE_ID = _ContactStageRewardV1SharedCore.STANCE_STAGE_ID

    def __init__(self, cfg: "RewardTermCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._stage_sensor_cfg = cfg.params.get("stage_sensor_cfg", SceneEntityCfg("contact_stage_filter"))
        self._tactile_sensor_cfg = cfg.params.get("tactile_sensor_cfg", SceneEntityCfg("foot_tactile"))
        self._asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self._num_feet = int(cfg.params.get("num_feet", 2))
        self._debug_context: dict[str, Any] = {}

        registry = getattr(env, "_contact_stage_reward_v1_shared_cores", None)
        if registry is None:
            registry = {}
            setattr(env, "_contact_stage_reward_v1_shared_cores", registry)
        self._shared_core_key = self._make_shared_core_key(cfg.params)
        core = registry.get(self._shared_core_key)
        if core is None:
            core = _ContactStageRewardV1SharedCore(
                env=env,
                stage_sensor_cfg=self._stage_sensor_cfg,
                tactile_sensor_cfg=self._tactile_sensor_cfg,
                asset_cfg=self._asset_cfg,
                num_feet=self._num_feet,
            )
            registry[self._shared_core_key] = core
        self._shared_core = core

    @staticmethod
    def _make_shared_core_key(params: dict[str, Any]) -> tuple[Any, ...]:
        return (
            _scene_entity_signature(params.get("stage_sensor_cfg", SceneEntityCfg("contact_stage_filter"))),
            _scene_entity_signature(params.get("tactile_sensor_cfg", SceneEntityCfg("foot_tactile"))),
            _scene_entity_signature(params.get("asset_cfg", SceneEntityCfg("robot"))),
            str(params.get("swing_command_name", "base_velocity")),
            float(params.get("swing_vel_threshold", 0.15)),
            float(params.get("swing_h_ref", 0.07)),
            float(params.get("pre_vz_ref", 0.24)),
            float(params.get("pre_az_ref", 7.0)),
            float(params.get("pre_az_filter_alpha", 0.7)),
            float(params.get("land_dF_ref", 3500.0)),
            float(params.get("body_weight", 420.0)),
            float(params.get("cop_margin_max", 0.038)),
            float(params.get("delta_cop_ref", 0.01)),
            float(params.get("contact_quality_eps", 1e-6)),
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._shared_core.reset(env_ids)

    def get_debug_dict(self, env_id: int) -> dict[str, torch.Tensor]:
        return self._shared_core.get_debug_dict(env_id, self._debug_context)

    def _compute_warmup_scale(
        self,
        env: "ManagerBasedRLEnv",
        enable_stage_reward_warmup: bool,
        stage_reward_warmup_steps: int,
    ) -> float:
        warmup_scale = 1.0
        if enable_stage_reward_warmup:
            warmup_steps = max(int(stage_reward_warmup_steps), 0)
            if warmup_steps > 0:
                warmup_iteration_getter = getattr(env, "get_stage_reward_warmup_iteration", None)
                if callable(warmup_iteration_getter):
                    warmup_progress = float(max(int(warmup_iteration_getter()), 0))
                else:
                    common_steps_raw = getattr(env, "common_step_counter", None)
                    warmup_progress = float(max(int(common_steps_raw), 0)) if common_steps_raw is not None else 0.0
                warmup_scale = min(warmup_progress / float(warmup_steps), 1.0)
        return warmup_scale

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
        enable_landing_dense_reward: bool = False,
        w_land_dense_F: float = 0.0,
        w_land_dense_dF: float = 0.0,
        land_dense_F_ref: float = 1.0,
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
        enable_debug_buffers: bool = False,
    ) -> torch.Tensor:
        del stage_sensor_cfg, tactile_sensor_cfg, asset_cfg
        if not enable_stage_reward_v1:
            return _zero_reward(env)

        metrics = self._shared_core.compute(
            env=env,
            swing_command_name=swing_command_name,
            swing_vel_threshold=swing_vel_threshold,
            swing_h_ref=swing_h_ref,
            pre_vz_ref=pre_vz_ref,
            pre_az_ref=pre_az_ref,
            pre_az_filter_alpha=pre_az_filter_alpha,
            land_dF_ref=land_dF_ref,
            body_weight=body_weight,
            cop_margin_max=cop_margin_max,
            delta_cop_ref=delta_cop_ref,
            contact_quality_eps=contact_quality_eps,
            enable_debug_buffers=enable_debug_buffers,
        )
        if metrics is None:
            return _zero_reward(env)

        alpha_sw = metrics["alpha_sw"]
        alpha_pre = metrics["alpha_pre"]
        alpha_land = metrics["alpha_land"]
        alpha_st = metrics["alpha_st"]

        reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
        if enable_swing_clearance_reward:
            reward += (alpha_sw * float(w_swing_h) * metrics["r_sw_h"]).sum(dim=-1)
        if enable_prelanding_reward:
            r_pre = float(w_pre_v) * metrics["r_pre_v"] + float(w_pre_a) * metrics["r_pre_a"]
            reward += torch.nan_to_num(alpha_pre * r_pre, nan=0.0, posinf=0.0, neginf=0.0).sum(dim=-1)
        if enable_contact_quality_reward:
            enable_land = 1.0 if bool(enable_contact_quality_on_landing) else 0.0
            enable_st = 1.0 if bool(enable_contact_quality_on_stance) else 0.0
            gamma_land = max(float(gamma_land), 0.0)
            gamma_st = max(float(gamma_st), 0.0)
            contact_phase_weight = torch.nan_to_num(
                enable_land * gamma_land * alpha_land + enable_st * gamma_st * alpha_st,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            r_contact_base = torch.nan_to_num(
                float(w_cop) * metrics["r_cop"] + float(w_area) * metrics["r_area"],
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            reward += torch.nan_to_num(contact_phase_weight * r_contact_base, nan=0.0, posinf=0.0, neginf=0.0).sum(dim=-1)
        if enable_stance_delta_cop_reward:
            reward += (alpha_st * float(w_st_delta_cop) * metrics["r_st_delta_cop"]).sum(dim=-1)
        if enable_landing_dense_reward:
            land_dense_F_ref = max(float(land_dense_F_ref), 1e-6)
            landing_dense_F = -torch.square(
                torch.clamp(metrics["landing_force_norm_step"] - land_dense_F_ref, min=0.0)
            )
            landing_dense_dF = -torch.square(torch.clamp(metrics["landing_dF_norm_step"] - 1.0, min=0.0))
            landing_dense = torch.nan_to_num(
                alpha_land
                * (
                    float(w_land_dense_F) * landing_dense_F
                    + float(w_land_dense_dF) * landing_dense_dF
                ),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            reward += landing_dense.sum(dim=-1)

        if enable_landing_event_penalty:
            landing_penalty = torch.where(
                metrics["landing_event_mask"],
                -float(w_land_F) * metrics["landing_F_peak_norm"]
                - float(w_land_dF) * metrics["landing_dF_peak_norm"]
                - float(w_land_rho) * metrics["landing_rho_peak_event"],
                torch.zeros_like(metrics["landing_F_peak_norm"]),
            )
            reward += torch.nan_to_num(landing_penalty, nan=0.0, posinf=0.0, neginf=0.0).sum(dim=-1)

        warmup_scale = self._compute_warmup_scale(env, enable_stage_reward_warmup, stage_reward_warmup_steps)
        reward = torch.nan_to_num(reward * float(warmup_scale), nan=0.0, posinf=0.0, neginf=0.0)

        self._debug_context = {
            "enable_contact_quality_reward": enable_contact_quality_reward,
            "enable_contact_quality_on_landing": enable_contact_quality_on_landing,
            "enable_contact_quality_on_stance": enable_contact_quality_on_stance,
            "gamma_land": gamma_land,
            "gamma_st": gamma_st,
            "w_cop": w_cop,
            "w_area": w_area,
            "enable_landing_event_penalty": enable_landing_event_penalty,
            "w_land_F": w_land_F,
            "w_land_dF": w_land_dF,
            "w_land_rho": w_land_rho,
            "enable_landing_dense_reward": enable_landing_dense_reward,
            "w_land_dense_F": w_land_dense_F,
            "w_land_dense_dF": w_land_dense_dF,
            "land_dense_F_ref": land_dense_F_ref,
        }

        if enable_self_check:
            if not torch.isfinite(metrics["az"]).all():
                raise RuntimeError("contact_stage_reward_v1 produced invalid az.")
            if not torch.isfinite(reward).all():
                raise RuntimeError("contact_stage_reward_v1 produced NaN/inf reward.")
            if enable_landing_event_penalty and not torch.isfinite(self._shared_core._landing_F_peak).all():
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

    
