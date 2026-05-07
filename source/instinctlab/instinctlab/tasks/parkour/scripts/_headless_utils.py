from __future__ import annotations

import copy
import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Callable

import torch

from instinct_rl.utils.utils import get_subobs_by_components, get_subobs_size
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.io import load_pickle, load_yaml
from isaaclab_tasks.utils import get_checkpoint_path

import instinctlab.tasks.parkour.mdp as parkour_mdp
from instinctlab.sensors.contact_stage.contact_stage_cfg import ContactStageCfg
from instinctlab.sensors.foot_tactile import (
    FootTactileCfg,
    FootTactileDiffusionCfg,
    FootTactileNoiseCfg,
    FootTactileThresholdRandomizationCfg,
)
from instinctlab.tasks.parkour.config.g1.foot_tactile_geometry import make_ankle_roll_foot_tactile_template_cfg


@dataclass
class CheckpointInfo:
    resume_path: str
    log_dir: str
    log_root_path: str


@dataclass
class RunningStats:
    count: int = 0
    sum: float = 0.0
    min: float = float("inf")
    max: float = float("-inf")

    def update(self, value: float) -> None:
        if not math.isfinite(value):
            return
        self.count += 1
        self.sum += value
        self.min = value if value < self.min else self.min
        self.max = value if value > self.max else self.max

    def mean(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0

    def as_dict(self) -> dict[str, float | int | None]:
        if self.count == 0:
            return {"count": 0, "min": None, "max": None, "mean": None}
        return {"count": self.count, "min": self.min, "max": self.max, "mean": self.mean()}


@dataclass
class EvalState:
    stage_sensor: Any
    tactile_sensor: Any
    num_feet: int
    num_stages: int
    stage_name_map: dict[int, str]
    stage_ids: dict[str, int]
    stage_metric_names: dict[int, tuple[str, ...]]
    tactile_body_ids: list[int]
    body_has_polygon: torch.Tensor
    body_polygons: list[torch.Tensor | None]
    done_mask: torch.Tensor
    reached_target_mask: torch.Tensor
    time_out_mask: torch.Tensor
    failed_other_mask: torch.Tensor
    episode_len: torch.Tensor
    episode_return: torch.Tensor
    volume_points_triggered_any: torch.Tensor
    volume_points_trigger_steps: torch.Tensor
    volume_points_max_penetration: torch.Tensor
    prev_vz: torch.Tensor
    az_filt: torch.Tensor
    az_initialized: torch.Tensor
    prev_stance_cop: torch.Tensor
    stance_cop_initialized: torch.Tensor
    landing_active_prev: torch.Tensor
    landing_force_peak: torch.Tensor
    pre_az_filter_alpha: float
    global_stage_stats: dict[int, dict[str, RunningStats]]
    per_env_stage_stats: list[dict[int, dict[str, RunningStats]]]
    output_dir: str
    output_prefix: str


def _is_config_object(value: Any) -> bool:
    return hasattr(value, "__dataclass_fields__") and not isinstance(value, type)


def _patch_missing_config_fields(target_cfg: Any, reference_cfg: Any, path: str, added_items: list[str]) -> None:
    if not _is_config_object(target_cfg) or not _is_config_object(reference_cfg):
        return

    for field_name in reference_cfg.__dataclass_fields__:
        ref_value = getattr(reference_cfg, field_name)
        field_path = f"{path}.{field_name}" if path else field_name
        if not hasattr(target_cfg, field_name):
            setattr(target_cfg, field_name, copy.deepcopy(ref_value))
            added_items.append(field_path)
            continue

        target_value = getattr(target_cfg, field_name)
        if _is_config_object(target_value) and _is_config_object(ref_value):
            _patch_missing_config_fields(target_value, ref_value, field_path, added_items)


def resolve_checkpoint_info(args_cli, agent_cfg, *, script_name: str) -> CheckpointInfo:
    log_root_path = os.path.abspath(os.path.join("logs", "instinct_rl", agent_cfg.experiment_name))

    if args_cli.checkpoint_path is not None:
        resume_path = os.path.abspath(args_cli.checkpoint_path)
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"--checkpoint_path not found: {resume_path}")
        log_dir = os.path.dirname(resume_path)
        agent_cfg.load_run = "__direct__"
        print(f"[INFO] Loading checkpoint directly: {resume_path}")
        return CheckpointInfo(resume_path=resume_path, log_dir=log_dir, log_root_path=log_root_path)

    agent_cfg.load_run = args_cli.load_run
    if agent_cfg.load_run is None:
        raise RuntimeError(
            f"[ERROR] {script_name} requires a checkpoint. Please specify --checkpoint_path or --load_run."
        )

    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if os.path.isabs(agent_cfg.load_run):
        resume_path = get_checkpoint_path(
            os.path.dirname(agent_cfg.load_run),
            os.path.basename(agent_cfg.load_run),
            agent_cfg.load_checkpoint,
        )
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    return CheckpointInfo(resume_path=resume_path, log_dir=os.path.dirname(resume_path), log_root_path=log_root_path)


def load_env_and_agent_cfg(args_cli, env_cfg, agent_cfg, log_dir: str, *, prefer_saved: bool) -> tuple[Any, dict]:
    params_dir = os.path.join(log_dir, "params")
    env_pkl_path = os.path.join(params_dir, "env.pkl")
    agent_yaml_path = os.path.join(params_dir, "agent.yaml")

    current_env_cfg = copy.deepcopy(env_cfg)
    agent_cfg_dict = agent_cfg.to_dict()

    ignore_saved_env_cfg = bool(getattr(args_cli, "ignore_saved_env_cfg", False))

    if ignore_saved_env_cfg and os.path.isfile(env_pkl_path):
        print(f"[Compat] Ignoring saved env config and using current task config: {env_pkl_path}")
    elif args_cli.env_cfg or (prefer_saved and os.path.isfile(env_pkl_path)):
        if not os.path.isfile(env_pkl_path):
            raise FileNotFoundError(f"Saved env config not found: {env_pkl_path}")
        env_cfg = load_pickle(env_pkl_path)
        print(f"[Compat] Loaded env config: {env_pkl_path}")
        added_items: list[str] = []
        _patch_missing_config_fields(env_cfg, current_env_cfg, "", added_items)
        if added_items:
            print("[Compat] Patched legacy env config fields: " + ", ".join(added_items))

    if args_cli.agent_cfg or (prefer_saved and os.path.isfile(agent_yaml_path)):
        if not os.path.isfile(agent_yaml_path):
            raise FileNotFoundError(f"Saved agent config not found: {agent_yaml_path}")
        agent_cfg_dict = load_yaml(agent_yaml_path)
        print(f"[Compat] Loaded agent config: {agent_yaml_path}")

    return env_cfg, agent_cfg_dict


def apply_runtime_env_overrides(args_cli, env_cfg) -> None:
    if hasattr(env_cfg, "sim"):
        if hasattr(env_cfg.sim, "device"):
            env_cfg.sim.device = args_cli.device
        if hasattr(env_cfg.sim, "use_fabric"):
            env_cfg.sim.use_fabric = not args_cli.disable_fabric
    if args_cli.num_envs is not None and hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "num_envs"):
        env_cfg.scene.num_envs = args_cli.num_envs
        terrain_cfg = getattr(env_cfg.scene, "terrain", None)
        if terrain_cfg is not None and hasattr(terrain_cfg, "num_envs"):
            terrain_cfg.num_envs = args_cli.num_envs


def apply_current_eval_scene_overrides(args_cli, env_cfg, current_scene_env_cfg) -> None:
    if not args_cli.eval_use_current_scene or current_scene_env_cfg is None:
        return

    applied_items: list[str] = []
    if hasattr(current_scene_env_cfg, "episode_length_s"):
        env_cfg.episode_length_s = current_scene_env_cfg.episode_length_s
        applied_items.append("episode_length_s")

    current_scene_cfg = getattr(current_scene_env_cfg, "scene", None)
    target_scene_cfg = getattr(env_cfg, "scene", None)
    if current_scene_cfg is not None and target_scene_cfg is not None and hasattr(current_scene_cfg, "terrain"):
        target_scene_cfg.terrain = copy.deepcopy(current_scene_cfg.terrain)
        applied_items.append("scene.terrain")

    if hasattr(current_scene_env_cfg, "commands"):
        env_cfg.commands = copy.deepcopy(current_scene_env_cfg.commands)
        applied_items.append("commands")

    if hasattr(current_scene_env_cfg, "events"):
        env_cfg.events = copy.deepcopy(current_scene_env_cfg.events)
        applied_items.append("events")

    if hasattr(current_scene_env_cfg, "terminations"):
        env_cfg.terminations = copy.deepcopy(current_scene_env_cfg.terminations)
        applied_items.append("terminations")

    if applied_items:
        print("[EvalScene] Applied current task scene overrides: " + ", ".join(applied_items))


def ensure_foot_tactile_cfg_schema_compat(env_cfg) -> None:
    scene_cfg = getattr(env_cfg, "scene", None)
    if scene_cfg is None:
        return

    tactile_cfg = getattr(scene_cfg, "foot_tactile", None)
    if tactile_cfg is None:
        return

    added_items: list[str] = []
    if not hasattr(tactile_cfg, "threshold_randomization_cfg"):
        tactile_cfg.threshold_randomization_cfg = FootTactileThresholdRandomizationCfg(enable=False)
        added_items.append("scene.foot_tactile.threshold_randomization_cfg")

    if added_items:
        print("[Compat] Patched legacy foot_tactile config: " + ", ".join(added_items))


def ensure_eval_stage_tactile_compat(env_cfg) -> None:
    scene_cfg = getattr(env_cfg, "scene", None)
    events_cfg = getattr(env_cfg, "events", None)
    if scene_cfg is None or events_cfg is None:
        return

    added_items: list[str] = []
    if getattr(scene_cfg, "foot_tactile", None) is None:
        scene_cfg.foot_tactile = FootTactileCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link",
            template_cfg=make_ankle_roll_foot_tactile_template_cfg(),
            taxel_z_offset=-0.038,
            raycast_offset=0.0003,
            pad_thickness=0.0014,
            max_support_dist=0.004,
            support_weight_band=0.004,
            support_weight_rho=50.0,
            alignment_gate_a0=0.3,
            alignment_gate_q=1.5,
            align_mix=0.4,
            min_force_threshold=5.0,
            active_taxel_threshold=0.5,
            diffusion_cfg=FootTactileDiffusionCfg(
                enable_neighbor_diffusion=True,
                diffusion_knn=4,
                diffusion_alpha=0.10,
                diffusion_iters=1,
                preserve_total_force_after_diffusion=True,
            ),
            noise_cfg=FootTactileNoiseCfg(
                enable=False,
                force_relative_error_max=0.08,
                delay_prob=0.0,
                max_delay_frames=2,
            ),
            update_period=0.02,
            debug_vis=False,
        )
        added_items.append("scene.foot_tactile")

    if getattr(scene_cfg, "contact_stage_filter", None) is None:
        scene_cfg.contact_stage_filter = ContactStageCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link",
            update_period=0.02,
            debug_vis=False,
        )
        added_items.append("scene.contact_stage_filter")

    if getattr(events_cfg, "bind_foot_tactile", None) is None:
        events_cfg.bind_foot_tactile = EventTerm(
            func=parkour_mdp.bind_foot_tactile,
            mode="startup",
            params={
                "tactile_cfg": SceneEntityCfg("foot_tactile"),
                "contact_forces_cfg": SceneEntityCfg("contact_forces_foot"),
            },
        )
        added_items.append("events.bind_foot_tactile")

    if getattr(events_cfg, "bind_contact_stage_filter", None) is None:
        events_cfg.bind_contact_stage_filter = EventTerm(
            func=parkour_mdp.bind_contact_stage_filter,
            mode="startup",
            params={
                "stage_cfg": SceneEntityCfg("contact_stage_filter"),
                "tactile_cfg": SceneEntityCfg("foot_tactile"),
            },
        )
        added_items.append("events.bind_contact_stage_filter")

    ensure_foot_tactile_cfg_schema_compat(env_cfg)

    if added_items:
        print("[EvalCompat] Injected eval-only sensor config: " + ", ".join(added_items))


def inject_target_reached_termination(env_cfg) -> None:
    env_cfg.terminations.target_reached = DoneTerm(
        func=parkour_mdp.reached_target_termination,
        params={"command_name": "base_velocity"},
    )


def _get_depth_component_names(agent_cfg_source) -> list[str]:
    if isinstance(agent_cfg_source, dict):
        return list(agent_cfg_source["policy"]["encoder_configs"]["depth_encoder"]["component_names"])
    return list(agent_cfg_source.policy.encoder_configs.depth_encoder.component_names)


def build_parkour_onnx_policy(env, agent_cfg_source, model_dir: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"ONNX export directory not found: {model_dir}")
    try:
        from .onnxer import load_parkour_onnx_model
    except ImportError:
        from onnxer import load_parkour_onnx_model
    depth_component_names = _get_depth_component_names(agent_cfg_source)
    return load_parkour_onnx_model(
        model_dir=model_dir,
        get_subobs_func=lambda obs: get_subobs_by_components(
            obs,
            depth_component_names,
            env.get_obs_segments(),
            temporal=True,
        ),
        depth_shape=env.get_obs_segments()["depth_image"],
        proprio_slice=slice(
            0,
            get_subobs_size(
                env.get_obs_segments(),
                [
                    "base_lin_vel",
                    "base_ang_vel",
                    "projected_gravity",
                    "velocity_commands",
                    "joint_pos",
                    "joint_vel",
                    "actions",
                    "foot_contact_state",
                ],
            ),
        ),
    )


def list_export_artifacts(export_dir: str) -> list[str]:
    if not os.path.isdir(export_dir):
        return []
    return sorted(
        entry
        for entry in os.listdir(export_dir)
        if os.path.isfile(os.path.join(export_dir, entry)) and (entry.endswith(".onnx") or entry.endswith(".npz"))
    )


def export_policy_and_validate(ppo_runner, env, agent_cfg_source, export_dir: str) -> dict[str, Any]:
    if env.unwrapped.num_envs != 1:
        raise ValueError("Exporting to ONNX is only supported for single environment.")

    os.makedirs(export_dir, exist_ok=True)
    obs, _ = env.get_observations()
    ppo_runner.export_as_onnx(obs, export_dir)

    artifacts = list_export_artifacts(export_dir)
    if not any(name.endswith(".onnx") for name in artifacts):
        raise RuntimeError(f"No ONNX artifacts were generated in: {export_dir}")

    torch_policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    onnx_policy = build_parkour_onnx_policy(env, agent_cfg_source, export_dir)

    with torch.inference_mode():
        torch_actions = torch_policy(obs)
        onnx_actions = onnx_policy(obs)
    diff = (onnx_actions - torch_actions).abs()
    if not torch.isfinite(diff).all():
        raise RuntimeError("ONNX validation produced non-finite action differences.")

    return {
        "export_dir": export_dir,
        "artifacts": artifacts,
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
    }


def _iter_reward_groups(reward_manager) -> list[str | None]:
    active_terms = getattr(reward_manager, "active_terms", None)
    if isinstance(active_terms, dict):
        return list(active_terms.keys())
    return [None]


def _find_stage_reward_cfg(env) -> object | None:
    reward_manager = getattr(env.unwrapped, "reward_manager", None)
    if reward_manager is None:
        return None
    candidate_names = (
        "stage_pre_v_v1",
        "stage_pre_a_v1",
        "stage_stance_cop_v1",
        "stage_stance_area_v1",
        "stage_stance_delta_cop_v1",
        "stage_landing_f_v1",
        "stage_landing_df_v1",
        "stage_landing_rho_v1",
        "stage_swing_clearance_v1",
    )
    for group_name in _iter_reward_groups(reward_manager):
        for term_name in candidate_names:
            try:
                if group_name is None:
                    return reward_manager.get_term_cfg(term_name)
                return reward_manager.get_term_cfg(term_name, group_name=group_name)
            except Exception:
                continue
    return None


def _make_stage_metric_stats(stage_metric_names: dict[int, tuple[str, ...]]) -> dict[int, dict[str, RunningStats]]:
    return {
        stage_id: {metric_name: RunningStats() for metric_name in metric_names}
        for stage_id, metric_names in stage_metric_names.items()
    }


def _infer_body_sides(body_names: list[str]) -> list[str]:
    sides: list[str] = []
    for name in body_names:
        lower = name.lower()
        if "left" in lower or lower.startswith("l_") or "_l_" in lower:
            sides.append("left")
        elif "right" in lower or lower.startswith("r_") or "_r_" in lower:
            sides.append("right")
        else:
            sides.append("left")
    return sides


def _point_in_polygon_even_odd(points_xy: torch.Tensor, polygon_xy: torch.Tensor) -> torch.Tensor:
    x = points_xy[:, 0:1]
    y = points_xy[:, 1:2]
    x1 = polygon_xy[:, 0].unsqueeze(0)
    y1 = polygon_xy[:, 1].unsqueeze(0)
    x2 = torch.roll(x1, shifts=-1, dims=1)
    y2 = torch.roll(y1, shifts=-1, dims=1)
    cond = (y1 > y) != (y2 > y)
    x_inter = (x2 - x1) * (y - y1) / ((y2 - y1) + 1e-12) + x1
    hits = cond & (x < x_inter)
    return (hits.sum(dim=1) % 2) == 1


def _distance_point_to_polygon_boundary(points_xy: torch.Tensor, polygon_xy: torch.Tensor) -> torch.Tensor:
    seg_start = polygon_xy
    seg_end = torch.roll(polygon_xy, shifts=-1, dims=0)
    seg_vec = seg_end - seg_start
    rel = points_xy.unsqueeze(1) - seg_start.unsqueeze(0)
    denom = (seg_vec * seg_vec).sum(dim=-1).clamp_min(1e-12).unsqueeze(0)
    t = ((rel * seg_vec.unsqueeze(0)).sum(dim=-1) / denom).clamp(0.0, 1.0)
    proj = seg_start.unsqueeze(0) + t.unsqueeze(-1) * seg_vec.unsqueeze(0)
    dist = torch.norm(points_xy.unsqueeze(1) - proj, dim=-1)
    return dist.min(dim=1).values


def _signed_distance_to_polygon(points_xy: torch.Tensor, polygon_xy: torch.Tensor) -> torch.Tensor:
    unsigned = _distance_point_to_polygon_boundary(points_xy, polygon_xy)
    inside = _point_in_polygon_even_odd(points_xy, polygon_xy)
    return torch.where(inside, unsigned, -unsigned)


def _resolve_stage_to_tactile_ids(stage_sensor, tactile_sensor, num_feet: int) -> list[int]:
    if tactile_sensor is None:
        return list(range(max(num_feet, 0)))
    stage_names = list(getattr(stage_sensor, "body_names", [])) if stage_sensor is not None else []
    tactile_names = list(getattr(tactile_sensor, "body_names", []))
    num_tactile = int(getattr(tactile_sensor, "num_bodies", len(tactile_names)))
    if num_tactile <= 0:
        return [0 for _ in range(num_feet)]

    if not stage_names or not tactile_names:
        ids = list(range(min(num_feet, num_tactile)))
    else:
        tactile_name_to_id = {name: idx for idx, name in enumerate(tactile_names)}
        ids = []
        for stage_name in stage_names[:num_feet]:
            ids.append(int(tactile_name_to_id.get(stage_name, len(ids))))

    if not ids:
        ids = list(range(min(num_feet, num_tactile)))
    while len(ids) < num_feet:
        ids.append(len(ids))
    return [int(min(max(idx, 0), num_tactile - 1)) for idx in ids[:num_feet]]


def _build_eval_outline_cache(tactile_sensor, tactile_ids: list[int], device: torch.device) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
    num_feet = len(tactile_ids)
    body_has_polygon = torch.zeros(num_feet, device=device, dtype=torch.bool)
    body_polygons: list[torch.Tensor | None] = [None for _ in range(num_feet)]
    if tactile_sensor is None:
        return body_has_polygon, body_polygons

    template_cfg = getattr(tactile_sensor.cfg, "template_cfg", None)
    if template_cfg is None:
        return body_has_polygon, body_polygons
    left_outline = getattr(template_cfg, "left_outline_xy", None)
    right_outline = getattr(template_cfg, "right_outline_xy", None)
    if left_outline is None or right_outline is None:
        return body_has_polygon, body_polygons

    left_outline_t = torch.tensor(left_outline, device=device, dtype=torch.float32)
    right_outline_t = torch.tensor(right_outline, device=device, dtype=torch.float32)
    if (
        left_outline_t.ndim != 2
        or right_outline_t.ndim != 2
        or left_outline_t.shape[-1] != 2
        or right_outline_t.shape[-1] != 2
        or left_outline_t.shape[0] < 3
        or right_outline_t.shape[0] < 3
    ):
        return body_has_polygon, body_polygons

    body_names = list(getattr(tactile_sensor, "body_names", []))
    body_sides = getattr(tactile_sensor, "_body_sides", None)
    if body_sides is None or len(body_sides) != len(body_names):
        body_sides = _infer_body_sides(body_names)

    for foot_id, tactile_id in enumerate(tactile_ids):
        side = body_sides[tactile_id] if tactile_id < len(body_sides) else "left"
        body_polygons[foot_id] = left_outline_t if side == "left" else right_outline_t
        body_has_polygon[foot_id] = True
    return body_has_polygon, body_polygons


def _compute_eval_cop_margin(
    cop_b: torch.Tensor,
    body_has_polygon: torch.Tensor,
    body_polygons: list[torch.Tensor | None],
) -> torch.Tensor:
    num_envs, num_feet, _ = cop_b.shape
    signed_margin = torch.full((num_envs, num_feet), float("nan"), device=cop_b.device, dtype=cop_b.dtype)
    for foot_id in range(num_feet):
        if not bool(body_has_polygon[foot_id]):
            continue
        polygon = body_polygons[foot_id]
        if polygon is None:
            continue
        signed_margin[:, foot_id] = _signed_distance_to_polygon(cop_b[:, foot_id, :], polygon.to(cop_b.dtype))
    return signed_margin


def _resolve_eval_stage_ids(stage_name_map: dict[int, str], num_stages: int) -> dict[str, int]:
    default_ids = {
        "swing": 0,
        "prelanding": 1,
        "landing": 2,
        "stance": 3,
    }
    resolved = default_ids.copy()
    for stage_id, stage_name in stage_name_map.items():
        stage_key = str(stage_name).replace(" ", "").replace("_", "").lower()
        if stage_key == "swing":
            resolved["swing"] = int(stage_id)
        elif stage_key == "prelanding":
            resolved["prelanding"] = int(stage_id)
        elif stage_key == "landing":
            resolved["landing"] = int(stage_id)
        elif stage_key == "stance":
            resolved["stance"] = int(stage_id)
    for key, value in tuple(resolved.items()):
        resolved[key] = int(min(max(value, 0), max(num_stages - 1, 0)))
    return resolved


def _reduce_step_reward(rewards: torch.Tensor) -> torch.Tensor:
    if rewards.ndim == 1:
        return rewards
    return rewards.sum(dim=-1)


def initialize_eval_state(args_cli, env, log_dir: str, resume_path: str) -> EvalState:
    if args_cli.eval_max_steps <= 0:
        raise ValueError("--eval_max_steps must be > 0.")
    if args_cli.eval_progress_every <= 0:
        raise ValueError("--eval_progress_every must be > 0.")
    if "contact_stage_filter" not in env.unwrapped.scene.sensors:
        raise RuntimeError("Eval requires 'contact_stage_filter' sensor in the scene.")

    stage_sensor = env.unwrapped.scene.sensors["contact_stage_filter"]
    tactile_sensor = env.unwrapped.scene.sensors.get("foot_tactile", None)
    if tactile_sensor is None:
        raise RuntimeError("Eval requires 'foot_tactile' sensor in the scene.")

    num_feet = int(stage_sensor.num_bodies)
    num_stages = int(getattr(stage_sensor, "NUM_STAGES", 4))
    if hasattr(stage_sensor, "stage_name_map"):
        stage_name_map = {int(k): str(v) for k, v in stage_sensor.stage_name_map().items()}
    else:
        stage_name_map = {stage_id: f"Stage{stage_id}" for stage_id in range(num_stages)}
    stage_ids = _resolve_eval_stage_ids(stage_name_map, num_stages)
    stage_metric_names = {
        stage_ids["prelanding"]: ("vz", "az"),
        stage_ids["landing"]: ("impact_force_peak",),
        stage_ids["stance"]: ("delta_cop", "cop_margin", "contact_area_ratio"),
    }
    tactile_body_ids = _resolve_stage_to_tactile_ids(stage_sensor, tactile_sensor, num_feet)
    body_has_polygon, body_polygons = _build_eval_outline_cache(tactile_sensor, tactile_body_ids, env.device)

    pre_az_filter_alpha = 0.7
    stage_reward_cfg = _find_stage_reward_cfg(env)
    if stage_reward_cfg is not None:
        pre_az_filter_alpha = float(stage_reward_cfg.params.get("pre_az_filter_alpha", pre_az_filter_alpha))

    output_dir = args_cli.eval_output_dir or os.path.join(log_dir, "eval")
    os.makedirs(output_dir, exist_ok=True)
    if args_cli.eval_output_prefix:
        output_prefix = args_cli.eval_output_prefix
    else:
        output_prefix = f"{os.path.splitext(os.path.basename(resume_path))[0]}_first_episode_{env.num_envs}env"

    print(
        f"[EvalMode] enabled. num_envs={env.num_envs}, num_feet={num_feet},"
        f" num_stages={num_stages}, output_dir={output_dir}"
    )

    return EvalState(
        stage_sensor=stage_sensor,
        tactile_sensor=tactile_sensor,
        num_feet=num_feet,
        num_stages=num_stages,
        stage_name_map=stage_name_map,
        stage_ids=stage_ids,
        stage_metric_names=stage_metric_names,
        tactile_body_ids=tactile_body_ids,
        body_has_polygon=body_has_polygon,
        body_polygons=body_polygons,
        done_mask=torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        reached_target_mask=torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        time_out_mask=torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        failed_other_mask=torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        episode_len=torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
        episode_return=torch.zeros(env.num_envs, dtype=torch.float32, device=env.device),
        volume_points_triggered_any=torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        volume_points_trigger_steps=torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
        volume_points_max_penetration=torch.zeros(env.num_envs, dtype=torch.float32, device=env.device),
        prev_vz=torch.zeros((env.num_envs, num_feet), dtype=torch.float32, device=env.device),
        az_filt=torch.zeros((env.num_envs, num_feet), dtype=torch.float32, device=env.device),
        az_initialized=torch.zeros((env.num_envs, num_feet), dtype=torch.bool, device=env.device),
        prev_stance_cop=torch.zeros((env.num_envs, num_feet, 2), dtype=torch.float32, device=env.device),
        stance_cop_initialized=torch.zeros((env.num_envs, num_feet), dtype=torch.bool, device=env.device),
        landing_active_prev=torch.zeros((env.num_envs, num_feet), dtype=torch.bool, device=env.device),
        landing_force_peak=torch.zeros((env.num_envs, num_feet), dtype=torch.float32, device=env.device),
        pre_az_filter_alpha=pre_az_filter_alpha,
        global_stage_stats=_make_stage_metric_stats(stage_metric_names),
        per_env_stage_stats=[_make_stage_metric_stats(stage_metric_names) for _ in range(env.num_envs)],
        output_dir=output_dir,
        output_prefix=output_prefix,
    )


def update_eval_state(eval_state: EvalState, env, rewards, dones, infos, timestep: int, args_cli) -> bool:
    eval_snapshot = infos.get("eval_pre_reset", None)
    if not isinstance(eval_snapshot, dict):
        raise RuntimeError("Eval requires eval_pre_reset snapshot from InstinctRlEnv.step().")

    snapshot_done = eval_snapshot.get("done")
    if not torch.is_tensor(snapshot_done):
        snapshot_done = dones > 0
    snapshot_done = snapshot_done.to(dtype=torch.bool)

    active_env_ids_t = torch.nonzero(~eval_state.done_mask, as_tuple=False).squeeze(-1)
    if active_env_ids_t.numel() > 0:
        step_reward = _reduce_step_reward(rewards)
        step_reward = torch.nan_to_num(step_reward, nan=0.0, posinf=0.0, neginf=0.0)
        eval_state.episode_return[active_env_ids_t] += step_reward[active_env_ids_t]
        eval_state.episode_len[active_env_ids_t] += 1

        stage_ids = eval_snapshot["contact_stage/dominant_stage_id"]
        vz_data = torch.nan_to_num(eval_snapshot["contact_stage/foot_vz"], nan=0.0, posinf=0.0, neginf=0.0)
        total_force_data = torch.nan_to_num(
            eval_snapshot["contact_stage/total_force"], nan=0.0, posinf=0.0, neginf=0.0
        ).clamp_min(0.0)
        landing_active_data = eval_snapshot["contact_stage/landing_window_active"].to(dtype=torch.bool)

        tactile_cop_all = torch.nan_to_num(
            eval_snapshot["foot_tactile/cop_b"],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        tactile_area_all = torch.nan_to_num(
            eval_snapshot["foot_tactile/contact_area_ratio"],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        cop_data = tactile_cop_all[:, eval_state.tactile_body_ids, :]
        contact_area_data = tactile_area_all[:, eval_state.tactile_body_ids]
        cop_margin_data = _compute_eval_cop_margin(cop_data, eval_state.body_has_polygon, eval_state.body_polygons)

        current_vz = vz_data[:, :eval_state.num_feet]
        az_data = torch.zeros_like(current_vz)
        active_prev_vz = eval_state.prev_vz[active_env_ids_t]
        active_prev_az = eval_state.az_filt[active_env_ids_t]
        active_initialized = eval_state.az_initialized[active_env_ids_t]
        az_raw_active = (current_vz[active_env_ids_t] - active_prev_vz) / max(float(env.unwrapped.step_dt), 1e-6)
        az_raw_active = torch.where(active_initialized, az_raw_active, torch.zeros_like(az_raw_active))
        az_active = torch.where(
            active_initialized,
            float(eval_state.pre_az_filter_alpha) * az_raw_active
            + (1.0 - float(eval_state.pre_az_filter_alpha)) * active_prev_az,
            torch.zeros_like(az_raw_active),
        )
        az_data[active_env_ids_t] = torch.nan_to_num(az_active, nan=0.0, posinf=0.0, neginf=0.0)
        eval_state.prev_vz[active_env_ids_t] = current_vz[active_env_ids_t]
        eval_state.az_filt[active_env_ids_t] = az_data[active_env_ids_t]
        eval_state.az_initialized[active_env_ids_t] = True

        penetration_depth_max = eval_snapshot.get("volume_points/max_penetration_depth")
        if not torch.is_tensor(penetration_depth_max):
            penetration_depth_max = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
        penetration_depth_max = torch.nan_to_num(
            penetration_depth_max,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)
        penetration_trigger = penetration_depth_max > 0.0
        eval_state.volume_points_triggered_any[active_env_ids_t] |= penetration_trigger[active_env_ids_t]
        eval_state.volume_points_trigger_steps[active_env_ids_t] += penetration_trigger[active_env_ids_t].to(
            dtype=torch.long
        )
        eval_state.volume_points_max_penetration[active_env_ids_t] = torch.maximum(
            eval_state.volume_points_max_penetration[active_env_ids_t],
            penetration_depth_max[active_env_ids_t],
        )

        active_env_ids = active_env_ids_t.tolist()
        for env_id in active_env_ids:
            for foot_id in range(eval_state.num_feet):
                force_value = float(total_force_data[env_id, foot_id].item())
                was_landing_active = bool(eval_state.landing_active_prev[env_id, foot_id].item())
                is_landing_active = bool(landing_active_data[env_id, foot_id].item())
                if is_landing_active:
                    eval_state.landing_force_peak[env_id, foot_id] = max(
                        float(eval_state.landing_force_peak[env_id, foot_id].item()),
                        force_value,
                    )
                landing_finalize = (was_landing_active and not is_landing_active) or (
                    bool(snapshot_done[env_id].item()) and (was_landing_active or is_landing_active)
                )
                if landing_finalize:
                    peak_force_value = float(eval_state.landing_force_peak[env_id, foot_id].item())
                    if is_landing_active:
                        peak_force_value = max(peak_force_value, force_value)
                    for stage_stats in (eval_state.per_env_stage_stats[env_id], eval_state.global_stage_stats):
                        stage_stats[eval_state.stage_ids["landing"]]["impact_force_peak"].update(peak_force_value)
                    eval_state.landing_force_peak[env_id, foot_id] = 0.0
                elif not is_landing_active:
                    eval_state.landing_force_peak[env_id, foot_id] = 0.0
                eval_state.landing_active_prev[env_id, foot_id] = is_landing_active and not bool(
                    snapshot_done[env_id].item()
                )

                stage_id = int(stage_ids[env_id, foot_id].item())
                if stage_id not in eval_state.stage_metric_names:
                    eval_state.stance_cop_initialized[env_id, foot_id] = False
                    continue

                if stage_id == eval_state.stage_ids["prelanding"]:
                    values = {
                        "vz": float(vz_data[env_id, foot_id].item()),
                        "az": float(az_data[env_id, foot_id].item()),
                    }
                elif stage_id == eval_state.stage_ids["stance"]:
                    values = {
                        "cop_margin": float(cop_margin_data[env_id, foot_id].item()),
                        "contact_area_ratio": float(contact_area_data[env_id, foot_id].item()),
                    }
                    if bool(eval_state.stance_cop_initialized[env_id, foot_id].item()):
                        delta_cop_value = float(
                            torch.norm(
                                cop_data[env_id, foot_id] - eval_state.prev_stance_cop[env_id, foot_id],
                                p=2,
                            ).item()
                        )
                        values["delta_cop"] = delta_cop_value
                    eval_state.prev_stance_cop[env_id, foot_id] = cop_data[env_id, foot_id]
                    eval_state.stance_cop_initialized[env_id, foot_id] = True
                else:
                    values = {}
                    eval_state.stance_cop_initialized[env_id, foot_id] = False

                for metric_name, metric_value in values.items():
                    eval_state.per_env_stage_stats[env_id][stage_id][metric_name].update(metric_value)
                    eval_state.global_stage_stats[stage_id][metric_name].update(metric_value)

                if stage_id != eval_state.stage_ids["stance"]:
                    eval_state.stance_cop_initialized[env_id, foot_id] = False

    new_done_t = torch.nonzero(snapshot_done & (~eval_state.done_mask), as_tuple=False).squeeze(-1)
    if new_done_t.numel() > 0:
        done_ids = new_done_t.tolist()
        target_reached = eval_snapshot.get("termination/target_reached")
        if not torch.is_tensor(target_reached):
            target_reached = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        target_reached = target_reached.to(dtype=torch.bool)
        for env_id in done_ids:
            eval_state.done_mask[env_id] = True
            reached_target = bool(target_reached[env_id].item())
            timed_out = bool(eval_snapshot["time_outs"][env_id].item())
            eval_state.reached_target_mask[env_id] = reached_target
            eval_state.time_out_mask[env_id] = timed_out and not reached_target
            eval_state.failed_other_mask[env_id] = (not reached_target) and (not timed_out)

    if timestep % args_cli.eval_progress_every == 0:
        completed = int(eval_state.done_mask.sum().item())
        success = int(eval_state.reached_target_mask.sum().item())
        success_rate = success / completed if completed > 0 else 0.0
        print(
            f"[EvalMode step={timestep}] completed={completed}/{env.num_envs},"
            f" reached_target={success}, success_rate={success_rate:.4f}"
        )

    if bool(torch.all(eval_state.done_mask)):
        print(f"[EvalMode] collected first episode for all {env.num_envs} environments.")
        return True
    if timestep >= args_cli.eval_max_steps:
        print(
            f"[EvalMode] reached --eval_max_steps={args_cli.eval_max_steps}."
            f" completed={int(eval_state.done_mask.sum().item())}/{env.num_envs}"
        )
        return True
    return False


def write_eval_outputs(eval_state: EvalState, env) -> dict[str, Any]:
    completed = int(eval_state.done_mask.sum().item())
    success = int(eval_state.reached_target_mask.sum().item())
    success_rate = success / completed if completed > 0 else 0.0

    completed_mask = eval_state.done_mask.detach().cpu().numpy().tolist()
    reached_target_mask = eval_state.reached_target_mask.detach().cpu().numpy().tolist()
    time_out_mask = eval_state.time_out_mask.detach().cpu().numpy().tolist()
    failed_other_mask = eval_state.failed_other_mask.detach().cpu().numpy().tolist()
    episode_len = eval_state.episode_len.detach().cpu().numpy().tolist()
    episode_return = eval_state.episode_return.detach().cpu().numpy().tolist()
    volume_points_triggered_any = eval_state.volume_points_triggered_any.detach().cpu().numpy().tolist()
    volume_points_trigger_steps = eval_state.volume_points_trigger_steps.detach().cpu().numpy().tolist()
    volume_points_max_penetration = eval_state.volume_points_max_penetration.detach().cpu().numpy().tolist()
    completed_lens = [episode_len[idx] for idx in range(env.num_envs) if completed_mask[idx]]
    completed_rets = [episode_return[idx] for idx in range(env.num_envs) if completed_mask[idx]]
    completed_volume_trigger_steps = [
        volume_points_trigger_steps[idx] for idx in range(env.num_envs) if completed_mask[idx]
    ]
    completed_volume_penetration = [
        volume_points_max_penetration[idx] for idx in range(env.num_envs) if completed_mask[idx]
    ]
    completed_triggered_any = [bool(volume_points_triggered_any[idx]) for idx in range(env.num_envs) if completed_mask[idx]]

    mean_episode_len = float(sum(completed_lens) / len(completed_lens)) if completed_lens else 0.0
    mean_episode_return = float(sum(completed_rets) / len(completed_rets)) if completed_rets else 0.0
    mean_volume_trigger_steps = (
        float(sum(completed_volume_trigger_steps) / len(completed_volume_trigger_steps))
        if completed_volume_trigger_steps
        else 0.0
    )
    mean_volume_max_penetration = (
        float(sum(completed_volume_penetration) / len(completed_volume_penetration))
        if completed_volume_penetration
        else 0.0
    )
    max_volume_max_penetration = max(completed_volume_penetration) if completed_volume_penetration else 0.0
    volume_triggered_count = sum(int(v) for v in completed_triggered_any)

    summary = {
        "num_envs": int(env.num_envs),
        "num_completed": completed,
        "num_incomplete": int(env.num_envs - completed),
        "success_count": success,
        "success_rate": success_rate,
        "success_reason_counts": {
            "reached_target": int(eval_state.reached_target_mask.sum().item()),
            "time_out": int(eval_state.time_out_mask.sum().item()),
            "failed_other": int(eval_state.failed_other_mask.sum().item()),
        },
        "mean_episode_length": mean_episode_len,
        "mean_episode_return": mean_episode_return,
        "metrics_by_stage": {
            eval_state.stage_name_map.get(stage_id, f"Stage{stage_id}"): list(metric_names)
            for stage_id, metric_names in eval_state.stage_metric_names.items()
        },
        "volume_points": {
            "triggered_env_count": int(volume_triggered_count),
            "triggered_env_rate": float(volume_triggered_count / completed) if completed > 0 else 0.0,
            "mean_trigger_steps": mean_volume_trigger_steps,
            "mean_max_penetration_depth": mean_volume_max_penetration,
            "max_max_penetration_depth": float(max_volume_max_penetration),
        },
        "stage_metrics_global": {
            eval_state.stage_name_map.get(stage_id, f"Stage{stage_id}"): {
                metric_name: eval_state.global_stage_stats[stage_id][metric_name].as_dict()
                for metric_name in metric_names
            }
            for stage_id, metric_names in eval_state.stage_metric_names.items()
        },
    }

    summary_path = os.path.join(eval_state.output_dir, f"{eval_state.output_prefix}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    per_env_path = os.path.join(eval_state.output_dir, f"{eval_state.output_prefix}_per_env_first_episode.csv")
    with open(per_env_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "env_id",
                "completed",
                "reached_target",
                "timed_out",
                "failed_other",
                "episode_length_steps",
                "episode_return_sum",
                "volume_points_triggered_any",
                "volume_points_trigger_steps",
                "volume_points_max_penetration_depth",
            ],
        )
        writer.writeheader()
        for env_id in range(env.num_envs):
            writer.writerow(
                {
                    "env_id": env_id,
                    "completed": int(completed_mask[env_id]),
                    "reached_target": int(reached_target_mask[env_id]),
                    "timed_out": int(time_out_mask[env_id]),
                    "failed_other": int(failed_other_mask[env_id]),
                    "episode_length_steps": int(episode_len[env_id]),
                    "episode_return_sum": float(episode_return[env_id]),
                    "volume_points_triggered_any": int(volume_points_triggered_any[env_id]),
                    "volume_points_trigger_steps": int(volume_points_trigger_steps[env_id]),
                    "volume_points_max_penetration_depth": float(volume_points_max_penetration[env_id]),
                }
            )

    global_stage_path = os.path.join(eval_state.output_dir, f"{eval_state.output_prefix}_stage_global_stats.csv")
    with open(global_stage_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["stage_id", "stage_name", "metric", "count", "min", "max", "mean"],
        )
        writer.writeheader()
        for stage_id, metric_names in eval_state.stage_metric_names.items():
            stage_name = eval_state.stage_name_map.get(stage_id, f"Stage{stage_id}")
            for metric_name in metric_names:
                row = eval_state.global_stage_stats[stage_id][metric_name].as_dict()
                writer.writerow(
                    {
                        "stage_id": stage_id,
                        "stage_name": stage_name,
                        "metric": metric_name,
                        "count": row["count"],
                        "min": row["min"],
                        "max": row["max"],
                        "mean": row["mean"],
                    }
                )

    per_env_stage_path = os.path.join(eval_state.output_dir, f"{eval_state.output_prefix}_stage_per_env_stats.csv")
    with open(per_env_stage_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["env_id", "stage_id", "stage_name", "metric", "count", "min", "max", "mean"],
        )
        writer.writeheader()
        for env_id in range(env.num_envs):
            for stage_id, metric_names in eval_state.stage_metric_names.items():
                stage_name = eval_state.stage_name_map.get(stage_id, f"Stage{stage_id}")
                for metric_name in metric_names:
                    row = eval_state.per_env_stage_stats[env_id][stage_id][metric_name].as_dict()
                    writer.writerow(
                        {
                            "env_id": env_id,
                            "stage_id": stage_id,
                            "stage_name": stage_name,
                            "metric": metric_name,
                            "count": row["count"],
                            "min": row["min"],
                            "max": row["max"],
                            "mean": row["mean"],
                        }
                    )

    print(
        f"[EvalMode] summary saved to {summary_path}\n"
        f"[EvalMode] per-env stats saved to {per_env_path}\n"
        f"[EvalMode] global stage stats saved to {global_stage_path}\n"
        f"[EvalMode] per-env stage stats saved to {per_env_stage_path}\n"
        f"[EvalMode] completed={completed}/{env.num_envs}, reached_target={success}, success_rate={success_rate:.4f},"
        f" mean_episode_len={mean_episode_len:.2f}, mean_episode_return={mean_episode_return:.4f}"
    )

    return {
        "summary_path": summary_path,
        "per_env_path": per_env_path,
        "global_stage_path": global_stage_path,
        "per_env_stage_path": per_env_stage_path,
        "completed": completed,
        "success": success,
        "success_rate": success_rate,
        "mean_episode_len": mean_episode_len,
        "mean_episode_return": mean_episode_return,
    }
