"""Script to play a checkpoint if an RL agent from Instinct-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import copy
import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass

sys.path.append(os.path.join(os.getcwd(), "scripts", "instinct_rl"))

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with Instinct-RL.")
parser.add_argument("--video", action="store_true", default=True, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=1000, help="Length of the recorded video (in steps).")
parser.add_argument("--video_start_step", type=int, default=0, help="Start step for the simulation.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--exportonnx", action="store_true", default=False, help="Export policy as ONNX model.")
parser.add_argument("--useonnx", action="store_true", default=False, help="Use the exported ONNX model for inference.")
parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode.")
parser.add_argument("--no_resume", default=None, action="store_true", help="Force play in no resume mode.")
# custom play arguments
parser.add_argument("--env_cfg", action="store_true", default=False, help="Load configuration from file.")
parser.add_argument("--agent_cfg", action="store_true", default=False, help="Load configuration from file.")
parser.add_argument("--sample", action="store_true", default=False, help="Sample actions instead of using the policy.")
parser.add_argument("--zero_act_until", type=int, default=0, help="Zero actions until this timestep.")
parser.add_argument("--keyboard_control", action="store_true", default=False, help="Enable keyboard control.")
parser.add_argument("--keyboard_linvel_step", type=float, default=0.5, help="Linear velocity change per keyboard step.")
parser.add_argument("--keyboard_angvel", type=float, default=1.0, help="Angular velocity set by keyboard.")
parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to a .pt checkpoint file.")
parser.add_argument("--foot_debug_full", action="store_true", default=False, help="Print full foot tactile debug each step.")
parser.add_argument("--foot_metrics", action="store_true", default=False, help="Print per-step CoP/Delta-CoP/ContactArea and window averages.")
parser.add_argument("--foot_metrics_window", type=int, default=200, help="Window size (steps) for average CoP/Delta-CoP/ContactArea.")
parser.add_argument("--foot_metrics_print_every", type=int, default=1, help="Print foot metrics every N steps.")
parser.add_argument("--foot_metrics_env_id", type=int, default=0, help="Environment index used for foot metrics.")
parser.add_argument("--h_eff_debug", action="store_true", default=False, help="Print per-step left/right foot h_eff from contact_stage_filter.")
parser.add_argument("--h_eff_print_every", type=int, default=1, help="Print h_eff every N steps.")
parser.add_argument("--h_eff_env_id", type=int, default=0, help="Environment index used for h_eff debug output.")
parser.add_argument("--stage_debug", action="store_true", default=False, help="Print compact per-foot stage gating debug periodically.")
parser.add_argument("--stage_debug_print_every", type=int, default=20, help="Print stage debug every N steps.")
parser.add_argument("--stage_debug_env_id", type=int, default=0, help="Environment index used for stage debug output.")
parser.add_argument("--eval_mode", action="store_true", default=False, help="Evaluate first episode of each env and dump stage-wise metrics.")
parser.add_argument(
    "--eval_use_current_scene",
    action="store_true",
    default=False,
    help=(
        "When eval_mode is enabled, override checkpoint scene semantics with the current task config "
        "(terrain, commands, events, terminations, episode length) while preserving policy-compatible "
        "obs/network contracts."
    ),
)
parser.add_argument("--eval_max_steps", type=int, default=20000, help="Maximum simulation steps in eval mode before forced stop.")
parser.add_argument("--eval_progress_every", type=int, default=200, help="Print eval progress every N steps.")
parser.add_argument("--eval_output_dir", type=str, default=None, help="Output directory for eval logs. Default: <log_dir>/eval.")
parser.add_argument("--eval_output_prefix", type=str, default="", help="Prefix of eval output files. Default is auto-generated.")
parser.add_argument("--print_colliders", action="store_true", default=False, help="Print collider prims for scene inspection.")

# append Instinct-RL cli arguments
cli_args.add_instinct_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import carb.input
import omni.appwindow as omni_appwindow
from carb.input import KeyboardEventType
from instinct_rl.runners import OnPolicyRunner
from instinct_rl.utils.utils import get_obs_slice, get_subobs_by_components, get_subobs_size

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import load_pickle, load_yaml
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

import instinctlab.tasks.parkour.mdp as parkour_mdp
from instinctlab.sensors.contact_stage.contact_stage_cfg import ContactStageCfg
from instinctlab.sensors.foot_tactile import FootTactileCfg, FootTactileDiffusionCfg, FootTactileNoiseCfg
from instinctlab.tasks.parkour.config.g1.foot_tactile_geometry import make_ankle_roll_foot_tactile_template_cfg

# Import extensions to set up environment tasks
from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper
from instinctlab.utils.wrappers.instinct_rl import InstinctRlOnPolicyRunnerCfg


#print("[INFO] Available Instinct-Parkour tasks:", [k for k in gym.registry.keys() if 'Instinct-Parkour' in k])

# wait for attach if in debug mode
if args_cli.debug:
    # import typing; typing.TYPE_CHECKING = True
    import debugpy

    ip_address = ("0.0.0.0", 6789)
    print("Process: " + " ".join(sys.argv[:]))
    print("Is waiting for attach at address: %s:%d" % ip_address, flush=True)
    debugpy.listen(ip_address)
    debugpy.wait_for_client()
    debugpy.breakpoint()


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


def _iter_reward_groups(reward_manager) -> list[str | None]:
    active_terms = getattr(reward_manager, "active_terms", None)
    if isinstance(active_terms, dict):
        return list(active_terms.keys())
    return [None]


def _find_stage_reward_debug_term(env) -> object | None:
    reward_manager = getattr(env.unwrapped, "reward_manager", None)
    if reward_manager is None:
        return None
    debug_term_candidates = (
        "stage_reward_v1",
        "stage_landing_f_v1",
        "stage_landing_df_v1",
        "stage_landing_rho_v1",
        "stage_swing_clearance_v1",
        "stage_pre_v_v1",
        "stage_pre_a_v1",
        "stage_stance_cop_v1",
        "stage_stance_area_v1",
        "stage_stance_delta_cop_v1",
    )
    for group_name in _iter_reward_groups(reward_manager):
        for term_name in debug_term_candidates:
            try:
                if group_name is None:
                    stage_reward_term_cfg = reward_manager.get_term_cfg(term_name)
                else:
                    stage_reward_term_cfg = reward_manager.get_term_cfg(term_name, group_name=group_name)
            except Exception:
                continue
            candidate_term = stage_reward_term_cfg.func
            if hasattr(candidate_term, "get_debug_dict"):
                return candidate_term
    return None


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


def _maybe_load_saved_eval_configs(log_dir: str, env_cfg, agent_cfg_dict, agent_cfg):
    """Prefer the checkpoint's saved configs during eval to preserve old obs/network contracts."""
    if not args_cli.eval_mode:
        return env_cfg, agent_cfg_dict

    params_dir = os.path.join(log_dir, "params")
    env_pkl_path = os.path.join(params_dir, "env.pkl")
    agent_yaml_path = os.path.join(params_dir, "agent.yaml")

    if not args_cli.env_cfg and os.path.isfile(env_pkl_path):
        env_cfg = load_pickle(env_pkl_path)
        print(f"[EvalCompat] Auto-loaded saved env config: {env_pkl_path}")

    if not args_cli.agent_cfg and os.path.isfile(agent_yaml_path):
        agent_cfg_dict = load_yaml(agent_yaml_path)
        print(f"[EvalCompat] Auto-loaded saved agent config: {agent_yaml_path}")
    elif agent_cfg_dict is None:
        agent_cfg_dict = agent_cfg.to_dict()

    return env_cfg, agent_cfg_dict


def _apply_runtime_env_overrides(env_cfg) -> None:
    """Re-apply runtime CLI overrides after loading a saved env config."""
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


def _apply_current_eval_scene_overrides(env_cfg, current_scene_env_cfg) -> None:
    """Apply current task scene semantics onto a checkpoint-compatible env config."""
    if not (args_cli.eval_mode and args_cli.eval_use_current_scene):
        return
    if current_scene_env_cfg is None:
        return

    applied_items: list[str] = []
    if hasattr(current_scene_env_cfg, "episode_length_s"):
        env_cfg.episode_length_s = current_scene_env_cfg.episode_length_s
        applied_items.append("episode_length_s")

    current_scene_cfg = getattr(current_scene_env_cfg, "scene", None)
    target_scene_cfg = getattr(env_cfg, "scene", None)
    if current_scene_cfg is not None and target_scene_cfg is not None:
        if hasattr(current_scene_cfg, "terrain"):
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


def _ensure_eval_stage_tactile_compat(env_cfg) -> None:
    """Inject eval-only tactile/stage sensors into old checkpoints that predate these signals."""
    if not args_cli.eval_mode:
        return

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

    if added_items:
        print("[EvalCompat] Injected eval-only sensor config: " + ", ".join(added_items))


def _print_foot_contact_obs_sample(obs: torch.Tensor, env, env_id: int = 0) -> None:
    obs_segments = env.get_obs_segments()
    if "foot_contact_state" not in obs_segments:
        print("[ObsDebug] 'foot_contact_state' is not in policy observation segments.")
        return

    env_id = max(0, min(env_id, env.num_envs - 1))
    term_slice, term_shape = get_obs_slice(obs_segments, "foot_contact_state")
    term_flat = obs[env_id, term_slice].detach().cpu()
    feature_dim = 5 if term_flat.numel() % 5 == 0 else (4 if term_flat.numel() % 4 == 0 else 0)
    if feature_dim == 0:
        print(
            f"[ObsDebug] unexpected 'foot_contact_state' flattened size={term_flat.numel()}, "
            f"shape_meta={term_shape}"
        )
        return

    foot_tensor = term_flat.view(-1, feature_dim)
    foot_names = []
    if "contact_stage_filter" in env.unwrapped.scene.sensors:
        foot_names = list(env.unwrapped.scene.sensors["contact_stage_filter"].body_names)
    elif "foot_tactile" in env.unwrapped.scene.sensors:
        foot_names = list(env.unwrapped.scene.sensors["foot_tactile"].body_names)

    print(
        f"[ObsDebug] foot_contact_state sample: env={env_id}, segment_shape={term_shape}, "
        f"flat_dim={term_flat.numel()}"
    )
    for foot_id in range(foot_tensor.shape[0]):
        foot_name = foot_names[foot_id] if foot_id < len(foot_names) else f"foot_{foot_id}"
        if feature_dim == 5:
            area, f_over_bw, cop_x, cop_y, vz = foot_tensor[foot_id].tolist()
            print(
                f"[ObsDebug env={env_id} {foot_name}] "
                f"area={area:.4f}, F/BW={f_over_bw:.4f}, COPnorm=({cop_x:.4f},{cop_y:.4f}), vz_norm={vz:.4f}"
            )
        else:
            area, f_over_bw, cop_x, cop_y = foot_tensor[foot_id].tolist()
            print(
                f"[ObsDebug env={env_id} {foot_name}] "
                f"area={area:.4f}, F/BW={f_over_bw:.4f}, COPnorm=({cop_x:.4f},{cop_y:.4f})"
            )


def _print_stage_debug_sample(env, timestep: int, env_id: int, stage_reward_debug_term: object | None) -> None:
    if "contact_stage_filter" not in env.unwrapped.scene.sensors:
        print("[StageDebug] 'contact_stage_filter' sensor is not available.")
        return

    stage_sensor = env.unwrapped.scene.sensors["contact_stage_filter"]
    env_id = max(0, min(env_id, env.num_envs - 1))
    foot_names = list(stage_sensor.body_names)

    print(f"[StageDebug step={timestep} env={env_id}]")
    for foot_id in range(stage_sensor.num_bodies):
        print(stage_sensor.get_stage_debug_string(env_id, foot_id))

    if stage_reward_debug_term is None or not hasattr(stage_reward_debug_term, "get_debug_dict"):
        return

    stage_rew_dbg = stage_reward_debug_term.get_debug_dict(env_id)
    for foot_id in range(stage_sensor.num_bodies):
        foot_name = foot_names[foot_id] if foot_id < len(foot_names) else f"foot_{foot_id}"
        alpha_sw = float(stage_rew_dbg["alpha_sw"][foot_id].item())
        alpha_pre = float(stage_rew_dbg["alpha_pre"][foot_id].item())
        alpha_land = float(stage_rew_dbg["alpha_land"][foot_id].item())
        alpha_st = float(stage_rew_dbg["alpha_st"][foot_id].item())
        r_pre_v = float(stage_rew_dbg["r_pre_v"][foot_id].item())
        r_pre_a = float(stage_rew_dbg["r_pre_a"][foot_id].item())
        r_contact_base = float(stage_rew_dbg["r_contact_base"][foot_id].item())
        r_contact = float(stage_rew_dbg["r_contact"][foot_id].item())
        contact_phase_weight = float(stage_rew_dbg["contact_phase_weight"][foot_id].item())
        area_ratio = float(stage_rew_dbg["contact_area_ratio"][foot_id].item())
        cop_margin = float(stage_rew_dbg["cop_margin"][foot_id].item())
        landing_window = int(stage_rew_dbg["landing_window"][foot_id].item())
        print(
            f"[StageReward step={timestep} env={env_id} foot={foot_id}({foot_name})] "
            f"alpha=({alpha_sw:.3f},{alpha_pre:.3f},{alpha_land:.3f},{alpha_st:.3f}) "
            f"r_pre=({r_pre_v:.3f},{r_pre_a:.3f}) "
            f"r_contact_base={r_contact_base:.3f} phase_w={contact_phase_weight:.3f} "
            f"r_contact={r_contact:.3f} area={area_ratio:.3f} cop_margin={cop_margin:.3f} "
            f"landing_window={landing_window}"
        )


def main():
    """Play with Instinct-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    current_eval_scene_env_cfg = copy.deepcopy(env_cfg)
    agent_cfg: InstinctRlOnPolicyRunnerCfg = cli_args.parse_instinct_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "instinct_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.checkpoint_path is not None:
        resume_path = os.path.abspath(args_cli.checkpoint_path)
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"--checkpoint_path not found: {resume_path}")
        log_dir = os.path.dirname(resume_path)
        agent_cfg.load_run = "__direct__"   # 只要保证后面 ppo_runner.load 会执行即可
        print(f"[INFO] Loading checkpoint directly: {resume_path}")
    else:
        agent_cfg.load_run = args_cli.load_run
        if agent_cfg.load_run is not None:
            print(f"[INFO] Loading experiment from directory: {log_root_path}")
            if os.path.isabs(agent_cfg.load_run):
                resume_path = get_checkpoint_path(
                    os.path.dirname(agent_cfg.load_run), os.path.basename(agent_cfg.load_run), agent_cfg.load_checkpoint
                )
            else:
                resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
            log_dir = os.path.dirname(resume_path)
        elif not args_cli.no_resume:
            raise RuntimeError(
                f"\033[91m[ERROR] No checkpoint specified and play.py resumes from a checkpoint by default. Please specify"
                f" a checkpoint to resume from using --load_run or use --no_resume to disable this behavior.\033[0m"
            )
        else:
            print(f"[INFO] No experiment directory specified. Using default: {log_root_path}")
            log_dir = os.path.join(log_root_path, agent_cfg.run_name + "_play")
            resume_path = "model_scratch.pt"
    if args_cli.env_cfg:
        env_cfg = load_pickle(os.path.join(log_dir, "params", "env.pkl"))
    if args_cli.agent_cfg:
        agent_cfg_dict = load_yaml(os.path.join(log_dir, "params", "agent.yaml"))
    else:
        agent_cfg_dict = agent_cfg.to_dict()

    env_cfg, agent_cfg_dict = _maybe_load_saved_eval_configs(log_dir, env_cfg, agent_cfg_dict, agent_cfg)
    _apply_current_eval_scene_overrides(env_cfg, current_eval_scene_env_cfg)
    _apply_runtime_env_overrides(env_cfg)
    _ensure_eval_stage_tactile_compat(env_cfg)

    if args_cli.keyboard_control:
        env_cfg.scene.num_envs = 1
        env_cfg.episode_length_s = 1e10

    if args_cli.eval_mode:
        env_cfg.terminations.target_reached = DoneTerm(
            func=parkour_mdp.reached_target_termination,
            params={"command_name": "base_velocity"},
        )

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.print_colliders:
        from pxr import UsdPhysics, PhysxSchema
        import omni.usd as omni_usd

        stage = omni_usd.get_context().get_stage()

        def is_collider(prim):
            return prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(PhysxSchema.PhysxCollisionAPI)

        print("=== COLLIDERS containing 'ground' ===")
        for prim in stage.Traverse():
            p = prim.GetPath().pathString
            if "ground" in p.lower() and is_collider(prim):
                print("COLLIDER:", p)
        print("=== COLLIDERS containing 'terrain' ===")
        for prim in stage.Traverse():
            p = prim.GetPath().pathString
            if "terrain" in p.lower() and is_collider(prim):
                print("COLLIDER:", p)
        print("=== COLLIDERS under L_foot_tactile ===")
        for prim in stage.Traverse():
            p = prim.GetPath().pathString
            if "/Robot/L_foot_tactile" in p and is_collider(prim):
                print("COLLIDER:", p)
        print("=== COLLIDERS under R_foot_tactile ===")
        for prim in stage.Traverse():
            p = prim.GetPath().pathString
            if "/Robot/R_foot_tactile" in p and is_collider(prim):
                print("COLLIDER:", p)
        print("=== ALL COLLIDERS ===")
        for prim in stage.Traverse():
            p = prim.GetPath().pathString
            if is_collider(prim):
                print("PRIM:", p, "is COLLIDER")
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == args_cli.video_start_step,
            "video_length": args_cli.video_length,
            "disable_logger": True,
            "name_prefix": f"model_{resume_path.split('_')[-1].split('.')[0]}",
        }
        print("[INFO] Recording videos during playing.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for instinct-rl
    env = InstinctRlVecEnvWrapper(env)
    env.unwrapped.configure_eval_pre_reset_snapshot(args_cli.eval_mode)
    import inspect

    # t = env.unwrapped.scene.sensors["foot_tactile"]
    # print("[FootTactile] loaded from:", inspect.getfile(type(t)))
    # print("[FootTactile] debug_print =", getattr(t.cfg, "debug_print", None))
    # print("[FootTactile] update_period =", getattr(t.cfg, "update_period", None))

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
    if agent_cfg.load_run is not None:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    if args_cli.sample:
        policy = ppo_runner.alg.actor_critic.act
    else:
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    if agent_cfg.load_run is not None:
        export_model_dir = os.path.join(log_dir, "exported")
        if args_cli.exportonnx:
            assert env.unwrapped.num_envs == 1, "Exporting to ONNX is only supported for single environment."
            if not os.path.exists(export_model_dir):
                os.makedirs(export_model_dir)
            obs, _ = env.get_observations()
            ppo_runner.alg.actor_critic.export_as_onnx(obs, export_model_dir)

    # use the exported model for inference
    if args_cli.useonnx:
        from onnxer import load_parkour_onnx_model

        # NOTE: This is only applicable with parkour task
        onnx_policy = load_parkour_onnx_model(
            model_dir=os.path.join(log_dir, "exported"),
            get_subobs_func=lambda obs: get_subobs_by_components(
                obs,
                agent_cfg.policy.encoder_configs.depth_encoder.component_names,
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

    override_command = torch.zeros(env.num_envs, 3, device=env.device)
    command_obs_slice = get_obs_slice(env.get_obs_segments(), "velocity_commands")

    def on_keyboard_input(e):
        if e.input == carb.input.KeyboardInput.W:
            if e.type == KeyboardEventType.KEY_PRESS or e.type == KeyboardEventType.KEY_REPEAT:
                override_command[:, 0] += args_cli.keyboard_linvel_step
        if e.input == carb.input.KeyboardInput.S:
            if e.type == KeyboardEventType.KEY_PRESS or e.type == KeyboardEventType.KEY_REPEAT:
                override_command[:, 2] = 0.0
        if e.input == carb.input.KeyboardInput.F:
            if e.type == KeyboardEventType.KEY_PRESS or e.type == KeyboardEventType.KEY_REPEAT:
                override_command[:, 2] = args_cli.keyboard_angvel
        if e.input == carb.input.KeyboardInput.G:
            if e.type == KeyboardEventType.KEY_PRESS or e.type == KeyboardEventType.KEY_REPEAT:
                override_command[:, 2] = -args_cli.keyboard_angvel
        if e.input == carb.input.KeyboardInput.X:
            if e.type == KeyboardEventType.KEY_PRESS or e.type == KeyboardEventType.KEY_REPEAT:
                override_command[:] = 0.0

    app_window = omni_appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard()
    input = carb.input.acquire_input_interface()
    input.subscribe_to_keyboard_events(keyboard, on_keyboard_input)

    # reset environment
    obs, _ = env.get_observations()
    _print_foot_contact_obs_sample(obs, env, env_id=args_cli.foot_metrics_env_id)
    timestep = 0
    metric_window_count = 0
    metric_prev_cop = None
    metric_sum_cop = torch.zeros((2, 2), dtype=torch.float64, device="cpu")
    metric_sum_delta_cop = torch.zeros((2, 2), dtype=torch.float64, device="cpu")
    metric_sum_delta_cop_norm = torch.zeros((2,), dtype=torch.float64, device="cpu")
    metric_sum_contact_area = torch.zeros((2,), dtype=torch.float64, device="cpu")
    stage_reward_debug_term = None
    stage_reward_debug_lookup_done = False
    eval_stage_sensor = None
    eval_tactile_sensor = None
    eval_stage_name_map: dict[int, str] = {}
    eval_stage_ids: dict[str, int] = {}
    eval_num_feet = 0
    eval_num_stages = 0
    eval_stage_metric_names: dict[int, tuple[str, ...]] = {}
    eval_tactile_body_ids: list[int] = []
    eval_body_has_polygon = torch.zeros(0, dtype=torch.bool, device=env.device)
    eval_body_polygons: list[torch.Tensor | None] = []
    eval_done_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    eval_reached_target_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    eval_time_out_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    eval_failed_other_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    eval_episode_len = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    eval_episode_return = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    eval_volume_points_triggered_any = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    eval_volume_points_trigger_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    eval_volume_points_max_penetration = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    eval_prev_vz = torch.zeros((env.num_envs, 0), dtype=torch.float32, device=env.device)
    eval_az_filt = torch.zeros((env.num_envs, 0), dtype=torch.float32, device=env.device)
    eval_az_initialized = torch.zeros((env.num_envs, 0), dtype=torch.bool, device=env.device)
    eval_prev_stance_cop = torch.zeros((env.num_envs, 0, 2), dtype=torch.float32, device=env.device)
    eval_stance_cop_initialized = torch.zeros((env.num_envs, 0), dtype=torch.bool, device=env.device)
    eval_landing_active_prev = torch.zeros((env.num_envs, 0), dtype=torch.bool, device=env.device)
    eval_landing_force_peak = torch.zeros((env.num_envs, 0), dtype=torch.float32, device=env.device)
    eval_pre_az_filter_alpha = 0.7
    eval_global_stage_stats = {}
    eval_per_env_stage_stats: list[dict[int, dict[str, RunningStats]]] = []
    eval_output_dir = None
    eval_output_prefix = None

    if args_cli.foot_metrics:
        if args_cli.foot_metrics_window <= 0:
            raise ValueError("--foot_metrics_window must be > 0.")
        if args_cli.foot_metrics_print_every <= 0:
            raise ValueError("--foot_metrics_print_every must be > 0.")
        if args_cli.foot_metrics_env_id < 0 or args_cli.foot_metrics_env_id >= env.num_envs:
            raise ValueError(
                f"--foot_metrics_env_id must be in [0, {env.num_envs - 1}], got {args_cli.foot_metrics_env_id}."
            )
        if "foot_tactile" not in env.unwrapped.scene.sensors:
            raise RuntimeError("--foot_metrics requires 'foot_tactile' sensor in the scene.")
    if args_cli.h_eff_debug:
        if args_cli.h_eff_print_every <= 0:
            raise ValueError("--h_eff_print_every must be > 0.")
        if args_cli.h_eff_env_id < 0 or args_cli.h_eff_env_id >= env.num_envs:
            raise ValueError(
                f"--h_eff_env_id must be in [0, {env.num_envs - 1}], got {args_cli.h_eff_env_id}."
            )
        if "contact_stage_filter" not in env.unwrapped.scene.sensors:
            raise RuntimeError("--h_eff_debug requires 'contact_stage_filter' sensor in the scene.")
    if args_cli.stage_debug:
        if args_cli.stage_debug_print_every <= 0:
            raise ValueError("--stage_debug_print_every must be > 0.")
        if args_cli.stage_debug_env_id < 0 or args_cli.stage_debug_env_id >= env.num_envs:
            raise ValueError(
                f"--stage_debug_env_id must be in [0, {env.num_envs - 1}], got {args_cli.stage_debug_env_id}."
            )
        if "contact_stage_filter" not in env.unwrapped.scene.sensors:
            raise RuntimeError("--stage_debug requires 'contact_stage_filter' sensor in the scene.")
    if args_cli.eval_mode:
        if args_cli.eval_max_steps <= 0:
            raise ValueError("--eval_max_steps must be > 0.")
        if args_cli.eval_progress_every <= 0:
            raise ValueError("--eval_progress_every must be > 0.")
        if "contact_stage_filter" not in env.unwrapped.scene.sensors:
            raise RuntimeError("--eval_mode requires 'contact_stage_filter' sensor in the scene.")

        eval_stage_sensor = env.unwrapped.scene.sensors["contact_stage_filter"]
        eval_tactile_sensor = env.unwrapped.scene.sensors.get("foot_tactile", None)
        if eval_tactile_sensor is None:
            raise RuntimeError("--eval_mode requires 'foot_tactile' sensor in the scene.")
        eval_num_feet = int(eval_stage_sensor.num_bodies)
        eval_num_stages = int(getattr(eval_stage_sensor, "NUM_STAGES", 4))
        if hasattr(eval_stage_sensor, "stage_name_map"):
            eval_stage_name_map = {int(k): str(v) for k, v in eval_stage_sensor.stage_name_map().items()}
        else:
            eval_stage_name_map = {stage_id: f"Stage{stage_id}" for stage_id in range(eval_num_stages)}
        eval_stage_ids = _resolve_eval_stage_ids(eval_stage_name_map, eval_num_stages)
        eval_stage_metric_names = {
            eval_stage_ids["prelanding"]: ("vz", "az"),
            eval_stage_ids["landing"]: ("impact_force_peak",),
            eval_stage_ids["stance"]: ("delta_cop", "cop_margin", "contact_area_ratio"),
        }
        eval_tactile_body_ids = _resolve_stage_to_tactile_ids(eval_stage_sensor, eval_tactile_sensor, eval_num_feet)
        eval_body_has_polygon, eval_body_polygons = _build_eval_outline_cache(
            eval_tactile_sensor,
            eval_tactile_body_ids,
            env.device,
        )
        eval_prev_vz = torch.zeros((env.num_envs, eval_num_feet), dtype=torch.float32, device=env.device)
        eval_az_filt = torch.zeros((env.num_envs, eval_num_feet), dtype=torch.float32, device=env.device)
        eval_az_initialized = torch.zeros((env.num_envs, eval_num_feet), dtype=torch.bool, device=env.device)
        eval_prev_stance_cop = torch.zeros((env.num_envs, eval_num_feet, 2), dtype=torch.float32, device=env.device)
        eval_stance_cop_initialized = torch.zeros((env.num_envs, eval_num_feet), dtype=torch.bool, device=env.device)
        eval_landing_active_prev = torch.zeros((env.num_envs, eval_num_feet), dtype=torch.bool, device=env.device)
        eval_landing_force_peak = torch.zeros((env.num_envs, eval_num_feet), dtype=torch.float32, device=env.device)
        stage_reward_cfg = _find_stage_reward_cfg(env)
        if stage_reward_cfg is not None:
            eval_pre_az_filter_alpha = float(stage_reward_cfg.params.get("pre_az_filter_alpha", eval_pre_az_filter_alpha))

        eval_global_stage_stats = _make_stage_metric_stats(eval_stage_metric_names)
        eval_per_env_stage_stats = [
            _make_stage_metric_stats(eval_stage_metric_names) for _ in range(env.num_envs)
        ]

        eval_output_dir = args_cli.eval_output_dir or os.path.join(log_dir, "eval")
        os.makedirs(eval_output_dir, exist_ok=True)
        if args_cli.eval_output_prefix:
            eval_output_prefix = args_cli.eval_output_prefix
        else:
            eval_output_prefix = f"{os.path.splitext(os.path.basename(resume_path))[0]}_first_episode_{env.num_envs}env"
        print(
            f"[EvalMode] enabled. num_envs={env.num_envs}, num_feet={eval_num_feet},"
            f" num_stages={eval_num_stages}, output_dir={eval_output_dir}"
        )

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            if args_cli.keyboard_control:
                obs[:, command_obs_slice[0]] = override_command.repeat(1, command_obs_slice[1][0] // 3)
            actions = policy(obs)
            if args_cli.useonnx:
                torch_actions = actions
                actions = onnx_policy(obs)
                if (actions - torch_actions).abs().max() > 1e-5:
                    print(
                        "[INFO]: ONNX model and PyTorch model have a difference of"
                        f" {(actions - torch_actions).abs().max()} in actions at joint"
                        f" {((actions - torch_actions).abs() > 1e-5).nonzero(as_tuple=True)[0]}"
                    )
            if timestep < args_cli.zero_act_until:
                actions[:] = 0.0



            # env stepping
            obs, rewards, dones, infos = env.step(actions)
            if args_cli.eval_mode:
                assert eval_stage_sensor is not None
                eval_snapshot = infos.get("eval_pre_reset", None)
                if not isinstance(eval_snapshot, dict):
                    raise RuntimeError("--eval_mode requires eval_pre_reset snapshot from InstinctRlEnv.step().")

                snapshot_done = eval_snapshot.get("done")
                if not torch.is_tensor(snapshot_done):
                    snapshot_done = dones > 0
                snapshot_done = snapshot_done.to(dtype=torch.bool)

                active_env_ids_t = torch.nonzero(~eval_done_mask, as_tuple=False).squeeze(-1)
                if active_env_ids_t.numel() > 0:
                    step_reward = _reduce_step_reward(rewards)
                    step_reward = torch.nan_to_num(step_reward, nan=0.0, posinf=0.0, neginf=0.0)
                    eval_episode_return[active_env_ids_t] += step_reward[active_env_ids_t]
                    eval_episode_len[active_env_ids_t] += 1

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
                    cop_data = tactile_cop_all[:, eval_tactile_body_ids, :]
                    contact_area_data = tactile_area_all[:, eval_tactile_body_ids]
                    cop_margin_data = _compute_eval_cop_margin(cop_data, eval_body_has_polygon, eval_body_polygons)

                    current_vz = vz_data[:, :eval_num_feet]
                    az_data = torch.zeros_like(current_vz)
                    active_prev_vz = eval_prev_vz[active_env_ids_t]
                    active_prev_az = eval_az_filt[active_env_ids_t]
                    active_initialized = eval_az_initialized[active_env_ids_t]
                    az_raw_active = (current_vz[active_env_ids_t] - active_prev_vz) / max(float(env.unwrapped.step_dt), 1e-6)
                    az_raw_active = torch.where(active_initialized, az_raw_active, torch.zeros_like(az_raw_active))
                    az_active = torch.where(
                        active_initialized,
                        float(eval_pre_az_filter_alpha) * az_raw_active + (1.0 - float(eval_pre_az_filter_alpha)) * active_prev_az,
                        torch.zeros_like(az_raw_active),
                    )
                    az_data[active_env_ids_t] = torch.nan_to_num(az_active, nan=0.0, posinf=0.0, neginf=0.0)
                    eval_prev_vz[active_env_ids_t] = current_vz[active_env_ids_t]
                    eval_az_filt[active_env_ids_t] = az_data[active_env_ids_t]
                    eval_az_initialized[active_env_ids_t] = True

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
                    eval_volume_points_triggered_any[active_env_ids_t] |= penetration_trigger[active_env_ids_t]
                    eval_volume_points_trigger_steps[active_env_ids_t] += penetration_trigger[active_env_ids_t].to(dtype=torch.long)
                    eval_volume_points_max_penetration[active_env_ids_t] = torch.maximum(
                        eval_volume_points_max_penetration[active_env_ids_t],
                        penetration_depth_max[active_env_ids_t],
                    )

                    active_env_ids = active_env_ids_t.tolist()
                    for env_id in active_env_ids:
                        for foot_id in range(eval_num_feet):
                            force_value = float(total_force_data[env_id, foot_id].item())
                            was_landing_active = bool(eval_landing_active_prev[env_id, foot_id].item())
                            is_landing_active = bool(landing_active_data[env_id, foot_id].item())
                            if is_landing_active:
                                eval_landing_force_peak[env_id, foot_id] = max(
                                    float(eval_landing_force_peak[env_id, foot_id].item()),
                                    force_value,
                                )
                            landing_finalize = (was_landing_active and not is_landing_active) or (
                                bool(snapshot_done[env_id].item()) and (was_landing_active or is_landing_active)
                            )
                            if landing_finalize:
                                peak_force_value = float(eval_landing_force_peak[env_id, foot_id].item())
                                if is_landing_active:
                                    peak_force_value = max(peak_force_value, force_value)
                                for stage_stats in (eval_per_env_stage_stats[env_id], eval_global_stage_stats):
                                    stage_stats[eval_stage_ids["landing"]]["impact_force_peak"].update(peak_force_value)
                                eval_landing_force_peak[env_id, foot_id] = 0.0
                            elif not is_landing_active:
                                eval_landing_force_peak[env_id, foot_id] = 0.0
                            eval_landing_active_prev[env_id, foot_id] = is_landing_active and not bool(
                                snapshot_done[env_id].item()
                            )

                            stage_id = int(stage_ids[env_id, foot_id].item())
                            if stage_id not in eval_stage_metric_names:
                                eval_stance_cop_initialized[env_id, foot_id] = False
                                continue

                            if stage_id == eval_stage_ids["prelanding"]:
                                values = {
                                    "vz": float(vz_data[env_id, foot_id].item()),
                                    "az": float(az_data[env_id, foot_id].item()),
                                }
                            elif stage_id == eval_stage_ids["stance"]:
                                values = {
                                    "cop_margin": float(cop_margin_data[env_id, foot_id].item()),
                                    "contact_area_ratio": float(contact_area_data[env_id, foot_id].item()),
                                }
                                if bool(eval_stance_cop_initialized[env_id, foot_id].item()):
                                    delta_cop_value = float(
                                        torch.norm(
                                            cop_data[env_id, foot_id] - eval_prev_stance_cop[env_id, foot_id],
                                            p=2,
                                        ).item()
                                    )
                                    values["delta_cop"] = delta_cop_value
                                eval_prev_stance_cop[env_id, foot_id] = cop_data[env_id, foot_id]
                                eval_stance_cop_initialized[env_id, foot_id] = True
                            else:
                                values = {}
                                eval_stance_cop_initialized[env_id, foot_id] = False

                            for metric_name, metric_value in values.items():
                                eval_per_env_stage_stats[env_id][stage_id][metric_name].update(metric_value)
                                eval_global_stage_stats[stage_id][metric_name].update(metric_value)

                            if stage_id != eval_stage_ids["stance"]:
                                eval_stance_cop_initialized[env_id, foot_id] = False

                new_done_t = torch.nonzero(snapshot_done & (~eval_done_mask), as_tuple=False).squeeze(-1)
                if new_done_t.numel() > 0:
                    done_ids = new_done_t.tolist()
                    target_reached = eval_snapshot.get("termination/target_reached")
                    if not torch.is_tensor(target_reached):
                        target_reached = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
                    target_reached = target_reached.to(dtype=torch.bool)
                    for env_id in done_ids:
                        eval_done_mask[env_id] = True
                        reached_target = bool(target_reached[env_id].item())
                        timed_out = bool(eval_snapshot["time_outs"][env_id].item())
                        eval_reached_target_mask[env_id] = reached_target
                        eval_time_out_mask[env_id] = timed_out and not reached_target
                        eval_failed_other_mask[env_id] = (not reached_target) and (not timed_out)

                if timestep % args_cli.eval_progress_every == 0:
                    completed = int(eval_done_mask.sum().item())
                    success = int(eval_reached_target_mask.sum().item())
                    success_rate = success / completed if completed > 0 else 0.0
                    print(
                        f"[EvalMode step={timestep}] completed={completed}/{env.num_envs},"
                        f" reached_target={success}, success_rate={success_rate:.4f}"
                    )

                if bool(torch.all(eval_done_mask)):
                    print(f"[EvalMode] collected first episode for all {env.num_envs} environments.")
                    break
                if timestep >= args_cli.eval_max_steps:
                    print(
                        f"[EvalMode] reached --eval_max_steps={args_cli.eval_max_steps}."
                        f" completed={int(eval_done_mask.sum().item())}/{env.num_envs}"
                    )
                    break
            if args_cli.foot_metrics:
                sensor = env.unwrapped.scene.sensors["foot_tactile"]
                env_id = args_cli.foot_metrics_env_id
                cop_xy = sensor.data.cop_b[env_id].detach().cpu().to(dtype=torch.float64)  # (2,2)
                contact_area = sensor.data.contact_area_ratio[env_id].detach().cpu().to(dtype=torch.float64)  # (2,)
                if metric_prev_cop is None:
                    delta_cop_xy = torch.zeros_like(cop_xy)
                else:
                    delta_cop_xy = cop_xy - metric_prev_cop
                delta_cop_norm = torch.linalg.norm(delta_cop_xy, dim=-1)  # (2,)

                metric_sum_cop += cop_xy
                metric_sum_delta_cop += delta_cop_xy
                metric_sum_delta_cop_norm += delta_cop_norm
                metric_sum_contact_area += contact_area
                metric_window_count += 1
                metric_prev_cop = cop_xy.clone()

                if timestep % args_cli.foot_metrics_print_every == 0:
                    cop_l, cop_r = cop_xy[0], cop_xy[1]
                    dcop_l, dcop_r = delta_cop_xy[0], delta_cop_xy[1]
                    print(
                        f"[FootMetrics step={timestep} env={env_id}] "
                        f"CoP_L=({cop_l[0]:+.4f},{cop_l[1]:+.4f}) "
                        f"CoP_R=({cop_r[0]:+.4f},{cop_r[1]:+.4f}) "
                        f"dCoP_L=({dcop_l[0]:+.4f},{dcop_l[1]:+.4f})|{delta_cop_norm[0]:.4f} "
                        f"dCoP_R=({dcop_r[0]:+.4f},{dcop_r[1]:+.4f})|{delta_cop_norm[1]:.4f} "
                        f"Area_L={contact_area[0]:.4f} Area_R={contact_area[1]:.4f}"
                    )

                if metric_window_count >= args_cli.foot_metrics_window:
                    avg_cop = metric_sum_cop / metric_window_count
                    avg_delta_cop = metric_sum_delta_cop / metric_window_count
                    avg_delta_cop_norm = metric_sum_delta_cop_norm / metric_window_count
                    avg_contact_area = metric_sum_contact_area / metric_window_count
                    print(f"[FootMetrics AVG over {metric_window_count} steps | env={env_id}]")
                    print(
                        f"  avg_CoP_L=({avg_cop[0,0]:+.4f},{avg_cop[0,1]:+.4f}) "
                        f"avg_CoP_R=({avg_cop[1,0]:+.4f},{avg_cop[1,1]:+.4f})"
                    )
                    print(
                        f"  avg_dCoP_L=({avg_delta_cop[0,0]:+.4f},{avg_delta_cop[0,1]:+.4f})|{avg_delta_cop_norm[0]:.4f} "
                        f"avg_dCoP_R=({avg_delta_cop[1,0]:+.4f},{avg_delta_cop[1,1]:+.4f})|{avg_delta_cop_norm[1]:.4f}"
                    )
                    print(
                        f"  avg_contact_area_L={avg_contact_area[0]:.4f} "
                        f"avg_contact_area_R={avg_contact_area[1]:.4f}"
                    )
                    metric_window_count = 0
                    metric_sum_cop.zero_()
                    metric_sum_delta_cop.zero_()
                    metric_sum_delta_cop_norm.zero_()
                    metric_sum_contact_area.zero_()

            if args_cli.h_eff_debug and timestep % args_cli.h_eff_print_every == 0:
                stage_sensor = env.unwrapped.scene.sensors["contact_stage_filter"]
                env_id = args_cli.h_eff_env_id
                h_eff = stage_sensor.data.h_eff[env_id].detach().cpu().to(dtype=torch.float64)
                foot_names = list(stage_sensor.body_names)
                if h_eff.numel() >= 2:
                    left_name = foot_names[0] if len(foot_names) > 0 else "left_foot"
                    right_name = foot_names[1] if len(foot_names) > 1 else "right_foot"
                    print(
                        f"[h_eff step={timestep} env={env_id}] "
                        f"{left_name}={h_eff[0]:.4f}m {right_name}={h_eff[1]:.4f}m"
                    )
                else:
                    values = " ".join(
                        f"{foot_names[i] if i < len(foot_names) else f'foot_{i}'}={float(h_eff[i]):.4f}m"
                        for i in range(h_eff.numel())
                    )
                    print(f"[h_eff step={timestep} env={env_id}] {values}")

            if args_cli.stage_debug and timestep % args_cli.stage_debug_print_every == 0:
                if not stage_reward_debug_lookup_done:
                    stage_reward_debug_lookup_done = True
                    stage_reward_debug_term = _find_stage_reward_debug_term(env)
                _print_stage_debug_sample(env, timestep, args_cli.stage_debug_env_id, stage_reward_debug_term)

            if args_cli.foot_debug_full:
                print("--------------------------------Foot Debug--------------------------")
                # print(env.unwrapped.scene.sensors.keys())
                # print(type(env.unwrapped.scene.sensors["foot_tactile"]).__name__)
                sensor = env.unwrapped.scene.sensors["foot_tactile"]
                force_clean = sensor.data.taxel_force_clean
                force_diffused = sensor.data.taxel_force_diffused
                force_measured = sensor.data.taxel_force
                # print("Foot tactile sensor data:")
                # print("taxel_xy_b shape", sensor.data.taxel_xy_b.shape)
                # print("valid_taxel_count", sensor.data.valid_taxel_mask.sum(dim=-1))
                # print("support_valid_count", sensor.data.support_valid_mask.sum(dim=-1))
                # print("total_normal_force", sensor.data.total_normal_force)
                # print("contact_area_ratio", sensor.data.contact_area_ratio)
                # print("edge_force_ratio", sensor.data.edge_force_ratio)
                # print("taxel_weight_clean_sum", sensor.data.taxel_weight_clean.sum(dim=-1))
                # print("taxel_weight_aligned_sum", sensor.data.taxel_weight_aligned.sum(dim=-1))
                # print("taxel_force_clean_sum", force_clean.sum(dim=-1))
                # print("taxel_force_diffused_sum", force_diffused.sum(dim=-1))
                # print("taxel_force_measured_sum", force_measured.sum(dim=-1))
                # print("clean_force_conservation_error", force_clean.sum(dim=-1) - sensor.data.total_normal_force)
                # print("diffused_force_conservation_error", force_diffused.sum(dim=-1) - sensor.data.total_normal_force)
                # print("measured_force_error_vs_total", force_measured.sum(dim=-1) - sensor.data.total_normal_force)
                # print("taxel_force_measured", force_measured)
                debug_env_id = max(0, min(args_cli.foot_metrics_env_id, env.num_envs - 1))
                support_dist_env = sensor.data.support_dist[debug_env_id]
                support_valid_env = sensor.data.support_valid_mask[debug_env_id]
                body_names = sensor.body_names
                # for body_id, body_name in enumerate(body_names):
                #     print(
                #         f"support_dist[{debug_env_id}][{body_name}]",
                #         support_dist_env[body_id].detach().cpu().numpy(),
                #     )
                #     print(
                #         f"support_valid_mask[{debug_env_id}][{body_name}]",
                #         support_valid_env[body_id].detach().cpu().numpy(),
                #     )
                if "contact_stage_filter" in env.unwrapped.scene.sensors:
                    stage_sensor = env.unwrapped.scene.sensors["contact_stage_filter"]
                    print("contact_stage dominant_stage_id", stage_sensor.data.dominant_stage_id)
                    for foot_id in range(stage_sensor.num_bodies):
                        print(stage_sensor.get_stage_debug_string(debug_env_id, foot_id))
                    if not stage_reward_debug_lookup_done:
                        stage_reward_debug_lookup_done = True
                        stage_reward_debug_term = _find_stage_reward_debug_term(env)
                    if stage_reward_debug_term is not None:
                        stage_rew_dbg = stage_reward_debug_term.get_debug_dict(debug_env_id)
                        foot_names = stage_sensor.body_names
                        for foot_id in range(stage_sensor.num_bodies):
                            foot_name = foot_names[foot_id] if foot_id < len(foot_names) else f"foot_{foot_id}"
                            alpha_sw = float(stage_rew_dbg["alpha_sw"][foot_id].item())
                            alpha_pre = float(stage_rew_dbg["alpha_pre"][foot_id].item())
                            alpha_land = float(stage_rew_dbg["alpha_land"][foot_id].item())
                            alpha_st = float(stage_rew_dbg["alpha_st"][foot_id].item())
                            r_sw_h = float(stage_rew_dbg["r_sw_h"][foot_id].item())
                            r_pre_v = float(stage_rew_dbg["r_pre_v"][foot_id].item())
                            r_pre_a = float(stage_rew_dbg["r_pre_a"][foot_id].item())
                            r_cop = float(stage_rew_dbg["r_cop"][foot_id].item())
                            r_area = float(stage_rew_dbg["r_area"][foot_id].item())
                            r_st_delta_cop = float(stage_rew_dbg["r_st_delta_cop"][foot_id].item())
                            delta_cop = float(stage_rew_dbg["delta_cop"][foot_id].item())
                            r_contact = float(stage_rew_dbg["r_contact"][foot_id].item())
                            contact_phase_weight = float(stage_rew_dbg["contact_phase_weight"][foot_id].item())
                            landing_event_penalty = float(stage_rew_dbg["landing_event_penalty"][foot_id].item())
                            pre_vz = float(stage_rew_dbg["vz"][foot_id].item())
                            landing_f_peak = float(stage_rew_dbg["landing_F_peak"][foot_id].item())
                            landing_df_peak = float(stage_rew_dbg["landing_dF_peak"][foot_id].item())
                            landing_rho_peak = float(stage_rew_dbg["landing_rho_peak_max"][foot_id].item())
                            cop_margin = float(stage_rew_dbg["cop_margin"][foot_id].item())
                            area_ratio = float(stage_rew_dbg["contact_area_ratio"][foot_id].item())
                            landing_window = int(stage_rew_dbg["landing_window"][foot_id].item())
                            enable_land = float(stage_rew_dbg["enable_land"][foot_id].item())
                            enable_st = float(stage_rew_dbg["enable_st"][foot_id].item())
                            gamma_land = float(stage_rew_dbg["gamma_land"][foot_id].item())
                            gamma_st = float(stage_rew_dbg["gamma_st"][foot_id].item())
                            print(
                                f"[StageRewardV1 env={debug_env_id} foot={foot_id}({foot_name})] "
                                f"alpha_sw={alpha_sw:.4f} alpha_pre={alpha_pre:.4f} alpha_land={alpha_land:.4f} "
                                f"alpha_st={alpha_st:.4f} "
                                f"r_sw_h={r_sw_h:.4f} r_pre_v={r_pre_v:.4f} r_pre_a={r_pre_a:.4f} "
                                f"r_cop={r_cop:.4f} r_area={r_area:.4f} "
                                f"r_st_delta_cop={r_st_delta_cop:.4f} delta_cop={delta_cop:.4f} "
                                f"r_contact={r_contact:.4f} "
                                f"phase_w={contact_phase_weight:.4f} "
                                f"enable=(L:{enable_land:.0f},S:{enable_st:.0f}) "
                                f"gamma=(L:{gamma_land:.2f},S:{gamma_st:.2f}) "
                                f"landing_event_penalty={landing_event_penalty:.4f} "
                                f"vz={pre_vz:.4f} landing_window={landing_window} "
                                f"F_peak={landing_f_peak:.4f} dF_peak={landing_df_peak:.4f} "
                                f"rho_peak={landing_rho_peak:.4f} cop_margin={cop_margin:.4f} area={area_ratio:.4f}"
                            )
                print("-----------------------------------------------------------------------")
                if "contact_forces_foot" in env.unwrapped.scene.sensors:
                    contact_try = env.unwrapped.scene.sensors["contact_forces_foot"]
                    cs = contact_try.data
                    nf = cs.net_forces_w
                    print("contact net_forces_w shape:", tuple(nf.shape))
                    print("contact net_forces_w[0] =", nf[0].detach().cpu().numpy())
                    print("-------------------------------------------------------------------------------")

        timestep += 1

        # exit the loop if video_length is meet
        if args_cli.video and not args_cli.eval_mode:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    if args_cli.foot_metrics and metric_window_count > 0:
        env_id = args_cli.foot_metrics_env_id
        avg_cop = metric_sum_cop / metric_window_count
        avg_delta_cop = metric_sum_delta_cop / metric_window_count
        avg_delta_cop_norm = metric_sum_delta_cop_norm / metric_window_count
        avg_contact_area = metric_sum_contact_area / metric_window_count
        print(f"[FootMetrics AVG over last {metric_window_count} steps | env={env_id}]")
        print(
            f"  avg_CoP_L=({avg_cop[0,0]:+.4f},{avg_cop[0,1]:+.4f}) "
            f"avg_CoP_R=({avg_cop[1,0]:+.4f},{avg_cop[1,1]:+.4f})"
        )
        print(
            f"  avg_dCoP_L=({avg_delta_cop[0,0]:+.4f},{avg_delta_cop[0,1]:+.4f})|{avg_delta_cop_norm[0]:.4f} "
            f"avg_dCoP_R=({avg_delta_cop[1,0]:+.4f},{avg_delta_cop[1,1]:+.4f})|{avg_delta_cop_norm[1]:.4f}"
        )
        print(
            f"  avg_contact_area_L={avg_contact_area[0]:.4f} "
            f"avg_contact_area_R={avg_contact_area[1]:.4f}"
        )
    if args_cli.eval_mode:
        assert eval_output_dir is not None
        assert eval_output_prefix is not None
        completed = int(eval_done_mask.sum().item())
        success = int(eval_reached_target_mask.sum().item())
        success_rate = success / completed if completed > 0 else 0.0

        completed_mask = eval_done_mask.detach().cpu().numpy().tolist()
        reached_target_mask = eval_reached_target_mask.detach().cpu().numpy().tolist()
        time_out_mask = eval_time_out_mask.detach().cpu().numpy().tolist()
        failed_other_mask = eval_failed_other_mask.detach().cpu().numpy().tolist()
        episode_len = eval_episode_len.detach().cpu().numpy().tolist()
        episode_return = eval_episode_return.detach().cpu().numpy().tolist()
        volume_points_triggered_any = eval_volume_points_triggered_any.detach().cpu().numpy().tolist()
        volume_points_trigger_steps = eval_volume_points_trigger_steps.detach().cpu().numpy().tolist()
        volume_points_max_penetration = eval_volume_points_max_penetration.detach().cpu().numpy().tolist()
        completed_lens = [episode_len[idx] for idx in range(env.num_envs) if completed_mask[idx]]
        completed_rets = [episode_return[idx] for idx in range(env.num_envs) if completed_mask[idx]]
        completed_volume_trigger_steps = [volume_points_trigger_steps[idx] for idx in range(env.num_envs) if completed_mask[idx]]
        completed_volume_penetration = [volume_points_max_penetration[idx] for idx in range(env.num_envs) if completed_mask[idx]]
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
                "reached_target": int(eval_reached_target_mask.sum().item()),
                "time_out": int(eval_time_out_mask.sum().item()),
                "failed_other": int(eval_failed_other_mask.sum().item()),
            },
            "mean_episode_length": mean_episode_len,
            "mean_episode_return": mean_episode_return,
            "metrics_by_stage": {
                eval_stage_name_map.get(stage_id, f"Stage{stage_id}"): list(metric_names)
                for stage_id, metric_names in eval_stage_metric_names.items()
            },
            "volume_points": {
                "triggered_env_count": int(volume_triggered_count),
                "triggered_env_rate": float(volume_triggered_count / completed) if completed > 0 else 0.0,
                "mean_trigger_steps": mean_volume_trigger_steps,
                "mean_max_penetration_depth": mean_volume_max_penetration,
                "max_max_penetration_depth": float(max_volume_max_penetration),
            },
            "stage_metrics_global": {
                eval_stage_name_map.get(stage_id, f"Stage{stage_id}"): {
                    metric_name: eval_global_stage_stats[stage_id][metric_name].as_dict()
                    for metric_name in metric_names
                }
                for stage_id, metric_names in eval_stage_metric_names.items()
            },
        }

        summary_path = os.path.join(eval_output_dir, f"{eval_output_prefix}_summary.json")
        with open(summary_path, "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)

        per_env_path = os.path.join(eval_output_dir, f"{eval_output_prefix}_per_env_first_episode.csv")
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

        global_stage_path = os.path.join(eval_output_dir, f"{eval_output_prefix}_stage_global_stats.csv")
        with open(global_stage_path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=["stage_id", "stage_name", "metric", "count", "min", "max", "mean"],
            )
            writer.writeheader()
            for stage_id, metric_names in eval_stage_metric_names.items():
                stage_name = eval_stage_name_map.get(stage_id, f"Stage{stage_id}")
                for metric_name in metric_names:
                    row = eval_global_stage_stats[stage_id][metric_name].as_dict()
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

        per_env_stage_path = os.path.join(eval_output_dir, f"{eval_output_prefix}_stage_per_env_stats.csv")
        with open(per_env_stage_path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=["env_id", "stage_id", "stage_name", "metric", "count", "min", "max", "mean"],
            )
            writer.writeheader()
            for env_id in range(env.num_envs):
                for stage_id, metric_names in eval_stage_metric_names.items():
                    stage_name = eval_stage_name_map.get(stage_id, f"Stage{stage_id}")
                    for metric_name in metric_names:
                        row = eval_per_env_stage_stats[env_id][stage_id][metric_name].as_dict()
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

    # close the simulator
    env.close()

    if args_cli.video:
        subprocess.run(
            [
                "code",
                "-r",
                os.path.join(log_dir, "videos", "play", f"model_{resume_path.split('_')[-1].split('.')[0]}-step-0.mp4"),
            ]
        )


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
