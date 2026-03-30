"""Script to play a checkpoint if an RL agent from Instinct-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
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
parser.add_argument("--eval_mode", action="store_true", default=False, help="Evaluate first episode of each env and dump stage-wise metrics.")
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
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import load_pickle, load_yaml
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

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
    for term_name in debug_term_candidates:
        try:
            stage_reward_term_cfg = reward_manager.get_term_cfg(term_name)
        except Exception:
            continue
        candidate_term = stage_reward_term_cfg.func
        if hasattr(candidate_term, "get_debug_dict"):
            return candidate_term
    return None


def _make_stage_metric_stats(num_stages: int, metric_names: tuple[str, ...]) -> dict[int, dict[str, RunningStats]]:
    return {
        stage_id: {metric_name: RunningStats() for metric_name in metric_names}
        for stage_id in range(num_stages)
    }


def _print_foot_contact_obs_sample(obs: torch.Tensor, env, env_id: int = 0) -> None:
    obs_segments = env.get_obs_segments()
    if "foot_contact_state" not in obs_segments:
        print("[ObsDebug] 'foot_contact_state' is not in policy observation segments.")
        return

    env_id = max(0, min(env_id, env.num_envs - 1))
    term_slice, term_shape = get_obs_slice(obs_segments, "foot_contact_state")
    term_flat = obs[env_id, term_slice].detach().cpu()
    if term_flat.numel() % 8 != 0:
        print(
            f"[ObsDebug] unexpected 'foot_contact_state' flattened size={term_flat.numel()}, "
            f"shape_meta={term_shape}"
        )
        return

    foot_tensor = term_flat.view(-1, 8)
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
        F, cop_x, cop_y, area, theta, vz, rho_peak, rho_fore = foot_tensor[foot_id].tolist()
        print(
            f"[ObsDebug env={env_id} {foot_name}] "
            f"F={F:.4f}, COP2D=({cop_x:.4f},{cop_y:.4f}), area={area:.4f}, "
            f"theta={theta:.4f}, vz={vz:.4f}, rho_peak={rho_peak:.4f}, rho_fore={rho_fore:.4f}"
        )


def main():
    """Play with Instinct-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
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

    if args_cli.keyboard_control:
        env_cfg.scene.num_envs = 1
        env_cfg.episode_length_s = 1e10

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
    eval_stage_name_map: dict[int, str] = {}
    eval_num_feet = 0
    eval_num_stages = 0
    eval_metric_names = ("vz", "az", "cop_margin", "contact_area", "f_peak", "df_peak")
    eval_done_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    eval_success_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    eval_episode_len = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    eval_episode_return = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    eval_used_time_outs_flag = False
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
    if args_cli.eval_mode:
        if args_cli.eval_max_steps <= 0:
            raise ValueError("--eval_max_steps must be > 0.")
        if args_cli.eval_progress_every <= 0:
            raise ValueError("--eval_progress_every must be > 0.")
        if "contact_stage_filter" not in env.unwrapped.scene.sensors:
            raise RuntimeError("--eval_mode requires 'contact_stage_filter' sensor in the scene.")

        stage_reward_debug_term = _find_stage_reward_debug_term(env)
        stage_reward_debug_lookup_done = True
        if stage_reward_debug_term is None:
            raise RuntimeError("--eval_mode requires stage reward term with get_debug_dict() for az/cop metrics.")

        eval_stage_sensor = env.unwrapped.scene.sensors["contact_stage_filter"]
        eval_num_feet = int(eval_stage_sensor.num_bodies)
        eval_num_stages = int(getattr(eval_stage_sensor, "NUM_STAGES", 4))
        if hasattr(eval_stage_sensor, "stage_name_map"):
            eval_stage_name_map = {int(k): str(v) for k, v in eval_stage_sensor.stage_name_map().items()}
        else:
            eval_stage_name_map = {stage_id: f"Stage{stage_id}" for stage_id in range(eval_num_stages)}

        eval_global_stage_stats = _make_stage_metric_stats(eval_num_stages, eval_metric_names)
        eval_per_env_stage_stats = [
            _make_stage_metric_stats(eval_num_stages, eval_metric_names) for _ in range(env.num_envs)
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
                assert stage_reward_debug_term is not None
                active_env_ids_t = torch.nonzero(~eval_done_mask, as_tuple=False).squeeze(-1)
                if active_env_ids_t.numel() > 0:
                    step_reward = rewards if rewards.ndim == 1 else rewards.sum(dim=-1)
                    step_reward = torch.nan_to_num(step_reward, nan=0.0, posinf=0.0, neginf=0.0)
                    eval_episode_return[active_env_ids_t] += step_reward[active_env_ids_t]
                    eval_episode_len[active_env_ids_t] += 1

                    stage_ids = eval_stage_sensor.data.dominant_stage_id
                    vz_data = torch.nan_to_num(eval_stage_sensor.data.foot_vz, nan=0.0, posinf=0.0, neginf=0.0)
                    az_data = torch.nan_to_num(stage_reward_debug_term._debug_az, nan=0.0, posinf=0.0, neginf=0.0)
                    cop_margin_data = torch.nan_to_num(
                        stage_reward_debug_term._debug_cop_margin, nan=0.0, posinf=0.0, neginf=0.0
                    )
                    contact_area_data = torch.nan_to_num(
                        stage_reward_debug_term._debug_contact_area_ratio, nan=0.0, posinf=0.0, neginf=0.0
                    )
                    f_peak_data = torch.nan_to_num(stage_reward_debug_term._landing_F_peak, nan=0.0, posinf=0.0, neginf=0.0)
                    df_peak_data = torch.nan_to_num(
                        stage_reward_debug_term._landing_dF_peak, nan=0.0, posinf=0.0, neginf=0.0
                    )

                    active_env_ids = active_env_ids_t.tolist()
                    for env_id in active_env_ids:
                        for foot_id in range(eval_num_feet):
                            stage_id = int(stage_ids[env_id, foot_id].item())
                            if stage_id < 0 or stage_id >= eval_num_stages:
                                continue
                            values = {
                                "vz": float(vz_data[env_id, foot_id].item()),
                                "az": float(az_data[env_id, foot_id].item()),
                                "cop_margin": float(cop_margin_data[env_id, foot_id].item()),
                                "contact_area": float(contact_area_data[env_id, foot_id].item()),
                                "f_peak": float(f_peak_data[env_id, foot_id].item()),
                                "df_peak": float(df_peak_data[env_id, foot_id].item()),
                            }
                            for metric_name, metric_value in values.items():
                                eval_per_env_stage_stats[env_id][stage_id][metric_name].update(metric_value)
                                eval_global_stage_stats[stage_id][metric_name].update(metric_value)

                new_done_t = torch.nonzero((dones > 0) & (~eval_done_mask), as_tuple=False).squeeze(-1)
                if new_done_t.numel() > 0:
                    done_ids = new_done_t.tolist()
                    time_outs = infos.get("time_outs", None)
                    for env_id in done_ids:
                        eval_done_mask[env_id] = True
                        if time_outs is not None:
                            eval_success_mask[env_id] = bool(time_outs[env_id].item())
                            eval_used_time_outs_flag = True

                if timestep % args_cli.eval_progress_every == 0:
                    completed = int(eval_done_mask.sum().item())
                    success = int(eval_success_mask.sum().item())
                    success_rate = success / completed if completed > 0 else 0.0
                    print(
                        f"[EvalMode step={timestep}] completed={completed}/{env.num_envs},"
                        f" success={success}, success_rate={success_rate:.4f}"
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
        success = int(eval_success_mask.sum().item())
        success_rate = success / completed if completed > 0 else 0.0

        completed_mask = eval_done_mask.detach().cpu().numpy().tolist()
        success_mask = eval_success_mask.detach().cpu().numpy().tolist()
        episode_len = eval_episode_len.detach().cpu().numpy().tolist()
        episode_return = eval_episode_return.detach().cpu().numpy().tolist()
        completed_lens = [episode_len[idx] for idx in range(env.num_envs) if completed_mask[idx]]
        completed_rets = [episode_return[idx] for idx in range(env.num_envs) if completed_mask[idx]]

        mean_episode_len = float(sum(completed_lens) / len(completed_lens)) if completed_lens else 0.0
        mean_episode_return = float(sum(completed_rets) / len(completed_rets)) if completed_rets else 0.0

        summary = {
            "num_envs": int(env.num_envs),
            "num_completed": completed,
            "num_incomplete": int(env.num_envs - completed),
            "success_count": success,
            "success_rate": success_rate,
            "mean_episode_length": mean_episode_len,
            "mean_episode_return": mean_episode_return,
            "used_time_outs_flag_for_success": bool(eval_used_time_outs_flag),
            "metrics": list(eval_metric_names),
            "stage_metrics_global": {
                eval_stage_name_map.get(stage_id, f"Stage{stage_id}"): {
                    metric_name: eval_global_stage_stats[stage_id][metric_name].as_dict()
                    for metric_name in eval_metric_names
                }
                for stage_id in range(eval_num_stages)
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
                    "success",
                    "episode_length_steps",
                    "episode_return_sum",
                ],
            )
            writer.writeheader()
            for env_id in range(env.num_envs):
                writer.writerow(
                    {
                        "env_id": env_id,
                        "completed": int(completed_mask[env_id]),
                        "success": int(success_mask[env_id]),
                        "episode_length_steps": int(episode_len[env_id]),
                        "episode_return_sum": float(episode_return[env_id]),
                    }
                )

        global_stage_path = os.path.join(eval_output_dir, f"{eval_output_prefix}_stage_global_stats.csv")
        with open(global_stage_path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=["stage_id", "stage_name", "metric", "count", "min", "max", "mean"],
            )
            writer.writeheader()
            for stage_id in range(eval_num_stages):
                stage_name = eval_stage_name_map.get(stage_id, f"Stage{stage_id}")
                for metric_name in eval_metric_names:
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
                for stage_id in range(eval_num_stages):
                    stage_name = eval_stage_name_map.get(stage_id, f"Stage{stage_id}")
                    for metric_name in eval_metric_names:
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
            f"[EvalMode] completed={completed}/{env.num_envs}, success={success}, success_rate={success_rate:.4f},"
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
