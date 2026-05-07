"""Headless first-episode evaluation for Parkour checkpoints."""

from __future__ import annotations

import argparse
import copy
import os
import sys

sys.path.append(os.path.join(os.getcwd(), "scripts", "instinct_rl"))

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Run headless first-episode evaluation for a Parkour policy.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to a .pt checkpoint file.")
parser.add_argument("--useonnx", action="store_true", default=False, help="Use ONNX policy in the evaluation loop.")
parser.add_argument("--env_cfg", action="store_true", default=False, help="Load saved env config from the checkpoint.")
parser.add_argument("--agent_cfg", action="store_true", default=False, help="Load saved agent config from the checkpoint.")
parser.add_argument(
    "--eval_use_current_scene",
    action="store_true",
    default=False,
    help=(
        "Override checkpoint scene semantics with the current task config while preserving checkpoint-compatible "
        "observation and network contracts."
    ),
)
parser.add_argument("--eval_max_steps", type=int, default=20000, help="Maximum simulation steps before forced stop.")
parser.add_argument("--eval_progress_every", type=int, default=200, help="Print eval progress every N steps.")
parser.add_argument("--eval_output_dir", type=str, default=None, help="Output directory for eval logs.")
parser.add_argument("--eval_output_prefix", type=str, default="", help="Prefix of eval output files.")

cli_args.add_instinct_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
if hasattr(args_cli, "enable_cameras"):
    args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from instinct_rl.runners import OnPolicyRunner
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_tasks.utils import parse_env_cfg

from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper

try:
    from ._headless_utils import (
        apply_current_eval_scene_overrides,
        apply_runtime_env_overrides,
        build_parkour_onnx_policy,
        ensure_eval_stage_tactile_compat,
        ensure_foot_tactile_cfg_schema_compat,
        initialize_eval_state,
        inject_target_reached_termination,
        load_env_and_agent_cfg,
        resolve_checkpoint_info,
        update_eval_state,
        write_eval_outputs,
    )
except ImportError:
    from _headless_utils import (
        apply_current_eval_scene_overrides,
        apply_runtime_env_overrides,
        build_parkour_onnx_policy,
        ensure_eval_stage_tactile_compat,
        ensure_foot_tactile_cfg_schema_compat,
        initialize_eval_state,
        inject_target_reached_termination,
        load_env_and_agent_cfg,
        resolve_checkpoint_info,
        update_eval_state,
        write_eval_outputs,
    )


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    current_eval_scene_env_cfg = copy.deepcopy(env_cfg)
    agent_cfg = cli_args.parse_instinct_rl_cfg(args_cli.task, args_cli)
    checkpoint_info = resolve_checkpoint_info(args_cli, agent_cfg, script_name="eval.py")

    env_cfg, agent_cfg_dict = load_env_and_agent_cfg(
        args_cli,
        env_cfg,
        agent_cfg,
        checkpoint_info.log_dir,
        prefer_saved=True,
    )
    apply_current_eval_scene_overrides(args_cli, env_cfg, current_eval_scene_env_cfg)
    apply_runtime_env_overrides(args_cli, env_cfg)
    ensure_foot_tactile_cfg_schema_compat(env_cfg)
    ensure_eval_stage_tactile_compat(env_cfg)
    inject_target_reached_termination(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = InstinctRlVecEnvWrapper(env)
    env.unwrapped.configure_eval_pre_reset_snapshot(True)

    ppo_runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device=args_cli.device)
    print(f"[INFO]: Loading model checkpoint from: {checkpoint_info.resume_path}")
    ppo_runner.load(checkpoint_info.resume_path)

    if args_cli.useonnx:
        onnx_dir = os.path.join(checkpoint_info.log_dir, "exported")
        print(f"[INFO] Evaluating with ONNX policy from: {onnx_dir}")
        policy = build_parkour_onnx_policy(env, agent_cfg_dict, onnx_dir)
    else:
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    obs, _ = env.get_observations()
    eval_state = initialize_eval_state(args_cli, env, checkpoint_info.log_dir, checkpoint_info.resume_path)

    timestep = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, rewards, dones, infos = env.step(actions)
            should_stop = update_eval_state(eval_state, env, rewards, dones, infos, timestep, args_cli)
            if should_stop:
                break
        timestep += 1

    write_eval_outputs(eval_state, env)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
