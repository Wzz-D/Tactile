"""Headless ONNX export for Parkour checkpoints."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.path.join(os.getcwd(), "scripts", "instinct_rl"))

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Export a trained Parkour policy to ONNX without launching viewer/video.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Must be 1 for export.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to a .pt checkpoint file.")
parser.add_argument("--env_cfg", action="store_true", default=False, help="Load saved env config from the checkpoint.")
parser.add_argument("--agent_cfg", action="store_true", default=False, help="Load saved agent config from the checkpoint.")
parser.add_argument(
    "--ignore_saved_env_cfg",
    action="store_true",
    default=False,
    help="Do not load params/env.pkl from the run directory; use the current task env config instead.",
)
parser.add_argument("--export_dir", type=str, default=None, help="Directory to write exported ONNX artifacts.")

cli_args.add_instinct_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
if hasattr(args_cli, "enable_cameras"):
    args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

from instinct_rl.runners import OnPolicyRunner
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_tasks.utils import parse_env_cfg

from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper

try:
    from ._headless_utils import (
        apply_runtime_env_overrides,
        ensure_foot_tactile_cfg_schema_compat,
        export_policy_and_validate,
        load_env_and_agent_cfg,
        resolve_checkpoint_info,
    )
except ImportError:
    from _headless_utils import (
        apply_runtime_env_overrides,
        ensure_foot_tactile_cfg_schema_compat,
        export_policy_and_validate,
        load_env_and_agent_cfg,
        resolve_checkpoint_info,
    )


def main() -> None:
    if args_cli.num_envs not in (None, 1):
        raise ValueError("expolicy.py only supports --num_envs=1.")
    args_cli.num_envs = 1

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg = cli_args.parse_instinct_rl_cfg(args_cli.task, args_cli)
    checkpoint_info = resolve_checkpoint_info(args_cli, agent_cfg, script_name="expolicy.py")

    env_cfg, agent_cfg_dict = load_env_and_agent_cfg(
        args_cli,
        env_cfg,
        agent_cfg,
        checkpoint_info.log_dir,
        prefer_saved=True,
    )
    apply_runtime_env_overrides(args_cli, env_cfg)
    ensure_foot_tactile_cfg_schema_compat(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = InstinctRlVecEnvWrapper(env)

    ppo_runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device=args_cli.device)
    print(f"[INFO]: Loading model checkpoint from: {checkpoint_info.resume_path}")
    ppo_runner.load(checkpoint_info.resume_path)

    export_dir = os.path.abspath(args_cli.export_dir) if args_cli.export_dir else os.path.join(
        checkpoint_info.log_dir, "exported"
    )
    summary = export_policy_and_validate(ppo_runner, env, agent_cfg_dict, export_dir)

    print(f"[Export] saved artifacts to: {summary['export_dir']}")
    print("[Export] artifacts: " + ", ".join(summary["artifacts"]))
    print(
        f"[Export] ONNX validation: max_abs_diff={summary['max_abs_diff']:.6e}, "
        f"mean_abs_diff={summary['mean_abs_diff']:.6e}"
    )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
