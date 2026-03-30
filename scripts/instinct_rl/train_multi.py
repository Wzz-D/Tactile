# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with Instinct-RL (GPU-offset friendly for multi-tmux runs)."""

import argparse
import multiprocessing as mp
import os
import sys

import torch
from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with Instinct-RL (multi-job GPU offset mode).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--logroot", type=str, default=None, help="Override default log root path, typically `log/instinct_rl/.`"
)
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed",
    action="store_true",
    default=False,
    help=(
        "Reserved for compatibility. In this script, torchrun detection is handled via env vars, "
        "and AppLauncher distributed mode is intentionally not forced."
    ),
)
parser.add_argument(
    "--local-rank",
    type=int,
    help="Local rank for distributed training. No need to add manually, it will be set automatically in the script.",
)
parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode.")
# train.py specific arguments
parser.add_argument("--cprofile", action="store_true", default=False, help="Enable cProfile.")

# train_multi.py specific arguments
parser.add_argument(
    "--gpu_offset",
    type=int,
    default=0,
    help=(
        "Global GPU offset for this tmux job. With torchrun local_rank in {0,1}, "
        "actual graphics/physics GPU is (gpu_offset + local_rank)."
    ),
)
parser.add_argument(
    "--disable_renderer_mgpu",
    dest="disable_renderer_mgpu",
    action="store_true",
    default=True,
    help="Disable renderer multi-GPU flags in Kit args (default: enabled for safety in concurrent jobs).",
)
parser.add_argument(
    "--enable_renderer_mgpu",
    dest="disable_renderer_mgpu",
    action="store_false",
    help="Allow renderer multi-GPU flags (not recommended for multi-tmux concurrent training).",
)

# append Instinct-RL cli arguments
cli_args.add_instinct_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()


# runtime-resolved mapping (used later in main)
RUNTIME_LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0")) if "LOCAL_RANK" in os.environ else 0
RUNTIME_GLOBAL_GPU_ID: int | None = None
RUNTIME_TORCH_DEVICE: str | None = None
RUNTIME_TORCH_CUDA_INDEX: int | None = None


def _append_kit_args(base: str, extra: str) -> str:
    base = (base or "").strip()
    extra = extra.strip()
    if not base:
        return extra
    return f"{base} {extra}"


def _parse_cuda_visible_devices() -> list[int] | None:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    if raw.lower() in {"all", "none"}:
        return None

    tokens = [x.strip() for x in raw.split(",") if x.strip()]
    if not tokens:
        return None
    try:
        return [int(x) for x in tokens]
    except ValueError:
        # UUID-style CUDA_VISIBLE_DEVICES is not handled here.
        return None


def _resolve_rank_gpu_mapping() -> None:
    """Resolve global graphics GPU and torch CUDA index for this process.

    Key idea for concurrent 4x tmux jobs:
    - AppLauncher/Kit gets a global GPU id (for renderer/physics)
    - PyTorch/agent gets a torch-visible CUDA index (usually local to CUDA_VISIBLE_DEVICES)
    """

    global RUNTIME_LOCAL_RANK, RUNTIME_GLOBAL_GPU_ID, RUNTIME_TORCH_DEVICE, RUNTIME_TORCH_CUDA_INDEX

    # default (single-process execution)
    RUNTIME_LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0")) if "LOCAL_RANK" in os.environ else 0
    device_explicit = getattr(args_cli, "device_explicit", False)

    if "LOCAL_RANK" not in os.environ:
        if args_cli.gpu_offset != 0 and not device_explicit:
            args_cli.device = f"cuda:{args_cli.gpu_offset}"

        if isinstance(args_cli.device, str) and args_cli.device.startswith("cuda"):
            if ":" in args_cli.device:
                gpu_id = int(args_cli.device.split(":")[-1])
            else:
                gpu_id = 0
            RUNTIME_GLOBAL_GPU_ID = gpu_id
            RUNTIME_TORCH_DEVICE = args_cli.device
            RUNTIME_TORCH_CUDA_INDEX = gpu_id
        return

    # torchrun mode
    local_rank = int(os.environ["LOCAL_RANK"])
    RUNTIME_LOCAL_RANK = local_rank

    # target global GPU id for renderer/physics in this process
    target_global_gpu = args_cli.gpu_offset + local_rank

    # In global-offset mode, CUDA_VISIBLE_DEVICES remaps ordinals to [0..N-1].
    # That conflicts with global GPU ids (e.g. target_global_gpu=2 with CVD=2,3 -> invalid ordinal).
    # We intentionally drop CVD here and rely on explicit global GPU ids from --gpu_offset.
    if args_cli.gpu_offset != 0 and os.environ.get("CUDA_VISIBLE_DEVICES"):
        old_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        print(
            "[WARN] Detected CUDA_VISIBLE_DEVICES with --gpu_offset != 0; "
            f"ignoring CUDA_VISIBLE_DEVICES='{old_cvd}' and using global GPU ids from --gpu_offset."
        )

    visible = _parse_cuda_visible_devices()
    if visible is None:
        # No CUDA_VISIBLE_DEVICES remap, torch index matches global index.
        torch_cuda_index = target_global_gpu
    else:
        # CUDA_VISIBLE_DEVICES remap exists.
        # Preferred case: offset+local_rank directly appears in visible list.
        if target_global_gpu in visible:
            torch_cuda_index = visible.index(target_global_gpu)
        elif local_rank < len(visible):
            # Fallback to position-based mapping.
            fallback_global = visible[local_rank]
            print(
                "[WARN] gpu_offset/local_rank target GPU is not present in CUDA_VISIBLE_DEVICES. "
                f"Fallback to visible[{local_rank}]={fallback_global}."
            )
            target_global_gpu = fallback_global
            torch_cuda_index = local_rank
        else:
            raise RuntimeError(
                "Cannot resolve GPU mapping: local_rank exceeds CUDA_VISIBLE_DEVICES length. "
                f"local_rank={local_rank}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
            )

    # Critical: keep AppLauncher out of its distributed GPU override path.
    # We do distributed training ourselves later via torch.distributed using env://.
    args_cli.distributed = False
    args_cli.device = f"cuda:{target_global_gpu}"

    RUNTIME_GLOBAL_GPU_ID = target_global_gpu
    RUNTIME_TORCH_DEVICE = f"cuda:{torch_cuda_index}"
    RUNTIME_TORCH_CUDA_INDEX = torch_cuda_index

    print(
        "[INFO] Resolved GPU mapping: "
        f"local_rank={local_rank}, gpu_offset={args_cli.gpu_offset}, "
        f"global_gpu={RUNTIME_GLOBAL_GPU_ID}, torch_device={RUNTIME_TORCH_DEVICE}, "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
    )


def _set_torch_cuda_device_prelaunch() -> None:
    """Pin this process to the resolved CUDA device before AppLauncher starts."""
    if RUNTIME_TORCH_CUDA_INDEX is None:
        return
    if not torch.cuda.is_available():
        print(
            "[WARN] torch.cuda is not available while a CUDA device was requested. "
            f"requested_index={RUNTIME_TORCH_CUDA_INDEX}"
        )
        return
    torch.cuda.set_device(RUNTIME_TORCH_CUDA_INDEX)


def _debug_pre_launch() -> None:
    rank = os.environ.get("RANK", "<unset>")
    local_rank = os.environ.get("LOCAL_RANK", "<unset>")
    world_size = os.environ.get("WORLD_SIZE", "<unset>")
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    current_device = torch.cuda.current_device() if torch.cuda.is_available() else "<cuda_unavailable>"
    print(
        "[DEBUG][PRE_LAUNCH] "
        f"pid={os.getpid()} rank={rank} local_rank={local_rank} world_size={world_size} "
        f"CUDA_VISIBLE_DEVICES={cuda_visible} torch_cuda_current_device={current_device} "
        f"torch_cuda_device_count={cuda_count} planned_app_device={args_cli.device} "
        f"planned_torch_device={RUNTIME_TORCH_DEVICE} app_global_gpu={RUNTIME_GLOBAL_GPU_ID}"
    )


def _debug_post_launch(app_launcher: AppLauncher) -> None:
    rank = os.environ.get("RANK", "<unset>")
    local_rank = os.environ.get("LOCAL_RANK", "<unset>")
    world_size = os.environ.get("WORLD_SIZE", "<unset>")
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    sim_cfg = getattr(app_launcher, "_sim_app_config", dict())
    physics_gpu = sim_cfg.get("physics_gpu", "<unknown>") if isinstance(sim_cfg, dict) else "<unknown>"
    active_gpu = sim_cfg.get("active_gpu", "<unknown>") if isinstance(sim_cfg, dict) else "<unknown>"
    current_device = torch.cuda.current_device() if torch.cuda.is_available() else "<cuda_unavailable>"
    print(
        "[DEBUG][POST_LAUNCH] "
        f"pid={os.getpid()} rank={rank} local_rank={local_rank} world_size={world_size} "
        f"CUDA_VISIBLE_DEVICES={cuda_visible} app_device_id={getattr(app_launcher, 'device_id', '<unknown>')} "
        f"app_local_rank={getattr(app_launcher, 'local_rank', '<unset>')} "
        f"app_global_rank={getattr(app_launcher, 'global_rank', '<unset>')} "
        f"app_physics_gpu={physics_gpu} app_active_gpu={active_gpu} "
        f"torch_cuda_current_device={current_device} planned_app_device={args_cli.device} "
        f"planned_torch_device={RUNTIME_TORCH_DEVICE}"
    )


# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# make renderer behavior safer in concurrent multi-job setting
if args_cli.disable_renderer_mgpu:
    args_cli.kit_args = _append_kit_args(
        args_cli.kit_args,
        "--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/maxGpuCount=1",
    )

# resolve per-rank GPU mapping BEFORE app launch
_resolve_rank_gpu_mapping()
_set_torch_cuda_device_prelaunch()
_debug_pre_launch()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
_debug_post_launch(app_launcher)

"""Rest everything follows."""

import gymnasium as gym
import torch.distributed as dist
from datetime import datetime

from instinct_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper
from instinctlab.utils.wrappers.instinct_rl import InstinctRlOnPolicyRunnerCfg

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

# Import extensions to set up environment tasks
import instinctlab.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


# set affinity in multiprocessing
def auto_affinity():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    total_cores = mp.cpu_count()
    num_cores = max(total_cores // max(world_size, 1), 1)
    start = rank * num_cores
    end = min((rank + 1) * num_cores, total_cores)
    core_range = range(start, end)
    core_mask = ",".join(map(str, core_range))
    os.system(f"taskset -cp {core_mask} {os.getpid()}")
    print("Affinity auto updated to:", core_mask, "for rank:", rank)


@hydra_task_config(args_cli.task, "instinct_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: InstinctRlOnPolicyRunnerCfg):
    """Train with Instinct-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_instinct_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed

    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    # prepare configs for distributed training
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        auto_affinity()

        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ.get("RANK", local_rank))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        env_cfg.seed += local_rank

        # torch device can be CUDA-visible index, while AppLauncher GPU is global id
        if RUNTIME_TORCH_DEVICE is not None and RUNTIME_TORCH_DEVICE.startswith("cuda"):
            torch.cuda.set_device(int(RUNTIME_TORCH_DEVICE.split(":")[-1]))
            env_cfg.sim.device = RUNTIME_TORCH_DEVICE
            agent_cfg.device = RUNTIME_TORCH_DEVICE
        else:
            env_cfg.sim.device = f"cuda:{local_rank}"
            agent_cfg.device = f"cuda:{local_rank}"

        print(
            "[INFO] Distributed training with "
            f"local_rank={local_rank}, rank={rank}, world_size={world_size}, "
            f"app_global_gpu={RUNTIME_GLOBAL_GPU_ID}, torch_device={agent_cfg.device}"
        )
        current_device = torch.cuda.current_device() if torch.cuda.is_available() else "<cuda_unavailable>"
        print(
            "[DEBUG][POST_LAUNCH] "
            f"pid={os.getpid()} rank={rank} local_rank={local_rank} world_size={world_size} "
            f"env_cfg_sim_device={env_cfg.sim.device} agent_cfg_device={agent_cfg.device} "
            f"torch_cuda_current_device={current_device}"
        )

    # specify directory for logging experiments
    if args_cli.logroot is None:
        log_root_path = os.path.join("logs", "instinct_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
    else:
        log_root_path = args_cli.logroot

    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
    if getattr(env_cfg, "run_name", None):
        log_dir += f"_{env_cfg.run_name}"
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
        for h_args in hydra_args:
            log_dir += "_"
            log_dir += h_args.split("=")[0].split(".")[-1]
            log_dir += "-"
            log_dir += h_args.split("=")[1]
    log_dir = os.path.join(log_root_path, log_dir)

    if agent_cfg.resume:
        if os.path.isabs(agent_cfg.load_run):
            resume_path = get_checkpoint_path(
                os.path.dirname(agent_cfg.load_run), os.path.basename(agent_cfg.load_run), agent_cfg.load_checkpoint
            )  # type: ignore
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] Resuming experiment from directory: {resume_path}")
        resume_run_name = os.path.basename(os.path.dirname(resume_path))
        log_dir += f"_from{resume_run_name.split('_')[0]}_{resume_run_name.split('_')[1]}"

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    env_runtime_device = getattr(env.unwrapped, "device", "<unknown>")
    print(
        "[DEBUG][POST_LAUNCH] "
        f"pid={os.getpid()} rank={os.environ.get('RANK', '<unset>')} local_rank={os.environ.get('LOCAL_RANK', '<unset>')} "
        f"env_cfg_sim_device={env_cfg.sim.device} agent_cfg_device={agent_cfg.device} "
        f"runtime_env_device={env_runtime_device}"
    )
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for instinct-rl
    env = InstinctRlVecEnvWrapper(env)

    # create runner from instinct-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    if not ("LOCAL_RANK" in os.environ and dist.get_rank() > 0):
        # prevent dumping the config in non-rank-0 process
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
        dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    if args_cli.cprofile:
        import cProfile

        cprofile = cProfile.Profile()
        print(
            "Profiling enabled, a .profile file will be saved in the log directory after the program successfully"
            " finished."
        )
        cprofile.enable()

    # run training
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=getattr(agent_cfg, "init_at_random_ep_len", False),
    )

    if args_cli.cprofile:
        cprofile.disable()
        cprofile.dump_stats(os.path.join(log_dir, "cprofile_stats.profile"))

    if "LOCAL_RANK" in os.environ:
        dist.destroy_process_group()
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
