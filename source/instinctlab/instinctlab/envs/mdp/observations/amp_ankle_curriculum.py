from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.observations import joint_pos_rel, joint_vel_rel
from isaaclab.managers import SceneEntityCfg

from .reference_as_state import (
    joint_pos_rel_reference_as_state,
    joint_vel_rel_reference_as_state,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


DEFAULT_ANKLE_JOINT_PATTERNS: tuple[str, ...] = (
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
)


def _get_training_iteration(env: ManagerBasedEnv) -> int:
    getter = getattr(env, "get_stage_reward_warmup_iteration", None)
    if callable(getter):
        try:
            return int(getter())
        except Exception:
            return 0
    return 0


def _compute_alpha(iteration: int, recover_start_iter: int, recover_end_iter: int) -> float:
    start_iter = int(recover_start_iter)
    end_iter = int(recover_end_iter)
    denom = max(end_iter - start_iter, 1)
    alpha = float(iteration - start_iter) / float(denom)
    return max(0.0, min(1.0, alpha))


def _get_selected_joint_ids(joint_ids, total_joints: int) -> list[int]:
    if joint_ids is None:
        return list(range(total_joints))
    if isinstance(joint_ids, slice):
        return list(range(total_joints))[joint_ids]
    if torch.is_tensor(joint_ids):
        return [int(v) for v in joint_ids.detach().cpu().tolist()]
    return [int(v) for v in joint_ids]


def _cached_ankle_local_joint_indices(
    env: ManagerBasedEnv,
    asset_name: str,
    asset_cfg: SceneEntityCfg,
    ankle_joint_patterns: Sequence[str],
) -> list[int]:
    cache = getattr(env, "_amp_ankle_joint_idx_cache", None)
    if cache is None:
        cache = {}
        setattr(env, "_amp_ankle_joint_idx_cache", cache)

    key = (asset_name, id(asset_cfg), tuple(ankle_joint_patterns))
    cached = cache.get(key)
    if cached is not None:
        return cached

    asset = env.scene[asset_name]
    joint_names = list(asset.joint_names)
    selected_joint_ids = _get_selected_joint_ids(asset_cfg.joint_ids, len(joint_names))
    selected_joint_names = [joint_names[idx] for idx in selected_joint_ids]
    local_joint_ids, _ = asset.find_joints(
        list(ankle_joint_patterns),
        joint_subset=selected_joint_names,
        preserve_order=True,
    )
    local_joint_ids = [int(idx) for idx in local_joint_ids]
    cache[key] = local_joint_ids
    return local_joint_ids


def _apply_ankle_curriculum(
    obs: torch.Tensor,
    ankle_local_joint_ids: Sequence[int],
    alpha: float,
) -> torch.Tensor:
    if not ankle_local_joint_ids:
        return obs
    if alpha >= 1.0:
        return obs

    out = obs.clone()
    out[:, list(ankle_local_joint_ids)] *= float(alpha)
    return out


def joint_pos_rel_amp_ankle_curriculum(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    recover_start_iter: int = 8000,
    recover_end_iter: int = 10000,
    ankle_joint_patterns: Sequence[str] = DEFAULT_ANKLE_JOINT_PATTERNS,
) -> torch.Tensor:
    obs = joint_pos_rel(env, asset_cfg=asset_cfg)
    alpha = _compute_alpha(_get_training_iteration(env), recover_start_iter, recover_end_iter)
    ankle_local_joint_ids = _cached_ankle_local_joint_indices(
        env=env,
        asset_name=asset_cfg.name,
        asset_cfg=asset_cfg,
        ankle_joint_patterns=ankle_joint_patterns,
    )
    return _apply_ankle_curriculum(obs, ankle_local_joint_ids, alpha)


def joint_vel_rel_amp_ankle_curriculum(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    recover_start_iter: int = 8000,
    recover_end_iter: int = 10000,
    ankle_joint_patterns: Sequence[str] = DEFAULT_ANKLE_JOINT_PATTERNS,
) -> torch.Tensor:
    obs = joint_vel_rel(env, asset_cfg=asset_cfg)
    alpha = _compute_alpha(_get_training_iteration(env), recover_start_iter, recover_end_iter)
    ankle_local_joint_ids = _cached_ankle_local_joint_indices(
        env=env,
        asset_name=asset_cfg.name,
        asset_cfg=asset_cfg,
        ankle_joint_patterns=ankle_joint_patterns,
    )
    return _apply_ankle_curriculum(obs, ankle_local_joint_ids, alpha)


def joint_pos_rel_reference_as_state_amp_ankle_curriculum(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    recover_start_iter: int = 8000,
    recover_end_iter: int = 10000,
    ankle_joint_patterns: Sequence[str] = DEFAULT_ANKLE_JOINT_PATTERNS,
) -> torch.Tensor:
    obs = joint_pos_rel_reference_as_state(env, asset_cfg=asset_cfg, robot_cfg=robot_cfg)
    alpha = _compute_alpha(_get_training_iteration(env), recover_start_iter, recover_end_iter)
    ankle_local_joint_ids = _cached_ankle_local_joint_indices(
        env=env,
        asset_name=asset_cfg.name,
        asset_cfg=asset_cfg,
        ankle_joint_patterns=ankle_joint_patterns,
    )
    return _apply_ankle_curriculum(obs, ankle_local_joint_ids, alpha)


def joint_vel_rel_reference_as_state_amp_ankle_curriculum(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    mask: bool = True,
    recover_start_iter: int = 8000,
    recover_end_iter: int = 10000,
    ankle_joint_patterns: Sequence[str] = DEFAULT_ANKLE_JOINT_PATTERNS,
) -> torch.Tensor:
    obs = joint_vel_rel_reference_as_state(
        env,
        asset_cfg=asset_cfg,
        robot_cfg=robot_cfg,
        mask=mask,
    )
    alpha = _compute_alpha(_get_training_iteration(env), recover_start_iter, recover_end_iter)
    ankle_local_joint_ids = _cached_ankle_local_joint_indices(
        env=env,
        asset_name=asset_cfg.name,
        asset_cfg=asset_cfg,
        ankle_joint_patterns=ankle_joint_patterns,
    )
    return _apply_ankle_curriculum(obs, ankle_local_joint_ids, alpha)
