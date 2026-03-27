from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pxr import UsdGeom

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedEnv


def push_by_setting_velocity_without_stand(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Push the asset by setting the root velocity to a random value within the given ranges. No pushing when standing still."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # velocities
    vel_w = asset.data.root_vel_w[env_ids]
    # sample random velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    add_vel = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device)
    lin_vel = torch.norm(env.command_manager.get_command(command_name)[env_ids, :2], dim=1) > 0.15
    ang_vel = torch.abs(env.command_manager.get_command(command_name)[env_ids, 2]) > 0.15
    should_push = torch.logical_or(lin_vel, ang_vel).float().unsqueeze(-1)

    vel_w += add_vel * should_push
    # set the velocities into the physics simulation
    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)


def _load_usd_mesh_trimesh(stage, prim_path: str):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Support mesh prim not found: {prim_path}")

    mesh = UsdGeom.Mesh(prim)
    if not mesh:
        raise RuntimeError(f"Prim is not a UsdGeom.Mesh: {prim_path}")

    pts = mesh.GetPointsAttr().Get()
    face_counts = mesh.GetFaceVertexCountsAttr().Get()
    face_indices = mesh.GetFaceVertexIndicesAttr().Get()

    if pts is None or face_counts is None or face_indices is None:
        raise RuntimeError(f"Mesh has missing topology attrs: {prim_path}")

    verts = np.asarray(pts, dtype=np.float32)

    # triangulate polygons (fan triangulation)
    tris = []
    idx = 0
    for n in face_counts:
        n = int(n)
        if n < 3:
            idx += n
            continue
        v0 = int(face_indices[idx + 0])
        for k in range(1, n - 1):
            v1 = int(face_indices[idx + k])
            v2 = int(face_indices[idx + k + 1])
            tris.append([v0, v1, v2])
        idx += n
    faces = np.asarray(tris, dtype=np.int32)

    if faces.shape[0] == 0:
        raise RuntimeError(f"Mesh triangulation produced 0 triangles: {prim_path}")

    # a lightweight "trimesh-like" object
    class _Tri:
        def __init__(self, v, f):
            self.vertices = v
            self.faces = f

    return _Tri(verts, faces)


def bind_foot_tactile(
    env,
    env_ids,
    tactile_cfg=SceneEntityCfg("foot_tactile"),
    contact_forces_cfg=SceneEntityCfg("contact_forces_foot"),
    support_mesh_prim_path="/World/ground/edges/mesh",
):
    tactile = env.scene[tactile_cfg.name]

    # resolve contact sensor robustly
    key = contact_forces_cfg.name if isinstance(contact_forces_cfg, SceneEntityCfg) else str(contact_forces_cfg)
    try:
        contact_sensor = env.scene[key]
    except KeyError:
        for fallback in ("contact_forces_foot", "contact_forces"):
            try:
                contact_sensor = env.scene[fallback]
                break
            except KeyError:
                contact_sensor = None

    tactile.bind_contact_sensor(contact_sensor)

    # register support mesh from USD prim
    stage = env.scene.stage  # InteractiveScene exposes stage
    tri = _load_usd_mesh_trimesh(stage, support_mesh_prim_path)
    tactile.register_support_mesh_from_trimesh(tri, device=str(tactile.device))


def bind_contact_stage_filter(
    env,
    env_ids,
    stage_cfg=SceneEntityCfg("contact_stage_filter"),
    tactile_cfg=SceneEntityCfg("foot_tactile"),
    support_mesh_prim_path="/World/ground/edges/mesh",
):
    stage_filter = env.scene[stage_cfg.name]

    tactile = None
    try:
        tactile = env.scene[tactile_cfg.name]
    except KeyError:
        tactile = None
    if tactile is None:
        raise RuntimeError("ContactStageFilter startup requires 'foot_tactile' sensor to be present.")

    stage_filter.bind_tactile_sensor(tactile)

    stage = env.scene.stage
    tri = _load_usd_mesh_trimesh(stage, support_mesh_prim_path)
    stage_filter.register_support_mesh_from_trimesh(tri, device=str(stage_filter.device))
