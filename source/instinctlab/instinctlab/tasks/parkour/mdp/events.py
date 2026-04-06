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


def _is_usd_mesh_prim(prim) -> bool:
    return bool(prim and prim.IsValid() and prim.GetTypeName() == "Mesh")


def _unique_nonempty_strings(paths: list[str | None]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if not path:
            continue
        normalized = path.rstrip("/") or path
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _support_mesh_sort_key(path: str) -> tuple[int, int, int, int, str]:
    lower = path.lower()
    return (
        0 if lower.endswith("/mesh") else 1,
        0 if "/terrain" in lower else 1,
        1 if "/visual" in lower else 0,
        len(lower),
        lower,
    )


def _find_mesh_descendants(stage, root_path: str) -> list[str]:
    root_path = root_path.rstrip("/")
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim or not root_prim.IsValid():
        return []

    prefix = root_path + "/"
    matches: list[str] = []
    for prim in stage.Traverse():
        prim_path = prim.GetPath().pathString
        if (prim_path == root_path or prim_path.startswith(prefix)) and prim.GetTypeName() == "Mesh":
            matches.append(prim_path)
    return sorted(matches, key=_support_mesh_sort_key)


def _resolve_support_mesh_prim_path(env, support_mesh_prim_path: str | None) -> str:
    """Resolve the support mesh from the terrain importer before falling back to ad hoc stage paths."""

    stage = env.scene.stage
    if support_mesh_prim_path:
        explicit_prim = stage.GetPrimAtPath(support_mesh_prim_path)
        if _is_usd_mesh_prim(explicit_prim):
            return support_mesh_prim_path

    terrain = getattr(env.scene, "terrain", None)
    terrain_prim_paths = list(getattr(terrain, "terrain_prim_paths", []) or [])
    search_roots = _unique_nonempty_strings(
        [
            support_mesh_prim_path,
            support_mesh_prim_path.rsplit("/", 1)[0] if support_mesh_prim_path and support_mesh_prim_path.endswith("/mesh") else None,
            *terrain_prim_paths,
            "/World/ground/terrain",
            "/World/ground",
        ]
    )

    for root_path in search_roots:
        exact_mesh_path = f"{root_path}/mesh"
        exact_mesh_prim = stage.GetPrimAtPath(exact_mesh_path)
        if _is_usd_mesh_prim(exact_mesh_prim):
            if support_mesh_prim_path and exact_mesh_path != support_mesh_prim_path:
                print(f"[INFO] Auto-resolved support mesh prim from '{support_mesh_prim_path}' to '{exact_mesh_path}'.")
            return exact_mesh_path

    mesh_candidates: list[str] = []
    for root_path in search_roots:
        mesh_candidates.extend(_find_mesh_descendants(stage, root_path))
    mesh_candidates = sorted(_unique_nonempty_strings(mesh_candidates), key=_support_mesh_sort_key)
    if mesh_candidates:
        chosen_path = mesh_candidates[0]
        if support_mesh_prim_path and chosen_path != support_mesh_prim_path:
            print(f"[INFO] Auto-resolved support mesh prim from '{support_mesh_prim_path}' to '{chosen_path}'.")
        return chosen_path

    raise RuntimeError(
        "Unable to resolve a support mesh prim. "
        f"requested={support_mesh_prim_path!r}, "
        f"terrain_prim_paths={terrain_prim_paths}, "
        f"search_roots={search_roots}"
    )


_PARKOUR_TERRAIN_SUPPORT_MESH_PRIM_PATH: str | None = None


def bind_foot_tactile(
    env,
    env_ids,
    tactile_cfg=SceneEntityCfg("foot_tactile"),
    contact_forces_cfg=SceneEntityCfg("contact_forces_foot"),
    support_mesh_prim_path=_PARKOUR_TERRAIN_SUPPORT_MESH_PRIM_PATH,
):
    tactile = env.scene[tactile_cfg.name]
    tactile_body_names = list(getattr(tactile, "body_names", []))

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
    if contact_sensor is not None and tactile_body_names and hasattr(contact_sensor, "body_names"):
        contact_body_ids = tuple(getattr(tactile, "_contact_body_ids", tuple()))
        contact_body_names = [str(contact_sensor.body_names[body_id]) for body_id in contact_body_ids]
        print(
            "[INFO] FootTactile binding: "
            f"tactile_bodies={tactile_body_names}, contact_bodies={contact_body_names}"
        )

    stage = env.scene.stage  # InteractiveScene exposes stage
    resolved_support_mesh_prim_path = _resolve_support_mesh_prim_path(env, support_mesh_prim_path)
    tri = _load_usd_mesh_trimesh(stage, resolved_support_mesh_prim_path)
    tactile.register_support_mesh_from_trimesh(tri, device=str(tactile.device))


def bind_contact_stage_filter(
    env,
    env_ids,
    stage_cfg=SceneEntityCfg("contact_stage_filter"),
    tactile_cfg=SceneEntityCfg("foot_tactile"),
    support_mesh_prim_path=_PARKOUR_TERRAIN_SUPPORT_MESH_PRIM_PATH,
):
    stage_filter = env.scene[stage_cfg.name]
    stage_body_names = list(getattr(stage_filter, "body_names", []))

    tactile = None
    try:
        tactile = env.scene[tactile_cfg.name]
    except KeyError:
        tactile = None
    if tactile is None:
        raise RuntimeError("ContactStageFilter startup requires 'foot_tactile' sensor to be present.")

    stage_filter.bind_tactile_sensor(tactile)
    if stage_body_names and hasattr(tactile, "body_names"):
        tactile_body_ids = tuple(getattr(stage_filter, "_tactile_body_ids", tuple()))
        tactile_body_names = [str(tactile.body_names[body_id]) for body_id in tactile_body_ids]
        print(
            "[INFO] ContactStageFilter binding: "
            f"stage_bodies={stage_body_names}, tactile_bodies={tactile_body_names}"
        )

    stage = env.scene.stage
    resolved_support_mesh_prim_path = _resolve_support_mesh_prim_path(env, support_mesh_prim_path)
    tri = _load_usd_mesh_trimesh(stage, resolved_support_mesh_prim_path)
    stage_filter.register_support_mesh_from_trimesh(tri, device=str(stage_filter.device))
