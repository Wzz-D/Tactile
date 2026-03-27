from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional, Union

import numpy as np
import torch

import omni.physics.tensors.impl.api as physx

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors.sensor_base import SensorBase
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh

from .core import (
    apply_alignment_gating,
    build_knn_diffusion_matrix,
    build_taxel_pos_w,
    compute_bandexp_weights,
    compute_basic_force_stats,
    compute_contact_area_ratio,
    compute_cop_b,
    compute_edge_force_ratio,
    compute_foot_normal_w,
    diffuse_taxel_force_knn,
    distribute_total_force_to_taxels,
    normalize_taxel_weights,
    point_near_polygon_edge_mask,
)
from .foot_tactile_data import FootTactileData
from .foot_tactile_noise import FootTactileNoiseModel


class FootTactile(SensorBase):
    """Virtual tactile insole sensor that estimates per-taxel normal force."""

    def __init__(self, cfg):
        super().__init__(cfg)

    @property
    def data(self) -> FootTactileData:
        self._update_outdated_buffers()
        return self._data

    @property
    def num_bodies(self) -> int:
        return self._num_bodies

    @property
    def body_physx_view(self) -> physx.RigidBodyView:
        return self._body_physx_view

    @property
    def body_names(self) -> list[str]:
        prim_paths = self.body_physx_view.prim_paths[: self.num_bodies]
        return [path.split("/")[-1] for path in prim_paths]

    def find_bodies(self, name_keys: Union[str, Sequence[str]], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        return string_utils.resolve_matching_names(name_keys, self.body_names, preserve_order)

    def bind_contact_sensor(self, contact_sensor: Any, body_ids: Optional[Sequence[int]] = None) -> None:
        """Bind a contact sensor used to provide aggregate per-foot contact force."""

        self._contact_sensor = contact_sensor
        if body_ids is None:
            if not hasattr(contact_sensor, "find_bodies"):
                raise ValueError("contact_sensor must provide body_ids or implement find_bodies().")
            resolved_ids, _ = contact_sensor.find_bodies(self.body_names, preserve_order=True)
            body_ids = resolved_ids

        body_ids = tuple(int(body_id) for body_id in body_ids)
        if len(body_ids) != self.num_bodies:
            raise ValueError(f"Expected {self.num_bodies} contact body ids, but got {len(body_ids)}.")
        self._contact_body_ids = body_ids

    def clear_contact_sensor_binding(self) -> None:
        self._contact_sensor = None
        self._contact_body_ids = None

    def register_support_mesh(self, warp_mesh: Any) -> None:
        """Register a pre-built Warp mesh for support queries."""

        self._support_mesh_wp = warp_mesh

    def register_support_mesh_from_trimesh(self, mesh: Any, device: Optional[str] = None) -> None:
        """Build and register a Warp mesh from a trimesh-like object."""

        if device is None:
            device = str(self.device)
        points = np.asarray(mesh.vertices, dtype=np.float32)
        indices = np.asarray(mesh.faces, dtype=np.int32)
        self._support_mesh_wp = convert_to_warp_mesh(points, indices, device=device)

    def clear_support_mesh(self) -> None:
        self._support_mesh_wp = None

    def reset(self, env_ids: Sequence[int] | None = None):
        super().reset(env_ids)
        if hasattr(self, "_noise_model") and self._noise_model is not None:
            self._noise_model.reset(env_ids)

    def _initialize_impl(self):
        super()._initialize_impl()
        self._physics_sim_view = physx.create_simulation_view(self._backend)
        self._physics_sim_view.set_subspace_roots("/")

        self._initialize_body_view()
        self._initialize_template()
        self._initialize_buffers()
        self._initialize_helpers()

    def _initialize_body_view(self) -> None:
        leaf_pattern = self.cfg.prim_path.rsplit("/", 1)[-1]
        template_prim_path = self._parent_prims[0].GetPath().pathString
        body_names = []
        for prim in sim_utils.find_matching_prims(template_prim_path + "/" + leaf_pattern):
            body_names.append(prim.GetPath().pathString.rsplit("/", 1)[-1])
        if not body_names:
            raise RuntimeError(f"Sensor at path '{self.cfg.prim_path}' could not find any rigid bodies.")

        body_names_regex = r"(" + "|".join(body_names) + r")"
        body_names_regex = f"{self.cfg.prim_path.rsplit('/', 1)[0]}/{body_names_regex}"
        body_names_glob = body_names_regex.replace(".*", "*")

        self._body_physx_view = self._physics_sim_view.create_rigid_body_view(body_names_glob)
        self._num_bodies = self.body_physx_view.count // self._num_envs
        if self._num_bodies != len(body_names):
            raise RuntimeError(
                "Failed to initialize foot tactile sensor for specified bodies."
                f"\n\tInput prim path    : {self.cfg.prim_path}"
                f"\n\tResolved prim paths: {body_names_regex}"
            )

    def _initialize_template(self) -> None:
        template = self.cfg.template_cfg.func(self.cfg.template_cfg)
        left_outline_xy = template["left_outline_xy"].to(self.device)
        right_outline_xy = template["right_outline_xy"].to(self.device)
        left_taxel_xy = template["left_taxel_xy"].to(self.device)
        right_taxel_xy = template["right_taxel_xy"].to(self.device)

        self._body_sides = self._resolve_body_sides(self.body_names)

        max_taxels = max(left_taxel_xy.shape[0], right_taxel_xy.shape[0])
        self._max_taxels = max_taxels

        self._template_taxel_xy_b = torch.zeros((self._num_bodies, max_taxels, 2), device=self.device)
        self._template_valid_mask = torch.zeros((self._num_bodies, max_taxels), dtype=torch.bool, device=self.device)
        self._template_edge_mask = torch.zeros((self._num_bodies, max_taxels), dtype=torch.bool, device=self.device)

        side_templates = {
            "left": (left_taxel_xy, left_outline_xy),
            "right": (right_taxel_xy, right_outline_xy),
        }
        for body_idx, side in enumerate(self._body_sides):
            taxel_xy, outline_xy = side_templates[side]
            taxel_count = taxel_xy.shape[0]
            self._template_taxel_xy_b[body_idx, :taxel_count] = taxel_xy
            self._template_valid_mask[body_idx, :taxel_count] = True
            self._template_edge_mask[body_idx, :taxel_count] = point_near_polygon_edge_mask(
                taxel_xy,
                outline_xy,
                self.cfg.edge_margin,
            )

    def _initialize_buffers(self) -> None:
        self._data = FootTactileData.make_zero(
            num_envs=self._num_envs,
            num_bodies=self._num_bodies,
            max_taxels=self._max_taxels,
            device=self.device,
        )
        self._data.taxel_xy_b[:] = self._template_taxel_xy_b
        self._data.valid_taxel_mask[:] = self._template_valid_mask
        self._data.edge_taxel_mask[:] = self._template_edge_mask

    def _initialize_helpers(self) -> None:
        self._foot_local_normal = torch.tensor(self.cfg.foot_local_normal, device=self.device, dtype=torch.float32)
        self._contact_sensor = None
        self._contact_body_ids = None
        self._support_mesh_wp = None

        self._diffusion_matrix = build_knn_diffusion_matrix(
            taxel_xy_b=self._data.taxel_xy_b,
            valid_taxel_mask=self._data.valid_taxel_mask,
            knn=self.cfg.diffusion_cfg.diffusion_knn,
        )

        self._noise_model = FootTactileNoiseModel(
            cfg=self.cfg.noise_cfg,
            num_envs=self._num_envs,
            num_bodies=self._num_bodies,
            max_taxels=self._max_taxels,
            device=self.device,
            dtype=self._data.taxel_force.dtype,
        )

    def _resolve_body_sides(self, body_names: list[str]) -> list[str]:
        body_sides = []
        for body_name in body_names:
            lower = body_name.lower()
            if "left" in lower:
                body_sides.append("left")
            elif "right" in lower:
                body_sides.append("right")
            else:
                raise RuntimeError(f"Cannot infer left/right side from tactile body name: {body_name}")
        return body_sides

    def _update_buffers_impl(self, env_ids: Union[Sequence[int], slice]):
        if not isinstance(env_ids, slice) and len(env_ids) == self._num_envs:
            env_ids = slice(None)

        self._refresh_body_state(env_ids)
        self._refresh_taxel_pose(env_ids)
        self._refresh_support_query(env_ids)
        self._refresh_total_normal_force(env_ids)
        self._refresh_taxel_force(env_ids)
        self._refresh_features(env_ids)

    def _refresh_body_state(self, env_ids: Union[Sequence[int], slice]) -> None:
        body_poses = self.body_physx_view.get_transforms().view(-1, self.num_bodies, 7)[env_ids]
        self._data.pos_w[env_ids] = body_poses[..., :3]
        self._data.quat_w[env_ids] = math_utils.convert_quat(body_poses[..., 3:], to="wxyz")
        self._data.foot_normal_w[env_ids] = compute_foot_normal_w(
            self._data.quat_w[env_ids],
            self._foot_local_normal,
            math_utils.quat_apply,
        )

    def _refresh_taxel_pose(self, env_ids: Union[Sequence[int], slice]) -> None:
        self._data.taxel_pos_w[env_ids] = build_taxel_pos_w(
            self._data.taxel_xy_b,
            self.cfg.taxel_z_offset,
            self._data.pos_w[env_ids],
            self._data.quat_w[env_ids],
            math_utils.transform_points,
        )

    def _refresh_support_query(self, env_ids: Union[Sequence[int], slice]) -> None:
        support_dist = self._data.support_dist[env_ids]
        support_valid_mask = self._data.support_valid_mask[env_ids]
        support_normal_w = self._data.support_normal_w[env_ids]
        support_alignment = self._data.support_alignment[env_ids]

        support_dist.fill_(float("inf"))
        support_valid_mask.zero_()
        support_normal_w.zero_()
        support_alignment.zero_()

        if self._support_mesh_wp is None:
            return

        taxel_pos_w = self._data.taxel_pos_w[env_ids]
        foot_normal_w = self._data.foot_normal_w[env_ids]
        valid_template_mask = self._data.valid_taxel_mask.unsqueeze(0).expand_as(support_valid_mask)

        ray_starts = taxel_pos_w + foot_normal_w.unsqueeze(-2) * self.cfg.raycast_offset
        ray_directions = -foot_normal_w.unsqueeze(-2).expand_as(taxel_pos_w)
        max_dist = self.cfg.max_support_dist + self.cfg.pad_thickness

        E, B, T, _ = ray_starts.shape
        ray_starts_f = ray_starts.reshape(E * B * T, 1, 3)
        ray_dirs_f = ray_directions.reshape(E * B * T, 1, 3)

        _, ray_dist_f, ray_normal_f, _ = raycast_mesh(
            ray_starts_f,
            ray_dirs_f,
            mesh=self._support_mesh_wp,
            max_dist=max_dist,
            return_distance=True,
            return_normal=True,
        )
        if ray_dist_f is None or ray_normal_f is None:
            return

        ray_dist = ray_dist_f.reshape(E, B, T)
        ray_normal = ray_normal_f.reshape(E, B, T, 3)

        valid_hits = torch.isfinite(ray_dist) & valid_template_mask
        support_gap = torch.clamp(ray_dist - self.cfg.raycast_offset, min=0.0)

        support_dist[valid_hits] = support_gap[valid_hits]
        support_valid_mask[valid_hits] = True
        support_normal_w[valid_hits] = ray_normal[valid_hits]

        alignment = torch.sum(ray_normal * foot_normal_w.unsqueeze(-2), dim=-1)
        support_alignment[valid_hits] = alignment[valid_hits]

        support_dist[:] = torch.nan_to_num(support_dist, nan=float("inf"), posinf=float("inf"), neginf=float("inf"))
        support_alignment[:] = torch.nan_to_num(support_alignment, nan=0.0, posinf=0.0, neginf=0.0)

    def _refresh_total_normal_force(self, env_ids: Union[Sequence[int], slice]) -> None:
        total_normal_force = self._data.total_normal_force[env_ids]
        total_normal_force.zero_()

        if self._contact_sensor is None or self._contact_body_ids is None:
            return

        net_forces_w_history = self._contact_sensor.data.net_forces_w_history
        current_forces_w = net_forces_w_history[env_ids, -1]
        current_forces_w = current_forces_w[:, self._contact_body_ids, :]
        projected_force = torch.sum(current_forces_w * self._data.foot_normal_w[env_ids], dim=-1)
        projected_force = torch.clamp(projected_force, min=0.0)
        projected_force = torch.where(
            projected_force >= self.cfg.min_force_threshold,
            projected_force,
            torch.zeros_like(projected_force),
        )
        total_normal_force[:] = torch.nan_to_num(projected_force, nan=0.0, posinf=0.0, neginf=0.0)

    def _refresh_taxel_force(self, env_ids: Union[Sequence[int], slice]) -> None:
        env_ids_t = self._resolve_env_ids_tensor(env_ids)

        band = self.cfg.support_weight_band
        if band is None:
            band = self.cfg.max_support_dist
        band = max(float(band), 1e-6)
        rho = max(float(self.cfg.support_weight_rho), 1.0001)

        weight_bandexp = compute_bandexp_weights(
            self._data.support_dist[env_ids],
            self._data.support_valid_mask[env_ids],
            band=band,
            rho=rho,
        )
        weight_aligned = apply_alignment_gating(
            weight_bandexp=weight_bandexp,
            support_alignment=self._data.support_alignment[env_ids],
            support_valid_mask=self._data.support_valid_mask[env_ids],
            a0=self.cfg.alignment_gate_a0,
            q=self.cfg.alignment_gate_q,
            align_mix=self.cfg.align_mix,
        )

        taxel_weight_clean = normalize_taxel_weights(weight_bandexp, self._data.valid_taxel_mask)
        taxel_weight_aligned = normalize_taxel_weights(weight_aligned, self._data.valid_taxel_mask)

        self._data.taxel_weight_clean[env_ids] = taxel_weight_clean
        self._data.taxel_weight_aligned[env_ids] = taxel_weight_aligned

        taxel_force_clean = distribute_total_force_to_taxels(
            total_normal_force=self._data.total_normal_force[env_ids],
            weights=taxel_weight_aligned,
            valid_taxel_mask=self._data.valid_taxel_mask,
        )
        taxel_force_clean = self._renormalize_force_to_total(
            taxel_force_clean,
            self._data.total_normal_force[env_ids],
        )
        self._data.taxel_force_clean[env_ids] = taxel_force_clean

        if self.cfg.diffusion_cfg.enable_neighbor_diffusion:
            taxel_force_diffused = diffuse_taxel_force_knn(
                taxel_force_clean=taxel_force_clean,
                diffusion_matrix=self._diffusion_matrix,
                valid_taxel_mask=self._data.valid_taxel_mask,
                alpha=self.cfg.diffusion_cfg.diffusion_alpha,
                diffusion_iters=self.cfg.diffusion_cfg.diffusion_iters,
                preserve_total_force=self.cfg.diffusion_cfg.preserve_total_force_after_diffusion,
            )
        else:
            taxel_force_diffused = taxel_force_clean.clone()

        if self.cfg.diffusion_cfg.preserve_total_force_after_diffusion:
            taxel_force_diffused = self._renormalize_force_to_total(
                taxel_force_diffused,
                self._data.total_normal_force[env_ids],
            )

        self._data.taxel_force_diffused[env_ids] = torch.nan_to_num(
            taxel_force_diffused, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp_min(0.0)

        taxel_force_measured = self._noise_model.apply(
            force_diffused=self._data.taxel_force_diffused[env_ids],
            valid_taxel_mask=self._data.valid_taxel_mask,
            env_ids=env_ids_t,
        )
        self._data.taxel_force[env_ids] = torch.nan_to_num(
            taxel_force_measured, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp_min(0.0)

    def _refresh_features(self, env_ids: Union[Sequence[int], slice]) -> None:
        measured_force = self._data.taxel_force[env_ids]

        self._data.contact_area_ratio[env_ids] = compute_contact_area_ratio(
            measured_force,
            self._data.valid_taxel_mask,
            self.cfg.active_taxel_threshold,
        )
        self._data.edge_force_ratio[env_ids] = compute_edge_force_ratio(
            measured_force,
            self._data.edge_taxel_mask,
            self._data.valid_taxel_mask,
        )
        peak_force, mean_force = compute_basic_force_stats(
            measured_force,
            self._data.valid_taxel_mask,
        )
        self._data.peak_force[env_ids] = peak_force
        self._data.mean_force[env_ids] = mean_force
        self._data.cop_b[env_ids] = compute_cop_b(
            measured_force,
            self._data.taxel_xy_b,
            self._data.valid_taxel_mask,
        )

    def _renormalize_force_to_total(self, force: torch.Tensor, total_normal_force: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Renormalize per-foot force to target totals while keeping invalid taxels at zero."""

        valid = self._data.valid_taxel_mask.unsqueeze(0).to(dtype=force.dtype)
        clean_force = torch.nan_to_num(force, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0) * valid

        target = torch.nan_to_num(total_normal_force, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0).unsqueeze(-1)
        current = clean_force.sum(dim=-1, keepdim=True, dtype=torch.float64).to(dtype=clean_force.dtype)

        valid_count = valid.sum(dim=-1, keepdim=True)
        uniform = torch.where(valid_count > 0.0, valid / valid_count.clamp_min(1.0), torch.zeros_like(valid))

        scaled = torch.where(current > eps, clean_force * (target / current), uniform * target)
        residual = target - scaled.sum(dim=-1, keepdim=True, dtype=torch.float64).to(dtype=scaled.dtype)
        has_valid = valid_count > 0.0
        residual = torch.where(has_valid, residual, torch.zeros_like(residual))
        if torch.any(has_valid):
            masked = torch.where(valid > 0.0, scaled, torch.full_like(scaled, -1.0))
            idx = masked.argmax(dim=-1, keepdim=True)
            scaled = scaled.scatter_add(-1, idx, residual)
        scaled = torch.where(target > eps, scaled, torch.zeros_like(scaled))
        scaled = scaled * valid
        scaled = torch.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
        return scaled.clamp_min(0.0)

    def _resolve_env_ids_tensor(self, env_ids: Union[Sequence[int], slice, torch.Tensor]) -> torch.Tensor:
        if isinstance(env_ids, slice):
            return torch.arange(self._num_envs, device=self.device, dtype=torch.long)[env_ids]
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long).flatten()
        return torch.as_tensor(list(env_ids), device=self.device, dtype=torch.long).flatten()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "taxel_visualizer"):
                self.taxel_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            self.taxel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "taxel_visualizer"):
                self.taxel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if self.body_physx_view is None:
            return

        points = self._data.taxel_pos_w.reshape(-1, 3)
        valid_mask = self._data.valid_taxel_mask.unsqueeze(0).expand(self._num_envs, -1, -1).reshape(-1)
        if not torch.any(valid_mask):
            return

        points = points[valid_mask]
        active_mask = (self._data.taxel_force > self.cfg.active_taxel_threshold).reshape(-1)[valid_mask]

        self.taxel_visualizer.visualize(
            translations=points,
            marker_indices=active_mask.long(),
        )

    def _invalidate_initialize_callback(self, event):
        super()._invalidate_initialize_callback(event)
        if hasattr(self, "taxel_visualizer"):
            delattr(self, "taxel_visualizer")
        self._physics_sim_view = None
        self._body_physx_view = None
        self._contact_sensor = None
        self._contact_body_ids = None
        self._support_mesh_wp = None
        self._noise_model = None
        self._diffusion_matrix = None
