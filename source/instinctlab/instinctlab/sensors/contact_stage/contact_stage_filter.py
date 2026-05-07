from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional, Union

import numpy as np
import torch

import omni.physics.tensors.impl.api as physx

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.sensors.sensor_base import SensorBase
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh

from .contact_stage_data import ContactStageData


class ContactStageFilter(SensorBase):
    """Independent per-foot 4-stage contact filter driven by tactile + kinematics."""

    STAGE_SWING = 0
    STAGE_PRELANDING = 1
    STAGE_LANDING = 2
    STAGE_STANCE = 3
    NUM_STAGES = 4

    STAGE_NAMES: tuple[str, ...] = (
        "Swing",
        "PreLanding",
        "Landing",
        "Stance",
    )

    def __init__(self, cfg):
        super().__init__(cfg)

    @property
    def data(self) -> ContactStageData:
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

    @classmethod
    def stage_name_map(cls) -> dict[int, str]:
        return {index: name for index, name in enumerate(cls.STAGE_NAMES)}

    @classmethod
    def stage_id_to_name(cls, stage_id: int) -> str:
        if 0 <= stage_id < cls.NUM_STAGES:
            return cls.STAGE_NAMES[stage_id]
        return f"Unknown({stage_id})"

    def get_debug_tensors(self) -> dict[str, torch.Tensor]:
        """Return read-only per-env/per-foot tensors used by stage gating and debugging."""

        stage_data = self.data
        stage_eligibility = torch.nan_to_num(
            stage_data.stage_eligibility,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        landing_window_active = stage_data.landing_window > 0
        h_zone_hit = stage_data.h_eff < float(self.cfg.h_zone)
        v_pre_hit = stage_data.foot_vz < -float(self.cfg.v_pre)
        force_on_hit = stage_data.total_force > float(self.cfg.contact_force_on)
        area_on_hit = stage_data.contact_area > float(self.cfg.contact_area_on)

        return {
            "h_eff": stage_data.h_eff,
            "foot_vz": stage_data.foot_vz,
            "total_force": stage_data.total_force,
            "total_force_filt": stage_data.total_force_filt,
            "contact_area": stage_data.contact_area,
            "contact_area_filt": stage_data.contact_area_filt,
            "dF": stage_data.dF,
            "dA": stage_data.dA,
            "contact_active": stage_data.contact_active,
            "contact_on_event": stage_data.contact_on_event,
            "contact_off_event": stage_data.contact_off_event,
            "landing_window": stage_data.landing_window,
            "landing_window_active": landing_window_active,
            "stage_eligibility": stage_eligibility,
            "dominant_stage_id": stage_data.dominant_stage_id,
            "h_zone_hit": h_zone_hit,
            "v_pre_hit": v_pre_hit,
            "force_on_hit": force_on_hit,
            "area_on_hit": area_on_hit,
            "E_sw": stage_eligibility[..., self.STAGE_SWING] > 0.5,
            "E_pre": stage_eligibility[..., self.STAGE_PRELANDING] > 0.5,
            "E_land": stage_eligibility[..., self.STAGE_LANDING] > 0.5,
            "E_st": stage_eligibility[..., self.STAGE_STANCE] > 0.5,
        }

    def get_debug_dict(self, env_id: int) -> dict[str, torch.Tensor]:
        """Return cloned per-foot tensors for one environment."""

        debug_tensors = self.get_debug_tensors()
        env_id = int(max(0, min(env_id, self._num_envs - 1)))
        return {name: tensor[env_id].detach().clone() for name, tensor in debug_tensors.items()}

    def get_stage_debug_string(self, env_id: int, foot_id: int) -> str:
        """Return compact per-foot debug info for stage filtering."""
        debug_dict = self.get_debug_dict(env_id)

        stage_id = int(debug_dict["dominant_stage_id"][foot_id].item())
        stage_name = self.stage_id_to_name(stage_id)
        eligibility = debug_dict["stage_eligibility"][foot_id].detach().cpu().tolist()

        contact_active = bool(debug_dict["contact_active"][foot_id].item())
        contact_on_event = bool(debug_dict["contact_on_event"][foot_id].item())
        contact_off_event = bool(debug_dict["contact_off_event"][foot_id].item())
        landing_window = int(debug_dict["landing_window"][foot_id].item())
        landing_window_active = bool(debug_dict["landing_window_active"][foot_id].item())
        tau_on = float(self._data.tau_on[env_id, foot_id].item())
        tau_off = float(self._data.tau_off[env_id, foot_id].item())
        tau_stage = float(self._data.tau_stage[env_id, foot_id].item())

        h_eff = float(debug_dict["h_eff"][foot_id].item())
        vz = float(debug_dict["foot_vz"][foot_id].item())
        force = float(debug_dict["total_force"][foot_id].item())
        force_filt = float(debug_dict["total_force_filt"][foot_id].item())
        area = float(debug_dict["contact_area"][foot_id].item())
        area_filt = float(debug_dict["contact_area_filt"][foot_id].item())
        d_force = float(debug_dict["dF"][foot_id].item())
        d_area = float(debug_dict["dA"][foot_id].item())
        rho_fore = float(self._data.rho_fore[env_id, foot_id].item())
        pre_core = bool(self._debug_pre_core[env_id, foot_id].item())
        h_zone_hit = bool(debug_dict["h_zone_hit"][foot_id].item())
        v_pre_hit = bool(debug_dict["v_pre_hit"][foot_id].item())
        force_on_hit = bool(debug_dict["force_on_hit"][foot_id].item())
        area_on_hit = bool(debug_dict["area_on_hit"][foot_id].item())

        return (
            f"ContactStage(env={env_id}, foot={foot_id}, stage={stage_name}, "
            f"contact_active={contact_active}, on_event={contact_on_event}, off_event={contact_off_event}, "
            f"landing_window={landing_window}, landing_active={landing_window_active}, tau_on={tau_on:.3f}, "
            f"tau_off={tau_off:.3f}, tau_stage={tau_stage:.3f}, h_eff={h_eff:.3f}, vz={vz:.3f}, "
            f"F={force:.3f}, Ff={force_filt:.3f}, A={area:.3f}, Af={area_filt:.3f}, "
            f"dF={d_force:.3f}, dA={d_area:.3f}, rho_fore={rho_fore:.3f}, pre_core={pre_core}, "
            f"hits=(h:{h_zone_hit},vz:{v_pre_hit},F:{force_on_hit},A:{area_on_hit}), "
            f"E={[round(v, 3) for v in eligibility]})"
        )

    def find_bodies(self, name_keys: Union[str, Sequence[str]], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        return string_utils.resolve_matching_names(name_keys, self.body_names, preserve_order)

    def bind_tactile_sensor(self, tactile_sensor: Any) -> None:
        self._tactile_sensor = tactile_sensor
        self._tactile_body_ids = tuple(range(self.num_bodies))
        stage_body_names = list(self.body_names)
        if hasattr(tactile_sensor, "find_bodies") and hasattr(tactile_sensor, "body_names"):
            ids, tactile_body_names = tactile_sensor.find_bodies(stage_body_names, preserve_order=True)
            if len(ids) == self.num_bodies:
                if list(tactile_body_names) != stage_body_names:
                    raise ValueError(
                        "ContactStageFilter tactile binding resolved unexpected body names: "
                        f"expected={stage_body_names}, got={list(tactile_body_names)}"
                    )
                self._tactile_body_ids = tuple(int(body_id) for body_id in ids)
        if hasattr(tactile_sensor, "body_names"):
            resolved_tactile_names = [str(tactile_sensor.body_names[body_id]) for body_id in self._tactile_body_ids]
            if resolved_tactile_names != stage_body_names:
                raise ValueError(
                    "ContactStageFilter tactile binding body-order mismatch: "
                    f"stage={stage_body_names}, tactile={resolved_tactile_names}"
                )
        self._geometry_ready = False

    def clear_tactile_sensor_binding(self) -> None:
        self._tactile_sensor = None
        self._tactile_body_ids = tuple(range(self.num_bodies))
        self._geometry_ready = False

    def register_support_mesh(self, warp_mesh: Any) -> None:
        self._support_mesh_wp = warp_mesh

    def register_support_mesh_from_trimesh(self, mesh: Any, device: Optional[str] = None) -> None:
        if device is None:
            device = str(self.device)
        points = np.asarray(mesh.vertices, dtype=np.float32)
        indices = np.asarray(mesh.faces, dtype=np.int32)
        self._support_mesh_wp = convert_to_warp_mesh(points, indices, device=device)

    def clear_support_mesh(self) -> None:
        self._support_mesh_wp = None

    def reset(self, env_ids: Sequence[int] | None = None):
        super().reset(env_ids)
        env_ids_t = self._resolve_env_ids_tensor(env_ids)

        self._data.h_eff[env_ids_t] = float(self.cfg.h_eff_max_dist)
        self._data.foot_vz[env_ids_t] = 0.0
        self._data.foot_theta[env_ids_t] = 0.0

        self._data.total_force[env_ids_t] = 0.0
        self._data.total_force_filt[env_ids_t] = 0.0
        self._data.dF[env_ids_t] = 0.0
        self._data.contact_area[env_ids_t] = 0.0
        self._data.contact_area_filt[env_ids_t] = 0.0
        self._data.dA[env_ids_t] = 0.0
        self._data.cop_ap[env_ids_t] = 0.0
        self._data.rho_peak[env_ids_t] = 0.0
        self._data.rho_fore[env_ids_t] = 0.0

        self._data.contact_active[env_ids_t] = False
        self._data.contact_on_event[env_ids_t] = False
        self._data.contact_off_event[env_ids_t] = False
        self._data.landing_window[env_ids_t] = 0
        self._data.tau_on[env_ids_t] = 0.0
        self._data.tau_off[env_ids_t] = 0.0
        self._data.tau_stage[env_ids_t] = 0.0

        self._data.stage_eligibility[env_ids_t] = 0.0
        self._data.dominant_stage_id[env_ids_t] = self.STAGE_SWING

        self._filter_initialized[env_ids_t] = False
        self._debug_pre_core[env_ids_t] = False
        self._contact_on_counter[env_ids_t] = 0
        self._contact_off_counter[env_ids_t] = 0
        self._foot_pos_w[env_ids_t] = 0.0

    def _initialize_impl(self):
        super()._initialize_impl()
        self._physics_sim_view = physx.create_simulation_view(self._backend)
        self._physics_sim_view.set_subspace_roots("/")

        self._initialize_body_view()
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
                "Failed to initialize contact stage filter for specified bodies."
                f"\n\tInput prim path    : {self.cfg.prim_path}"
                f"\n\tResolved prim paths: {body_names_regex}"
            )

    def _initialize_buffers(self) -> None:
        self._data = ContactStageData.make_zero(
            num_envs=self._num_envs,
            num_feet=self._num_bodies,
            num_stages=self.NUM_STAGES,
            device=self.device,
        )
        self._data.h_eff[:] = float(self.cfg.h_eff_max_dist)

        self._filter_initialized = torch.zeros((self._num_envs, self._num_bodies), device=self.device, dtype=torch.bool)
        self._debug_pre_core = torch.zeros((self._num_envs, self._num_bodies), device=self.device, dtype=torch.bool)
        self._contact_on_counter = torch.zeros((self._num_envs, self._num_bodies), device=self.device, dtype=torch.long)
        self._contact_off_counter = torch.zeros((self._num_envs, self._num_bodies), device=self.device, dtype=torch.long)
        self._foot_pos_w = torch.zeros((self._num_envs, self._num_bodies, 3), device=self.device, dtype=torch.float32)

    def _initialize_helpers(self) -> None:
        foot_normal = torch.tensor(self.cfg.foot_local_normal, device=self.device, dtype=torch.float32)
        normal_norm = torch.norm(foot_normal).clamp_min(1e-6)
        self._foot_local_normal = foot_normal / normal_norm

        gravity_dir = torch.tensor(self.cfg.gravity_dir_w, device=self.device, dtype=torch.float32)
        gravity_norm = torch.norm(gravity_dir).clamp_min(1e-6)
        self._gravity_dir_w = gravity_dir / gravity_norm
        self._up_dir_w = -self._gravity_dir_w

        self._support_mesh_wp = None
        self._tactile_sensor = None
        self._tactile_body_ids: tuple[int, ...] = tuple(range(self._num_bodies))

        self._forefoot_mask = torch.zeros((self._num_bodies, 1), device=self.device, dtype=torch.bool)
        self._cop_ap_scale = torch.ones((self._num_bodies,), device=self.device, dtype=torch.float32)
        self._geometry_ready = False

    def _update_buffers_impl(self, env_ids: Union[Sequence[int], slice]):
        if not isinstance(env_ids, slice) and len(env_ids) == self._num_envs:
            env_ids = slice(None)
        if not self.cfg.enable:
            return

        self._refresh_body_kinematics(env_ids)
        self._refresh_h_eff(env_ids)
        self._refresh_tactile_statistics(env_ids)
        self._refresh_contact_events(env_ids)
        self._refresh_stage_state(env_ids)
        self._run_self_checks(env_ids)

    def _refresh_body_kinematics(self, env_ids: Union[Sequence[int], slice]) -> None:
        body_poses = self.body_physx_view.get_transforms().view(-1, self.num_bodies, 7)[env_ids]
        body_vels = self.body_physx_view.get_velocities().view(-1, self.num_bodies, 6)[env_ids]

        foot_pos_w = body_poses[..., :3]
        foot_quat_w = math_utils.convert_quat(body_poses[..., 3:], to="wxyz")
        foot_vel_w = body_vels[..., :3]

        foot_normal_w = math_utils.quat_apply(
            foot_quat_w,
            self._foot_local_normal.view(1, 1, 3).expand_as(foot_vel_w),
        )

        self._foot_pos_w[env_ids] = foot_pos_w
        self._data.foot_vz[env_ids] = torch.sum(foot_vel_w * self._up_dir_w.view(1, 1, 3), dim=-1)

        normal_alignment_up = torch.sum(foot_normal_w * self._up_dir_w.view(1, 1, 3), dim=-1).clamp(-1.0, 1.0)
        self._data.foot_theta[env_ids] = torch.acos(normal_alignment_up)

        self._data.foot_vz[env_ids] = torch.nan_to_num(self._data.foot_vz[env_ids], nan=0.0, posinf=0.0, neginf=0.0)
        self._data.foot_theta[env_ids] = torch.nan_to_num(self._data.foot_theta[env_ids], nan=0.0, posinf=0.0, neginf=0.0)

    def _refresh_h_eff(self, env_ids: Union[Sequence[int], slice]) -> None:
        # `tensor[env_ids]` is a copy when `env_ids` is a tensor subset. Use local buffers and write them
        # back explicitly so partial sensor refreshes after reset do not silently drop updates.
        h_eff = torch.full_like(self._data.h_eff[env_ids], float(self.cfg.h_eff_max_dist))

        if self._support_mesh_wp is None:
            self._data.h_eff[env_ids] = h_eff
            return

        foot_pos_w = self._foot_pos_w[env_ids]
        offset = float(self.cfg.h_eff_raycast_offset)
        max_dist = float(self.cfg.h_eff_max_dist)

        ray_starts = foot_pos_w - self._gravity_dir_w.view(1, 1, 3) * offset
        ray_dirs = self._gravity_dir_w.view(1, 1, 3).expand_as(ray_starts)

        E, B, _ = ray_starts.shape
        ray_starts_f = ray_starts.reshape(E * B, 1, 3)
        ray_dirs_f = ray_dirs.reshape(E * B, 1, 3)

        _, ray_dist_f, _, _ = raycast_mesh(
            ray_starts_f,
            ray_dirs_f,
            mesh=self._support_mesh_wp,
            max_dist=max_dist + offset,
            return_distance=True,
            return_normal=False,
        )
        if ray_dist_f is None:
            self._data.h_eff[env_ids] = h_eff
            return

        ray_dist = ray_dist_f.reshape(E, B)
        valid_hits = torch.isfinite(ray_dist)
        h_local = torch.clamp(ray_dist - offset, min=0.0, max=max_dist)
        h_eff[valid_hits] = h_local[valid_hits]
        h_eff[:] = torch.nan_to_num(h_eff, nan=max_dist, posinf=max_dist, neginf=max_dist)
        self._data.h_eff[env_ids] = h_eff

    def _refresh_tactile_statistics(self, env_ids: Union[Sequence[int], slice]) -> None:
        total_force = torch.zeros_like(self._data.total_force[env_ids])
        total_force_filt_prev = self._data.total_force_filt[env_ids]
        contact_area = torch.zeros_like(self._data.contact_area[env_ids])
        contact_area_filt_prev = self._data.contact_area_filt[env_ids]
        cop_ap = torch.zeros_like(self._data.cop_ap[env_ids])
        rho_peak = torch.zeros_like(self._data.rho_peak[env_ids])
        rho_fore = torch.zeros_like(self._data.rho_fore[env_ids])

        if self._tactile_sensor is None:
            self._data.total_force[env_ids] = total_force
            self._data.total_force_filt[env_ids] = 0.0
            self._data.contact_area[env_ids] = contact_area
            self._data.contact_area_filt[env_ids] = 0.0
            self._data.cop_ap[env_ids] = cop_ap
            self._data.rho_peak[env_ids] = rho_peak
            self._data.rho_fore[env_ids] = rho_fore
            self._data.dF[env_ids] = 0.0
            self._data.dA[env_ids] = 0.0
            self._filter_initialized[env_ids] = False
            return

        self._ensure_geometry_from_tactile()

        tactile_data = self._tactile_sensor.data
        body_ids = list(self._tactile_body_ids)

        measured_force = tactile_data.taxel_force[env_ids][:, body_ids, :]
        valid_taxel_mask = tactile_data.valid_taxel_mask[body_ids]
        valid_mask = valid_taxel_mask.unsqueeze(0).to(dtype=measured_force.dtype)
        measured_force = torch.nan_to_num(measured_force, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0) * valid_mask

        total_force[:] = measured_force.sum(dim=-1)
        contact_area[:] = torch.nan_to_num(
            tactile_data.contact_area_ratio[env_ids][:, body_ids],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        cop_b = torch.nan_to_num(tactile_data.cop_b[env_ids][:, body_ids, :], nan=0.0, posinf=0.0, neginf=0.0)
        cop_ap[:] = cop_b[..., int(self.cfg.cop_ap_axis)] / self._cop_ap_scale.view(1, -1)

        peak_force = torch.nan_to_num(tactile_data.peak_force[env_ids][:, body_ids], nan=0.0, posinf=0.0, neginf=0.0)
        rho_peak[:] = peak_force / (total_force + float(self.cfg.eps))

        fore_mask = self._forefoot_mask.to(dtype=measured_force.dtype).unsqueeze(0)
        fore_force = (measured_force * fore_mask).sum(dim=-1)
        rho_fore[:] = fore_force / (total_force + float(self.cfg.eps))

        force_raw = torch.nan_to_num(total_force, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        area_raw = torch.nan_to_num(contact_area, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)

        prev_force_filt = total_force_filt_prev.clone()
        prev_area_filt = contact_area_filt_prev.clone()
        initialized = self._filter_initialized[env_ids]

        alpha = float(min(max(self.cfg.derivative_filter_alpha, 0.0), 1.0))
        force_filt_candidate = alpha * force_raw + (1.0 - alpha) * prev_force_filt
        area_filt_candidate = alpha * area_raw + (1.0 - alpha) * prev_area_filt
        force_filt = torch.where(initialized, force_filt_candidate, force_raw)
        area_filt = torch.where(initialized, area_filt_candidate, area_raw)

        dt = self._effective_dt()
        d_force = (force_filt - prev_force_filt) / dt
        d_area = (area_filt - prev_area_filt) / dt
        d_force = torch.where(initialized, d_force, torch.zeros_like(d_force))
        d_area = torch.where(initialized, d_area, torch.zeros_like(d_area))

        total_force_filt = force_filt
        contact_area_filt = area_filt
        self._data.total_force[env_ids] = torch.nan_to_num(total_force, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        self._data.total_force_filt[env_ids] = torch.nan_to_num(total_force_filt, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        self._data.contact_area[env_ids] = torch.nan_to_num(contact_area, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
        self._data.contact_area_filt[env_ids] = torch.nan_to_num(contact_area_filt, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
        self._data.cop_ap[env_ids] = torch.nan_to_num(cop_ap, nan=0.0, posinf=0.0, neginf=0.0)
        self._data.rho_peak[env_ids] = torch.nan_to_num(rho_peak, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        self._data.rho_fore[env_ids] = torch.nan_to_num(rho_fore, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        self._data.dF[env_ids] = d_force
        self._data.dA[env_ids] = d_area
        self._filter_initialized[env_ids] = True
        self._data.dF[env_ids] = torch.nan_to_num(self._data.dF[env_ids], nan=0.0, posinf=0.0, neginf=0.0)
        self._data.dA[env_ids] = torch.nan_to_num(self._data.dA[env_ids], nan=0.0, posinf=0.0, neginf=0.0)

    def _refresh_contact_events(self, env_ids: Union[Sequence[int], slice]) -> None:
        force = self._data.total_force[env_ids]
        area = self._data.contact_area[env_ids]

        active = self._data.contact_active[env_ids]
        on_counter = self._contact_on_counter[env_ids]
        off_counter = self._contact_off_counter[env_ids]
        landing_window = self._data.landing_window[env_ids]

        on_cond = (force > float(self.cfg.contact_force_on)) & (area > float(self.cfg.contact_area_on))
        off_cond = (force < float(self.cfg.contact_force_off)) & (area < float(self.cfg.contact_area_off))

        on_counter = torch.where((~active) & on_cond, on_counter + 1, torch.zeros_like(on_counter))
        off_counter = torch.where(active & off_cond, off_counter + 1, torch.zeros_like(off_counter))

        on_event = (~active) & (on_counter >= int(self.cfg.contact_on_frames))
        off_event = active & (off_counter >= int(self.cfg.contact_off_frames))

        active = torch.where(on_event, torch.ones_like(active, dtype=torch.bool), active)
        active = torch.where(off_event, torch.zeros_like(active, dtype=torch.bool), active)

        on_counter = torch.where(on_event, torch.zeros_like(on_counter), on_counter)
        off_counter = torch.where(off_event, torch.zeros_like(off_counter), off_counter)

        landing_window = torch.clamp(landing_window - 1, min=0)
        landing_window = torch.where(
            on_event,
            torch.full_like(landing_window, int(self.cfg.landing_window_frames)),
            landing_window,
        )

        dt = self._effective_dt()
        tau_on = self._data.tau_on[env_ids] + dt
        tau_off = self._data.tau_off[env_ids] + dt
        tau_on = torch.where(on_event, torch.zeros_like(tau_on), tau_on)
        tau_off = torch.where(off_event, torch.zeros_like(tau_off), tau_off)

        self._data.contact_active[env_ids] = active
        self._data.contact_on_event[env_ids] = on_event
        self._data.contact_off_event[env_ids] = off_event
        self._data.landing_window[env_ids] = landing_window
        self._data.tau_on[env_ids] = tau_on
        self._data.tau_off[env_ids] = tau_off
        self._contact_on_counter[env_ids] = on_counter
        self._contact_off_counter[env_ids] = off_counter

    def _refresh_stage_state(self, env_ids: Union[Sequence[int], slice]) -> None:
        h_eff = self._data.h_eff[env_ids]
        foot_vz = self._data.foot_vz[env_ids]
        contact_active = self._data.contact_active[env_ids]
        contact_on_event = self._data.contact_on_event[env_ids]
        landing_window = self._data.landing_window[env_ids]

        # Complete hard partition in 4 stages:
        # - no contact: Swing / PreLanding
        # - in contact: Landing / Stance
        C = contact_active
        W = contact_on_event | (landing_window > 0)
        N = h_eff < float(self.cfg.h_zone)
        D = foot_vz < -float(self.cfg.v_pre)

        E_pre = (~C) & N & D
        E_sw = (~C) & (~E_pre)
        E_land = C & W
        E_st = C & (~E_land)

        stage_mask = torch.stack((E_sw, E_pre, E_land, E_st), dim=-1)
        stage_eligibility = stage_mask.to(dtype=h_eff.dtype)
        dominant_stage_new = torch.argmax(stage_mask.to(dtype=torch.int64), dim=-1)
        dominant_stage_prev = self._data.dominant_stage_id[env_ids]

        dt = self._effective_dt()
        tau_stage_prev = self._data.tau_stage[env_ids]
        tau_stage = torch.where(
            dominant_stage_new == dominant_stage_prev,
            tau_stage_prev + dt,
            torch.zeros_like(tau_stage_prev),
        )

        self._data.stage_eligibility[env_ids] = stage_eligibility
        self._data.dominant_stage_id[env_ids] = dominant_stage_new
        self._data.tau_stage[env_ids] = tau_stage
        self._debug_pre_core[env_ids] = E_pre

    def _run_self_checks(self, env_ids: Union[Sequence[int], slice]) -> None:
        if not self.cfg.enable_self_check:
            return

        force_filt = self._data.total_force_filt[env_ids]
        area_filt = self._data.contact_area_filt[env_ids]
        d_force = self._data.dF[env_ids]
        d_area = self._data.dA[env_ids]
        h_eff = self._data.h_eff[env_ids]
        foot_vz = self._data.foot_vz[env_ids]
        contact_active = self._data.contact_active[env_ids]
        eligibility = self._data.stage_eligibility[env_ids]
        dominant = self._data.dominant_stage_id[env_ids]
        on_event = self._data.contact_on_event[env_ids]
        off_event = self._data.contact_off_event[env_ids]
        landing_window = self._data.landing_window[env_ids]
        tau_on = self._data.tau_on[env_ids]
        tau_off = self._data.tau_off[env_ids]
        tau_stage = self._data.tau_stage[env_ids]

        if not torch.isfinite(force_filt).all() or not torch.isfinite(area_filt).all():
            raise RuntimeError("ContactStageFilter filtered force/area contains NaN/inf")
        if not torch.isfinite(d_force).all() or not torch.isfinite(d_area).all():
            raise RuntimeError("ContactStageFilter dF/dA contains NaN/inf")

        if eligibility.shape[-1] != self.NUM_STAGES:
            raise RuntimeError("ContactStageFilter stage tensor shape mismatch with NUM_STAGES")

        eligibility_bool = eligibility > 0.5
        eligible_count = eligibility_bool.to(dtype=torch.int64).sum(dim=-1)
        if torch.any(eligible_count == 0):
            raise RuntimeError("ContactStageFilter stage partition contains all-zero rows")
        if torch.any(eligible_count > 1):
            raise RuntimeError("ContactStageFilter stage partition contains overlapping stages")
        if torch.any(eligible_count != 1):
            raise RuntimeError("ContactStageFilter stage partition is not one-hot")

        E_sw = eligibility_bool[..., self.STAGE_SWING]
        E_pre = eligibility_bool[..., self.STAGE_PRELANDING]
        E_land = eligibility_bool[..., self.STAGE_LANDING]
        E_st = eligibility_bool[..., self.STAGE_STANCE]

        expected_pre = (~contact_active) & (h_eff < float(self.cfg.h_zone)) & (foot_vz < -float(self.cfg.v_pre))
        expected_land = contact_active & (on_event | (landing_window > 0))
        if torch.any(E_pre != expected_pre):
            raise RuntimeError("ContactStageFilter PreLanding hard partition mismatch")
        if torch.any(E_land != expected_land):
            raise RuntimeError("ContactStageFilter Landing hard partition mismatch")
        if torch.any((~contact_active) != (E_sw | E_pre)):
            raise RuntimeError("ContactStageFilter no-contact branch is not partitioned by Swing/PreLanding")
        if torch.any(contact_active != (E_land | E_st)):
            raise RuntimeError("ContactStageFilter contact branch is not partitioned by Landing/Stance")

        dominant_from_mask = torch.argmax(eligibility_bool.to(dtype=torch.int64), dim=-1)
        if not torch.equal(dominant, dominant_from_mask):
            raise RuntimeError("ContactStageFilter dominant_stage_id must match hard partition")

        if torch.any(on_event & off_event):
            raise RuntimeError("ContactStageFilter has simultaneous contact_on_event and contact_off_event")

        if torch.any(on_event & (landing_window <= 0)):
            raise RuntimeError("ContactStageFilter landing_window must be positive on contact_on_event")
        if torch.any(landing_window < 0):
            raise RuntimeError("ContactStageFilter landing_window must be non-negative")
        if torch.any(landing_window > int(self.cfg.landing_window_frames)):
            raise RuntimeError("ContactStageFilter landing_window exceeded configured maximum")

        if torch.any(tau_on < 0.0) or torch.any(tau_off < 0.0) or torch.any(tau_stage < 0.0):
            raise RuntimeError("ContactStageFilter tau values must be non-negative")

    def _ensure_geometry_from_tactile(self) -> None:
        if self._geometry_ready or self._tactile_sensor is None:
            return

        tactile_data = self._tactile_sensor.data
        body_ids = list(self._tactile_body_ids)

        taxel_xy_b = tactile_data.taxel_xy_b[body_ids]
        valid_taxel_mask = tactile_data.valid_taxel_mask[body_ids]

        ap_axis = int(self.cfg.cop_ap_axis)
        ap_coord = taxel_xy_b[..., ap_axis]

        fore_mask = (ap_coord >= float(self.cfg.forefoot_split)) & valid_taxel_mask
        has_fore = fore_mask.any(dim=-1, keepdim=True)
        fore_mask = torch.where(has_fore, fore_mask, valid_taxel_mask)

        ap_abs = torch.abs(ap_coord) * valid_taxel_mask.float()
        cop_scale = ap_abs.max(dim=-1).values.clamp_min(1e-3)

        self._forefoot_mask = fore_mask
        self._cop_ap_scale = cop_scale
        self._geometry_ready = True

    def _effective_dt(self) -> float:
        if self.cfg.update_period > 0.0:
            return float(self.cfg.update_period)
        return float(self._sim_physics_dt)

    def _resolve_env_ids_tensor(self, env_ids: Union[Sequence[int], slice, torch.Tensor, None]) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self._num_envs, device=self.device, dtype=torch.long)
        if isinstance(env_ids, slice):
            return torch.arange(self._num_envs, device=self.device, dtype=torch.long)[env_ids]
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long).flatten()
        return torch.as_tensor(list(env_ids), device=self.device, dtype=torch.long).flatten()

    def _set_debug_vis_impl(self, debug_vis: bool):
        del debug_vis

    def _debug_vis_callback(self, event):
        del event

    def _invalidate_initialize_callback(self, event):
        super()._invalidate_initialize_callback(event)
        self._physics_sim_view = None
        self._body_physx_view = None
        self._support_mesh_wp = None
        self._tactile_sensor = None
        self._filter_initialized = None
        self._debug_pre_core = None
