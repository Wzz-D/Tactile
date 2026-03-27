from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn.functional as torch_f

import omni.physics.tensors.impl.api as physx

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.sensors.sensor_base import SensorBase
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh

from .contact_stage_data import ContactStageData


class ContactStageFilter(SensorBase):
    """Independent per-foot 5-stage contact filter driven by tactile + kinematics."""

    STAGE_SWING = 0
    STAGE_PRELANDING = 1
    STAGE_LANDING = 2
    STAGE_STANCE = 3
    STAGE_PUSHOFF = 4
    NUM_STAGES = 5

    STAGE_NAMES: tuple[str, ...] = (
        "Swing",
        "PreLanding",
        "Landing",
        "Stance",
        "PushOff",
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

    def get_stage_debug_string(self, env_id: int, foot_id: int) -> str:
        """Return compact per-foot debug info for stage filtering."""
        self._update_outdated_buffers()

        stage_id = int(self._data.dominant_stage_id[env_id, foot_id].item())
        stage_name = self.stage_id_to_name(stage_id)

        eligibility = self._data.stage_eligibility[env_id, foot_id].detach().cpu().tolist()
        quality = self._data.stage_quality[env_id, foot_id].detach().cpu().tolist()
        scores = self._data.stage_scores[env_id, foot_id].detach().cpu().tolist()
        belief = self._data.stage_belief[env_id, foot_id].detach().cpu().tolist()
        confidence = float(self._data.stage_confidence[env_id, foot_id].item())

        contact_active = bool(self._data.contact_active[env_id, foot_id].item())
        landing_window = int(self._data.landing_window[env_id, foot_id].item())
        land_trigger = bool(self._data.land_trigger[env_id, foot_id].item())
        land_maintain = bool(self._data.land_maintain[env_id, foot_id].item())
        land_active = bool(self._data.land_active[env_id, foot_id].item())
        stance_landing_scale = float(self._data.stance_during_landing_scale[env_id, foot_id].item())
        if land_trigger:
            land_mode = "trigger"
        elif land_maintain:
            land_mode = "maintain"
        else:
            land_mode = "none"
        tau_on = float(self._data.tau_on[env_id, foot_id].item())
        tau_off = float(self._data.tau_off[env_id, foot_id].item())
        tau_stage = float(self._data.tau_stage[env_id, foot_id].item())

        h_eff = float(self._data.h_eff[env_id, foot_id].item())
        vz = float(self._data.foot_vz[env_id, foot_id].item())
        force = float(self._data.total_force[env_id, foot_id].item())
        force_filt = float(self._data.total_force_filt[env_id, foot_id].item())
        area = float(self._data.contact_area[env_id, foot_id].item())
        area_filt = float(self._data.contact_area_filt[env_id, foot_id].item())
        d_force = float(self._data.dF[env_id, foot_id].item())
        d_area = float(self._data.dA[env_id, foot_id].item())
        rho_fore = float(self._data.rho_fore[env_id, foot_id].item())
        pre_core = bool(self._debug_pre_core[env_id, foot_id].item())

        return (
            f"ContactStage(env={env_id}, foot={foot_id}, stage={stage_name}, conf={confidence:.3f}, "
            f"contact_active={contact_active}, landing_window={landing_window}, tau_on={tau_on:.3f}, "
            f"tau_off={tau_off:.3f}, tau_stage={tau_stage:.3f}, h_eff={h_eff:.3f}, vz={vz:.3f}, "
            f"F={force:.3f}, Ff={force_filt:.3f}, A={area:.3f}, Af={area_filt:.3f}, "
            f"dF={d_force:.3f}, dA={d_area:.3f}, rho_fore={rho_fore:.3f}, pre_core={pre_core}, "
            f"E_land_trigger={land_trigger}, E_land_maintain={land_maintain}, E_land={land_active}, "
            f"landing_mode={land_mode}, stance_during_landing_scale={stance_landing_scale:.3f}, "
            f"E={[round(v, 3) for v in eligibility]}, Q={[round(v, 3) for v in quality]}, "
            f"scores={[round(v, 4) for v in scores]}, belief={[round(v, 4) for v in belief]})"
        )

    def find_bodies(self, name_keys: Union[str, Sequence[str]], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        return string_utils.resolve_matching_names(name_keys, self.body_names, preserve_order)

    def bind_tactile_sensor(self, tactile_sensor: Any) -> None:
        self._tactile_sensor = tactile_sensor
        self._tactile_body_ids = tuple(range(self.num_bodies))
        if hasattr(tactile_sensor, "find_bodies") and hasattr(tactile_sensor, "body_names"):
            ids, _ = tactile_sensor.find_bodies(self.body_names, preserve_order=True)
            if len(ids) == self.num_bodies:
                self._tactile_body_ids = tuple(int(body_id) for body_id in ids)
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
        self._data.land_trigger[env_ids_t] = False
        self._data.land_maintain[env_ids_t] = False
        self._data.land_active[env_ids_t] = False
        self._data.stance_during_landing_scale[env_ids_t] = 1.0

        self._data.stage_eligibility[env_ids_t] = 0.0
        self._data.stage_quality[env_ids_t] = 0.0
        self._data.stage_scores[env_ids_t] = 0.0
        self._data.stage_likelihood[env_ids_t] = 1.0 / float(self.NUM_STAGES)
        self._data.stage_prior[env_ids_t] = 1.0 / float(self.NUM_STAGES)
        self._data.stage_belief[env_ids_t] = 1.0 / float(self.NUM_STAGES)
        self._data.stage_confidence[env_ids_t] = 0.0
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

        self._predecessor_index = torch.tensor(
            [self.STAGE_PUSHOFF, self.STAGE_SWING, self.STAGE_PRELANDING, self.STAGE_LANDING, self.STAGE_STANCE],
            device=self.device,
            dtype=torch.long,
        )
        self._lambda_self = torch.tensor(
            [
                self.cfg.lambda_self_sw,
                self.cfg.lambda_self_pre,
                self.cfg.lambda_self_land,
                self.cfg.lambda_self_st,
                self.cfg.lambda_self_push,
            ],
            device=self.device,
            dtype=torch.float32,
        )
        self._lambda_prev = torch.tensor(
            [
                self.cfg.lambda_prev_sw,
                self.cfg.lambda_prev_pre,
                self.cfg.lambda_prev_land,
                self.cfg.lambda_prev_st,
                self.cfg.lambda_prev_push,
            ],
            device=self.device,
            dtype=torch.float32,
        )
        self._min_stage_duration = torch.tensor(
            [
                self.cfg.min_stage_duration_sw,
                self.cfg.min_stage_duration_pre,
                self.cfg.min_stage_duration_land,
                self.cfg.min_stage_duration_st,
                self.cfg.min_stage_duration_push,
            ],
            device=self.device,
            dtype=torch.float32,
        )

    def _update_buffers_impl(self, env_ids: Union[Sequence[int], slice]):
        if not isinstance(env_ids, slice) and len(env_ids) == self._num_envs:
            env_ids = slice(None)
        if not self.cfg.enable:
            return

        self._refresh_body_kinematics(env_ids)
        self._refresh_h_eff(env_ids)
        self._refresh_tactile_statistics(env_ids)
        self._refresh_contact_events(env_ids)
        self._refresh_stage_belief(env_ids)
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
        h_eff = self._data.h_eff[env_ids]
        h_eff.fill_(float(self.cfg.h_eff_max_dist))

        if self._support_mesh_wp is None:
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
            return

        ray_dist = ray_dist_f.reshape(E, B)
        valid_hits = torch.isfinite(ray_dist)
        h_local = torch.clamp(ray_dist - offset, min=0.0, max=max_dist)
        h_eff[valid_hits] = h_local[valid_hits]
        h_eff[:] = torch.nan_to_num(h_eff, nan=max_dist, posinf=max_dist, neginf=max_dist)

    def _refresh_tactile_statistics(self, env_ids: Union[Sequence[int], slice]) -> None:
        total_force = self._data.total_force[env_ids]
        total_force_filt = self._data.total_force_filt[env_ids]
        contact_area = self._data.contact_area[env_ids]
        contact_area_filt = self._data.contact_area_filt[env_ids]
        cop_ap = self._data.cop_ap[env_ids]
        rho_peak = self._data.rho_peak[env_ids]
        rho_fore = self._data.rho_fore[env_ids]

        total_force.zero_()
        contact_area.zero_()
        cop_ap.zero_()
        rho_peak.zero_()
        rho_fore.zero_()

        if self._tactile_sensor is None:
            total_force_filt.zero_()
            contact_area_filt.zero_()
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

        prev_force_filt = total_force_filt.clone()
        prev_area_filt = contact_area_filt.clone()
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

        total_force_filt[:] = force_filt
        contact_area_filt[:] = area_filt
        self._data.dF[env_ids] = d_force
        self._data.dA[env_ids] = d_area
        self._filter_initialized[env_ids] = True

        self._data.total_force[env_ids] = torch.nan_to_num(self._data.total_force[env_ids], nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        self._data.total_force_filt[env_ids] = torch.nan_to_num(self._data.total_force_filt[env_ids], nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        self._data.contact_area[env_ids] = torch.nan_to_num(self._data.contact_area[env_ids], nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
        self._data.contact_area_filt[env_ids] = torch.nan_to_num(self._data.contact_area_filt[env_ids], nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
        self._data.dF[env_ids] = torch.nan_to_num(self._data.dF[env_ids], nan=0.0, posinf=0.0, neginf=0.0)
        self._data.dA[env_ids] = torch.nan_to_num(self._data.dA[env_ids], nan=0.0, posinf=0.0, neginf=0.0)
        self._data.cop_ap[env_ids] = torch.nan_to_num(self._data.cop_ap[env_ids], nan=0.0, posinf=0.0, neginf=0.0)
        self._data.rho_peak[env_ids] = torch.nan_to_num(self._data.rho_peak[env_ids], nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        self._data.rho_fore[env_ids] = torch.nan_to_num(self._data.rho_fore[env_ids], nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)

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

    def _refresh_stage_belief(self, env_ids: Union[Sequence[int], slice]) -> None:
        force = self._data.total_force[env_ids]
        d_force = self._data.dF[env_ids]
        area = self._data.contact_area[env_ids]
        d_area = self._data.dA[env_ids]
        h_eff = self._data.h_eff[env_ids]
        foot_vz = self._data.foot_vz[env_ids]
        rho_fore = self._data.rho_fore[env_ids]
        contact_active = self._data.contact_active[env_ids]
        contact_on_event = self._data.contact_on_event[env_ids]
        landing_window = self._data.landing_window[env_ids]

        pre_core = (
            (force < float(self.cfg.F_pre))
            & (area < float(self.cfg.A_pre))
            & (~contact_active)
            & (h_eff < float(self.cfg.h_pre))
            & (foot_vz < -float(self.cfg.v_pre))
        )
        swing_base = (force < float(self.cfg.F_sw)) & (area < float(self.cfg.A_sw)) & (~contact_active)
        if self.cfg.swing_prelanding_yield_mode == "hard":
            E_sw = swing_base & (~pre_core)
            swing_soft_scale = torch.ones_like(force)
        else:
            E_sw = swing_base
            soft_scale = float(self.cfg.swing_prelanding_soft_scale)
            swing_soft_scale = torch.where(pre_core, torch.full_like(force, soft_scale), torch.ones_like(force))
        E_pre = pre_core

        # Landing now uses trigger + maintain:
        # - trigger keeps the strong impact semantics (F/A/dF/h + contact-on or active landing window gate).
        # - maintain keeps landing alive for a short window with softer F/A thresholds.
        E_land_trigger = (
            (force > float(self.cfg.F_land))
            & (area > float(self.cfg.A_land))
            & (d_force > float(self.cfg.dF_land))
            & (contact_on_event | (landing_window > 0))
            & (h_eff < float(self.cfg.h_land))
        )
        landing_trigger_start = E_land_trigger & (landing_window <= 0)
        landing_window = torch.where(
            landing_trigger_start,
            torch.full_like(landing_window, int(self.cfg.landing_window_frames)),
            landing_window,
        )
        self._data.landing_window[env_ids] = landing_window

        E_land_maintain = (
            (landing_window > 0)
            & contact_active
            & (force > float(self.cfg.F_land_maintain))
            & (area > float(self.cfg.A_land_maintain))
            & (h_eff < float(self.cfg.h_land))
        )
        E_land = E_land_trigger | E_land_maintain
        E_st = (
            (force > float(self.cfg.F_st))
            & (area > float(self.cfg.A_st))
            & contact_active
            & (h_eff < float(self.cfg.h_st))
        )
        E_push = (
            (force > float(self.cfg.F_push))
            & (area > float(self.cfg.A_push))
            & contact_active
            & (h_eff < float(self.cfg.h_push))
            & (d_force < -float(self.cfg.dF_push))
            & (d_area < -float(self.cfg.dA_push))
        )

        Q_sw = self._H(h_eff, self.cfg.h_sw, self.cfg.s_h) * swing_soft_scale
        Q_pre = torch.ones_like(force)
        Q_land = self._H(d_force, self.cfg.dF_land, self.cfg.s_dF)
        Q_st = (
            self._H(force, self.cfg.F_st, self.cfg.s_F)
            + self._H(area, self.cfg.A_st, self.cfg.s_A)
            + self._L(torch.abs(d_force), self.cfg.dF_st, self.cfg.s_dF)
        ) / 3.0
        landing_window_active = landing_window > 0
        stance_landing_scale = torch.where(
            landing_window_active,
            torch.full_like(force, float(self.cfg.stance_during_landing_scale)),
            torch.ones_like(force),
        )
        Q_st = Q_st * stance_landing_scale
        # Strengthen rho_fore dependency without adding hard gate:
        # low forefoot ratio gets suppressed more aggressively.
        Q_push_base = self._H(rho_fore, self.cfg.rho_fore0, self.cfg.s_rho)
        Q_push = Q_push_base * Q_push_base

        E_float = torch.stack((E_sw, E_pre, E_land, E_st, E_push), dim=-1).to(dtype=force.dtype)
        Q = torch.stack((Q_sw, Q_pre, Q_land, Q_st, Q_push), dim=-1)

        Q = torch.nan_to_num(Q, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
        stage_scores = E_float * Q
        stage_scores = torch.nan_to_num(stage_scores, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)

        belief_prev = self._data.stage_belief[env_ids]
        stage_prior = self._build_stage_prior(belief_prev)

        likelihood_from_score = torch.softmax(float(self.cfg.beta_score) * stage_scores, dim=-1)
        likelihood_from_score = torch.nan_to_num(
            likelihood_from_score,
            nan=1.0 / float(self.NUM_STAGES),
            posinf=0.0,
            neginf=0.0,
        )
        likelihood_from_score = self._normalize_prob(likelihood_from_score)

        has_stage_signal = (stage_scores > 0.0).any(dim=-1, keepdim=True)
        stage_likelihood = torch.where(has_stage_signal, likelihood_from_score, stage_prior)
        stage_likelihood = self._normalize_prob(stage_likelihood)

        belief_raw = self._normalize_prob(stage_likelihood * stage_prior)
        if self.cfg.enable_min_stage_duration:
            belief_raw = self._apply_min_stage_duration(env_ids, belief_raw)

        ema_stage = float(min(max(self.cfg.ema_stage, 0.0), 1.0))
        belief = (1.0 - ema_stage) * belief_prev + ema_stage * belief_raw
        belief = self._normalize_prob(belief)

        dominant_stage_new = torch.argmax(belief, dim=-1)
        dominant_stage_prev = self._data.dominant_stage_id[env_ids]

        dt = self._effective_dt()
        tau_stage_prev = self._data.tau_stage[env_ids]
        tau_stage = torch.where(
            dominant_stage_new == dominant_stage_prev,
            tau_stage_prev + dt,
            torch.zeros_like(tau_stage_prev),
        )

        entropy = -(belief * torch.log(belief.clamp_min(float(self.cfg.eps)))).sum(dim=-1)
        max_entropy = torch.log(torch.tensor(float(self.NUM_STAGES), device=belief.device, dtype=belief.dtype))
        confidence = 1.0 - entropy / max_entropy
        confidence = torch.nan_to_num(confidence, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        self._data.stage_eligibility[env_ids] = E_float
        self._data.stage_quality[env_ids] = Q
        self._data.stage_scores[env_ids] = stage_scores
        self._data.stage_likelihood[env_ids] = stage_likelihood
        self._data.stage_prior[env_ids] = stage_prior
        self._data.stage_belief[env_ids] = belief
        self._data.stage_confidence[env_ids] = confidence
        self._data.dominant_stage_id[env_ids] = dominant_stage_new
        self._data.tau_stage[env_ids] = tau_stage
        self._data.land_trigger[env_ids] = E_land_trigger
        self._data.land_maintain[env_ids] = E_land_maintain
        self._data.land_active[env_ids] = E_land
        self._data.stance_during_landing_scale[env_ids] = stance_landing_scale
        self._debug_pre_core[env_ids] = pre_core

    def _build_stage_prior(self, belief_prev: torch.Tensor) -> torch.Tensor:
        predecessor_belief = belief_prev[..., self._predecessor_index]
        prior_score = float(self.cfg.eps_prior)
        prior_score = prior_score + self._lambda_self.view(1, 1, -1) * belief_prev
        prior_score = prior_score + self._lambda_prev.view(1, 1, -1) * predecessor_belief
        prior_score = torch.nan_to_num(prior_score, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        return self._normalize_prob(prior_score)

    def _apply_min_stage_duration(self, env_ids: Union[Sequence[int], slice], belief_raw: torch.Tensor) -> torch.Tensor:
        dominant_prev = self._data.dominant_stage_id[env_ids]
        tau_stage = self._data.tau_stage[env_ids]

        min_duration = self._min_stage_duration[dominant_prev]
        valid_duration = min_duration > 0.0
        if not torch.any(valid_duration):
            return belief_raw

        dwell_ratio = ((min_duration - tau_stage) / min_duration.clamp_min(float(self.cfg.eps))).clamp(0.0, 1.0)
        stay_weight = (float(self.cfg.lambda_dwell) * dwell_ratio).clamp(0.0, 1.0)
        stay_weight = torch.where(valid_duration, stay_weight, torch.zeros_like(stay_weight))

        prev_one_hot = torch_f.one_hot(dominant_prev, num_classes=self.NUM_STAGES).to(dtype=belief_raw.dtype)
        belief_dwell = (1.0 - stay_weight.unsqueeze(-1)) * belief_raw + stay_weight.unsqueeze(-1) * prev_one_hot
        return self._normalize_prob(belief_dwell)

    def _run_self_checks(self, env_ids: Union[Sequence[int], slice]) -> None:
        if not self.cfg.enable_self_check:
            return

        force_filt = self._data.total_force_filt[env_ids]
        area_filt = self._data.contact_area_filt[env_ids]
        d_force = self._data.dF[env_ids]
        d_area = self._data.dA[env_ids]
        scores = self._data.stage_scores[env_ids]
        likelihood = self._data.stage_likelihood[env_ids]
        prior = self._data.stage_prior[env_ids]
        belief = self._data.stage_belief[env_ids]
        confidence = self._data.stage_confidence[env_ids]
        dominant = self._data.dominant_stage_id[env_ids]
        on_event = self._data.contact_on_event[env_ids]
        off_event = self._data.contact_off_event[env_ids]
        landing_window = self._data.landing_window[env_ids]
        land_trigger = self._data.land_trigger[env_ids]
        land_maintain = self._data.land_maintain[env_ids]
        land_active = self._data.land_active[env_ids]
        stance_landing_scale = self._data.stance_during_landing_scale[env_ids]
        tau_on = self._data.tau_on[env_ids]
        tau_off = self._data.tau_off[env_ids]
        tau_stage = self._data.tau_stage[env_ids]

        if not torch.isfinite(force_filt).all() or not torch.isfinite(area_filt).all():
            raise RuntimeError("ContactStageFilter filtered force/area contains NaN/inf")
        if not torch.isfinite(d_force).all() or not torch.isfinite(d_area).all():
            raise RuntimeError("ContactStageFilter dF/dA contains NaN/inf")

        if not torch.isfinite(scores).all() or not torch.isfinite(likelihood).all() or not torch.isfinite(prior).all() or not torch.isfinite(belief).all():
            raise RuntimeError("ContactStageFilter has NaN/inf in score/probability tensors")

        belief_sum_error = torch.abs(belief.sum(dim=-1) - 1.0)
        if torch.any(belief_sum_error > float(self.cfg.self_check_tol)):
            raise RuntimeError("ContactStageFilter belief rows are not normalized")

        if torch.any(confidence < -1e-6) or torch.any(confidence > 1.0 + 1e-6):
            raise RuntimeError("ContactStageFilter confidence out of [0, 1]")

        dominant_check = torch.argmax(belief, dim=-1)
        if not torch.equal(dominant, dominant_check):
            raise RuntimeError("ContactStageFilter dominant_stage_id mismatch with argmax(belief)")

        if torch.any(on_event & off_event):
            raise RuntimeError("ContactStageFilter has simultaneous contact_on_event and contact_off_event")

        if torch.any(on_event & (landing_window <= 0)):
            raise RuntimeError("ContactStageFilter landing_window must be positive on contact_on_event")
        if torch.any(landing_window < 0):
            raise RuntimeError("ContactStageFilter landing_window must be non-negative")
        if torch.any(landing_window > int(self.cfg.landing_window_frames)):
            raise RuntimeError("ContactStageFilter landing_window exceeded configured maximum")
        if torch.any(land_active != (land_trigger | land_maintain)):
            raise RuntimeError("ContactStageFilter land_active mismatch with trigger/maintain states")
        if not torch.isfinite(stance_landing_scale).all():
            raise RuntimeError("ContactStageFilter stance_during_landing_scale contains NaN/inf")
        if torch.any(stance_landing_scale <= 0.0):
            raise RuntimeError("ContactStageFilter stance_during_landing_scale must stay positive")

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

    def _normalize_prob(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        denom = tensor.sum(dim=-1, keepdim=True)
        fallback = torch.full_like(tensor, 1.0 / float(self.NUM_STAGES))
        normalized = torch.where(denom > float(self.cfg.eps), tensor / denom, fallback)
        normalized = torch.nan_to_num(normalized, nan=1.0 / float(self.NUM_STAGES), posinf=0.0, neginf=0.0)
        return normalized.clamp_min(0.0)

    @staticmethod
    def _H(x: torch.Tensor, x0: float, s: float) -> torch.Tensor:
        return torch.sigmoid((x - float(x0)) / max(float(s), 1e-6))

    @staticmethod
    def _L(x: torch.Tensor, x0: float, s: float) -> torch.Tensor:
        return 1.0 - ContactStageFilter._H(x, x0, s)

    @staticmethod
    def _M(x: torch.Tensor, mu: float, s: float) -> torch.Tensor:
        sigma = max(float(s), 1e-6)
        return torch.exp(-0.5 * ((x - float(mu)) / sigma) ** 2)

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
