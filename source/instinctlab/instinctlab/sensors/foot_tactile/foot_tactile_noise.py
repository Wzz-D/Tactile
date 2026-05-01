from __future__ import annotations

from collections.abc import Sequence

import torch

from .foot_tactile_noise_cfg import FootTactileNoiseCfg


class FootTactileNoiseModel:
    """Stateful tactile measurement model with legacy and structured sim-to-real profiles."""

    def __init__(
        self,
        cfg: FootTactileNoiseCfg,
        num_envs: int,
        num_bodies: int,
        max_taxels: int,
        device: str | torch.device,
        dtype: torch.dtype,
        base_taxel_xy_b: torch.Tensor | None = None,
        valid_taxel_mask: torch.Tensor | None = None,
        edge_taxel_mask: torch.Tensor | None = None,
        body_sides: Sequence[str] | None = None,
    ):
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.num_bodies = int(num_bodies)
        self.max_taxels = int(max_taxels)
        self.device = torch.device(device)
        self.dtype = dtype

        generator_device = "cuda" if self.device.type == "cuda" else "cpu"
        self._generator = torch.Generator(device=generator_device)
        if self.cfg.seed is not None:
            self._generator.manual_seed(int(self.cfg.seed))

        self._base_taxel_xy_b = torch.zeros((self.num_bodies, self.max_taxels, 2), device=self.device, dtype=self.dtype)
        if base_taxel_xy_b is not None:
            self._base_taxel_xy_b.copy_(base_taxel_xy_b.to(device=self.device, dtype=self.dtype))

        self._valid_taxel_mask = torch.zeros((self.num_bodies, self.max_taxels), device=self.device, dtype=torch.bool)
        if valid_taxel_mask is not None:
            self._valid_taxel_mask.copy_(valid_taxel_mask.to(device=self.device, dtype=torch.bool))

        self._edge_taxel_mask = torch.zeros((self.num_bodies, self.max_taxels), device=self.device, dtype=torch.bool)
        if edge_taxel_mask is not None:
            self._edge_taxel_mask.copy_(edge_taxel_mask.to(device=self.device, dtype=torch.bool))

        self._body_is_left = torch.zeros((self.num_bodies,), device=self.device, dtype=torch.bool)
        if body_sides is not None:
            if len(body_sides) != self.num_bodies:
                raise ValueError(f"Expected {self.num_bodies} body sides, got {len(body_sides)}")
            self._body_is_left[:] = torch.tensor([side == "left" for side in body_sides], device=self.device, dtype=torch.bool)

        self._valid_count = self._valid_taxel_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(dtype=self.dtype)
        self._region_fore_mask, self._region_heel_mask, self._region_medial_mask, self._region_lateral_mask = (
            self._build_region_masks()
        )

        self._history_len = max(int(self.cfg.max_delay_frames), 0) + 1
        self._history_cursor = 0
        self._delay_history = torch.zeros(
            (self.num_envs, self._history_len, self.num_bodies, self.max_taxels),
            device=self.device,
            dtype=self.dtype,
        )
        self._delay_initialized = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self._burst_remaining = torch.zeros((self.num_envs, self.num_bodies), device=self.device, dtype=torch.int64)

        transport_cfg = self.cfg.transport_profile_cfg
        self._structured_history_len = max(int(transport_cfg.delay.max_delay_frames), 0) + 1
        self._structured_history_cursor = 0
        self._structured_delay_history = torch.zeros(
            (self.num_envs, self._structured_history_len, self.num_bodies, self.max_taxels),
            device=self.device,
            dtype=self.dtype,
        )
        self._structured_delay_initialized = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self._frame_hold_prev = torch.zeros((self.num_envs, self.num_bodies, self.max_taxels), device=self.device, dtype=self.dtype)
        self._frame_hold_initialized = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self._hysteresis_state = torch.zeros((self.num_envs, self.num_bodies, self.max_taxels), device=self.device, dtype=self.dtype)
        self._hysteresis_initialized = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self._drift_total = torch.zeros((self.num_envs, self.num_bodies, 1), device=self.device, dtype=self.dtype)

        self._measured_taxel_xy_b = self._base_taxel_xy_b.unsqueeze(0).expand(self.num_envs, -1, -1, -1).clone()
        self._per_foot_gain = torch.ones((self.num_envs, self.num_bodies, 1), device=self.device, dtype=self.dtype)
        self._per_foot_bias_total = torch.zeros((self.num_envs, self.num_bodies, 1), device=self.device, dtype=self.dtype)
        self._regional_gain = torch.ones((self.num_envs, self.num_bodies, self.max_taxels), device=self.device, dtype=self.dtype)
        self._dead_taxel_mask = torch.zeros((self.num_envs, self.num_bodies, self.max_taxels), device=self.device, dtype=torch.bool)
        self._weak_patch_scale = torch.ones((self.num_envs, self.num_bodies, self.max_taxels), device=self.device, dtype=self.dtype)

        self.reset(None)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Reset noise states for selected environments."""

        env_ids_t = self._as_env_ids(env_ids)
        if env_ids_t.numel() == 0:
            return

        self._delay_history[env_ids_t] = 0.0
        self._delay_initialized[env_ids_t] = False
        self._burst_remaining[env_ids_t] = 0

        self._structured_delay_history[env_ids_t] = 0.0
        self._structured_delay_initialized[env_ids_t] = False
        self._frame_hold_prev[env_ids_t] = 0.0
        self._frame_hold_initialized[env_ids_t] = False
        self._hysteresis_state[env_ids_t] = 0.0
        self._hysteresis_initialized[env_ids_t] = False
        self._drift_total[env_ids_t] = 0.0

        self._per_foot_gain[env_ids_t] = 1.0
        self._per_foot_bias_total[env_ids_t] = 0.0
        self._regional_gain[env_ids_t] = 1.0
        self._dead_taxel_mask[env_ids_t] = False
        self._weak_patch_scale[env_ids_t] = 1.0
        self._measured_taxel_xy_b[env_ids_t] = self._base_taxel_xy_b.unsqueeze(0).expand(env_ids_t.numel(), -1, -1, -1)

        if self.cfg.enable and self.cfg.use_structured_profiles:
            self._sample_measurement_profile(env_ids_t)

    def get_measured_taxel_xy_b(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> torch.Tensor:
        env_ids_t = self._as_env_ids(env_ids)
        if env_ids_t.numel() == 0:
            return self._measured_taxel_xy_b[:0]
        return self._measured_taxel_xy_b[env_ids_t]

    def apply(
        self,
        force_diffused: torch.Tensor,
        valid_taxel_mask: torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor,
    ) -> torch.Tensor:
        """Apply tactile measurement noise to diffused force."""

        env_ids_t = self._as_env_ids(env_ids)
        if env_ids_t.numel() == 0:
            return force_diffused

        measured = torch.nan_to_num(force_diffused, nan=0.0, posinf=0.0, neginf=0.0).clone()
        measured = measured.clamp_min(0.0)

        valid = valid_taxel_mask.unsqueeze(0).expand_as(measured)
        valid_f = valid.to(dtype=measured.dtype)
        measured = measured * valid_f

        if not self.cfg.enable:
            return measured

        pre_noise_total = measured.sum(dim=-1, keepdim=True)
        if self.cfg.use_structured_profiles:
            measured = self._apply_structured_profiles(measured, env_ids_t)
        else:
            measured = self._apply_legacy_profiles(measured, valid, env_ids_t)

        measured = torch.nan_to_num(measured, nan=0.0, posinf=0.0, neginf=0.0)
        measured = measured.clamp_min(float(self.cfg.clip_min))
        if self.cfg.clip_max is not None:
            measured = measured.clamp_max(float(self.cfg.clip_max))

        measured = measured * valid_f
        measured = measured.clamp_min(0.0)

        if self.cfg.preserve_total_force_after_noise:
            measured = self._preserve_total_force(measured, pre_noise_total)
            measured = measured * valid_f

        measured = torch.nan_to_num(measured, nan=0.0, posinf=0.0, neginf=0.0)
        measured = measured.clamp_min(0.0)
        measured = measured * valid_f
        return measured

    def _apply_structured_profiles(self, force: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        measured = self._apply_measurement_profile(force, env_ids)
        measured = self._apply_transport_profile(measured, env_ids)
        return measured

    def _apply_measurement_profile(self, force: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        out = force.clone()
        valid_f = self._valid_taxel_mask.unsqueeze(0).to(dtype=out.dtype)
        profile_cfg = self.cfg.measurement_profile_cfg

        out = out * self._per_foot_gain[env_ids]

        if profile_cfg.per_foot_bias.enable:
            per_taxel_bias = self._per_foot_bias_total[env_ids] / self._valid_count.view(1, self.num_bodies, 1)
            out = out + per_taxel_bias * valid_f

        out = out * self._regional_gain[env_ids]
        out = out * self._weak_patch_scale[env_ids]
        out = torch.where(self._dead_taxel_mask[env_ids], torch.zeros_like(out), out)

        if profile_cfg.range_based_force.enable:
            out = self._apply_range_based_noise(out)
        if profile_cfg.residual_gaussian.enable:
            out = self._apply_residual_gaussian(
                out,
                profile_cfg.residual_gaussian.multiplicative_std,
                profile_cfg.residual_gaussian.additive_std,
            )
        if profile_cfg.deadzone.enable and profile_cfg.deadzone.force_threshold > 0.0:
            out = torch.where(out >= float(profile_cfg.deadzone.force_threshold), out, torch.zeros_like(out))
        if profile_cfg.soft_saturation.enable and profile_cfg.soft_saturation.saturation_force > 0.0:
            sat_force = float(profile_cfg.soft_saturation.saturation_force)
            alpha = float(min(max(profile_cfg.soft_saturation.compression_alpha, 0.0), 1.0))
            above = torch.clamp(out - sat_force, min=0.0)
            out = torch.where(out > sat_force, sat_force + alpha * above, out)
        if profile_cfg.quantization.enable and profile_cfg.quantization.quantization_step is not None:
            step = float(profile_cfg.quantization.quantization_step)
            if step > 0.0:
                out = torch.round(out / step) * step

        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        out = out.clamp_min(0.0) * valid_f
        return out

    def _apply_transport_profile(self, force: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        out = force
        transport_cfg = self.cfg.transport_profile_cfg

        if transport_cfg.delay.enable:
            out = self._apply_structured_delay(out, env_ids)
        if transport_cfg.frame_hold.enable:
            out = self._apply_frame_hold(out, env_ids)
        if transport_cfg.hysteresis.enable:
            out = self._apply_hysteresis(out, env_ids)
        if transport_cfg.drift.enable:
            out = self._apply_drift(out, env_ids)
        if transport_cfg.sparse_dropout.enable:
            out = self._apply_sparse_transport_dropout(out)

        valid_f = self._valid_taxel_mask.unsqueeze(0).to(dtype=out.dtype)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0) * valid_f

    def _apply_range_based_noise(self, force: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg.measurement_profile_cfg.range_based_force
        out = force
        low_thr = float(cfg.low_force_threshold)
        high_thr = float(max(cfg.high_force_threshold, low_thr))

        low_mask = force < low_thr
        mid_mask = (force >= low_thr) & (force < high_thr)
        high_mask = force >= high_thr
        noise = torch.zeros_like(force)

        if cfg.low_force_multiplicative_std > 0.0:
            noise = torch.where(
                low_mask,
                self._randn(force.shape, dtype=force.dtype) * float(cfg.low_force_multiplicative_std),
                noise,
            )
        if cfg.mid_force_multiplicative_std > 0.0:
            mid_noise = self._randn(force.shape, dtype=force.dtype) * float(cfg.mid_force_multiplicative_std)
            noise = torch.where(mid_mask, mid_noise, noise)
        if cfg.high_force_multiplicative_std > 0.0:
            high_noise = self._randn(force.shape, dtype=force.dtype) * float(cfg.high_force_multiplicative_std)
            noise = torch.where(high_mask, high_noise, noise)
        out = out * (1.0 + noise)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def _apply_residual_gaussian(self, force: torch.Tensor, multiplicative_std: float, additive_std: float) -> torch.Tensor:
        out = force
        if multiplicative_std > 0.0:
            mult_noise = self._randn(force.shape, dtype=force.dtype) * float(multiplicative_std)
            out = out * (1.0 + mult_noise)
        if additive_std > 0.0:
            out = out + self._randn(force.shape, dtype=force.dtype) * float(additive_std)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def _apply_structured_delay(self, force: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg.transport_profile_cfg.delay
        max_delay = max(int(cfg.max_delay_frames), 0)
        delay_prob = float(min(max(cfg.delay_prob, 0.0), 1.0))
        if max_delay <= 0 or self._structured_history_len <= 1:
            return force

        not_initialized = ~self._structured_delay_initialized[env_ids]
        if torch.any(not_initialized):
            init_env_ids = env_ids[not_initialized]
            init_force = force[not_initialized].unsqueeze(1).expand(-1, self._structured_history_len, -1, -1)
            self._structured_delay_history[init_env_ids] = init_force
            self._structured_delay_initialized[init_env_ids] = True

        self._structured_delay_history[env_ids, self._structured_history_cursor] = force

        num_envs, num_bodies, _ = force.shape
        delay_steps = torch.randint(
            low=0,
            high=max_delay + 1,
            size=(num_envs, num_bodies),
            device=self.device,
            generator=self._generator,
        )
        if delay_prob < 1.0:
            use_delay = self._rand((num_envs, num_bodies), dtype=torch.float32) < delay_prob
            delay_steps = torch.where(use_delay, delay_steps, torch.zeros_like(delay_steps))

        history_indices = (self._structured_history_cursor - delay_steps) % self._structured_history_len
        history = self._structured_delay_history[env_ids]
        env_axis = torch.arange(num_envs, device=self.device).unsqueeze(1).expand(num_envs, num_bodies)
        body_axis = torch.arange(num_bodies, device=self.device).unsqueeze(0).expand(num_envs, num_bodies)
        delayed_force = history[env_axis, history_indices, body_axis]
        self._structured_history_cursor = (self._structured_history_cursor + 1) % self._structured_history_len
        return torch.nan_to_num(delayed_force, nan=0.0, posinf=0.0, neginf=0.0)

    def _apply_frame_hold(self, force: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg.transport_profile_cfg.frame_hold
        hold_prob = float(min(max(cfg.hold_prob, 0.0), 1.0))
        if hold_prob <= 0.0:
            return force

        not_initialized = ~self._frame_hold_initialized[env_ids]
        if torch.any(not_initialized):
            init_env_ids = env_ids[not_initialized]
            self._frame_hold_prev[init_env_ids] = force[not_initialized]
            self._frame_hold_initialized[init_env_ids] = True

        hold_mask = self._rand((force.shape[0], force.shape[1]), dtype=torch.float32) < hold_prob
        out = torch.where(hold_mask.unsqueeze(-1), self._frame_hold_prev[env_ids], force)
        self._frame_hold_prev[env_ids] = out
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def _apply_hysteresis(self, force: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg.transport_profile_cfg.hysteresis
        alpha = float(min(max(cfg.alpha, 0.0), 1.0))
        if alpha >= 1.0:
            self._hysteresis_state[env_ids] = force
            self._hysteresis_initialized[env_ids] = True
            return force

        not_initialized = ~self._hysteresis_initialized[env_ids]
        if torch.any(not_initialized):
            init_env_ids = env_ids[not_initialized]
            self._hysteresis_state[init_env_ids] = force[not_initialized]
            self._hysteresis_initialized[init_env_ids] = True

        out = alpha * force + (1.0 - alpha) * self._hysteresis_state[env_ids]
        self._hysteresis_state[env_ids] = out
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def _apply_drift(self, force: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg.transport_profile_cfg.drift
        if cfg.per_foot_step_std <= 0.0:
            return force

        drift = self._drift_total[env_ids]
        drift = drift + self._randn(drift.shape, dtype=force.dtype) * float(cfg.per_foot_step_std)
        if cfg.max_abs_drift is not None and cfg.max_abs_drift > 0.0:
            drift = drift.clamp(min=-float(cfg.max_abs_drift), max=float(cfg.max_abs_drift))
        self._drift_total[env_ids] = drift
        per_taxel_drift = drift / self._valid_count.view(1, self.num_bodies, 1)
        out = force + per_taxel_drift * self._valid_taxel_mask.unsqueeze(0).to(dtype=force.dtype)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def _apply_sparse_transport_dropout(self, force: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg.transport_profile_cfg.sparse_dropout
        drop_prob = float(min(max(cfg.per_foot_dropout_prob, 0.0), 1.0))
        if drop_prob <= 0.0:
            return force
        drop_mask = self._rand((force.shape[0], force.shape[1]), dtype=torch.float32) < drop_prob
        return torch.where(drop_mask.unsqueeze(-1), torch.zeros_like(force), force)

    def _sample_measurement_profile(self, env_ids: torch.Tensor) -> None:
        measurement_cfg = self.cfg.measurement_profile_cfg
        num_envs = env_ids.numel()

        if measurement_cfg.per_foot_gain.enable:
            low, high = measurement_cfg.per_foot_gain.gain_range
            self._per_foot_gain[env_ids] = self._rand_uniform((num_envs, self.num_bodies, 1), low, high, self.dtype)

        if measurement_cfg.per_foot_bias.enable and measurement_cfg.per_foot_bias.bias_std > 0.0:
            self._per_foot_bias_total[env_ids] = self._randn((num_envs, self.num_bodies, 1), dtype=self.dtype) * float(
                measurement_cfg.per_foot_bias.bias_std
            )

        if measurement_cfg.regional_gain.enable:
            self._regional_gain[env_ids] = self._sample_regional_gain(num_envs, measurement_cfg)

        if measurement_cfg.dead_taxel.enable and measurement_cfg.dead_taxel.dead_taxel_prob > 0.0:
            dead_mask = self._rand((num_envs, self.num_bodies, self.max_taxels), dtype=torch.float32) < float(
                measurement_cfg.dead_taxel.dead_taxel_prob
            )
            self._dead_taxel_mask[env_ids] = dead_mask & self._valid_taxel_mask.unsqueeze(0)

        if measurement_cfg.weak_patch.enable and measurement_cfg.weak_patch.patch_prob > 0.0:
            self._weak_patch_scale[env_ids] = self._sample_weak_patch_scale(env_ids, measurement_cfg)

        if measurement_cfg.taxel_geometry_perturb.enable:
            self._measured_taxel_xy_b[env_ids] = self._sample_measured_taxel_xy(num_envs, measurement_cfg)

    def _sample_regional_gain(self, num_envs: int, measurement_cfg) -> torch.Tensor:
        gain_map = torch.ones((num_envs, self.num_bodies, self.max_taxels), device=self.device, dtype=self.dtype)
        regional_cfg = measurement_cfg.regional_gain
        fore_gain = self._rand_uniform((num_envs, self.num_bodies, 1), *regional_cfg.forefoot_gain_range, self.dtype)
        heel_gain = self._rand_uniform((num_envs, self.num_bodies, 1), *regional_cfg.heel_gain_range, self.dtype)
        medial_gain = self._rand_uniform((num_envs, self.num_bodies, 1), *regional_cfg.medial_gain_range, self.dtype)
        lateral_gain = self._rand_uniform((num_envs, self.num_bodies, 1), *regional_cfg.lateral_gain_range, self.dtype)
        ap_gain = torch.where(self._region_fore_mask.unsqueeze(0), fore_gain, heel_gain)
        ml_gain = torch.where(self._region_medial_mask.unsqueeze(0), medial_gain, lateral_gain)
        gain_map = gain_map * ap_gain * ml_gain
        edge_gain = self._rand_uniform((num_envs, self.num_bodies, 1), *regional_cfg.edge_gain_range, self.dtype)
        gain_map = torch.where(self._edge_taxel_mask.unsqueeze(0), gain_map * edge_gain, gain_map)
        gain_map = gain_map * self._valid_taxel_mask.unsqueeze(0).to(dtype=self.dtype)
        gain_map = torch.where(self._valid_taxel_mask.unsqueeze(0), gain_map, torch.ones_like(gain_map))
        return gain_map

    def _sample_weak_patch_scale(self, env_ids: torch.Tensor, measurement_cfg) -> torch.Tensor:
        scale = torch.ones((env_ids.numel(), self.num_bodies, self.max_taxels), device=self.device, dtype=self.dtype)
        patch_cfg = measurement_cfg.weak_patch
        radius = float(max(patch_cfg.patch_radius, 1e-6))
        for local_env in range(env_ids.numel()):
            for body_id in range(self.num_bodies):
                if self._rand((1,), dtype=torch.float32).item() >= float(patch_cfg.patch_prob):
                    continue
                valid_idx = torch.nonzero(self._valid_taxel_mask[body_id], as_tuple=False).squeeze(-1)
                if valid_idx.numel() == 0:
                    continue
                center_pick = torch.randint(
                    low=0,
                    high=int(valid_idx.numel()),
                    size=(1,),
                    device=self.device,
                    generator=self._generator,
                )
                center_idx = int(valid_idx[int(center_pick.item())].item())
                center_xy = self._base_taxel_xy_b[body_id, center_idx]
                dist = torch.norm(self._base_taxel_xy_b[body_id] - center_xy, dim=-1)
                attenuation = float(
                    self._rand_uniform((1,), patch_cfg.attenuation_range[0], patch_cfg.attenuation_range[1], self.dtype).item()
                )
                patch_mask = (dist <= radius) & self._valid_taxel_mask[body_id]
                scale[local_env, body_id, patch_mask] = attenuation
        return scale

    def _sample_measured_taxel_xy(self, num_envs: int, measurement_cfg) -> torch.Tensor:
        geom_cfg = measurement_cfg.taxel_geometry_perturb
        measured_xy = self._base_taxel_xy_b.unsqueeze(0).expand(num_envs, -1, -1, -1).clone()
        if geom_cfg.per_foot_xy_offset_std > 0.0:
            offset = self._randn((num_envs, self.num_bodies, 1, 2), dtype=self.dtype) * float(geom_cfg.per_foot_xy_offset_std)
            measured_xy = measured_xy + offset
        if geom_cfg.per_taxel_xy_jitter_std > 0.0:
            jitter = self._randn((num_envs, self.num_bodies, self.max_taxels, 2), dtype=self.dtype) * float(
                geom_cfg.per_taxel_xy_jitter_std
            )
            measured_xy = measured_xy + jitter
        valid_mask = self._valid_taxel_mask.unsqueeze(0).unsqueeze(-1)
        measured_xy = torch.where(valid_mask, measured_xy, self._base_taxel_xy_b.unsqueeze(0).expand_as(measured_xy))
        return measured_xy

    def _build_region_masks(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ap_coord = self._base_taxel_xy_b[..., 0]
        lr_coord = self._base_taxel_xy_b[..., 1]
        fore_mask = self._valid_taxel_mask & (ap_coord >= float(self.cfg.measurement_profile_cfg.regional_gain.forefoot_split))
        heel_mask = self._valid_taxel_mask & (~fore_mask)
        medial_mask = torch.where(self._body_is_left.unsqueeze(-1), lr_coord < 0.0, lr_coord > 0.0) & self._valid_taxel_mask
        lateral_mask = self._valid_taxel_mask & (~medial_mask)
        return fore_mask, heel_mask, medial_mask, lateral_mask

    def _apply_legacy_profiles(self, force: torch.Tensor, valid_mask: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        measured = force
        measured = self._apply_relative_uniform_error(measured)
        measured = self._apply_gaussian_noise(measured)
        measured = self._apply_dropout(measured, valid_mask, env_ids)
        measured = self._apply_delay(measured, env_ids)

        measured = torch.nan_to_num(measured, nan=0.0, posinf=0.0, neginf=0.0)
        step = self.cfg.quantization_step
        if step is not None and step > 0.0:
            measured = torch.round(measured / float(step)) * float(step)
        return measured

    def _apply_relative_uniform_error(self, force: torch.Tensor) -> torch.Tensor:
        rel_max = max(float(self.cfg.force_relative_error_max), 0.0)
        if rel_max <= 0.0:
            return force

        scale = 1.0 + (2.0 * self._rand(force.shape, dtype=force.dtype) - 1.0) * rel_max
        out = force * scale
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def _apply_gaussian_noise(self, force: torch.Tensor) -> torch.Tensor:
        out = force
        mult_std = float(self.cfg.multiplicative_std)
        if mult_std > 0.0:
            mult_noise = self._randn(force.shape, dtype=force.dtype) * mult_std
            out = out * (1.0 + mult_noise)

        add_std = float(self.cfg.additive_std)
        if add_std > 0.0:
            out = out + self._randn(force.shape, dtype=force.dtype) * add_std

        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def _apply_dropout(self, force: torch.Tensor, valid_mask: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        num_envs, num_bodies, _ = force.shape
        foot_drop = torch.zeros((num_envs, num_bodies), device=self.device, dtype=torch.bool)

        foot_prob = float(min(max(self.cfg.per_foot_dropout_prob, 0.0), 1.0))
        if foot_prob > 0.0:
            foot_drop = foot_drop | (self._rand((num_envs, num_bodies), dtype=torch.float32) < foot_prob)

        foot_drop = foot_drop | self._sample_burst_foot_dropout(env_ids, num_bodies)

        taxel_prob = float(min(max(self.cfg.per_taxel_dropout_prob, 0.0), 1.0))
        if taxel_prob > 0.0:
            taxel_drop = self._rand(force.shape, dtype=torch.float32) < taxel_prob
        else:
            taxel_drop = torch.zeros_like(force, dtype=torch.bool)

        drop_mask = (taxel_drop | foot_drop.unsqueeze(-1)) & valid_mask
        out = torch.where(drop_mask, torch.zeros_like(force), force)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def _sample_burst_foot_dropout(self, env_ids: torch.Tensor, num_bodies: int) -> torch.Tensor:
        burst_prob = float(min(max(self.cfg.burst_dropout_prob, 0.0), 1.0))
        if burst_prob <= 0.0:
            return torch.zeros((env_ids.numel(), num_bodies), device=self.device, dtype=torch.bool)

        remaining = self._burst_remaining[env_ids]
        active = remaining > 0
        foot_drop = active.clone()

        remaining = torch.where(active, remaining - 1, remaining)

        min_frames = max(int(self.cfg.burst_dropout_min_frames), 1)
        max_frames = max(int(self.cfg.burst_dropout_max_frames), min_frames)

        can_start = remaining == 0
        start = (self._rand(remaining.shape, dtype=torch.float32) < burst_prob) & can_start
        if torch.any(start):
            durations = torch.randint(
                min_frames,
                max_frames + 1,
                remaining.shape,
                device=self.device,
                generator=self._generator,
            )
            remaining = torch.where(start, durations - 1, remaining)
            foot_drop = foot_drop | start

        self._burst_remaining[env_ids] = remaining
        return foot_drop

    def _apply_delay(self, force: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        max_delay = max(int(self.cfg.max_delay_frames), 0)
        delay_prob = float(min(max(self.cfg.delay_prob, 0.0), 1.0))

        if max_delay <= 0 or self._history_len <= 1:
            return force

        not_initialized = ~self._delay_initialized[env_ids]
        if torch.any(not_initialized):
            init_env_ids = env_ids[not_initialized]
            init_force = force[not_initialized].unsqueeze(1).expand(-1, self._history_len, -1, -1)
            self._delay_history[init_env_ids] = init_force
            self._delay_initialized[init_env_ids] = True

        self._delay_history[env_ids, self._history_cursor] = force

        num_envs, num_bodies, _ = force.shape
        delay_steps = torch.randint(
            low=0,
            high=max_delay + 1,
            size=(num_envs, num_bodies),
            device=self.device,
            generator=self._generator,
        )
        if delay_prob < 1.0:
            use_delay = self._rand((num_envs, num_bodies), dtype=torch.float32) < delay_prob
            delay_steps = torch.where(use_delay, delay_steps, torch.zeros_like(delay_steps))

        history_indices = (self._history_cursor - delay_steps) % self._history_len
        history = self._delay_history[env_ids]
        env_axis = torch.arange(num_envs, device=self.device).unsqueeze(1).expand(num_envs, num_bodies)
        body_axis = torch.arange(num_bodies, device=self.device).unsqueeze(0).expand(num_envs, num_bodies)
        delayed_force = history[env_axis, history_indices, body_axis]

        self._history_cursor = (self._history_cursor + 1) % self._history_len
        return torch.nan_to_num(delayed_force, nan=0.0, posinf=0.0, neginf=0.0)

    def _preserve_total_force(self, force: torch.Tensor, target_total: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        current_total = force.sum(dim=-1, keepdim=True)
        scale = torch.where(current_total > eps, target_total / current_total, torch.zeros_like(current_total))
        out = force * scale
        out = torch.where(target_total > eps, out, torch.zeros_like(out))
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def _rand_uniform(self, shape: tuple[int, ...] | torch.Size, low: float, high: float, dtype: torch.dtype) -> torch.Tensor:
        low = float(low)
        high = float(high)
        if high < low:
            low, high = high, low
        if abs(high - low) < 1e-12:
            return torch.full(shape, low, device=self.device, dtype=dtype)
        return low + (high - low) * self._rand(shape, dtype=dtype)

    def _as_env_ids(self, env_ids: Sequence[int] | torch.Tensor | slice | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)[env_ids]
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long).flatten()
        return torch.as_tensor(list(env_ids), device=self.device, dtype=torch.long).flatten()

    def _rand(self, shape: tuple[int, ...] | torch.Size, dtype: torch.dtype) -> torch.Tensor:
        return torch.rand(shape, device=self.device, dtype=dtype, generator=self._generator)

    def _randn(self, shape: tuple[int, ...] | torch.Size, dtype: torch.dtype) -> torch.Tensor:
        return torch.randn(shape, device=self.device, dtype=dtype, generator=self._generator)
