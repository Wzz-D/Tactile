from __future__ import annotations

from collections.abc import Sequence

import torch

from .foot_tactile_noise_cfg import FootTactileNoiseCfg


class FootTactileNoiseModel:
    """Stateful tactile measurement noise model with dropout and random delay."""

    def __init__(
        self,
        cfg: FootTactileNoiseCfg,
        num_envs: int,
        num_bodies: int,
        max_taxels: int,
        device: str | torch.device,
        dtype: torch.dtype,
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

        self._history_len = max(int(self.cfg.max_delay_frames), 0) + 1
        self._history_cursor = 0
        self._delay_history = torch.zeros(
            (self.num_envs, self._history_len, self.num_bodies, self.max_taxels),
            device=self.device,
            dtype=self.dtype,
        )
        self._delay_initialized = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)

        self._burst_remaining = torch.zeros((self.num_envs, self.num_bodies), device=self.device, dtype=torch.int64)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Reset noise states for selected environments."""

        env_ids_t = self._as_env_ids(env_ids)
        self._delay_history[env_ids_t] = 0.0
        self._delay_initialized[env_ids_t] = False
        self._burst_remaining[env_ids_t] = 0

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

        measured = self._apply_relative_uniform_error(measured)
        measured = self._apply_gaussian_noise(measured)
        measured = self._apply_dropout(measured, valid, env_ids_t)
        measured = self._apply_delay(measured, env_ids_t)

        measured = torch.nan_to_num(measured, nan=0.0, posinf=0.0, neginf=0.0)

        clip_min = float(self.cfg.clip_min)
        measured = measured.clamp_min(clip_min)

        if self.cfg.clip_max is not None:
            measured = measured.clamp_max(float(self.cfg.clip_max))

        step = self.cfg.quantization_step
        if step is not None and step > 0.0:
            measured = torch.round(measured / float(step)) * float(step)

        measured = measured * valid_f
        measured = measured.clamp_min(0.0)

        if self.cfg.preserve_total_force_after_noise:
            measured = self._preserve_total_force(measured, pre_noise_total)
            measured = measured * valid_f

        measured = torch.nan_to_num(measured, nan=0.0, posinf=0.0, neginf=0.0)
        measured = measured.clamp_min(0.0)
        measured = measured * valid_f
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
