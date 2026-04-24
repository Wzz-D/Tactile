import torch
from collections.abc import Sequence

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.ui.widgets import ManagerLiveVisualizer

from instinctlab.managers import DummyRewardCfg, MultiRewardCfg, MultiRewardManager
from instinctlab.monitors import MonitorManager


class InstinctRlEnv(ManagerBasedRLEnv):
    """This class adds additional logging mechanism on sensors to get more
    comprehensive running statistics.
    """

    def configure_stage_reward_warmup_counter(self, rollout_horizon_steps: int, iteration_offset: int = 0) -> None:
        """Configure how stage-reward warmup maps env steps back to PPO iterations.

        Args:
            rollout_horizon_steps: Number of env.step calls collected per training iteration.
            iteration_offset: Runner iteration offset, used when resuming from checkpoints.
        """
        self._stage_reward_warmup_rollout_horizon = max(int(rollout_horizon_steps), 1)
        self._stage_reward_warmup_iteration_offset = max(int(iteration_offset), 0)

    def get_stage_reward_warmup_iteration(self) -> int:
        """Return the current training iteration seen by reward terms.

        `common_step_counter` is incremented once per env.step() call before rewards are computed.
        We map it back to the rollout iteration so that warmup aligns with TensorBoard's iteration axis.
        """
        rollout_horizon = max(int(getattr(self, "_stage_reward_warmup_rollout_horizon", 1)), 1)
        iteration_offset = max(int(getattr(self, "_stage_reward_warmup_iteration_offset", 0)), 0)
        common_steps = max(int(getattr(self, "common_step_counter", 0)), 0)
        rollout_iteration = max(common_steps - 1, 0) // rollout_horizon
        return iteration_offset + rollout_iteration

    def load_managers(self):

        # check and routing the reward manager to the multi reward manager
        if isinstance(self.cfg.rewards, MultiRewardCfg):
            reward_group_cfg = self.cfg.rewards
            self.cfg.rewards = DummyRewardCfg()
        super().load_managers()
        # replace the parent class's reward manager
        if "reward_group_cfg" in locals():
            self.cfg.rewards = reward_group_cfg
            self.reward_manager = MultiRewardManager(self.cfg.rewards, self)
            print("[INFO] Multi-Reward Manager: ", self.reward_manager)

        self.monitor_manager = MonitorManager(self.cfg.monitors, self)
        print("[INFO] Monitor Manager: ", self.monitor_manager)
        self._log_sensor_binding_check()

    def configure_eval_pre_reset_snapshot(self, enabled: bool = True) -> None:
        """Enable raw pre-reset snapshots used by play.py eval mode.

        These snapshots are cloned before env resets so terminal-step sensor values remain available
        to downstream evaluation code.
        """
        self._enable_eval_pre_reset_snapshot = bool(enabled)

    def setup_manager_visualizers(self):
        super().setup_manager_visualizers()
        self.manager_visualizers["monitor_manager"] = ManagerLiveVisualizer(manager=self.monitor_manager)

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        # process actions
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        self.extras["step"] = self.extras.get("step", {})
        if bool(getattr(self.cfg, "enable_pre_reset_sensor_snapshot_stats", True)):
            self.extras["step"].update(self._collect_pre_reset_sensor_snapshot())
        if bool(getattr(self, "_enable_eval_pre_reset_snapshot", False)):
            self.extras["eval_pre_reset"] = self._collect_eval_pre_reset_snapshot()
        else:
            self.extras.pop("eval_pre_reset", None)

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)

            self._reset_idx(reset_env_ids)
            self.scene.write_data_to_sim()
            self.sim.forward()

            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

            self.recorder_manager.record_post_reset(reset_env_ids)

        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        self.obs_buf = self.observation_manager.compute()

        return_ = (self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras)
        monitor_infos = self.monitor_manager.update(dt=self.step_dt)
        self.extras["step"] = self.extras.get("step", {})
        self.extras["step"].update(monitor_infos)
        return return_

    def _reset_idx(self, env_ids: Sequence[int]):
        monitor_infos = self.monitor_manager.reset(env_ids, is_episode=True)
        return_ = super()._reset_idx(env_ids)
        self.extras["log"] = self.extras.get("log", {})
        self.extras["log"].update(monitor_infos)
        return return_

    def _log_sensor_binding_check(self) -> None:
        tactile_sensor = self.scene.sensors.get("foot_tactile", None)
        stage_sensor = self.scene.sensors.get("contact_stage_filter", None)
        if tactile_sensor is None or stage_sensor is None:
            print("[INFO] Sensor binding check skipped: foot_tactile or contact_stage_filter missing.")
            return

        bound_tactile_sensor = getattr(stage_sensor, "_tactile_sensor", None)
        same_object = bound_tactile_sensor is tactile_sensor
        stage_body_names = list(getattr(stage_sensor, "body_names", []))
        tactile_body_ids = tuple(getattr(stage_sensor, "_tactile_body_ids", tuple()))
        tactile_body_names = []
        if hasattr(tactile_sensor, "body_names"):
            tactile_body_names = [str(tactile_sensor.body_names[idx]) for idx in tactile_body_ids]

        print(
            "[INFO] Sensor binding check: "
            f"same_object={same_object}, "
            f"stage_body_names={stage_body_names}, "
            f"tactile_body_ids={tactile_body_ids}, "
            f"tactile_body_names={tactile_body_names}, "
            f"scene_tactile_id={id(tactile_sensor)}, "
            f"bound_tactile_id={id(bound_tactile_sensor) if bound_tactile_sensor is not None else None}"
        )

    def _collect_pre_reset_sensor_snapshot(self) -> dict[str, torch.Tensor]:
        stats: dict[str, torch.Tensor] = {}
        tactile_sensor = self.scene.sensors.get("foot_tactile", None)
        stage_sensor = self.scene.sensors.get("contact_stage_filter", None)

        same_object = float(stage_sensor is not None and getattr(stage_sensor, "_tactile_sensor", None) is tactile_sensor)
        stats["PreReset/Binding/stage_tactile_same_object"] = torch.tensor(same_object, device=self.device)

        if tactile_sensor is not None:
            tactile_data = tactile_sensor.data
            total_force = getattr(tactile_data, "total_normal_force", None)
            taxel_force = getattr(tactile_data, "taxel_force", None)
            contact_area_ratio = getattr(tactile_data, "contact_area_ratio", None)
            support_valid_mask = getattr(tactile_data, "support_valid_mask", None)
            support_dist = getattr(tactile_data, "support_dist", None)
            valid_taxel_mask = getattr(tactile_data, "valid_taxel_mask", None)

            self._append_metric_mean_var(stats, total_force, "PreReset/FootTactile/total_normal_force")
            self._append_metric_mean_var(stats, contact_area_ratio, "PreReset/FootTactile/contact_area_ratio")

            if torch.is_tensor(taxel_force) and taxel_force.numel() > 0:
                taxel_force_f = torch.nan_to_num(taxel_force.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
                taxel_force_sum = taxel_force_f.sum(dim=-1)
                self._append_metric_mean_var(stats, taxel_force_sum, "PreReset/FootTactile/taxel_force_sum")
                if torch.is_tensor(total_force) and total_force.shape == taxel_force_sum.shape:
                    total_force_f = torch.nan_to_num(total_force.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
                    force_delta = taxel_force_sum - total_force_f
                    stats["PreReset/FootTactile/force_consistency_abs_mean"] = force_delta.abs().mean()

            if (
                torch.is_tensor(support_valid_mask)
                and torch.is_tensor(support_dist)
                and torch.is_tensor(valid_taxel_mask)
                and support_valid_mask.ndim == 3
                and support_dist.shape == support_valid_mask.shape
            ):
                template_valid = valid_taxel_mask.unsqueeze(0).expand_as(support_valid_mask)
                support_valid = support_valid_mask & template_valid
                valid_count = template_valid.sum(dim=-1).clamp_min(1)
                support_ratio = support_valid.to(dtype=torch.float32).sum(dim=-1) / valid_count.to(dtype=torch.float32)
                self._append_metric_mean_var(stats, support_ratio, "PreReset/FootTactile/support_valid_ratio")

                support_dist_f = torch.nan_to_num(support_dist.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
                support_valid_f = support_valid.to(dtype=torch.float32)
                support_dist_sum = (support_dist_f * support_valid_f).sum(dim=-1)
                support_valid_count = support_valid_f.sum(dim=-1)
                support_dist_mean = torch.where(
                    support_valid_count > 0.0,
                    support_dist_sum / support_valid_count.clamp_min(1.0),
                    torch.zeros_like(support_dist_sum),
                )
                self._append_metric_mean_var(stats, support_dist_mean, "PreReset/FootTactile/support_dist_valid_mean")

        if stage_sensor is not None and hasattr(stage_sensor, "get_debug_tensors"):
            debug_tensors = stage_sensor.get_debug_tensors()
            for metric_name in ("h_eff", "foot_vz", "total_force", "contact_area"):
                self._append_metric_mean_var(stats, debug_tensors.get(metric_name), f"PreReset/ContactStage/{metric_name}")
            for metric_name in ("contact_active", "contact_on_event", "contact_off_event", "landing_window_active", "E_pre", "E_land", "E_st", "E_sw"):
                values = debug_tensors.get(metric_name)
                if torch.is_tensor(values) and values.numel() > 0:
                    values_f = torch.nan_to_num(values.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
                    stats[f"PreReset/ContactStage/{metric_name}_mean"] = values_f.mean()

            if tactile_sensor is not None:
                tactile_body_ids = tuple(getattr(stage_sensor, "_tactile_body_ids", tuple()))
                tactile_data = tactile_sensor.data
                if tactile_body_ids:
                    direct_total_force = getattr(tactile_data, "total_normal_force", None)
                    direct_contact_area = getattr(tactile_data, "contact_area_ratio", None)
                    direct_taxel_force = getattr(tactile_data, "taxel_force", None)

                    if torch.is_tensor(direct_total_force) and direct_total_force.ndim == 2:
                        direct_total_force = direct_total_force[:, tactile_body_ids]
                        self._append_metric_mean_var(stats, direct_total_force, "PreReset/StageInput/direct_total_force")
                        stage_total_force = debug_tensors.get("total_force")
                        if torch.is_tensor(stage_total_force) and stage_total_force.shape == direct_total_force.shape:
                            delta_force = stage_total_force.to(dtype=torch.float32) - direct_total_force.to(dtype=torch.float32)
                            stats["PreReset/StageInput/direct_minus_stage_force_abs_mean"] = delta_force.abs().mean()

                    if torch.is_tensor(direct_taxel_force) and direct_taxel_force.ndim == 3:
                        direct_taxel_force_sum = torch.nan_to_num(
                            direct_taxel_force[:, tactile_body_ids, :].to(dtype=torch.float32),
                            nan=0.0,
                            posinf=0.0,
                            neginf=0.0,
                        ).clamp_min(0.0).sum(dim=-1)
                        self._append_metric_mean_var(stats, direct_taxel_force_sum, "PreReset/StageInput/direct_taxel_force_sum")

                    if torch.is_tensor(direct_contact_area) and direct_contact_area.ndim == 2:
                        direct_contact_area = direct_contact_area[:, tactile_body_ids]
                        self._append_metric_mean_var(stats, direct_contact_area, "PreReset/StageInput/direct_contact_area")
                        stage_contact_area = debug_tensors.get("contact_area")
                        if torch.is_tensor(stage_contact_area) and stage_contact_area.shape == direct_contact_area.shape:
                            delta_area = stage_contact_area.to(dtype=torch.float32) - direct_contact_area.to(dtype=torch.float32)
                            stats["PreReset/StageInput/direct_minus_stage_area_abs_mean"] = delta_area.abs().mean()

        return stats

    def _collect_eval_pre_reset_snapshot(self) -> dict[str, torch.Tensor]:
        """Collect raw per-env tensors before env reset for eval-only consumers."""
        snapshot: dict[str, torch.Tensor] = {
            "done": self.reset_buf.detach().clone(),
            "terminated": self.reset_terminated.detach().clone(),
            "time_outs": self.reset_time_outs.detach().clone(),
        }

        termination_manager = getattr(self, "termination_manager", None)
        if termination_manager is not None:
            for term_name in ("target_reached",):
                try:
                    snapshot[f"termination/{term_name}"] = termination_manager.get_term(term_name).detach().clone()
                except Exception:
                    continue

        stage_sensor = self.scene.sensors.get("contact_stage_filter", None)
        if stage_sensor is not None:
            stage_data = stage_sensor.data
            snapshot["contact_stage/dominant_stage_id"] = stage_data.dominant_stage_id.detach().clone()
            snapshot["contact_stage/foot_vz"] = stage_data.foot_vz.detach().clone()
            snapshot["contact_stage/total_force"] = stage_data.total_force.detach().clone()
            snapshot["contact_stage/contact_area"] = stage_data.contact_area.detach().clone()
            snapshot["contact_stage/landing_window_active"] = (stage_data.landing_window > 0).detach().clone()

        tactile_sensor = self.scene.sensors.get("foot_tactile", None)
        if tactile_sensor is not None:
            tactile_data = tactile_sensor.data
            snapshot["foot_tactile/cop_b"] = tactile_data.cop_b.detach().clone()
            snapshot["foot_tactile/contact_area_ratio"] = tactile_data.contact_area_ratio.detach().clone()

        volume_sensor = self.scene.sensors.get("leg_volume_points", None)
        if volume_sensor is not None:
            penetration_offset = getattr(volume_sensor.data, "penetration_offset", None)
            if torch.is_tensor(penetration_offset) and penetration_offset.numel() > 0:
                penetration_depth = torch.norm(penetration_offset, dim=-1)
                penetration_depth = penetration_depth.flatten(1, -1)
                snapshot["volume_points/max_penetration_depth"] = penetration_depth.max(dim=1).values.detach().clone()
            else:
                snapshot["volume_points/max_penetration_depth"] = torch.zeros(
                    self.num_envs, device=self.device, dtype=torch.float32
                )

        return snapshot

    def _append_metric_mean_var(self, stats: dict[str, torch.Tensor], values, prefix: str) -> None:
        if not torch.is_tensor(values) or values.numel() == 0:
            return
        values_f = torch.nan_to_num(values.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        stats[f"{prefix}_mean"] = values_f.mean()
        stats[f"{prefix}_var"] = values_f.var(unbiased=False)

    """
    Properties.
    """

    @property
    def num_rewards(self) -> int:
        return getattr(self.reward_manager, "num_rewards", 1)
