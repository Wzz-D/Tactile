from __future__ import annotations

import gymnasium as gym
import torch
from typing import TYPE_CHECKING, Dict

from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnvCfg

from instinct_rl.env import VecEnv


class InstinctRlVecEnvWrapper(VecEnv):
    """Wraps around Isaac Lab environment for Instinct-RL library
    Reference:
       https://github.com/project-instinct/instinct_rl/blob/master/instinct_rl/env/vec_env.py
    """

    def __init__(self, env: ManagerBasedRLEnv):
        """Initializes the wrapper.

        Note:
            The wrapper calls :meth:`reset` at the start since the RSL-RL runner does not call reset.

        Args:
            env: The environment to wrap around.

        Raises:
            ValueError: When the environment is not an instance of :class:`ManagerBasedRLEnv` or :class:`DirectRLEnv`.
        """
        # check that input is valid
        if not isinstance(env.unwrapped, ManagerBasedRLEnv) and not isinstance(env.unwrapped, DirectRLEnv):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv or DirectRLEnv. Environment type:"
                f" {type(env)}"
            )
        # initialize the wrapper
        self.env = env
        # store information required by wrapper
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length
        if hasattr(self.unwrapped, "action_manager"):
            self.num_actions = self.unwrapped.action_manager.total_action_dim
        else:
            self.num_actions = gym.spaces.flatdim(self.unwrapped.single_action_space)
        if hasattr(self.unwrapped, "observation_manager"):
            num_obs_dims = self.unwrapped.observation_manager.group_obs_term_dim["policy"]
            num_obs_dims = [torch.prod(torch.tensor(dim, device="cpu")).item() for dim in num_obs_dims]
            self.num_obs = int(sum(num_obs_dims))
        else:
            # Not checked for DiectRlEnv
            self.num_obs = gym.spaces.flatdim(self.unwrapped.single_observation_space["policy"])
        # -- privileged observations
        if (
            hasattr(self.unwrapped, "observation_manager")
            and "critic" in self.unwrapped.observation_manager.group_obs_dim
        ):
            num_obs_dims = self.unwrapped.observation_manager.group_obs_term_dim["critic"]
            num_obs_dims = [torch.prod(torch.tensor(dim, device="cpu")).item() for dim in num_obs_dims]
            self.num_critic_obs = int(sum(num_obs_dims))
        elif hasattr(self.unwrapped, "num_states") and "critic" in self.unwrapped.single_observation_space:
            # Not checked for DiectRlEnv
            self.num_critic_obs = gym.spaces.flatdim(self.unwrapped.single_observation_space["critic"])
        else:
            self.num_critic_obs = None
        # reset at the start since the Instinct-RL runner does not call reset
        self.env.reset()

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    """
    Properties -- Gym.Wrapper
    """

    @property
    def cfg(self) -> ManagerBasedRLEnvCfg:
        """Returns the configuration class instance of the environment."""
        return self.unwrapped.cfg

    @property
    def render_mode(self) -> str | None:
        """Returns the :attr:`Env` :attr:`render_mode`."""
        return self.env.render_mode

    @property
    def observation_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`observation_space`."""
        return self.env.observation_space

    @property
    def action_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`action_space`."""
        return self.env.action_space

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def unwrapped(self) -> ManagerBasedRLEnv | DirectRLEnv:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped

    """
    Properties
    """

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Returns the current observations of the environment."""
        if hasattr(self.unwrapped, "observation_manager"):
            obs_pack = self.unwrapped.observation_manager.compute()
        else:
            obs_pack = self.unwrapped._get_observations()
        obs_pack = self._flatten_all_obs_groups(obs_pack)
        return obs_pack["policy"], {"observations": obs_pack}

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """The episode length buffer."""
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        """Set the episode length buffer.

        Note:
            This is needed to perform random initialization of episode lengths in RSL-RL.
        """
        self.unwrapped.episode_length_buf = value

    @property
    def num_rewards(self) -> int:
        return self.unwrapped.num_rewards

    """
    Operations - MDP
    """

    def seed(self, seed: int = -1) -> int:  # noqa: D102
        return self.unwrapped.seed(seed)

    def reset(self) -> tuple[torch.Tensor, dict]:  # noqa: D102
        # reset the environment
        obs_pack, _ = self.env.reset()
        obs_pack = self._flatten_all_obs_groups(obs_pack)
        # return observations
        return obs_pack["policy"], {"observations": obs_pack}

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # record step information
        obs_pack, rew, terminated, truncated, extras = self.env.step(actions)
        extras["step"] = extras.get("step", {})
        extras["step"].update(self._compute_reset_step_stats(terminated, truncated))
        extras["step"].update(self._compute_foot_contact_state_step_stats(obs_pack))
        extras["step"].update(self._compute_foot_tactile_step_stats())
        extras["step"].update(self._compute_contact_stage_step_stats())
        obs_pack = self._flatten_all_obs_groups(obs_pack)
        # compute dones for compatibility with RSL-RL
        dones = (terminated | truncated).to(dtype=torch.long)
        # move extra observations to the extras dict

        extras["observations"] = obs_pack
        # move time out information to the extras dict
        # this is only needed for infinite horizon tasks
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        # return the step information
        obs = obs_pack["policy"]
        if isinstance(rew, dict):
            # returned by multi-reward manager
            rew = self._stack_rewards(rew)
        else:
            # returned by regular reward manager
            # make sure rewards are always in shape of [batch, num_rewards]
            rew = rew.unsqueeze(1)
        return obs, rew, dones, extras

    def close(self):  # noqa: D102
        return self.env.close()

    """
    Operations -- Instinct-RL
    """

    def get_obs_segments(self, group_name: str = "policy"):
        obs_term_names = self.unwrapped.observation_manager.active_terms[group_name]
        obs_term_dims = self.unwrapped.observation_manager.group_obs_term_dim[group_name]
        return self._get_obs_segments(obs_term_names, obs_term_dims)

    def _get_obs_segments(self, obs_term_names, obs_term_dims):
        # assuming the computed obs_term_dim is in the same order as the obs_cfg
        # From Python 3.6+, dictionaries are ordered by insertion order
        obs_segments = dict()
        for term_name, term_dim in zip(obs_term_names, obs_term_dims):
            obs_segments[term_name] = term_dim

        return obs_segments

    def get_obs_format(self) -> dict[str, dict[str, tuple]]:
        """Returns the observation information for all observation groups.
        Using this interface, so that, the algorithm / policy should not access env directly.
        But let the runner access env to get critical information.
        """
        obs_format = dict()
        for group_name in self.unwrapped.observation_manager.active_terms.keys():
            obs_format[group_name] = self.get_obs_segments(group_name)
        return obs_format

    """
    Internal Helpers
    """

    def _flatten_obs_group(self, obs_group: dict) -> torch.Tensor:
        """Considering observation_manager only concatenate observation terms of 1D tensors,
        this function flattens the observation terms of different shape and concatenate into
        a single tensor.
        """
        obs = []
        for obs_term, obs_value in obs_group.items():
            obs.append(obs_value.flatten(start_dim=1))
        obs = torch.cat(obs, dim=1)
        return obs

    def _flatten_all_obs_groups(self, obs_pack: dict) -> dict:
        obs_pack_ = dict()
        for obs_group_name, obs_group in obs_pack.items():
            obs_pack_[obs_group_name] = self._flatten_obs_group(obs_group) if isinstance(obs_group, dict) else obs_group
        return obs_pack_

    def _stack_rewards(self, rewards_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        rewards = []
        for reward_term, reward_value in rewards_dict.items():
            rewards.append(reward_value)
        rewards = torch.stack(rewards, dim=-1)
        return rewards

    def _compute_foot_contact_state_step_stats(self, obs_pack: dict) -> dict[str, torch.Tensor]:
        """Compute per-dimension mean/variance of latest foot_contact_state frame for TensorBoard."""
        policy_obs = obs_pack.get("policy")
        if not isinstance(policy_obs, dict) or "foot_contact_state" not in policy_obs:
            return {}

        term = policy_obs["foot_contact_state"]
        if not torch.is_tensor(term):
            return {}

        latest = self._extract_latest_foot_contact_frame(term)
        if latest is None or latest.numel() == 0:
            return {}

        latest = torch.nan_to_num(latest, nan=0.0, posinf=0.0, neginf=0.0)
        feature_names_by_dim = {
            4: (
                "contact_area_ratio",
                "F_over_body_weight",
                "cop_x_norm",
                "cop_y_norm",
            ),
            5: (
                "contact_area_ratio",
                "F_over_body_weight",
                "cop_x_norm",
                "cop_y_norm",
                "vz_norm",
            ),
        }
        feature_names = feature_names_by_dim.get(int(latest.shape[-1]), None)
        if feature_names is None:
            return {}

        stats = {}
        for feature_id, feature_name in enumerate(feature_names):
            values = latest[..., feature_id]
            stats[f"FootContactState/{feature_name}_mean"] = values.mean()
            stats[f"FootContactState/{feature_name}_var"] = values.var(unbiased=False)
        return stats

    def _compute_reset_step_stats(self, terminated: torch.Tensor, truncated: torch.Tensor) -> dict[str, torch.Tensor]:
        """Expose reset rates so post-reset diagnostics can be interpreted correctly."""

        if not torch.is_tensor(terminated) or not torch.is_tensor(truncated):
            return {}

        terminated_f = torch.nan_to_num(terminated.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        truncated_f = torch.nan_to_num(truncated.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        reset_f = torch.clamp(terminated_f + truncated_f, min=0.0, max=1.0)
        return {
            "Env/reset_rate": reset_f.mean(),
            "Env/reset_terminated_rate": terminated_f.mean(),
            "Env/reset_timeout_rate": truncated_f.mean(),
        }

    def _compute_foot_tactile_step_stats(self) -> dict[str, torch.Tensor]:
        """Compute direct FootTactile diagnostics from current sensor buffers."""

        tactile_sensor = self.unwrapped.scene.sensors.get("foot_tactile", None)
        if tactile_sensor is None:
            return {}

        tactile_data = tactile_sensor.data
        stats: dict[str, torch.Tensor] = {}

        total_force = getattr(tactile_data, "total_normal_force", None)
        taxel_force = getattr(tactile_data, "taxel_force", None)
        contact_area_ratio = getattr(tactile_data, "contact_area_ratio", None)
        support_valid_mask = getattr(tactile_data, "support_valid_mask", None)
        support_dist = getattr(tactile_data, "support_dist", None)
        valid_taxel_mask = getattr(tactile_data, "valid_taxel_mask", None)

        self._append_foot_tactile_float_stats(stats, total_force, "total_normal_force")
        self._append_foot_tactile_float_stats(stats, contact_area_ratio, "contact_area_ratio")

        if torch.is_tensor(taxel_force) and taxel_force.numel() > 0:
            taxel_force_f = torch.nan_to_num(taxel_force.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
            taxel_force_sum = taxel_force_f.sum(dim=-1)
            self._append_foot_tactile_float_stats(stats, taxel_force_sum, "taxel_force_sum")
            if torch.is_tensor(total_force) and total_force.shape == taxel_force_sum.shape:
                total_force_f = torch.nan_to_num(total_force.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
                force_delta = taxel_force_sum - total_force_f
                stats["FootTactile/force_consistency_signed_mean"] = force_delta.mean()
                stats["FootTactile/force_consistency_abs_mean"] = force_delta.abs().mean()
                stats["FootTactile/force_consistency_abs_max"] = force_delta.abs().max()

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
            self._append_foot_tactile_float_stats(stats, support_ratio, "support_valid_ratio")

            support_dist_f = torch.nan_to_num(support_dist.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
            support_valid_f = support_valid.to(dtype=torch.float32)
            support_dist_sum = (support_dist_f * support_valid_f).sum(dim=-1)
            support_valid_count = support_valid_f.sum(dim=-1)
            support_dist_mean = torch.where(
                support_valid_count > 0.0,
                support_dist_sum / support_valid_count.clamp_min(1.0),
                torch.zeros_like(support_dist_sum),
            )
            self._append_foot_tactile_float_stats(stats, support_dist_mean, "support_dist_valid_mean")

        return stats

    def _compute_contact_stage_step_stats(self) -> dict[str, torch.Tensor]:
        """Compute mean/rate diagnostics from ContactStageFilter for TensorBoard."""

        stage_sensor = self.unwrapped.scene.sensors.get("contact_stage_filter", None)
        if stage_sensor is None or not hasattr(stage_sensor, "get_debug_tensors"):
            return {}

        debug_tensors = stage_sensor.get_debug_tensors()
        if not isinstance(debug_tensors, dict) or len(debug_tensors) == 0:
            return {}

        stats: dict[str, torch.Tensor] = {}
        bool_metric_specs = (
            ("contact_active", "contact_active_mean", True),
            ("contact_on_event", "contact_on_event_rate", False),
            ("contact_off_event", "contact_off_event_rate", False),
            ("landing_window_active", "landing_window_active_rate", False),
            ("E_sw", "E_sw_rate", False),
            ("E_pre", "E_pre_rate", False),
            ("E_land", "E_land_rate", False),
            ("E_st", "E_st_rate", False),
            ("h_zone_hit", "h_zone_hit_rate", False),
            ("v_pre_hit", "v_pre_hit_rate", False),
            ("force_on_hit", "force_on_hit_rate", False),
            ("area_on_hit", "area_on_hit_rate", False),
        )
        for source_name, overall_name, include_var in bool_metric_specs:
            self._append_contact_stage_bool_stats(
                stats,
                debug_tensors.get(source_name),
                metric_name=source_name,
                overall_name=overall_name,
                include_var=include_var,
            )

        for metric_name in ("h_eff", "foot_vz", "total_force", "contact_area"):
            self._append_contact_stage_float_stats(stats, debug_tensors.get(metric_name), metric_name)
        return stats

    def _append_foot_tactile_float_stats(
        self,
        stats: dict[str, torch.Tensor],
        values: torch.Tensor | None,
        metric_name: str,
    ) -> None:
        if not torch.is_tensor(values) or values.numel() == 0:
            return
        values_f = torch.nan_to_num(values.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        stats[f"FootTactile/{metric_name}_mean"] = values_f.mean()
        stats[f"FootTactile/{metric_name}_var"] = values_f.var(unbiased=False)
        if values_f.ndim == 2:
            if values_f.shape[1] >= 1:
                stats[f"FootTactile/{metric_name}_left_mean"] = values_f[:, 0].mean()
            if values_f.shape[1] >= 2:
                stats[f"FootTactile/{metric_name}_right_mean"] = values_f[:, 1].mean()

    def _append_contact_stage_bool_stats(
        self,
        stats: dict[str, torch.Tensor],
        values: torch.Tensor | None,
        metric_name: str,
        overall_name: str,
        include_var: bool,
    ) -> None:
        if not torch.is_tensor(values) or values.numel() == 0:
            return
        values_f = torch.nan_to_num(values.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        stats[f"ContactStage/{overall_name}"] = values_f.mean()
        if include_var:
            stats[f"ContactStage/{metric_name}_var"] = values_f.var(unbiased=False)
        self._append_contact_stage_side_means(stats, metric_name, values_f)

    def _append_contact_stage_float_stats(
        self,
        stats: dict[str, torch.Tensor],
        values: torch.Tensor | None,
        metric_name: str,
    ) -> None:
        if not torch.is_tensor(values) or values.numel() == 0:
            return
        values_f = torch.nan_to_num(values.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        stats[f"ContactStage/{metric_name}_mean"] = values_f.mean()
        stats[f"ContactStage/{metric_name}_var"] = values_f.var(unbiased=False)
        self._append_contact_stage_side_means(stats, metric_name, values_f)

    def _append_contact_stage_side_means(
        self,
        stats: dict[str, torch.Tensor],
        metric_name: str,
        values: torch.Tensor,
    ) -> None:
        if values.ndim != 2:
            return
        if values.shape[1] >= 1:
            stats[f"ContactStage/{metric_name}_left_mean"] = values[:, 0].mean()
        if values.shape[1] >= 2:
            stats[f"ContactStage/{metric_name}_right_mean"] = values[:, 1].mean()

    def _extract_latest_foot_contact_frame(self, term: torch.Tensor) -> torch.Tensor | None:
        """Return latest frame as (num_envs, num_feet, feature_dim) from possible term layouts."""
        num_feet = self._get_foot_contact_num_feet()

        if term.ndim == 2:
            if num_feet <= 0:
                return None
            for feature_dim in (4, 5):
                if term.shape[1] % feature_dim != 0:
                    continue
                if term.shape[1] % (num_feet * feature_dim) != 0:
                    continue
                history_length = term.shape[1] // (num_feet * feature_dim)
                reshaped = term.reshape(term.shape[0], history_length, num_feet, feature_dim)
                return reshaped[:, -1, :, :]

        if term.ndim == 3 and term.shape[-1] in (4, 5):
            # (N, num_feet, feature_dim)
            return term

        if term.ndim == 4 and term.shape[-1] in (4, 5):
            # (N, history, num_feet, feature_dim)
            return term[:, -1, :, :]

        return None

    def _get_foot_contact_num_feet(self) -> int:
        stage_sensor = self.unwrapped.scene.sensors.get("contact_stage_filter", None)
        if stage_sensor is not None:
            return int(getattr(stage_sensor, "num_bodies", 0))
        tactile_sensor = self.unwrapped.scene.sensors.get("foot_tactile", None)
        if tactile_sensor is not None:
            return int(getattr(tactile_sensor, "num_bodies", 0))
        return 0
