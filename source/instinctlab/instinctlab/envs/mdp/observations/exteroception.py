from __future__ import annotations

import numpy as np
import torch
from typing import TYPE_CHECKING, Literal

import cv2

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.events import (  # This could be dangerous for code maintainability. Maybe optimize this import later.
    _randomize_prop_by_op,
)
from isaaclab.managers import ManagerTermBase, ManagerTermBaseCfg, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from isaaclab.sensors import Camera, RayCasterCamera, TiledCamera

    from instinctlab.sensors.grouped_ray_caster import GroupedRayCasterCamera
    from instinctlab.sensors.noisy_camera import NoisyGroupedRayCasterCamera




def foot_tactile(env, sensor_cfg: SceneEntityCfg = SceneEntityCfg("foot_tactile")):
    sensor = env.scene.sensors[sensor_cfg.name]   
    return sensor.data.force_N                    # (N, 2, 120)


_FOOT_CONTACT_STATE_DIM = 5
_FOOT_CONTACT_STATE_BODY_WEIGHT_CACHE_ATTR = "_foot_contact_state_body_weight_n"
_FOOT_CONTACT_STATE_COP_SCALE_CACHE_ATTR = "_foot_contact_state_cop_scales"
_FOOT_CONTACT_STATE_BODY_WEIGHT_FALLBACK_N = 327.68
_FOOT_CONTACT_STATE_VZ_REF = 0.24
_FOOT_CONTACT_STATE_FORCE_GATE = 0.05


def _get_or_cache_body_weight_n(env, asset_name: str = "robot") -> float:
    """Read body weight from runtime masses once and cache it on the env."""
    cached_value = getattr(env, _FOOT_CONTACT_STATE_BODY_WEIGHT_CACHE_ATTR, None)
    if isinstance(cached_value, (float, int)) and np.isfinite(cached_value) and float(cached_value) > 1e-6:
        return float(cached_value)

    body_weight_n = float(_FOOT_CONTACT_STATE_BODY_WEIGHT_FALLBACK_N)
    try:
        asset = env.scene[asset_name]
        masses = asset.root_physx_view.get_masses()
        masses_t = torch.as_tensor(masses, device=env.device, dtype=torch.float32)
        total_mass = masses_t[0].sum() if masses_t.ndim >= 2 else masses_t.sum()
        total_mass = torch.nan_to_num(total_mass, nan=0.0, posinf=0.0, neginf=0.0)
        total_mass_scalar = float(total_mass.item())
        if total_mass_scalar > 1e-6:
            body_weight_n = total_mass_scalar * 9.81
    except Exception:
        body_weight_n = float(_FOOT_CONTACT_STATE_BODY_WEIGHT_FALLBACK_N)

    body_weight_n = max(float(body_weight_n), 1e-6)
    setattr(env, _FOOT_CONTACT_STATE_BODY_WEIGHT_CACHE_ATTR, body_weight_n)
    return body_weight_n


def _outline_piecewise_scales(outline_xy: list[list[float]] | tuple[tuple[float, float], ...]) -> tuple[float, float, float, float]:
    points = np.asarray(outline_xy, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != 2:
        return (1.0, 1.0, 1.0, 1.0)

    x_vals = points[:, 0]
    y_vals = points[:, 1]
    x_neg = max(abs(float(np.min(x_vals))), 1e-6)
    x_pos = max(float(np.max(x_vals)), 1e-6)
    y_neg = max(abs(float(np.min(y_vals))), 1e-6)
    y_pos = max(float(np.max(y_vals)), 1e-6)
    return (x_neg, x_pos, y_neg, y_pos)


def _get_or_cache_cop_normalization_scales(env, tactile_sensor) -> torch.Tensor:
    """Build and cache per-foot COP piecewise normalization scales from foot outlines."""
    num_bodies = int(getattr(tactile_sensor, "num_bodies", 0))
    body_names = tuple(getattr(tactile_sensor, "body_names", []))
    cache = getattr(env, _FOOT_CONTACT_STATE_COP_SCALE_CACHE_ATTR, None)
    if isinstance(cache, dict):
        cached_scales = cache.get("scales")
        if (
            cache.get("num_bodies") == num_bodies
            and cache.get("body_names") == body_names
            and isinstance(cached_scales, torch.Tensor)
            and cached_scales.shape == (num_bodies, 4)
            and str(cached_scales.device) == str(env.device)
        ):
            return cached_scales

    from instinctlab.tasks.parkour.config.g1.foot_tactile_geometry import (
        LEFT_FOOT_OUTLINE_XY_ANKLE_ROLL,
        RIGHT_FOOT_OUTLINE_XY_ANKLE_ROLL,
    )

    left_scales = _outline_piecewise_scales(LEFT_FOOT_OUTLINE_XY_ANKLE_ROLL)
    right_scales = _outline_piecewise_scales(RIGHT_FOOT_OUTLINE_XY_ANKLE_ROLL)
    default_scales = tuple(0.5 * (left + right) for left, right in zip(left_scales, right_scales))

    scales = torch.empty((max(num_bodies, 0), 4), device=env.device, dtype=torch.float32)
    for body_id in range(num_bodies):
        body_name = body_names[body_id] if body_id < len(body_names) else ""
        body_name_l = body_name.lower()
        if "right" in body_name_l or body_name_l.startswith("r_"):
            x_neg, x_pos, y_neg, y_pos = right_scales
        elif "left" in body_name_l or body_name_l.startswith("l_"):
            x_neg, x_pos, y_neg, y_pos = left_scales
        elif body_id == 0:
            x_neg, x_pos, y_neg, y_pos = left_scales
        elif body_id == 1:
            x_neg, x_pos, y_neg, y_pos = right_scales
        else:
            x_neg, x_pos, y_neg, y_pos = default_scales

        scales[body_id, 0] = x_neg
        scales[body_id, 1] = x_pos
        scales[body_id, 2] = y_neg
        scales[body_id, 3] = y_pos

    setattr(
        env,
        _FOOT_CONTACT_STATE_COP_SCALE_CACHE_ATTR,
        {
            "num_bodies": num_bodies,
            "body_names": body_names,
            "scales": scales,
        },
    )
    return scales


def _piecewise_normalize_cop(
    cop_2d: torch.Tensor,
    tactile_ids: list[int],
    cop_scales: torch.Tensor,
    force_norm: torch.Tensor | None = None,
) -> torch.Tensor:
    """Piecewise COP normalization using left/right outline extents in ankle-roll frame."""
    if cop_2d.numel() == 0:
        return torch.zeros_like(cop_2d)

    if cop_scales.numel() == 0:
        cop_norm = torch.zeros_like(cop_2d)
    else:
        ids_t = torch.as_tensor(tactile_ids, device=cop_2d.device, dtype=torch.long)
        ids_t = ids_t.clamp_(0, cop_scales.shape[0] - 1)
        selected_scales = cop_scales.index_select(0, ids_t)

        x_neg = selected_scales[:, 0].view(1, -1).clamp_min(1e-6)
        x_pos = selected_scales[:, 1].view(1, -1).clamp_min(1e-6)
        y_neg = selected_scales[:, 2].view(1, -1).clamp_min(1e-6)
        y_pos = selected_scales[:, 3].view(1, -1).clamp_min(1e-6)

        cop_x = torch.nan_to_num(cop_2d[..., 0], nan=0.0, posinf=0.0, neginf=0.0)
        cop_y = torch.nan_to_num(cop_2d[..., 1], nan=0.0, posinf=0.0, neginf=0.0)
        cop_x_norm = torch.where(cop_x >= 0.0, cop_x / x_pos, cop_x / x_neg)
        cop_y_norm = torch.where(cop_y >= 0.0, cop_y / y_pos, cop_y / y_neg)
        cop_x_norm = torch.clamp(cop_x_norm, -1.0, 1.0)
        cop_y_norm = torch.clamp(cop_y_norm, -1.0, 1.0)

        # When contact is very weak, COP is unreliable and should not pollute the policy.
        if force_norm is not None:
            stable_contact = force_norm >= float(_FOOT_CONTACT_STATE_FORCE_GATE)
            cop_x_norm = torch.where(stable_contact, cop_x_norm, torch.zeros_like(cop_x_norm))
            cop_y_norm = torch.where(stable_contact, cop_y_norm, torch.zeros_like(cop_y_norm))

        cop_norm = torch.stack((cop_x_norm, cop_y_norm), dim=-1)

    return torch.nan_to_num(cop_norm, nan=0.0, posinf=0.0, neginf=0.0)


def _resolve_stage_to_tactile_ids(stage_sensor, tactile_sensor, num_feet: int) -> list[int]:
    stage_names = list(getattr(stage_sensor, "body_names", []))
    tactile_names = list(getattr(tactile_sensor, "body_names", []))
    num_tactile = int(getattr(tactile_sensor, "num_bodies", len(tactile_names)))
    if num_tactile <= 0:
        return [0 for _ in range(num_feet)]

    if not stage_names or not tactile_names:
        ids = list(range(min(num_feet, num_tactile)))
    else:
        tactile_name_to_id = {name: idx for idx, name in enumerate(tactile_names)}
        ids = []
        for stage_name in stage_names[:num_feet]:
            mapped_id = tactile_name_to_id.get(stage_name, len(ids))
            ids.append(int(mapped_id))

    if not ids:
        ids = list(range(min(num_feet, num_tactile)))
    while len(ids) < num_feet:
        ids.append(len(ids))
    return [int(min(max(idx, 0), num_tactile - 1)) for idx in ids[:num_feet]]


def foot_contact_state(
    env: "ManagerBasedEnv",
    stage_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_stage_filter"),
    tactile_sensor_cfg: SceneEntityCfg = SceneEntityCfg("foot_tactile"),
) -> torch.Tensor:
    """Per-foot low-dimensional contact state for actor/critic observations.

    Per-foot feature order (5 dims):
    [contact_area_ratio, F_over_body_weight, cop_x_norm, cop_y_norm, vz_norm]
    """
    stage_sensor = None
    tactile_sensor = None
    try:
        stage_sensor = env.scene.sensors[stage_sensor_cfg.name]
    except KeyError:
        stage_sensor = None
    try:
        tactile_sensor = env.scene.sensors[tactile_sensor_cfg.name]
    except KeyError:
        tactile_sensor = None

    num_envs = int(getattr(env, "num_envs", 0))
    num_feet = 2
    if stage_sensor is not None:
        num_feet = int(getattr(stage_sensor, "num_bodies", num_feet))
    elif tactile_sensor is not None:
        num_feet = int(getattr(tactile_sensor, "num_bodies", num_feet))

    if stage_sensor is None or tactile_sensor is None or num_envs <= 0 or num_feet <= 0:
        return torch.zeros((max(num_envs, 0), max(num_feet, 0), _FOOT_CONTACT_STATE_DIM), device=env.device, dtype=torch.float32)

    stage_data = stage_sensor.data
    tactile_data = tactile_sensor.data
    tactile_ids = _resolve_stage_to_tactile_ids(stage_sensor, tactile_sensor, num_feet)

    total_force = torch.nan_to_num(
        tactile_data.total_normal_force[:, tactile_ids],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clamp_min(0.0)
    cop_2d = torch.nan_to_num(
        tactile_data.cop_b[:, tactile_ids, :],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    contact_area_ratio = torch.nan_to_num(
        tactile_data.contact_area_ratio[:, tactile_ids],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)
    vz = torch.nan_to_num(
        stage_data.foot_vz[:, :num_feet],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # Force normalization by body weight keeps force channels comparable to bounded features.
    body_weight_n = _get_or_cache_body_weight_n(env, asset_name="robot")
    force_over_bw = torch.clamp(total_force / max(body_weight_n, 1e-6), min=0.0, max=1.5)

    # COP normalization uses asymmetric left/right outline extents (heel/toe and medial/lateral).
    cop_scales = _get_or_cache_cop_normalization_scales(env, tactile_sensor)
    cop_norm = _piecewise_normalize_cop(cop_2d, tactile_ids=tactile_ids, cop_scales=cop_scales, force_norm=force_over_bw)

    vz_norm = torch.clamp(vz / float(_FOOT_CONTACT_STATE_VZ_REF), min=-1.5, max=1.5)
    obs = torch.stack(
        (
            contact_area_ratio,
            force_over_bw,
            cop_norm[..., 0],
            cop_norm[..., 1],
            vz_norm,
        ),
        dim=-1,
    )
    return torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)



def _debug_visualize_image(
    image: torch.Tensor,
    scale_up_vis: int = 5,
    window_name: str = "vis_image",
) -> None:
    """Visualize images in a cv2 window for debugging purposes.

    This function normalizes images to [0, 255], handles different channel configurations,
    scales them up for better visualization, and displays them in an OpenCV window.

    Args:
        images: Image tensor in shape (H, W)
        scale_up_vis: The factor to scale up the image for better visualization if the
            resolution is too low. Defaults to 5.
        window_name: The name of the OpenCV window. Defaults to "vis_image".
    """
    # automatically normalize images to [0, 255]
    img = (image * 255.0 / image.max()).cpu().numpy().astype("uint8")  # (H, W)
    # Scale up the image for better visualization
    img = cv2.resize(img, (img.shape[1] * scale_up_vis, img.shape[0] * scale_up_vis), interpolation=cv2.INTER_AREA)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img)
    cv2.waitKey(1)


def visualizable_image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
    data_type: str = "rgb",
    debug_vis: bool = False,
    scale_up_vis: int = 5,
    history_skip_frames: int = 0,
) -> torch.Tensor:
    """Images of a specific datatype from the camera sensor.

    If the flag :attr:`normalize` is True, post-processing of the images are performed based on their
    data-types:

    - "rgb": Scales the image to (0, 1) and subtracts with the mean of the current image batch.
    - "depth" or "distance_to_camera" or "distance_to_plane": Replaces infinity values with zero.

    Args:
        env: The environment the cameras are placed within.
        sensor_cfg: The desired sensor to read from. Defaults to SceneEntityCfg("camera").
        data_type: The data type to pull from the desired camera. Defaults to "rgb".
        convert_perspective_to_orthogonal: Whether to orthogonalize perspective depth images.
            This is used only when the data type is "distance_to_camera". Defaults to False.
        normalize: Whether to normalize the images. This depends on the selected data type.
            Defaults to True.
        debug_vis: Whether to visualize the images in an animated cv2 window. Defaults to False.
        scale_up_vis: The factor to scale up the image for better visualization if the resolution is too low. Defaults to 5.
        history_skip_frames: The number of frames to skip to downsample the history data.
            For example, if the input sequence is 0,1,2,3,4,5,6; and history_skip_frames is 2,
            the output sequence will be 0,3,6. NOTE: This is only supported when 'history' is in the data_type.
            Defaults to 0. NOTE: It is recommended to set the camera update_period to env_step_dt.

    Returns:
        The images produced at the last time-step
    """
    # extract the used quantities (to enable type-hinting)
    sensor: TiledCamera | Camera | RayCasterCamera | GroupedRayCasterCamera | NoisyGroupedRayCasterCamera = (
        env.scene.sensors[sensor_cfg.name]
    )

    # obtain the input image
    images = sensor.data.output[data_type].clone()  # (N, H, W, C) or (N, history, H, W, C)
    if "history" in data_type:
        # NOTE: Only depth-related data types with history are supported. where C = 1.
        images = images.squeeze(
            -1
        )  # (N, history, H, W, C) -> (N, history, H, W), images[:, -1] shall be the latest frame.
        if history_skip_frames > 0:
            images = images[:, ::history_skip_frames, :, :]
    else:
        images = images.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

    # rgb/depth image normalization
    # NOTE: rgb image is not tested yet.

    if debug_vis:
        # (N, C, H, W) -> (C, H, N, W) -> (C*H, N*W)
        _debug_visualize_image(
            images.permute(1, 2, 0, 3).flatten(start_dim=0, end_dim=1).flatten(start_dim=1, end_dim=2), scale_up_vis
        )

    return images


class delayed_visualizable_image(ManagerTermBase):
    """A callable class that could sample delayed images from camera sensor that has history data. This is initially
    designed to use NoisyGroupedRayCasterCamera. The output shape will always be (N, num_output_frames, H, W) for now.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.sensor_cfg = cfg.params.get("sensor_cfg", SceneEntityCfg("camera"))
        self.data_type = cfg.params["data_type"]  # must provide the data_type, must have "history" in the data_type
        assert "history" in self.data_type, "data_type must have 'history' in it"
        self.sensor: NoisyGroupedRayCasterCamera = env.scene.sensors[self.sensor_cfg.name]
        self.delayed_frame_ranges = cfg.params.get("delayed_frame_ranges", (0, 0))  # (min_delay, max_delay)
        # not recommended for gaussian distribution, but it is supported.
        self.delayed_frame_distribution: Literal["uniform", "log_uniform"] = cfg.params.get(
            "delayed_frame_distribution", "uniform"
        )
        self._num_delayed_frames = torch.zeros(env.num_envs, device=env.device)  # depending on the sensor update period
        self.history_skip_frames = max(cfg.params.get("history_skip_frames", 1), 1)
        # if greater than 0, the output data from this observation term will have history dimension, else no history dimension.
        self.num_output_frames = max(cfg.params.get("num_output_frames", 0), 1)
        assert len(self.sensor.data.output[self.data_type].shape) >= 5, (
            f"sensor data of type {self.data_type} should have (N, history, H, W, C) shape, but got"
            f" {self.sensor.data.output[self.data_type].shape}"
        )
        self.sensor_history_length = self.sensor.data.output[self.data_type].shape[1]

        # build frame offset based on num_output_frames and history_skip_frames
        # use reverse order because [:, -1] gets the latest frame in sensor data. frame_offset[0] should be the largest
        # to return the oldest frame in the output.
        self.frame_offset = torch.flip(
            torch.arange(
                0,
                self.num_output_frames * self.history_skip_frames,
                self.history_skip_frames,
                device=env.device,
            ),
            dims=(0,),
        )  # (num_output_frames,)

        self.check_delay_bounds()

    def check_delay_bounds(self) -> None:
        """
        Check if the delayed frame ranges are within the bounds of the sensor history length.
        If not, raise an error.
        """
        max_delayed_frames = self.delayed_frame_ranges[1]
        frames_needed_if_no_delay = (self.num_output_frames - 1) * self.history_skip_frames + 1
        if (frames_needed_if_no_delay + max_delayed_frames) > self.sensor_history_length:
            raise ValueError(
                "The delayed frame ranges are too large for the sensor history length. The maximum delayed frames is"
                f" {max_delayed_frames}, but the frames needed if no delay is {frames_needed_if_no_delay}, which is"
                f" {frames_needed_if_no_delay + max_delayed_frames}."
            )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """
        Reset the delayed frame ranges.
        """
        if env_ids is None:
            env_ids = slice(None)
        self._num_delayed_frames[env_ids] = _randomize_prop_by_op(
            self._num_delayed_frames[env_ids].unsqueeze(-1),
            self.delayed_frame_ranges,
            None,
            slice(None),
            operation="abs",
            distribution=self.delayed_frame_distribution,
        ).squeeze(
            -1
        )  # (N,)

    def __call__(
        self,
        env: ManagerBasedEnv,
        data_type: str,
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
        history_skip_frames: int = 0,
        num_output_frames: int = 0,
        delayed_frame_ranges: tuple[int, int] = (0, 0),
        delayed_frame_distribution: Literal["uniform", "log_uniform"] = "uniform",
        debug_vis: bool = False,
        scale_up_vis: int = 5,
    ) -> torch.Tensor:
        """
        Get the delayed frames from the sensor data.
        """
        # obtain the input image
        images = self.sensor.data.output[self.data_type].clone()  # (N, history, H, W, C)
        # NOTE: Only depth-related data types with history are supported for now. where C = 1.
        images = images.squeeze(
            -1
        )  # (N, history, H, W, C) -> (N, history, H, W), images[:, -1] shall be the latest frame.
        # get the delayed frames
        frame_indices = (
            self.sensor_history_length - self.frame_offset.unsqueeze(0) - self._num_delayed_frames.unsqueeze(1) - 1
        )  # (N, num_output_frames)
        frame_indices = frame_indices.to(torch.long)
        # final safety check to avoid frame_indices being out of bounds
        assert (frame_indices >= 0).all(), f"frame_indices should be non-negative, but got {frame_indices}"
        assert (
            frame_indices < self.sensor_history_length
        ).all(), f"frame_indices should be less than the sensor history length {self.sensor_history_length}"
        # Use advanced indexing: create batch indices and use them together with frame_indices
        batch_indices = (
            torch.arange(images.shape[0], device=images.device)
            .unsqueeze(1)
            .expand(-1, frame_indices.shape[1])
            .to(torch.long)
        )
        delayed_frames = images[batch_indices, frame_indices]  # (N, num_output_frames, H, W)
        if debug_vis:
            # (N, num_output_frames, H, W) -> (num_output_frames, H, N, W) -> (num_output_frames * H, N * W)
            _debug_visualize_image(
                delayed_frames.permute(1, 2, 0, 3).flatten(start_dim=0, end_dim=1).flatten(start_dim=1, end_dim=2),
                scale_up_vis,
            )
        return delayed_frames  # still delayed_frames[:, -1] shall be the latest frame.
