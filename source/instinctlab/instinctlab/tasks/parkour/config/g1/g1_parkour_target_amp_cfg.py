import copy
import os

from isaaclab.envs import ViewerCfg
from isaaclab.utils import configclass

import instinctlab.tasks.parkour.mdp as mdp
from instinctlab.assets.unitree_g1 import (
    G1_29DOF_TORSOBASE_POPSICLE_CFG,
    G1_29Dof_TorsoBase_symmetric_augmentation_joint_mapping,
    G1_29Dof_TorsoBase_symmetric_augmentation_joint_reverse_buf,
    beyondmimic_g1_29dof_actuators,
    beyondmimic_g1_29dof_delayed_actuators,
)
from instinctlab.motion_reference import MotionReferenceManagerCfg
from instinctlab.motion_reference.motion_files.amass_motion_cfg import AmassMotionCfg as AmassMotionCfgBase
from instinctlab.motion_reference.utils import motion_interpolate_bilinear
from instinctlab.tasks.parkour.config.parkour_env_cfg import DualCriticRewardsCfg, ROUGH_TERRAINS_CFG, ParkourEnvCfg

from instinctlab.sensors.contact_stage import ContactStageCfg
from instinctlab.sensors.foot_tactile import FootTactileCfg, FootTactileDiffusionCfg, FootTactileNoiseCfg
from instinctlab.tasks.parkour.config.g1.foot_tactile_geometry import make_ankle_roll_foot_tactile_template_cfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

__file_dir__ = os.path.dirname(os.path.realpath(__file__))
G1_CFG = copy.deepcopy(G1_29DOF_TORSOBASE_POPSICLE_CFG)
G1_CFG.spawn.merge_fixed_joints = True
G1_CFG.init_state.pos = (0.0, 0.0, 0.9)
G1_with_shoe_CFG = copy.deepcopy(G1_CFG)
G1_with_shoe_CFG.spawn.asset_path = os.path.abspath(
    f"{__file_dir__}/../../urdf/g1_29dof_torsoBase_popsicle_with_shoe_copy.urdf"
)
G1_with_shoe_CFG.spawn.collider_type = "convex_decomposition"


@configclass
class AmassMotionCfg(AmassMotionCfgBase):
    path = os.path.expanduser("/home/future/instinct/Datasets/data&model/parkour_motion_reference")
    retargetting_func = None
    filtered_motion_selection_filepath = os.path.expanduser("/home/future/instinct/Datasets/data&model/parkour_motion_reference/parkour_motion_without_run.yaml")
    print(filtered_motion_selection_filepath)
    print('+++++++++++++++++++++++++++++++++++++++++++++++++')
    motion_start_from_middle_range = [0.0, 0.9]
    motion_start_height_offset = 0.0
    ensure_link_below_zero_ground = False
    buffer_device = "output_device"
    motion_interpolate_func = motion_interpolate_bilinear
    velocity_estimation_method = "frontward"


motion_reference_cfg = MotionReferenceManagerCfg(
    prim_path="{ENV_REGEX_NS}/Robot/torso_link",
    robot_model_path=G1_CFG.spawn.asset_path,
    reference_prim_path="/World/envs/env_.*/RobotReference/torso_link",
    symmetric_augmentation_link_mapping=[0, 1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12],
    symmetric_augmentation_joint_mapping=G1_29Dof_TorsoBase_symmetric_augmentation_joint_mapping,
    symmetric_augmentation_joint_reverse_buf=G1_29Dof_TorsoBase_symmetric_augmentation_joint_reverse_buf,
    frame_interval_s=0.02,
    update_period=0.02,
    num_frames=10,
    motion_buffers={
        "run_walk": AmassMotionCfg(),
    },
    link_of_interests=[
        "pelvis",
        "torso_link",
        "left_shoulder_roll_link",
        "right_shoulder_roll_link",
        "left_elbow_link",
        "right_elbow_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_hip_roll_link",
        "right_hip_roll_link",
        "left_knee_link",
        "right_knee_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    ],
    mp_split_method="Even",
)


ROUGH_TERRAINS_CFG_PLAY = copy.deepcopy(ROUGH_TERRAINS_CFG)
for sub_terrain_name, sub_terrain_cfg in ROUGH_TERRAINS_CFG_PLAY.sub_terrains.items():
    sub_terrain_cfg.wall_prob = [0.0, 0.0, 0.0, 0.0]


@configclass
class G1ParkourRoughEnvCfg(ParkourEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG
        self.scene.robot = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.actuators = beyondmimic_g1_29dof_delayed_actuators
        self.scene.motion_reference = motion_reference_cfg


class ShoeConfigMixin:
    def apply_shoe_config(self):
        self.scene.robot = G1_with_shoe_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.leg_volume_points.points_generator.z_min = -0.063
        self.scene.leg_volume_points.points_generator.z_max = -0.023
        if hasattr(self.rewards, "rewards") and getattr(self.rewards.rewards, "feet_at_plane", None) is not None:
            self.rewards.rewards.feet_at_plane.params["height_offset"] = 0.058
        if hasattr(self.rewards, "sparse") and getattr(self.rewards.sparse, "feet_at_plane", None) is not None:
            self.rewards.sparse.feet_at_plane.params["height_offset"] = 0.058
        structured_noise_cfg = FootTactileNoiseCfg(
            enable=True,
            use_structured_profiles=True,
        )
        # First training version: keep the tactile randomization structured and low-frequency.
        # This targets calibration/latency/stage-boundary mismatch without aggressively distorting
        # total force or CoP semantics on day one.
        structured_noise_cfg.measurement_profile_cfg.per_foot_gain.enable = True
        structured_noise_cfg.measurement_profile_cfg.per_foot_gain.gain_range = (0.97, 1.03)
        structured_noise_cfg.measurement_profile_cfg.regional_gain.enable = True
        structured_noise_cfg.measurement_profile_cfg.regional_gain.forefoot_gain_range = (0.98, 1.02)
        structured_noise_cfg.measurement_profile_cfg.regional_gain.heel_gain_range = (0.98, 1.02)
        structured_noise_cfg.measurement_profile_cfg.regional_gain.medial_gain_range = (0.98, 1.02)
        structured_noise_cfg.measurement_profile_cfg.regional_gain.lateral_gain_range = (0.98, 1.02)
        structured_noise_cfg.measurement_profile_cfg.regional_gain.edge_gain_range = (0.96, 1.00)
        structured_noise_cfg.measurement_profile_cfg.residual_gaussian.enable = True
        structured_noise_cfg.measurement_profile_cfg.residual_gaussian.multiplicative_std = 0.01
        structured_noise_cfg.measurement_profile_cfg.deadzone.enable = True
        structured_noise_cfg.measurement_profile_cfg.deadzone.force_threshold = 0.5
        structured_noise_cfg.transport_profile_cfg.delay.enable = True
        structured_noise_cfg.transport_profile_cfg.delay.delay_prob = 0.05
        structured_noise_cfg.transport_profile_cfg.delay.max_delay_frames = 1
        structured_noise_cfg.transport_profile_cfg.frame_hold.enable = True
        structured_noise_cfg.transport_profile_cfg.frame_hold.hold_prob = 0.02
        structured_noise_cfg.transport_profile_cfg.hysteresis.enable = True
        structured_noise_cfg.transport_profile_cfg.hysteresis.alpha = 0.90
        # foot tactile sensor config
        foot_tactile_cfg = FootTactileCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link",
            template_cfg=make_ankle_roll_foot_tactile_template_cfg(),
            taxel_z_offset=-0.038,
            raycast_offset=0.0003,
            pad_thickness=0.0014,
            max_support_dist=0.004,
            support_weight_band=0.004,
            support_weight_rho=50.0,
            alignment_gate_a0=0.3,
            alignment_gate_q=1.5,
            align_mix=0.4,
            min_force_threshold=5.0,
            active_taxel_threshold=0.5,
            diffusion_cfg=FootTactileDiffusionCfg(
                enable_neighbor_diffusion=True,
                diffusion_knn=4,
                diffusion_alpha=0.10,
                diffusion_iters=1,
                preserve_total_force_after_diffusion=True,
            ),
            noise_cfg=structured_noise_cfg,
            update_period=0.02,
            debug_vis=False,
        )
        foot_tactile_cfg.threshold_randomization_cfg.enable = True
        foot_tactile_cfg.threshold_randomization_cfg.min_force_threshold_range = (4.5, 5.8)
        foot_tactile_cfg.threshold_randomization_cfg.active_taxel_threshold_range = (0.40, 0.60)
        self.scene.foot_tactile = foot_tactile_cfg
        contact_stage_cfg = ContactStageCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link",
            update_period=0.02,
            debug_vis=False,
            enable_self_check=False,
        )
        contact_stage_cfg.decision_randomization_cfg.enable = True
        contact_stage_cfg.decision_randomization_cfg.contact_force_on_range = (22.0, 28.0)
        contact_stage_cfg.decision_randomization_cfg.contact_force_off_range = (10.5, 13.5)
        contact_stage_cfg.decision_randomization_cfg.contact_area_on_range = (0.10, 0.14)
        contact_stage_cfg.decision_randomization_cfg.contact_area_off_range = (0.05, 0.07)
        contact_stage_cfg.decision_randomization_cfg.derivative_filter_alpha_range = (0.78, 0.82)
        self.scene.contact_stage_filter = contact_stage_cfg
        self.events.bind_foot_tactile = EventTerm(
            func=mdp.bind_foot_tactile,
            mode="startup",
            params={
                "tactile_cfg": SceneEntityCfg("foot_tactile"),
                "contact_forces_cfg": SceneEntityCfg("contact_forces_foot"),
            },
        )
        self.events.bind_contact_stage_filter = EventTerm(
            func=mdp.bind_contact_stage_filter,
            mode="startup",
            params={
                "stage_cfg": SceneEntityCfg("contact_stage_filter"),
                "tactile_cfg": SceneEntityCfg("foot_tactile"),
            },
        )


@configclass
class G1ParkourRoughEnvCfg_PLAY(G1ParkourRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG_PLAY
        # make a smaller scene for play
        self.scene.num_envs = 10
        self.viewer = ViewerCfg(
            eye=[4.0, 0.75, 1.0],
            lookat=[0.0, 0.75, 0.0],
            origin_type="asset_root",
            asset_name="robot",
        )

        self.scene.env_spacing = 2.5
        self.episode_length_s = 10
        self.terminations.root_height = None
        # spawn the robot randomly in the grid (instead of their terrain levels)
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 10
            self.scene.terrain.terrain_generator.num_cols = 10

        self.scene.leg_volume_points.debug_vis = True
        self.commands.base_velocity.debug_vis = True
        self.events.physics_material = None
        self.events.reset_robot_joints.params = {
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        }


@configclass
class G1ParkourEnvCfg(G1ParkourRoughEnvCfg, ShoeConfigMixin):
    def __post_init__(self):
        super().__post_init__()
        self.apply_shoe_config()


@configclass
class G1ParkourEnvCfg_PLAY(G1ParkourRoughEnvCfg_PLAY, ShoeConfigMixin):
    def __post_init__(self):
        super().__post_init__()
        self.apply_shoe_config()


@configclass
class G1ParkourDualCriticEnvCfg(G1ParkourEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards = DualCriticRewardsCfg()
        self.rewards.sparse.feet_at_plane.params["height_offset"] = 0.058


@configclass
class G1ParkourDualCriticEnvCfg_PLAY(G1ParkourEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.rewards = DualCriticRewardsCfg()
        self.rewards.sparse.feet_at_plane.params["height_offset"] = 0.058


@configclass
class G1ParkourEnvCfg_STAND_DEBUG(G1ParkourEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1e10

        # 彻底关闭终止
        self.terminations.time_out = None
        self.terminations.terrain_out_bound = None
        self.terminations.base_contact = None
        self.terminations.bad_orientation = None
        self.terminations.root_height = None


        self.events.physics_material = None
        # 1) 关掉 base 随机速度和姿态扰动
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }

        # 2) 关掉 joint reset 偏移
        self.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
