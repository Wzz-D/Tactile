from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.sensors import SensorBaseCfg
from isaaclab.utils import configclass

from .foot_tactile import FootTactile
from .foot_tactile_noise_cfg import FootTactileNoiseCfg
from .taxel_generator_cfg import FootTactileTemplateCfg

FOOT_TACTILE_VISUALIZER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/footTactile",
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.006,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 1.0)),
        ),
        "sphere_active": sim_utils.SphereCfg(
            radius=0.006,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.35, 0.1)),
        ),
    },
)


@configclass
class FootTactileDiffusionCfg:
    """Configuration for lightweight within-foot kNN diffusion."""

    enable_neighbor_diffusion: bool = True
    diffusion_knn: int = 4
    diffusion_alpha: float = 0.08
    diffusion_iters: int = 1
    preserve_total_force_after_diffusion: bool = True


@configclass
class FootTactileCfg(SensorBaseCfg):
    """Configuration for the virtual foot insole tactile sensor."""

    class_type: type = FootTactile

    filter_prim_paths_expr: list[str] = list()
    template_cfg: FootTactileTemplateCfg = MISSING

    foot_local_normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    taxel_z_offset: float = -0.057

    raycast_offset: float = 0.0003
    max_support_dist: float = 0.003
    pad_thickness: float = 0.0014

    support_weight_band: float | None = None
    support_weight_rho: float = 50.0

    alignment_gate_a0: float = 0.3
    alignment_gate_q: float = 1.5
    align_mix: float = 0.4

    min_force_threshold: float = 5.0
    active_taxel_threshold: float = 1.0

    edge_margin: float = 0.01

    diffusion_cfg: FootTactileDiffusionCfg = FootTactileDiffusionCfg()
    noise_cfg: FootTactileNoiseCfg = FootTactileNoiseCfg()

    visualizer_cfg: VisualizationMarkersCfg = FOOT_TACTILE_VISUALIZER_CFG
