from __future__ import annotations

from dataclasses import MISSING, dataclass

import torch


@dataclass
class FootTactileData:
    """Data container for the foot tactile sensor."""

    pos_w: torch.Tensor = MISSING
    quat_w: torch.Tensor = MISSING
    foot_normal_w: torch.Tensor = MISSING

    taxel_xy_b: torch.Tensor = MISSING
    valid_taxel_mask: torch.Tensor = MISSING
    edge_taxel_mask: torch.Tensor = MISSING

    taxel_pos_w: torch.Tensor = MISSING
    support_dist: torch.Tensor = MISSING
    support_valid_mask: torch.Tensor = MISSING
    support_normal_w: torch.Tensor = MISSING
    support_alignment: torch.Tensor = MISSING

    taxel_weight_clean: torch.Tensor = MISSING
    taxel_weight_aligned: torch.Tensor = MISSING

    total_normal_force: torch.Tensor = MISSING
    taxel_force_clean: torch.Tensor = MISSING
    taxel_force_diffused: torch.Tensor = MISSING
    taxel_force: torch.Tensor = MISSING

    contact_area_ratio: torch.Tensor = MISSING
    edge_force_ratio: torch.Tensor = MISSING
    peak_force: torch.Tensor = MISSING
    mean_force: torch.Tensor = MISSING
    cop_b: torch.Tensor = MISSING

    @staticmethod
    def make_zero(
        num_envs: int,
        num_bodies: int,
        max_taxels: int,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "FootTactileData":
        return FootTactileData(
            pos_w=torch.zeros((num_envs, num_bodies, 3), device=device, dtype=dtype),
            quat_w=torch.zeros((num_envs, num_bodies, 4), device=device, dtype=dtype),
            foot_normal_w=torch.zeros((num_envs, num_bodies, 3), device=device, dtype=dtype),
            taxel_xy_b=torch.zeros((num_bodies, max_taxels, 2), device=device, dtype=dtype),
            valid_taxel_mask=torch.zeros((num_bodies, max_taxels), device=device, dtype=torch.bool),
            edge_taxel_mask=torch.zeros((num_bodies, max_taxels), device=device, dtype=torch.bool),
            taxel_pos_w=torch.zeros((num_envs, num_bodies, max_taxels, 3), device=device, dtype=dtype),
            support_dist=torch.full((num_envs, num_bodies, max_taxels), float("inf"), device=device, dtype=dtype),
            support_valid_mask=torch.zeros((num_envs, num_bodies, max_taxels), device=device, dtype=torch.bool),
            support_normal_w=torch.zeros((num_envs, num_bodies, max_taxels, 3), device=device, dtype=dtype),
            support_alignment=torch.zeros((num_envs, num_bodies, max_taxels), device=device, dtype=dtype),
            taxel_weight_clean=torch.zeros((num_envs, num_bodies, max_taxels), device=device, dtype=dtype),
            taxel_weight_aligned=torch.zeros((num_envs, num_bodies, max_taxels), device=device, dtype=dtype),
            total_normal_force=torch.zeros((num_envs, num_bodies), device=device, dtype=dtype),
            taxel_force_clean=torch.zeros((num_envs, num_bodies, max_taxels), device=device, dtype=dtype),
            taxel_force_diffused=torch.zeros((num_envs, num_bodies, max_taxels), device=device, dtype=dtype),
            taxel_force=torch.zeros((num_envs, num_bodies, max_taxels), device=device, dtype=dtype),
            contact_area_ratio=torch.zeros((num_envs, num_bodies), device=device, dtype=dtype),
            edge_force_ratio=torch.zeros((num_envs, num_bodies), device=device, dtype=dtype),
            peak_force=torch.zeros((num_envs, num_bodies), device=device, dtype=dtype),
            mean_force=torch.zeros((num_envs, num_bodies), device=device, dtype=dtype),
            cop_b=torch.zeros((num_envs, num_bodies, 2), device=device, dtype=dtype),
        )
