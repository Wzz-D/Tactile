from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .taxel_generator_cfg import ExplicitFootTactileTemplateCfg


def explicit_foot_tactile_template_generator(cfg: ExplicitFootTactileTemplateCfg) -> dict[str, torch.Tensor]:
    """Build left/right tactile templates from explicit local sole-plane points."""

    left_outline_xy = torch.tensor(cfg.left_outline_xy, dtype=torch.float32)
    right_outline_xy = torch.tensor(cfg.right_outline_xy, dtype=torch.float32)
    left_taxel_xy = torch.tensor(cfg.left_taxel_xy, dtype=torch.float32)
    right_taxel_xy = torch.tensor(cfg.right_taxel_xy, dtype=torch.float32)

    _validate_xy_points(left_outline_xy, "left_outline_xy", min_points=3)
    _validate_xy_points(right_outline_xy, "right_outline_xy", min_points=3)
    _validate_xy_points(left_taxel_xy, "left_taxel_xy", min_points=1)
    _validate_xy_points(right_taxel_xy, "right_taxel_xy", min_points=1)

    return {
        "left_outline_xy": left_outline_xy,
        "right_outline_xy": right_outline_xy,
        "left_taxel_xy": left_taxel_xy,
        "right_taxel_xy": right_taxel_xy,
    }


def _validate_xy_points(points_xy: torch.Tensor, name: str, min_points: int) -> None:
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2).")
    if points_xy.shape[0] < min_points:
        raise ValueError(f"{name} must contain at least {min_points} points.")
