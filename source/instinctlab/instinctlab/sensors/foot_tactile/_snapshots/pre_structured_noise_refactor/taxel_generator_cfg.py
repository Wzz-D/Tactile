from __future__ import annotations

from dataclasses import MISSING
from typing import Callable

from isaaclab.utils import configclass

from .taxel_generator import explicit_foot_tactile_template_generator


@configclass
class FootTactileTemplateCfg:
    """Base config for building local foot tactile templates."""

    func: Callable = MISSING  # type: ignore


@configclass
class ExplicitFootTactileTemplateCfg(FootTactileTemplateCfg):
    """Use explicit left/right outline and taxel points in local sole plane."""

    func: Callable = explicit_foot_tactile_template_generator

    left_outline_xy: list[list[float]] = MISSING
    right_outline_xy: list[list[float]] = MISSING
    left_taxel_xy: list[list[float]] = MISSING
    right_taxel_xy: list[list[float]] = MISSING
