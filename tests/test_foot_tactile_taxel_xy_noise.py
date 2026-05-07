from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def _install_import_stubs() -> None:
    isaaclab_module = types.ModuleType("isaaclab")
    isaaclab_utils_module = types.ModuleType("isaaclab.utils")
    isaaclab_utils_module.configclass = dataclass
    isaaclab_module.utils = isaaclab_utils_module
    sys.modules.setdefault("isaaclab", isaaclab_module)
    sys.modules.setdefault("isaaclab.utils", isaaclab_utils_module)

    for package_name in ("instinctlab", "instinctlab.sensors", "instinctlab.sensors.foot_tactile"):
        package = types.ModuleType(package_name)
        package.__path__ = []
        sys.modules.setdefault(package_name, package)


def _load_module(module_name: str, relative_path: str):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_install_import_stubs()
noise_cfg_module = _load_module(
    "instinctlab.sensors.foot_tactile.foot_tactile_noise_cfg",
    "source/instinctlab/instinctlab/sensors/foot_tactile/foot_tactile_noise_cfg.py",
)
noise_module = _load_module(
    "instinctlab.sensors.foot_tactile.foot_tactile_noise",
    "source/instinctlab/instinctlab/sensors/foot_tactile/foot_tactile_noise.py",
)
core_module = _load_module(
    "instinctlab.sensors.foot_tactile.core",
    "source/instinctlab/instinctlab/sensors/foot_tactile/core.py",
)

FootTactileNoiseCfg = noise_cfg_module.FootTactileNoiseCfg
FootTactileNoiseModel = noise_module.FootTactileNoiseModel
compute_cop_b = core_module.compute_cop_b


def test_taxel_xy_noise_disabled_matches_base_coordinates() -> None:
    base_xy = torch.tensor([[[0.0, 0.0], [0.02, 0.0], [0.0, 0.03]]], dtype=torch.float32)
    valid_mask = torch.tensor([[True, True, False]])
    cfg = FootTactileNoiseCfg(enable=True, force_relative_error_max=0.0, taxel_xy_noise_std=0.0)

    model = FootTactileNoiseModel(
        cfg=cfg,
        num_envs=2,
        num_bodies=1,
        max_taxels=3,
        device="cpu",
        dtype=torch.float32,
        base_taxel_xy_b=base_xy,
        valid_taxel_mask=valid_mask,
    )

    measured_xy = model.get_measured_taxel_xy_b()
    expected_xy = base_xy.unsqueeze(0).expand_as(measured_xy)

    assert measured_xy.shape == (2, 1, 3, 2)
    assert torch.allclose(measured_xy, expected_xy)


def test_taxel_xy_noise_is_episode_fixed_clipped_and_valid_taxel_only() -> None:
    base_xy = torch.tensor([[[0.0, 0.0], [0.02, 0.0], [0.0, 0.03]]], dtype=torch.float32)
    valid_mask = torch.tensor([[True, True, False]])
    cfg = FootTactileNoiseCfg(
        enable=True,
        seed=7,
        force_relative_error_max=0.0,
        taxel_xy_noise_std=0.01,
        taxel_xy_noise_clip=0.0015,
    )

    model = FootTactileNoiseModel(
        cfg=cfg,
        num_envs=2,
        num_bodies=1,
        max_taxels=3,
        device="cpu",
        dtype=torch.float32,
        base_taxel_xy_b=base_xy,
        valid_taxel_mask=valid_mask,
    )

    measured_xy_before = model.get_measured_taxel_xy_b().clone()
    force = torch.ones((2, 1, 3), dtype=torch.float32)
    model.apply(force, valid_mask, torch.arange(2))
    measured_xy_after = model.get_measured_taxel_xy_b()

    offset = measured_xy_before - base_xy.unsqueeze(0)
    assert torch.allclose(measured_xy_before, measured_xy_after)
    assert torch.any(offset[:, :, :2, :].abs() > 0.0)
    assert torch.all(offset[:, :, :2, :].abs() <= 0.0015 + 1e-7)
    assert torch.allclose(offset[:, :, 2:, :], torch.zeros_like(offset[:, :, 2:, :]))


def test_compute_cop_accepts_env_specific_measured_taxel_coordinates() -> None:
    taxel_force = torch.tensor([[[1.0, 1.0]], [[1.0, 1.0]]], dtype=torch.float32)
    valid_mask = torch.tensor([[True, True]])
    measured_xy = torch.tensor(
        [
            [[[0.0, 0.0], [2.0, 0.0]]],
            [[[1.0, 0.0], [3.0, 0.0]]],
        ],
        dtype=torch.float32,
    )

    cop_b = compute_cop_b(taxel_force, measured_xy, valid_mask)

    assert torch.allclose(cop_b[0, 0], torch.tensor([1.0, 0.0]), atol=1e-6)
    assert torch.allclose(cop_b[1, 0], torch.tensor([2.0, 0.0]), atol=1e-6)
