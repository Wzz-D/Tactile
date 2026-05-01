from __future__ import annotations

import copy
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "source" / "instinctlab" / "instinctlab"
FOOT_ROOT = SOURCE_ROOT / "sensors" / "foot_tactile"
CONTACT_ROOT = SOURCE_ROOT / "sensors" / "contact_stage"


def _simple_configclass(cls):
    annotations = getattr(cls, "__annotations__", {})
    defaults = {name: getattr(cls, name) for name in annotations if hasattr(cls, name)}

    def __init__(self, **kwargs):
        unknown = set(kwargs) - set(annotations)
        if unknown:
            raise TypeError(f"Unexpected config args: {sorted(unknown)}")
        for name in annotations:
            if name in kwargs:
                value = kwargs[name]
            elif name in defaults:
                value = copy.deepcopy(defaults[name])
            else:
                raise TypeError(f"Missing required config arg: {name}")
            setattr(self, name, value)

    cls.__init__ = __init__
    return cls


def _install_test_stubs() -> None:
    if "isaaclab" not in sys.modules:
        sys.modules["isaaclab"] = types.ModuleType("isaaclab")

    if "isaaclab.utils" not in sys.modules:
        utils_mod = types.ModuleType("isaaclab.utils")
        utils_mod.configclass = _simple_configclass
        sys.modules["isaaclab.utils"] = utils_mod

    if "isaaclab.sensors" not in sys.modules:
        sensors_mod = types.ModuleType("isaaclab.sensors")

        class SensorBaseCfg:
            def validate(self):
                return None

        sensors_mod.SensorBaseCfg = SensorBaseCfg
        sys.modules["isaaclab.sensors"] = sensors_mod

    if "isaaclab.sensors.sensor_base" not in sys.modules:
        sensor_base_mod = types.ModuleType("isaaclab.sensors.sensor_base")

        class SensorBase:
            def __init__(self, cfg):
                self.cfg = cfg

        sensor_base_mod.SensorBase = SensorBase
        sys.modules["isaaclab.sensors.sensor_base"] = sensor_base_mod

    if "isaaclab.markers" not in sys.modules:
        markers_mod = types.ModuleType("isaaclab.markers")

        class VisualizationMarkers:
            def __init__(self, cfg):
                self.cfg = cfg

            def set_visibility(self, visible: bool):
                self.visible = visible

            def visualize(self, **kwargs):
                self.last = kwargs

        class VisualizationMarkersCfg:
            def __init__(self, prim_path=None, markers=None):
                self.prim_path = prim_path
                self.markers = markers or {}

        markers_mod.VisualizationMarkers = VisualizationMarkers
        markers_mod.VisualizationMarkersCfg = VisualizationMarkersCfg
        sys.modules["isaaclab.markers"] = markers_mod

    if "isaaclab.sim" not in sys.modules:
        sim_mod = types.ModuleType("isaaclab.sim")

        class PreviewSurfaceCfg:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class SphereCfg:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        sim_mod.PreviewSurfaceCfg = PreviewSurfaceCfg
        sim_mod.SphereCfg = SphereCfg
        sim_mod.find_matching_prims = lambda expr: []
        sys.modules["isaaclab.sim"] = sim_mod

    if "isaaclab.utils.math" not in sys.modules:
        math_mod = types.ModuleType("isaaclab.utils.math")
        math_mod.convert_quat = lambda x, to=None: x
        math_mod.quat_apply = lambda quat, vec: vec
        math_mod.transform_points = lambda points, pos, quat: points + pos.unsqueeze(-2)
        sys.modules["isaaclab.utils.math"] = math_mod

    if "isaaclab.utils.string" not in sys.modules:
        string_mod = types.ModuleType("isaaclab.utils.string")
        string_mod.resolve_matching_names = lambda keys, names, preserve_order=False: (list(range(len(names))), names)
        sys.modules["isaaclab.utils.string"] = string_mod

    if "isaaclab.utils.warp" not in sys.modules:
        warp_mod = types.ModuleType("isaaclab.utils.warp")
        warp_mod.convert_to_warp_mesh = lambda *args, **kwargs: None
        warp_mod.raycast_mesh = lambda *args, **kwargs: (None, None, None, None)
        sys.modules["isaaclab.utils.warp"] = warp_mod

    if "omni" not in sys.modules:
        sys.modules["omni"] = types.ModuleType("omni")
    if "omni.physics" not in sys.modules:
        sys.modules["omni.physics"] = types.ModuleType("omni.physics")
    if "omni.physics.tensors" not in sys.modules:
        sys.modules["omni.physics.tensors"] = types.ModuleType("omni.physics.tensors")
    if "omni.physics.tensors.impl" not in sys.modules:
        sys.modules["omni.physics.tensors.impl"] = types.ModuleType("omni.physics.tensors.impl")
    if "omni.physics.tensors.impl.api" not in sys.modules:
        api_mod = types.ModuleType("omni.physics.tensors.impl.api")
        api_mod.create_simulation_view = lambda backend: None
        api_mod.RigidBodyView = object
        sys.modules["omni.physics.tensors.impl.api"] = api_mod

    for pkg_name, pkg_path in (
        ("instinctlab", SOURCE_ROOT),
        ("instinctlab.sensors", SOURCE_ROOT / "sensors"),
        ("instinctlab.sensors.foot_tactile", FOOT_ROOT),
        ("instinctlab.sensors.contact_stage", CONTACT_ROOT),
    ):
        if pkg_name not in sys.modules:
            module = types.ModuleType(pkg_name)
            module.__path__ = [str(pkg_path)]
            sys.modules[pkg_name] = module


def _load_module(module_name: str, file_path: Path):
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def loaded_modules():
    _install_test_stubs()
    modules = {}
    modules["core"] = _load_module("instinctlab.sensors.foot_tactile.core", FOOT_ROOT / "core.py")
    modules["noise_cfg"] = _load_module(
        "instinctlab.sensors.foot_tactile.foot_tactile_noise_cfg",
        FOOT_ROOT / "foot_tactile_noise_cfg.py",
    )
    modules["data"] = _load_module(
        "instinctlab.sensors.foot_tactile.foot_tactile_data",
        FOOT_ROOT / "foot_tactile_data.py",
    )
    modules["noise"] = _load_module(
        "instinctlab.sensors.foot_tactile.foot_tactile_noise",
        FOOT_ROOT / "foot_tactile_noise.py",
    )
    modules["foot_cfg"] = _load_module(
        "instinctlab.sensors.foot_tactile.foot_tactile_cfg",
        FOOT_ROOT / "foot_tactile_cfg.py",
    )
    modules["foot"] = _load_module(
        "instinctlab.sensors.foot_tactile.foot_tactile",
        FOOT_ROOT / "foot_tactile.py",
    )
    modules["contact_data"] = _load_module(
        "instinctlab.sensors.contact_stage.contact_stage_data",
        CONTACT_ROOT / "contact_stage_data.py",
    )
    modules["contact_filter"] = _load_module(
        "instinctlab.sensors.contact_stage.contact_stage_filter",
        CONTACT_ROOT / "contact_stage_filter.py",
    )
    modules["contact_cfg"] = _load_module(
        "instinctlab.sensors.contact_stage.contact_stage_cfg",
        CONTACT_ROOT / "contact_stage_cfg.py",
    )
    return modules


def test_snapshot_contains_pre_refactor_files():
    snapshot_dir = FOOT_ROOT / "_snapshots" / "pre_structured_noise_refactor"
    expected = {
        "__init__.py",
        "core.py",
        "foot_tactile.py",
        "foot_tactile_cfg.py",
        "foot_tactile_data.py",
        "foot_tactile_noise.py",
        "foot_tactile_noise_cfg.py",
        "taxel_generator.py",
        "taxel_generator_cfg.py",
    }
    assert snapshot_dir.is_dir()
    assert expected.issubset({path.name for path in snapshot_dir.iterdir() if path.is_file()})


def test_core_supports_tensor_thresholds_and_measured_xy(loaded_modules):
    core = loaded_modules["core"]

    taxel_force = torch.tensor([[[0.0, 1.5, 3.0]], [[0.0, 1.5, 3.0]]], dtype=torch.float32)
    valid_mask = torch.tensor([[True, True, True]])
    thresholds = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    ratio = core.compute_contact_area_ratio(taxel_force, valid_mask, thresholds)
    assert torch.allclose(ratio[:, 0], torch.tensor([2.0 / 3.0, 1.0 / 3.0]))

    taxel_xy = torch.tensor(
        [
            [[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]],
            [[[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]]],
        ],
        dtype=torch.float32,
    )
    cop = core.compute_cop_b(taxel_force, taxel_xy, valid_mask)
    assert torch.allclose(cop[0, 0], torch.tensor([5.0 / 3.0, 0.0]))
    assert torch.allclose(cop[1, 0], torch.tensor([0.0, 5.0 / 3.0]))


def test_noise_model_legacy_path_is_identity_when_disabled(loaded_modules):
    noise_cfg_mod = loaded_modules["noise_cfg"]
    noise_mod = loaded_modules["noise"]

    cfg = noise_cfg_mod.FootTactileNoiseCfg(enable=False)
    model = noise_mod.FootTactileNoiseModel(
        cfg=cfg,
        num_envs=2,
        num_bodies=1,
        max_taxels=3,
        device="cpu",
        dtype=torch.float32,
        base_taxel_xy_b=torch.zeros((1, 3, 2)),
        valid_taxel_mask=torch.tensor([[True, True, False]]),
        edge_taxel_mask=torch.tensor([[False, False, False]]),
        body_sides=["left"],
    )

    force = torch.tensor(
        [
            [[1.0, 2.0, 9.0]],
            [[3.0, 4.0, 8.0]],
        ],
        dtype=torch.float32,
    )
    measured = model.apply(force, torch.tensor([[True, True, False]]), env_ids=torch.tensor([0, 1]))
    expected = torch.tensor(
        [
            [[1.0, 2.0, 0.0]],
            [[3.0, 4.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(measured, expected)


def test_structured_measurement_profile_applies_gain_and_geometry(loaded_modules):
    noise_cfg_mod = loaded_modules["noise_cfg"]
    noise_mod = loaded_modules["noise"]

    cfg = noise_cfg_mod.FootTactileNoiseCfg(enable=True, use_structured_profiles=True)
    cfg.measurement_profile_cfg.per_foot_gain.enable = True
    cfg.measurement_profile_cfg.per_foot_gain.gain_range = (0.5, 0.5)
    cfg.measurement_profile_cfg.taxel_geometry_perturb.enable = True
    cfg.measurement_profile_cfg.taxel_geometry_perturb.per_foot_xy_offset_std = 0.02
    cfg.seed = 7

    base_xy = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]], dtype=torch.float32)
    valid_mask = torch.tensor([[True, True, True]])
    model = noise_mod.FootTactileNoiseModel(
        cfg=cfg,
        num_envs=1,
        num_bodies=1,
        max_taxels=3,
        device="cpu",
        dtype=torch.float32,
        base_taxel_xy_b=base_xy,
        valid_taxel_mask=valid_mask,
        edge_taxel_mask=torch.tensor([[False, False, True]]),
        body_sides=["left"],
    )

    force = torch.full((1, 1, 3), 4.0)
    measured = model.apply(force, valid_mask, env_ids=torch.tensor([0]))
    assert torch.allclose(measured, torch.full((1, 1, 3), 2.0))
    measured_xy = model.get_measured_taxel_xy_b(torch.tensor([0]))
    assert measured_xy.shape == (1, 1, 3, 2)
    assert not torch.allclose(measured_xy[0, 0], base_xy[0])


def test_foot_tactile_threshold_randomization_samples_ranges(loaded_modules):
    foot_cfg_mod = loaded_modules["foot_cfg"]
    foot_mod = loaded_modules["foot"]

    sensor = object.__new__(foot_mod.FootTactile)
    sensor.device = torch.device("cpu")
    sensor._num_envs = 3
    sensor._num_bodies = 2
    sensor._data = types.SimpleNamespace(taxel_force=torch.zeros((3, 2, 4), dtype=torch.float32))
    cfg = foot_cfg_mod.FootTactileCfg()
    cfg.threshold_randomization_cfg = foot_cfg_mod.make_tactile_threshold_randomization_cfg()
    sensor.cfg = cfg

    sensor._init_threshold_randomization()

    assert sensor._min_force_threshold.shape == (3, 2)
    assert sensor._active_taxel_threshold.shape == (3, 2)
    assert torch.all(sensor._min_force_threshold >= 4.0)
    assert torch.all(sensor._min_force_threshold <= 6.5)
    assert torch.all(sensor._active_taxel_threshold >= 0.8)
    assert torch.all(sensor._active_taxel_threshold <= 1.2)


def test_contact_stage_decision_randomization_preserves_threshold_order(loaded_modules):
    cfg_mod = loaded_modules["contact_cfg"]
    filter_mod = loaded_modules["contact_filter"]

    stage_filter = object.__new__(filter_mod.ContactStageFilter)
    stage_filter.device = torch.device("cpu")
    stage_filter._num_envs = 4
    stage_filter._num_bodies = 2
    cfg = cfg_mod.ContactStageCfg()
    cfg.decision_randomization_cfg = cfg_mod.make_contact_stage_noise_E_threshold_cfg()
    stage_filter.cfg = cfg

    stage_filter._init_decision_randomization()

    assert torch.all(stage_filter._contact_force_on_threshold > stage_filter._contact_force_off_threshold)
    assert torch.all(stage_filter._contact_area_on_threshold > stage_filter._contact_area_off_threshold)
    assert torch.all(stage_filter._derivative_filter_alpha >= 0.0)
    assert torch.all(stage_filter._derivative_filter_alpha <= 1.0)
