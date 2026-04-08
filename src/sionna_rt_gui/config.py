#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import logging
import os
import yaml
import json
import ast
import re

from omegaconf import OmegaConf
from sionna import rt
from sionna.rt.antenna_pattern import (
    antenna_pattern_registry,
    polarization_registry,
)

from . import DATA_DIR, CONFIGS_DIR

# ------------------------

NVIDIA_GREEN = (0.4627, 0.7255, 0.0, 1.0)
NVIDIA_GREEN_DARK = tuple(0.95 * c for c in NVIDIA_GREEN)
NVIDIA_GREEN_DARKER = tuple(0.9 * c for c in NVIDIA_GREEN)

# TODO: use a simpler name once Polyscope allows naming from Python.
DEFAULT_SLICE_PLANE_NAME = "Scene Slice Plane 0"

# ------------------------


@dataclass(kw_only=True)
class AntennaArrayConfig:
    """
    Class holding configuration parameters for an antenna array.
    This is needed because once created, antenna array objects don't
    expose those fields.
    """

    num_rows: int = 1
    num_cols: int = 1
    vertical_spacing: float = 0.5
    horizontal_spacing: float = 0.5
    pattern_i: int = antenna_pattern_registry.list().index("iso")
    polarization_i: int = polarization_registry.list().index("V")

    @property
    def pattern(self) -> str:
        return antenna_pattern_registry.list()[self.pattern_i]

    @property
    def polarization(self) -> str:
        return polarization_registry.list()[self.polarization_i]

    def create(self) -> rt.AntennaArray:
        return rt.PlanarArray(
            num_rows=self.num_rows,
            num_cols=self.num_cols,
            vertical_spacing=self.vertical_spacing,
            horizontal_spacing=self.horizontal_spacing,
            pattern=self.pattern,
            polarization=self.polarization,
        )


# ------------------------


@dataclass(kw_only=True)
class RadioMapConfig:
    auto_update: bool = True
    accumulate_max_samples_per_tx: int = int(1e10)

    # -- Computation
    center: tuple[float, float, float] | None = None
    orientation: tuple[float, float, float] | None = None
    size: tuple[float, float] | None = None
    cell_size: tuple[float, float] = (1.0, 1.0)
    measurement_surface: str | None = None
    # precoding_vec: tuple[mi.TensorXf, mi.TensorXf] | None = None
    log_samples_per_it: float = 8.0
    max_depth: int = 2
    los: bool = True
    specular_reflection: bool = True
    diffuse_reflection: bool = True
    refraction: bool = True
    diffraction: bool = False
    edge_diffraction: bool = False
    diffraction_lit_region: bool = True
    # seed: int = 42
    # rr_depth: int = -1
    # rr_prob: float = 0.95
    # stop_threshold: float | None = None

    # -- Display
    color_map: str = "viridis"
    use_alpha: bool = True
    vmin: float = -150
    vmax: float = -50
    show_colorbar: bool = True
    # When updating the radio map, upload new values directly from the device.
    # This is only supported when using a CUDA variant.
    use_direct_update_from_device: bool = True

    # -- Metric options
    metric: str = "sinr" # "path_gain", "rss", or "sinr"
    tx_power_dbm: float = 25.0
    noise_power_dbm: float = -120.0

    @property
    def samples_per_it(self) -> int:
        return int(10**self.log_samples_per_it)


# ------------------------


@dataclass(kw_only=True)
class PathsConfig:
    auto_update: bool = True

    # Minimum delay between path computations, which involves ray tracing.
    # Set this higher (e.g., 0.1-1.0) to reduce GPU load when animations are playing.
    min_update_delay_s: float = 0.0  # 0 = update every frame (default behavior)
    accumulate_max_samples_per_src: int = int(1e9)

    max_depth: int = 2
    max_num_paths_per_src: int = 1000000
    samples_per_src: int = 1000000
    synthetic_array: bool = True
    los: bool = True
    specular_reflection: bool = True
    diffuse_reflection: bool = False
    refraction: bool = False
    diffraction: bool = False
    edge_diffraction: bool = False
    diffraction_lit_region: bool = False

    # CIR
    compute_cir: bool = False
    num_taps: int = 10
    fft_size: int = 512
    subcarrier_spacing: float = 30e3
    l_min: int = 0
    l_max: int = 100
    num_time_steps: int = 1
    normalize: bool = False
    normalize_delays: bool = True
    snr_offset_db: float = -50

    @property
    def bandwidth(self) -> float:
        return self.fft_size * self.subcarrier_spacing

    @property
    def sampling_frequency(self) -> float:
        return self.subcarrier_spacing


# ------------------------


class RenderingMode(Enum):
    RASTERIZATION = 0
    RAY_TRACING = 1


RENDERING_MODE_NAMES = ["Rasterization", "Ray tracing"]
assert len(RENDERING_MODE_NAMES) == len(RenderingMode)


@dataclass(kw_only=True)
class RenderingConfig:
    mode: RenderingMode = RenderingMode.RASTERIZATION
    # Full resolution of the window at startup, not accounting for any DPI scaling.
    default_resolution: tuple[int, int] = (1920, 1080)
    # Current full resolution of the window, not accounting for any DPI scaling.
    current_resolution: tuple[int, int] = default_resolution
    # Relative resolution for ray traced rendering
    relative_resolution: float = 0.5
    spp_per_frame: int = 8
    max_accumulated_spp: int = 256

    # Use the OptiX denoiser (CUDA variant only)
    use_denoiser: bool = True

    envmap: str | None = os.path.join(
        DATA_DIR, "envmaps", "teufelsberg_ground_2_1k.exr"
    )
    # Brightness factor to apply to the environment map.
    envmap_factor: float = 1.0
    # Rotation of the environment map along the vertical axis, in degrees.
    envmap_rotation_deg: float = -60

    slice_plane_normal: tuple[float, float, float] = (0, 0, -1)
    # If None, the plane will be placed at the z-center of the scene bounding box.
    slice_plane_position: tuple[float, float, float] | None = None
    # default_slice_plane_enabled: bool = False
    default_slice_plane_enabled: bool = True

    @property
    def rendering_resolution(self) -> tuple[int, int]:
        """
        Resolution used for ray-traced rendering, accounting for `relative_resolution`
        but not any DPI scaling.
        """
        return (
            int(self.current_resolution[0] * self.relative_resolution),
            int(self.current_resolution[1] * self.relative_resolution),
        )


# ------------------------


@dataclass
class SiteConfig:
    name: str
    position: list[float]
    downtilt: float
    azimuths: list[float]
    power_dbm: float


@dataclass
class ScenarioConfig:
    sites: list[SiteConfig] = field(default_factory=list)
    road_segments: list[list[list[float]]] = field(default_factory=list)
    num_ue: int = 5
    carrier_frequency: float | None = None


# ------------------------


class GuiMode(Enum):
    HIDDEN = 0
    FULL = 1


GUI_MODE_NAMES = ["Hidden", "Full"]
assert len(GUI_MODE_NAMES) == len(GuiMode)

# ------------------------


@dataclass(kw_only=True)
class GuiConfig:
    title: str = "Sionna RT"
    config_path: str

    # Logging
    log_level: int = logging.INFO

    gui_mode: GuiMode = GuiMode.FULL
    show_polyscope_gui: bool = False
    show_help_window: bool = False
    use_live_reload: bool = False
    use_vsync: bool = True
    background_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    road_lift: float = 0.1

    # Either the path to an XML scene file, or the name of a built-in scene.
    scene_filename: str | None = None
    # Name of the built-in scene to load if no scene filename is provided.
    default_scene_filename: str = "simple_street_canyon_with_cars"
    # Whether to create an example scenario with radio devices.
    create_example_scenario: bool = False

    # Path to the scenario configuration file (e.g., a notebook or a YAML file).
    scenario_filename: str | None = None

    # Scenario configuration: sites, road segments, number of UEs.
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)

    # If set, override the radio materials' thickness property
    radio_material_thickness: float | None = None
    # If set, override the radio materials' scattering coefficient property
    radio_material_scattering_coefficient: float | None = None

    # Antenna arrays
    tx_array: AntennaArrayConfig = field(default_factory=AntennaArrayConfig)
    rx_array: AntennaArrayConfig = field(default_factory=AntennaArrayConfig)

    # Rendering
    rendering: RenderingConfig = field(default_factory=RenderingConfig)

    # Radio map
    radio_map: RadioMapConfig = field(default_factory=RadioMapConfig)

    # Paths
    paths: PathsConfig = field(default_factory=PathsConfig)

    def __post_init__(self):
        if self.scene_filename is None:
            self.scene_filename = self.default_scene_filename
            # Only add example radio devices if we're loading the default scene.
            self.create_example_scenario = True


# ------------------------


def extract_scenario_from_notebook(path: str) -> dict:
    """
    Extract scenario parameters (sites, road segments, etc.) from a Jupyter notebook.
    Also attempts to find a call to load_scene() to identify the XML scene file.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            nb = json.load(f)
    except Exception as e:
        logging.error(f"Failed to load notebook {path}: {e}")
        return {}

    code_lines = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            if isinstance(source, list):
                for line in source:
                    code_lines.append(line)
                    if not line.endswith("\n"):
                        code_lines.append("\n")
            else:
                code_lines.append(source)
                if not source.endswith("\n"):
                    code_lines.append("\n")

    code = "".join(code_lines)

    # Target variables mapping: notebook_name -> config_name
    targets = {
        "sites": "sites",
        "road_segments": "road_segments",
        "NUM_UE": "num_ue",
        "num_ue": "num_ue",
        "carrier_frequency": "carrier_frequency",
        "tx_array": "tx_array",
        "rx_array": "rx_array",
    }

    result = {}

    # Try to find the scene path: Look for load_scene("...")
    match = re.search(r'load_scene\s*\(\s*["\']([^"\']+)["\']\s*\)', code)
    if match:
        scene_filename = match.group(1)
        # If the scene file is not found, try to find it relative to the notebook
        if not os.path.exists(scene_filename):
            nb_dir = os.path.dirname(os.path.abspath(path))
            candidate = os.path.join(nb_dir, scene_filename)
            if os.path.exists(candidate):
                scene_filename = candidate
            else:
                # Also try one level up from the notebook, as often happens in project structures
                candidate = os.path.join(os.path.dirname(nb_dir), scene_filename)
                if os.path.exists(candidate):
                    scene_filename = candidate
        result["scene_filename"] = scene_filename

    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    # Check for simple name assignments (e.g., sites = ...)
                    target_id = None
                    if isinstance(target, ast.Name):
                        target_id = target.id
                    # Check for scene attribute assignments (e.g., scene.tx_array = ...)
                    elif (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "scene"
                    ):
                        target_id = target.attr

                    if target_id in targets:
                        config_key = targets[target_id]
                        # Handle literal values
                        try:
                            val = ast.literal_eval(node.value)
                            result[config_key] = val
                            continue
                        except:
                            pass

                        # Handle PlanarArray(...) calls for antenna arrays
                        if (
                            isinstance(node.value, ast.Call)
                            and isinstance(node.value.func, ast.Name)
                            and node.value.func.id == "PlanarArray"
                        ):
                            array_cfg = {}
                            # Mapping: PlanarArray arg -> AntennaArrayConfig field
                            arg_names = [
                                "num_rows",
                                "num_cols",
                                "vertical_spacing",
                                "horizontal_spacing",
                                "pattern",
                                "polarization",
                            ]
                            arg_map = {
                                "num_rows": "num_rows",
                                "num_cols": "num_cols",
                                "vertical_spacing": "vertical_spacing",
                                "horizontal_spacing": "horizontal_spacing",
                                "pattern": "pattern_i",
                                "polarization": "polarization_i",
                            }
                            # Enums for pattern and polarization indices
                            patterns = {
                                p: i
                                for i, p in enumerate(antenna_pattern_registry.list())
                            }
                            polarizations = {
                                p: i
                                for i, p in enumerate(polarization_registry.list())
                            }

                            # Positional arguments
                            for i, arg in enumerate(node.value.args):
                                if i < len(arg_names):
                                    field_name = arg_map[arg_names[i]]
                                    try:
                                        val = ast.literal_eval(arg)
                                        if (
                                            field_name == "pattern_i"
                                            and val in patterns
                                        ):
                                            val = patterns[val]
                                        elif (
                                            field_name == "polarization_i"
                                            and val in polarizations
                                        ):
                                            val = polarizations[val]
                                        array_cfg[field_name] = val
                                    except:
                                        pass

                            # Keyword arguments
                            for keyword in node.value.keywords:
                                if keyword.arg in arg_map:
                                    field_name = arg_map[keyword.arg]
                                    try:
                                        val = ast.literal_eval(keyword.value)
                                        if (
                                            field_name == "pattern_i"
                                            and val in patterns
                                        ):
                                            val = patterns[val]
                                        elif (
                                            field_name == "polarization_i"
                                            and val in polarizations
                                        ):
                                            val = polarizations[val]
                                        array_cfg[field_name] = val
                                    except:
                                        pass
                            result[config_key] = array_cfg
    except Exception as e:
        logging.warning(f"Failed to parse notebook code for scenario data: {e}")

    # Robust type conversion for OmegaConf
    if "num_ue" in result:
        try:
            result["num_ue"] = int(result["num_ue"])
        except:
            result.pop("num_ue")
    if "carrier_frequency" in result and result["carrier_frequency"] is not None:
        try:
            result["carrier_frequency"] = float(result["carrier_frequency"])
        except:
            result.pop("carrier_frequency")
    if "sites" in result and isinstance(result["sites"], list):
        new_sites = []
        for site in result["sites"]:
            try:
                if isinstance(site, dict):
                    if "position" in site:
                        site["position"] = [float(p) for p in site["position"]]
                    if "downtilt" in site:
                        site["downtilt"] = float(site["downtilt"])
                    if "azimuths" in site:
                        site["azimuths"] = [float(a) for a in site["azimuths"]]
                    if "power_dbm" in site:
                        site["power_dbm"] = float(site["power_dbm"])
                    new_sites.append(site)
            except:
                continue
        result["sites"] = new_sites
    if "road_segments" in result and isinstance(result["road_segments"], list):
        new_segments = []
        for segment in result["road_segments"]:
            try:
                # segment can be a tuple or list of two points
                new_segment = []
                for point in segment:
                    new_segment.append([float(coord) for coord in point])
                new_segments.append(new_segment)
            except:
                continue
        result["road_segments"] = new_segments

    return result


def load_config(
    config_path: str,
    scene_filename: str | None = None,
    scenario_filename: str | None = None,
) -> GuiConfig:
    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader

    candidates = [
        config_path,
        os.path.join(CONFIGS_DIR, config_path),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            with open(candidate, "r", encoding="utf-8") as f:
                loaded = yaml.load(f, Loader=Loader)
                # The config file might be empty.
                if loaded is None:
                    loaded = {}
            break
    else:
        raise FileNotFoundError(
            f"Config file not found: {config_path}, tried:\n- "
            + "\n- ".join(candidates)
        )

    loaded = OmegaConf.create(loaded)
    # Resolve interpolations, if any.
    OmegaConf.resolve(loaded)
    loaded["config_path"] = config_path

    if scene_filename is not None and scene_filename.endswith(".ipynb"):
        original_nb_path = scene_filename
        notebook_data = extract_scenario_from_notebook(scene_filename)
        if "scene_filename" in notebook_data:
            scene_filename = notebook_data.pop("scene_filename")

        # Move antenna arrays to top level to override GuiConfig fields
        for array_key in ["tx_array", "rx_array"]:
            if array_key in notebook_data:
                loaded[array_key] = notebook_data.pop(array_key)

        # If scenario_filename is not provided, use the data from the notebook
        if scenario_filename is None:
            loaded["scenario"] = notebook_data
            loaded["create_example_scenario"] = True
            loaded["scenario_filename"] = original_nb_path

    if scene_filename is not None:
        loaded["scene_filename"] = scene_filename

    if scenario_filename is not None:
        loaded["scenario_filename"] = scenario_filename
        if scenario_filename.endswith(".ipynb"):
            scenario_data = extract_scenario_from_notebook(scenario_filename)
            # Remove scene_filename from scenario data to avoid OmegaConf errors
            scenario_data.pop("scene_filename", None)
            # Move antenna arrays to top level to override GuiConfig fields
            for array_key in ["tx_array", "rx_array"]:
                if array_key in scenario_data:
                    loaded[array_key] = scenario_data.pop(array_key)
        else:
            with open(scenario_filename, "r", encoding="utf-8") as f:
                scenario_data = yaml.load(f, Loader=Loader)

        if scenario_data is not None:
            loaded["scenario"] = scenario_data
            # Automatically enable example scenario if a scenario file is provided.
            loaded["create_example_scenario"] = True

    merged = OmegaConf.merge(OmegaConf.structured(GuiConfig), loaded)
    return OmegaConf.to_object(merged)
