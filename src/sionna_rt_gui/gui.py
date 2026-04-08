#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import hashlib
import logging
import os
import re
import sys
import time

import drjit as dr
import mitsuba as mi
import numpy as np
import matplotlib.pyplot as plt
import polyscope as ps
import polyscope.imgui as psim
from sionna import rt
from sionna.rt.scene_utils import remove_objects_duplicate_vertices

from . import __version__ as GUI_VERSION
from .animation import AnimationConfig, animation_gui, animation_tick, LoopingMode
from .antenna_array import antenna_array_gui
from .config import (
    GuiConfig,
    RadioMapConfig,
    PathsConfig,
    GuiMode,
    RenderingMode,
    RENDERING_MODE_NAMES,
    NVIDIA_GREEN,
)
from .rendering import (
    render_scene,
    set_envmap_rotation,
    set_envmap_intensity,
    add_or_update_ray_traced_image_quantity,
)
from .rm_utils import radio_map_colorbar_to_image
from .ps_utils import (
    set_custom_imgui_style,
    set_polyscope_device_interop_funcs,
    supports_direct_update_from_device,
)
from .sionna_utils import (
    add_paths_to_polyscope,
    add_radio_map_to_polyscope,
    add_scene_to_polyscope,
    get_built_in_scenes,
    get_point_from_camera_center_ray,
    get_normal_for_path,
    set_or_update_radio_devices_polyscope,
)
from .selection import SelectionType, selection_gui

logger = logging.getLogger(__name__)

CTRL_OR_CMD = "Cmd" if sys.platform == "darwin" else "Ctrl"
HELP_WINDOW_TABLES = {
    "Camera controls": {
        "Left click + drag": "Camera rotation",
        "Right click + drag": "Camera panning",
        "Shift + left click + drag": "Camera panning",
        "Double left click": "Center camera rotation around clicked point",
        "Mouse scroll": "Camera zoom",
        f"{CTRL_OR_CMD} + shift + left click + drag": "Continuous camera zoom",
        "R": "Reset camera to initial position",
        "F": "Fit scene to camera",
    },
    "Mouse bindings": {
        "Left click": "Select a radio device",
        f"{CTRL_OR_CMD} + left click": "Add transmitter",
        f"{CTRL_OR_CMD} + right click": "Add receiver",
    },
    "Key bindings": {
        "H / ?": "Show this help window",
        "K": "Add transmitter at the current mouse position",
        "L": "Add receiver at the current mouse position",
        "M": "Show / hide radio map. Requires at least one transmitter.",
        "C": "Go to next rendering mode",
        "Tab": "Go to next GUI mode (can be used to hide the GUI)",
        "Esc": "Close help window or de-select current object",
        "Del (with item selected)": "Delete radio device",
        "Shift + R": "Reload application",
        f"{CTRL_OR_CMD} + Q": "Exit",
    },
    "Slice plane": {
        "S": "Toggle slice plane visibility (not supported in ray-traced rendering mode)",
        "Alt + left click drag": "Move slice plane along its normal",
    },
}


def _asciify_xml_paths(xml_content: str, scene_dir: str):
    """
    Finds file paths in Mitsuba XML that contain non-ASCII characters and
    replaces them with ASCII-safe paths using hard links.
    This is a workaround for Mitsuba/Windows issues with non-ASCII filenames.
    """
    # Match value="..." or filename="..." that contains at least one non-ASCII character
    pattern = re.compile(r'(value|filename)="(.*?[^\x00-\x7F].*?)"')

    # Create a hidden directory for ASCII links if we find any non-ASCII paths
    links_dir = os.path.join(scene_dir, ".sionna_rt_gui_links")

    def replace_match(match):
        attr = match.group(1)
        path = match.group(2)

        # Absolute path for checking file existence
        abs_path = os.path.normpath(os.path.join(scene_dir, path))
        if not os.path.isfile(abs_path):
            # If not relative to scene_dir, try absolute
            if os.path.isfile(path):
                abs_path = os.path.abspath(path)
            else:
                return match.group(0)  # Not a file on disk, keep as is

        # Create an ASCII-safe filename
        filename = os.path.basename(path)
        stem, ext = os.path.splitext(filename)

        # Use a hash of the full path to avoid collisions
        h = hashlib.md5(path.encode("utf-8")).hexdigest()[:8]
        safe_stem = "".join(c if ord(c) < 128 else "_" for c in stem)
        safe_filename = f"fixed_{h}_{safe_stem}{ext}"

        # Ensure the links directory exists
        try:
            os.makedirs(links_dir, exist_ok=True)
            abs_safe_path = os.path.join(links_dir, safe_filename)
            rel_safe_path = os.path.join(".sionna_rt_gui_links", safe_filename)
        except Exception:
            # Fallback: create in the same directory as the original file
            abs_safe_path = os.path.join(os.path.dirname(abs_path), safe_filename)
            rel_safe_path = os.path.join(os.path.dirname(path), safe_filename)

        # Create hard link if it doesn't exist
        if not os.path.exists(abs_safe_path):
            try:
                os.link(abs_path, abs_safe_path)
                logger.info(f"Created ASCII-safe hard link: {abs_safe_path}")
            except Exception as e:
                logger.warning(f"Failed to create hard link for {abs_path}: {e}")
                return match.group(0)  # Keep original if linking fails

        # Mitsuba parser uses / as separator, even on Windows
        rel_safe_path = rel_safe_path.replace("\\", "/")
        return f'{attr}="{rel_safe_path}"'

    return pattern.sub(replace_match, xml_content)


class SionnaRtGui:
    def __init__(self, cfg: GuiConfig):
        self.cfg = cfg

        # --- Sionna RT
        # Scene
        built_in_scenes = get_built_in_scenes()
        self.known_scene_names = ["None"] + list(built_in_scenes.keys())
        self.known_scene_paths = [None] + list(built_in_scenes.values())
        self.current_scene_idx: int = 0
        self.load_scene_requested: str | None = None
        self.scene: rt.Scene | None = None

        # Radio map results
        self.radio_map: rt.RadioMap | None = None
        self.rm_accumulated_samples: int = 0
        self.rm_color_map_options = [v for v in plt.colormaps() if not v.endswith("_r")]
        self.rm_color_map_index = self.rm_color_map_options.index(
            self.cfg.radio_map.color_map
        )
        self.rm_colorbar: np.ndarray | None = None
        self.rm_colorbar_texture_id: int | None = None
        # Due to a current limitation, we can't have alpha on radio maps when using
        # direct updates from the device.
        self.cfg.radio_map.use_alpha = not (
            self.cfg.radio_map.use_direct_update_from_device
            and supports_direct_update_from_device()
        )

        # Paths results
        self.paths: rt.Paths | None = None
        # Shape: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, l_max - l_min + 1]
        self.paths_taps: np.ndarray | None = None
        self.paths_cir: tuple[np.ndarray, np.ndarray] | None = None
        # Used to throttle path computations
        self._last_paths_update_time: float = 0.0

        # --- Rendering
        self.render_cache: dict = None
        self.ray_traced_img: mi.TensorXf | None = None
        self.ray_traced_depth: mi.TensorXf | None = None
        self.previous_camera_pose: np.ndarray = ps.get_camera_view_matrix()
        self.rendering_accumulated_samples: int = 0
        self.reset_accumulation_requested: bool = False
        self.denoiser: mi.OptixDenoiser | None = None
        self.slice_plane: ps.SlicePlane | None = None

        # --- Animation state
        self.animation_config: AnimationConfig = AnimationConfig()

        # --- Inputs state
        self.last_mouse_pos: mi.ScalarVector2f | None = None

        self.snapshot_load_requested: bool = False
        self.code_reload_requested: bool = False

        # --- Selections
        self.selected_object: rt.SceneObject | None = None
        self.selected_type: SelectionType | None = None
        self.prev_gizmo_to_world: np.ndarray | None = None

        # --- Polyscope setup
        # Can be used to derive e.g. random seeds.
        self.ps_groups: dict[str, ps.Group] = {}
        self.frame_i: int = 0
        self.was_mouse_dragging: bool = False
        self.home_camera_to_world: np.ndarray | None = None

        # Pre-init settings
        ps.set_program_name(self.cfg.title)
        # Window size and position will be loaded from the last run from `.polyscope.ini`
        ps.set_use_prefs_file(True)
        ps.set_enable_vsync(cfg.use_vsync)
        # On some machines (especially with remote access), VSync doesn't do anything,
        # so we cap fps as well.
        ps.set_max_fps(60)
        ps.set_user_gui_is_on_right_side(False)
        ps.set_open_imgui_window_for_user_callback(False)
        ps.set_verbosity(1)
        ps.set_build_default_gui_panels(self.cfg.show_polyscope_gui)
        ps.set_background_color(self.cfg.background_color)
        ps.set_ground_plane_mode("none")
        ps.set_window_resizable(True)
        ps.set_give_focus_on_show(True)
        ps.set_transparency_mode("pretty")
        ps.set_files_dropped_callback(self.on_files_dropped)
        set_custom_imgui_style()

        was_initialized = ps.is_initialized()
        if not was_initialized:
            ps.set_up_dir("z_up")
            ps.set_front_dir("y_front")
            ps.init()
            if supports_direct_update_from_device():
                set_polyscope_device_interop_funcs()

        # Set window size once UI scale is known (after init).
        self.ui_scale: float = ps.get_ui_scale()
        ps.set_window_size(
            self.cfg.rendering.default_resolution[0] * self.ui_scale,
            self.cfg.rendering.default_resolution[1] * self.ui_scale,
        )
        # Used to track window size changes. Includes DPI scaling.
        self.previous_window_resolution: tuple[int, int] = ps.get_window_size()
        # Note that our `set_window_size()` call may not succeed, e.g. if the window is
        # maximized. We adopt the effective resolution to make sure that we rendering with
        # the right aspect ratio, etc.
        self.cfg.rendering.current_resolution = tuple(
            v // self.ui_scale for v in self.previous_window_resolution
        )
        # Enable denoiser if requested.
        if "cuda" in mi.variant():
            self.set_use_denoiser(self.cfg.rendering.use_denoiser)
        else:
            # If there is no CUDA-compatible GPU, switch to rasterization (cheaper).
            self.cfg.rendering.mode = RenderingMode.RASTERIZATION

        # Used to throttle window size change handling. Set to None if no change is pending.
        self.last_window_size_changed_time: float | None = None

        # --- Load scene
        # TODO: preserve currently-selected scene across reloads
        if not self.cfg.scene_filename.endswith(".xml"):
            if self.cfg.scene_filename not in built_in_scenes:
                raise ValueError(
                    f'Scene "{self.cfg.scene_filename}" not found. Built-in scenes: {list(built_in_scenes.keys())}'
                )
            self.cfg.scene_filename = built_in_scenes[self.cfg.scene_filename]
        self.load_scene(
            self.cfg.scene_filename,
            # If the program was just reloaded (live coding), don't move the camera
            recenter_camera=not was_initialized,
        )

        # --- Slice plane
        ps.remove_all_slice_planes()
        plane = ps.add_scene_slice_plane()
        plane.set_pose(
            (
                self.cfg.rendering.slice_plane_position
                if self.cfg.rendering.slice_plane_position is not None
                else (0, 0, self.scene.mi_scene.bbox().center().z)
            ),
            self.cfg.rendering.slice_plane_normal,
        )
        plane.set_active(
            self.cfg.rendering.default_slice_plane_enabled
            and (self.cfg.rendering.mode == RenderingMode.RASTERIZATION)
        )
        self.slice_plane = plane

        if self.cfg.create_example_scenario:
            self.create_example_scenario(
                set_camera=not was_initialized, add_radio_map=False
            )

    def normalize_radio_materials(self):
        """
        MO-specific radio material normalization logic.
        Adapted from MO_v2.py
        """
        radio_materials = getattr(self.scene, "radio_materials", {})
        if not radio_materials:
            return

        def find_mat(prefix):
            # Try to find a material that matches the prefix, possibly with mat- or itu_ prefixes
            candidates = [prefix, f"itu_{prefix}", f"mat-{prefix}", f"mat-itu_{prefix}"]
            for c in candidates:
                if c in radio_materials:
                    return c
            return None

        fallback_name = find_mat("concrete") or next(iter(radio_materials), None)
        if fallback_name is None:
            return

        brick_name = find_mat("brick") or fallback_name
        glass_name = find_mat("glass") or fallback_name
        metal_name = find_mat("metal") or fallback_name
        marble_name = find_mat("marble") or fallback_name
        ground_name = find_mat("very_dry_ground") or find_mat("medium_dry_ground") or fallback_name

        objects = getattr(self.scene, "objects", {})
        for obj_name, obj in objects.items():
            # If the object already has a radio material that is an ITU material, keep it.
            curr_mat = getattr(obj, "radio_material", None)
            if curr_mat is not None:
                mat_name = ""
                if isinstance(curr_mat, str):
                    mat_name = curr_mat
                elif hasattr(curr_mat, "name"):
                    mat_name = curr_mat.name
                elif hasattr(curr_mat, "id"):
                    mat_name = curr_mat.id()

                # Strip common prefixes
                stripped_mat_name = mat_name
                if stripped_mat_name.startswith("mat-"):
                    stripped_mat_name = stripped_mat_name[4:]

                # If the object already has a radio material that is an ITU material,
                # we usually keep it. 
                if stripped_mat_name.lower().startswith("itu_"):
                    # We only override if it's very clearly a mismatch, but the user
                    # wants to distinguish ground from plane now.
                    # If it's marble, we'll let it stay marble (black) if it's a "plane".
                    continue
                else:
                    pass

            name = obj_name.lower()
            target = None

            if "brick" in name:
                target = brick_name
            elif "glass" in name or "window" in name:
                target = glass_name
            elif "metal" in name or "steel" in name or "roof" in name:
                target = metal_name
            elif "marble" in name or "white" in name or "road" in name or "route" in name:
                target = marble_name
            elif "ground" in name or "terrain" in name:
                target = ground_name
            elif "plane" in name:
                # Let it keep its original material if it has one (likely marble/black)
                # or map it to ground if it's truly unknown.
                if curr_mat is None:
                    target = ground_name
                else:
                    target = None
            elif "concrete" in name:
                target = fallback_name
            
            if target is None:
                continue

            try:
                obj.radio_material = radio_materials[target]
            except Exception:
                try:
                    obj.radio_material = target
                except Exception:
                    pass

    def update_material_color(self, mat_id_or_obj, new_color):
        """Update a material's color and synchronize with Polyscope meshes."""
        if self.scene is None:
            return

        if isinstance(mat_id_or_obj, str):
            mat_id = mat_id_or_obj
            mat = self.scene.radio_materials.get(mat_id)
        else:
            mat = mat_id_or_obj
            mat_id = getattr(mat, "id", lambda: "")()
            if not mat_id and hasattr(mat, "name"):
                mat_id = mat.name

        if mat is not None and hasattr(mat, "color"):
            mat.color = new_color

        if not mat_id:
            # Try to find the ID from radio_materials if we only have the object
            for name, m in self.scene.radio_materials.items():
                if m is mat:
                    mat_id = name
                    break

        if not mat_id:
            return

        def get_clean_id(id_str):
            if id_str.startswith("mat-itu_"):
                return id_str[8:]
            if id_str.startswith("itu_"):
                return id_str[4:]
            if id_str.startswith("mat-"):
                return id_str[4:]
            return id_str

        target_clean = get_clean_id(mat_id)
        logger.debug(
            f"Updating material color for '{mat_id}' (clean: '{target_clean}') to {new_color}"
        )

        # Update all meshes using this material
        for mesh in self.scene.mi_scene.shapes():
            mesh_mat = mesh.bsdf()
            mesh_mat_id = getattr(mesh_mat, "id", lambda: "")()
            if not mesh_mat_id and hasattr(mesh_mat, "name"):
                mesh_mat_id = mesh_mat.name

            if not mesh_mat_id:
                continue

            # Check if material matches by ID (original or stripped)
            if (
                (mesh_mat is mat)
                or (mesh_mat_id == mat_id)
                or (get_clean_id(mesh_mat_id) == target_clean)
            ):
                if ps.has_surface_mesh(mesh.id()):
                    ps.get_surface_mesh(mesh.id()).set_color(new_color)

    def create_example_scenario(
        self, set_camera: bool = True, add_radio_map: bool = True
    ):
        if set_camera:
            self.fit_camera_to_scene()
            self.home_camera_to_world = ps.get_camera_view_matrix()

        has_custom_scenario = (
            len(self.cfg.scenario.sites) > 0 or len(self.cfg.scenario.road_segments) > 0
        )

        if has_custom_scenario:
            # 1. Add transmitters from sites
            for site in self.cfg.scenario.sites:
                # SiteConfig from OmegaConf
                site_name = site.name
                site_pos = site.position
                site_downtilt = site.downtilt
                site_azimuths = site.azimuths
                site_power_dbm = site.power_dbm

                for i, az in enumerate(site_azimuths):
                    tx = rt.Transmitter(
                        name=f"{site_name}_sector_{i + 1}",
                        position=[float(p) for p in site_pos],
                        orientation=[
                            float(np.deg2rad(az)),
                            float(np.deg2rad(site_downtilt)),
                            0.0,
                        ],
                        power_dbm=float(site_power_dbm),
                    )
                    self.scene.add(tx)

            # 2. Add UEs along road segments
            if self.cfg.scenario.road_segments:
                # Reproducible seed for random UEs
                rng = np.random.default_rng(48)
                for i in range(self.cfg.scenario.num_ue):
                    seg_idx = rng.integers(len(self.cfg.scenario.road_segments))
                    s, e = self.cfg.scenario.road_segments[seg_idx]
                    s, e = np.array(s), np.array(e)

                    # Start at a random point along the entire segment
                    t_start = rng.random()
                    p_start = s + t_start * (e - s)

                    rx_name = f"rx-{i}"
                    rx = rt.Receiver(name=rx_name, position=p_start.tolist())
                    self.scene.add(rx)

                    traj = self.animation_config.trajectories[rx_name]
                    traj.add_point(s.tolist())
                    traj.add_point(e.tolist())
                    traj.enabled = True
                    # Set the initial distance based on the random start point
                    traj.distance = t_start * traj.total_distance()
                    # Randomize initial direction
                    traj.backward = rng.random() > 0.5
                    # Set looping mode to mirror (bounce back)
                    traj.looping_mode_i = int(LoopingMode.Mirror.value)

                self.animation_config.playing = True
                self.animation_config.speed_multiplier = 20.0

            # Update Polyscope for radio devices
            set_or_update_radio_devices_polyscope(
                self.scene.transmitters, True, self
            )
            set_or_update_radio_devices_polyscope(
                self.scene.receivers, False, self
            )

        else:
            # Default example transmitters
            for pos in [[-34.0, 13.0, 33.0]]:
                self.add_radio_device(
                    pos, is_transmitter=True, allow_auto_update=False
                )

                shifted = [pos[0] + 15, pos[1] - 14, pos[2] - 20]
                self.add_radio_device(
                    shifted, is_transmitter=False, allow_auto_update=False
                )

            # Default example animation
            if "rx-0" in self.scene.receivers:
                p = self.scene.get("rx-0").position.numpy().squeeze()
                traj = self.animation_config.trajectories["rx-0"]
                traj.add_point(p - [0, 30, 0])
                traj.add_point(p)
                traj.add_point(p + [40, 0, 0])
                traj.enabled = True
                traj.distance = 0.0
                self.animation_config.playing = True
                self.animation_config.speed_multiplier = 10.0

        if add_radio_map:
            self.set_radio_map(self.compute_radio_map(), show=True)
        if self.cfg.paths.auto_update:
            self.update_paths(show=True)

    def reset_and_setup_structures(self):
        # Clear Sionna state
        self.clear_radio_map()
        self.clear_selection()
        self.clear_paths()
        self.clear_ray_traced_image()
        self.animation_config.clear()

        # Clear Polyscope state
        ps.remove_all_structures()
        ps.remove_all_groups()

        self.ps_groups = {
            "scene": ps.create_group("Scene meshes"),
            "rd": ps.create_group("Radio devices"),
            "radio_maps": ps.create_group("Radio maps"),
            "paths": ps.create_group("Paths"),
        }

    def load_scene(self, scene_path: str, recenter_camera: bool = True):
        self.reset_and_setup_structures()
        # Fix for encoding issues on Windows with special characters in filenames (e.g. œ)
        try:
            with open(scene_path, "r", encoding="utf-8") as f:
                content = f.read()

            scene_dir = os.path.dirname(scene_path)
            # WORKAROUND: Mitsuba on Windows fails on non-ASCII filenames.
            # We create ASCII-safe hard links for those files and update the XML.
            content = _asciify_xml_paths(content, scene_dir)

            # Reproduce Sionna's load_scene logic but with explicit utf-8 encoding
            fres_old = mi.file_resolver()
            fres = mi.FileResolver(fres_old)
            fres.append(scene_dir)
            try:
                mi.set_file_resolver(fres)
                self.scene = rt.load_scene_from_string(content)
            finally:
                mi.set_file_resolver(fres_old)
        except Exception as e:
            logger.warning(
                f"Failed to load scene from string with utf-8, falling back to default: {e}"
            )
            self.scene = rt.load_scene(scene_path)

        remove_objects_duplicate_vertices(self.scene.mi_scene)

        self.normalize_radio_materials()

        self.scene.tx_array = self.cfg.tx_array.create()
        self.scene.rx_array = self.cfg.rx_array.create()

        if self.cfg.scenario.carrier_frequency is not None:
            self.scene.frequency = float(self.cfg.scenario.carrier_frequency)

        thickness = self.cfg.radio_material_thickness
        scattering_coefficient = self.cfg.radio_material_scattering_coefficient
        for sh in self.scene.mi_scene.shapes():
            bsdf = sh.bsdf()
            if isinstance(bsdf, rt.RadioMaterialBase):
                if thickness is not None:
                    bsdf.thickness = thickness
                if scattering_coefficient is not None:
                    bsdf.scattering_coefficient = scattering_coefficient

        self.cfg.scene_filename = scene_path

        # Add this scene to the list of known scene names, if missing
        if scene_path not in self.known_scene_paths:
            self.known_scene_names.append(scene_path)
            self.known_scene_paths.append(scene_path)

        try:
            self.current_scene_idx = self.known_scene_paths.index(
                self.cfg.scene_filename
            )
        except ValueError:
            self.current_scene_idx = 0

        add_scene_to_polyscope(self.scene, self.ps_groups, self.cfg.road_lift)
        self.set_rendering_mode(self.cfg.rendering.mode)
        if recenter_camera:
            self.fit_camera_to_scene()

        if (self.cfg.rendering.slice_plane_position is None) and (
            self.slice_plane is not None
        ):
            self.slice_plane.set_pose(
                (0, 0, self.scene.mi_scene.bbox().center().z),
                self.cfg.rendering.slice_plane_normal,
            )

    def on_files_dropped(self, files: list[str]):
        for file in files:
            if file.endswith(".xml"):
                print(f"[i] Loading dropped XML file: {file}")
                try:
                    self.load_scene(file)
                except Exception as e:
                    print(f'[!] Failed loading scene "{file}":\n{e}')
                    continue
                # Only handle one valid XML file
                break

    def move_camera_home(self):
        """Move camera to the home view, if any. Otherwise, fits the scene in the camera viewport."""
        if self.home_camera_to_world is not None:
            # We probably can't rely on ps.pick() since the buffers probably haven't been
            # updated yet with the new camera pose. So we trace our own ray to get the rotation
            # point of the turntable navigation.
            if ps.get_navigation_style() == "turntable":
                center = get_point_from_camera_center_ray(
                    self.scene, np.linalg.inv(self.home_camera_to_world)
                )
                if center is not None:
                    ps.set_view_center_raw(center)

            ps.set_camera_view_matrix(self.home_camera_to_world)

            ps.request_redraw()
        else:
            self.fit_camera_to_scene()

    def fit_camera_to_scene(self):
        """Move the camera to a position where most of the scene is visible."""
        fov_vertical_deg = ps.get_view_camera_parameters().get_fov_vertical_deg()
        bbox = self.scene.mi_scene.bbox()

        center = bbox.center()
        extents = bbox.extents()

        # Calculate the required distance to fit the scene in the vertical FOV
        # We use the larger of height or diagonal extent for safety margin
        scene_diagonal = np.linalg.norm(extents)
        scene_size = max(extents.z, scene_diagonal * 0.8)  # 0.8 for some margin

        # Convert FOV to radians and calculate required distance
        fov_vertical_rad = np.radians(fov_vertical_deg)
        distance = scene_size / (2.0 * np.tan(0.5 * fov_vertical_rad))
        distance = distance * 0.95

        # Position camera at a reasonable viewing angle.
        # Angle above horizontal
        es, ec = dr.sincos(np.radians(45))
        # Angle from positive X axis
        zs, zc = dr.sincos(np.radians(0))
        origin = [
            center.x + distance * ec * zc,
            center.y + distance * ec * zs,
            center.z + distance * es,
        ]
        target = [center.x - 0.2 * (center.x - origin[0]), center.y, center.z]

        ps.set_view_center_raw(target)
        ps.look_at(origin, target)

    # ------------------------

    def tick(self):
        if self.load_scene_requested is not None:
            # If load_scene_requested is a boolean True, it means we want to reload the current scene
            path_to_load = self.load_scene_requested
            if path_to_load is True:
                path_to_load = self.cfg.scene_filename
            
            if path_to_load is not None:
                try:
                    self.load_scene(path_to_load)
                except Exception as e:
                    print(f'[!] Failed loading scene "{path_to_load}":\n{e}')
            self.load_scene_requested = None

        self.process_inputs()

        # --- Resolution changes
        current_window_resolution = ps.get_window_size()
        now = time.time()
        if current_window_resolution != self.previous_window_resolution:
            self.last_window_size_changed_time = now
            self.previous_window_resolution = current_window_resolution
        if (self.last_window_size_changed_time is not None) and (
            now - self.last_window_size_changed_time > 0.3
        ):
            self.clear_ray_traced_image()
            self.set_rendering_resolution(current_window_resolution)
            self.last_window_size_changed_time = None

        # --- Rendering
        if self.cfg.rendering.mode == RenderingMode.RAY_TRACING:
            camera_changed = self.reset_accumulation_requested or not np.allclose(
                self.previous_camera_pose, ps.get_camera_view_matrix()
            )
            if camera_changed:
                self.rendering_reset_accumulation()
                self.reset_accumulation_requested = False

            if (
                self.rendering_accumulated_samples
                < self.cfg.rendering.max_accumulated_spp
            ):
                # TODO: we could potentially skip rendering depth in subsequent frames,
                #       since we only accumulate RGB.
                new_img, aovs, self.render_cache = render_scene(
                    self.cfg.rendering,
                    self.scene,
                    seed=self.frame_i,
                    camera_changed=camera_changed,
                    cache=self.render_cache,
                    use_denoiser=self.denoiser is not None,
                )
                if self.ray_traced_img is None:
                    self.ray_traced_img = new_img
                    self.ray_traced_depth = aovs[0]
                else:
                    t = self.cfg.rendering.spp_per_frame / (
                        self.rendering_accumulated_samples
                        + self.cfg.rendering.spp_per_frame
                    )
                    t = dr.opaque(mi.Float32, t)
                    self.ray_traced_img = (1 - t) * self.ray_traced_img + t * new_img
                    # Keep using 1spp depth, it looks better than accumulating.
                    self.ray_traced_depth = aovs[0]

                self.rendering_accumulated_samples += self.cfg.rendering.spp_per_frame

                if self.denoiser is not None:
                    to_sensor = self.render_cache["sensor"].world_transform().inverse()
                    self.ray_traced_img = self.denoiser(
                        self.ray_traced_img,
                        albedo=aovs[1],
                        normals=aovs[2],
                        to_sensor=to_sensor,
                    )

                add_or_update_ray_traced_image_quantity(
                    self.ray_traced_img, self.ray_traced_depth
                )

        # --- Automatic refinement of the radio map
        if self.radio_map is not None:
            if (
                self.rm_accumulated_samples
                < self.cfg.radio_map.accumulate_max_samples_per_tx
            ):
                rm_new = self.compute_radio_map()
                if rm_new is not None:
                    self.radio_map._pathgain_map += rm_new.path_gain
                    self.rm_accumulated_samples += (
                        self.cfg.radio_map.samples_per_it
                        // len(self.scene._transmitters)
                    )
                    # Note: vmin, vmax didn't change so we don't update the colorbar.
                    add_radio_map_to_polyscope(
                        "radio_map",
                        self.radio_map,
                        self.ps_groups,
                        self.cfg.radio_map,
                        direct_update_from_device=self.cfg.radio_map.use_direct_update_from_device,
                        use_alpha=self.cfg.radio_map.use_alpha,
                    )

        # --- Radio device animations
        animation_tick(self, psim.GetIO().DeltaTime)

        # --- GUI
        self.gui()
        self.frame_i += 1
        self.previous_camera_pose = ps.get_camera_view_matrix()

    # ------------------------

    def set_rendering_mode(self, mode: RenderingMode):
        self.cfg.rendering.mode = mode
        is_ray_tracing = self.cfg.rendering.mode == RenderingMode.RAY_TRACING

        if is_ray_tracing and (self.slice_plane is not None):
            self.slice_plane.set_active(False)

        if not is_ray_tracing:
            self.clear_ray_traced_image()

        # Hide Polyscope-side meshes if we are ray tracing.
        self.ps_groups["scene"].set_enabled(not is_ray_tracing)

    def set_use_denoiser(self, use_denoiser: bool):
        self.cfg.rendering.use_denoiser = use_denoiser

        # The integrator will be re-created, so we reset the rendering cache.
        self.render_cache = None

        if self.cfg.rendering.use_denoiser:
            self.denoiser = mi.OptixDenoiser(
                input_size=self.cfg.rendering.rendering_resolution,
                albedo=True,
                normals=True,
                temporal=False,
            )
        else:
            self.denoiser = None
        self.rendering_reset_accumulation()

    def rendering_reset_accumulation(self):
        self.rendering_accumulated_samples = 0
        if self.ray_traced_img is not None:
            self.ray_traced_img[:] = 0
            self.ray_traced_depth[:] = 0

    def clear_ray_traced_image(self):
        import polyscope_bindings as psb

        self.render_cache = None
        self.ray_traced_img = None
        self.ray_traced_depth = None
        self.rendering_accumulated_samples = 0
        psb.get_global_floating_quantity_structure().remove_quantity(
            "ray_traced_img", errorIfAbsent=False
        )

    def set_rendering_resolution(self, window_resolution: tuple[int, int]):
        self.cfg.rendering.current_resolution = tuple(
            v // self.ui_scale for v in window_resolution
        )
        self.clear_ray_traced_image()
        # Re-create denoiser for the new resolution, if appropriate
        self.set_use_denoiser(self.cfg.rendering.use_denoiser)

    # ------------------------

    def set_radio_map(self, radio_map: rt.RadioMap, show: bool = False):
        self.radio_map = radio_map
        self.rm_accumulated_samples = 0
        self.update_radio_map_colorbar()

        if show:
            add_radio_map_to_polyscope(
                "radio_map",
                self.radio_map,
                self.ps_groups,
                self.cfg.radio_map,
                direct_update_from_device=self.cfg.radio_map.use_direct_update_from_device,
                use_alpha=self.cfg.radio_map.use_alpha,
            )

    def update_radio_map_colorbar(self):
        # Draw the colorbar to an array.
        self.rm_colorbar = radio_map_colorbar_to_image(
            self.cfg.radio_map.color_map,
            self.cfg.radio_map.vmin,
            self.cfg.radio_map.vmax,
        )
        # Note: we upload the colorbar to a texture, even if it's not shown right now.
        ps.add_color_alpha_image_quantity(
            "rm_colorbar",
            values=self.rm_colorbar,
            enabled=True,
            image_origin="upper_left",
            show_fullscreen=False,
            # show_in_camera_billboard=True,
        )
        self.rm_colorbar_texture_id = ps.get_quantity_buffer(
            "rm_colorbar", "colors"
        ).get_texture_native_id()

    def compute_radio_map(self) -> rt.RadioMap | None:
        if not self.scene._transmitters:
            return None

        solver = rt.RadioMapSolver()
        samples_per_tx = self.cfg.radio_map.samples_per_it // len(
            self.scene._transmitters
        )
        return solver(
            self.scene,
            seed=self.frame_i,
            center=self.cfg.radio_map.center,
            orientation=self.cfg.radio_map.orientation,
            size=self.cfg.radio_map.size,
            cell_size=self.cfg.radio_map.cell_size,
            measurement_surface=self.cfg.radio_map.measurement_surface,
            # precoding_vec=self.cfg.radio_map.precoding_vec,
            samples_per_tx=samples_per_tx,
            max_depth=self.cfg.radio_map.max_depth,
            los=self.cfg.radio_map.los,
            specular_reflection=self.cfg.radio_map.specular_reflection,
            diffuse_reflection=self.cfg.radio_map.diffuse_reflection,
            refraction=self.cfg.radio_map.refraction,
            diffraction=self.cfg.radio_map.diffraction,
            edge_diffraction=self.cfg.radio_map.edge_diffraction,
            diffraction_lit_region=self.cfg.radio_map.diffraction_lit_region,
        )

    def has_visible_radio_map(self) -> tuple[bool, ps.SurfaceMesh]:
        rm_struct = None
        if ps.has_surface_mesh("radio_map"):
            rm_struct = ps.get_surface_mesh("radio_map")
            rm_visible = rm_struct.is_enabled()
        else:
            rm_visible = False

        return rm_visible, rm_struct

    # ------------------------

    def update_paths(self, clear_first: bool = False, show: bool = True):
        # Optionally throttle path computations to reduce load
        current_time = time.time()
        time_since_last_update = current_time - self._last_paths_update_time
        if time_since_last_update < self.cfg.paths.min_update_delay_s:
            # Skip this update
            return

        if clear_first:
            self.clear_paths()

        self.paths = self.compute_paths()
        self._last_paths_update_time = current_time
        if self.paths is None:
            return

        if show:
            add_paths_to_polyscope(self, self.paths, self.ps_groups)

        if self.cfg.paths.compute_cir:
            self.paths_taps = self.paths.taps(
                bandwidth=self.cfg.paths.bandwidth,
                l_min=self.cfg.paths.l_min,
                l_max=self.cfg.paths.l_max,
                sampling_frequency=self.cfg.paths.sampling_frequency,
                num_time_steps=self.cfg.paths.num_time_steps,
                normalize=self.cfg.paths.normalize,
                normalize_delays=self.cfg.paths.normalize_delays,
                out_type="numpy",
            )
            self.paths_cir = self.paths.cir(normalize_delays=True, out_type="numpy")

            # If self.paths_cir[0] has no elements, replace self.paths_taps with zeros, first element = 1e-10
            if np.sum(self.paths_cir[0]) == 0:
                # Set the first element to 1e-10
                self.paths_taps[0, 0, 0, 0, 0, 0] = 1e-10

    def compute_paths(self) -> rt.Paths | None:
        if not self.scene._transmitters or not self.scene._receivers:
            return None

        solver = rt.PathSolver()
        return solver(
            self.scene,
            max_depth=self.cfg.paths.max_depth,
            max_num_paths_per_src=self.cfg.paths.max_num_paths_per_src,
            samples_per_src=self.cfg.paths.samples_per_src,
            synthetic_array=self.cfg.paths.synthetic_array,
            los=self.cfg.paths.los,
            specular_reflection=self.cfg.paths.specular_reflection,
            diffuse_reflection=self.cfg.paths.diffuse_reflection,
            refraction=self.cfg.paths.refraction,
            diffraction=self.cfg.paths.diffraction,
            edge_diffraction=self.cfg.paths.edge_diffraction,
            diffraction_lit_region=self.cfg.paths.diffraction_lit_region,
            seed=12345,
        )

    # ------------------------

    def add_radio_device(
        self,
        position: list[float],
        is_transmitter: bool,
        allow_auto_update: bool = True,
    ) -> rt.RadioDevice:
        # TODO: controllable offset to the clicked surface (along normal?)
        existing_rd = (
            self.scene._transmitters if is_transmitter else self.scene._receivers
        )

        # Add actual radio device to Sionna scene
        prefix = "tx" if is_transmitter else "rx"
        free_index = len(existing_rd)
        while f"{prefix}-{free_index}" in existing_rd:
            free_index += 1

        new_rd = (rt.Transmitter if is_transmitter else rt.Receiver)(
            name=f"{prefix}-{free_index}",
            position=[float(p) for p in position],
            orientation=[0.0, 0.0, 0.0],
        )
        self.scene.add(new_rd)

        set_or_update_radio_devices_polyscope(
            existing_rd,
            is_transmitter,
            self,
        )

        if (
            allow_auto_update
            and self.cfg.radio_map.auto_update
            and is_transmitter
            and (self.radio_map is not None)
        ):
            self.set_radio_map(self.compute_radio_map(), show=True)
        if allow_auto_update and self.cfg.paths.auto_update:
            self.update_paths(show=True)

        return new_rd

    def remove_object(
        self, object: rt.SceneObject, selected_type: SelectionType
    ) -> None:
        match selected_type:
            case SelectionType.Transmitter:
                del self.scene._transmitters[object.name]
            case SelectionType.Receiver:
                del self.scene._receivers[object.name]
            case _:
                print(f"[!] Unexpected selection type: {selected_type}")
                pass

        if object.name in self.animation_config.trajectories:
            del self.animation_config.trajectories[object.name]

    def clear_radio_devices(self) -> None:
        for name in self.scene._transmitters.keys():
            if name in self.animation_config.trajectories:
                del self.animation_config.trajectories[name]
        for name in self.scene._receivers.keys():
            if name in self.animation_config.trajectories:
                del self.animation_config.trajectories[name]

        self.scene._transmitters.clear()
        self.scene._receivers.clear()
        for name in self.ps_groups["rd"].get_child_structure_names():
            ps.get_point_cloud(name).remove()
        if self.selected_type in (SelectionType.Transmitter, SelectionType.Receiver):
            self.clear_selection()

        if self.cfg.radio_map.auto_update:
            self.clear_radio_map()
        if self.cfg.paths.auto_update:
            self.clear_paths()

    def reset_radio_map(self):
        """
        Attempts to reset the radio map to zero, as well as the accumulation counter.
        However, if the number of transmitters has changed, the radio map will be
        completely removed (and re-computed if auto-updates are enabled).
        """
        self.rm_accumulated_samples = 0
        if self.radio_map is None:
            return

        if len(self.scene._transmitters) != self.radio_map.num_tx:
            self.clear_radio_map()
            if self.cfg.radio_map.auto_update:
                self.set_radio_map(self.compute_radio_map(), show=True)
            return

        self.radio_map._pathgain_map *= 0.0

    def clear_radio_map(self):
        self.radio_map = None
        self.rm_accumulated_samples = 0
        self.rm_colorbar = None
        self.rm_colorbar_texture_id = None
        if ps.has_surface_mesh("radio_map"):
            ps.get_surface_mesh("radio_map").remove()

    def clear_paths(self):
        self.paths = None
        self.paths_taps = None
        self.paths_cir = None
        if ps.has_curve_network("paths"):
            ps.get_curve_network("paths").remove()

    # ------------------------

    def set_slice_plane_active(self, active: bool):
        if self.slice_plane is None:
            return
        self.slice_plane.set_active(active)

        if active and self.cfg.rendering.mode == RenderingMode.RAY_TRACING:
            # Switch to rasterization mode when activating the slice plane.
            self.set_rendering_mode(RenderingMode.RASTERIZATION)

    # ------------------------

    def process_inputs(self):
        imgui_io = psim.GetIO()
        if imgui_io.WantCaptureKeyboard:
            # If the user is e.g. typing in a text field, don't process
            # keyboard or mouse inputs.
            return

        allow_click = not imgui_io.WantCaptureMouse
        has_left_click = allow_click and psim.IsMouseClicked(psim.ImGuiMouseButton_Left)
        has_mouse_drag = psim.IsMouseDragging(
            psim.ImGuiMouseButton_Left
        ) or psim.IsMouseDragging(psim.ImGuiMouseButton_Right)
        has_left_release = allow_click and psim.IsMouseReleased(
            psim.ImGuiMouseButton_Left
        )
        has_right_click = allow_click and psim.IsMouseClicked(
            psim.ImGuiMouseButton_Right
        )
        has_active_item = psim.IsAnyItemActive()

        # TODO: +/- to zoom in/out
        # TODO: keyboard shortcuts to move around (WASD + QE)
        # TODO: keyboard shortcuts to rotate the envmap?

        # K/L or (Ctrl + left/right click): add transmitter/receiver
        has_k = psim.IsKeyPressed(psim.ImGuiKey(ps.get_key_code("K")))
        has_l = psim.IsKeyPressed(psim.ImGuiKey(ps.get_key_code("L")))
        if (
            has_k
            or has_l
            or (
                imgui_io.KeyCtrl
                and (has_left_click or has_right_click)
                and not imgui_io.KeyShift
            )
        ):
            is_transmitter = imgui_io.MouseClicked[0] or has_k

            origin = ps.get_camera_view_matrix()[:3, 3]
            rd_position = ps.screen_coords_to_world_position(imgui_io.MousePos)

            if np.all(np.isfinite(rd_position)):
                normal = get_normal_for_path(self.scene, origin, rd_position)
                if normal is None:
                    normal = np.array([0, 0, 1])
                # Make sure that normal is oriented towards the camera
                if np.dot(normal, rd_position - origin) < 0:
                    normal = -normal
                rd_position += 1.5 * normal

                self.add_radio_device(rd_position, is_transmitter)

        # Plain left click: object selection
        if has_left_release and not (
            imgui_io.KeyCtrl
            or imgui_io.KeyShift
            or imgui_io.KeyAlt
            or has_mouse_drag
            or self.was_mouse_dragging
        ):
            self.process_pick_result(ps.pick(screen_coords=imgui_io.MousePos))

        # Shift + R: reload code
        if imgui_io.KeyShift and psim.IsKeyPressed(
            psim.ImGuiKey(ps.get_key_code("R")), repeat=False
        ):
            self.code_reload_requested = True

        # R: reset camera to initial position
        if psim.IsKeyPressed(psim.ImGuiKey(ps.get_key_code("R")), repeat=False):
            self.move_camera_home()
        # F: fit scene in camera viewport
        if psim.IsKeyPressed(psim.ImGuiKey(ps.get_key_code("F")), repeat=False):
            self.fit_camera_to_scene()

        # C: go to next rendering mode.
        if psim.IsKeyPressed(psim.ImGuiKey(ps.get_key_code("C")), repeat=False):
            self.set_rendering_mode(
                RenderingMode((self.cfg.rendering.mode.value + 1) % len(RenderingMode))
            )

        # H / ?: show help window
        if psim.IsKeyPressed(psim.ImGuiKey(ps.get_key_code("H")), repeat=False) or (
            imgui_io.KeyShift
            and psim.IsKeyPressed(psim.ImGuiKey(ps.get_key_code("/")), repeat=False)
        ):
            self.cfg.show_help_window = not self.cfg.show_help_window

        # M: toggle radio map computation and display
        if psim.IsKeyPressed(psim.ImGuiKey(ps.get_key_code("M")), repeat=False):
            rm_visible, rm_struct = self.has_visible_radio_map()

            if rm_visible:
                # Hide radio map (but don't delete it, we may want to show it later)
                self.cfg.radio_map.auto_update = False
                rm_struct.set_enabled(False)
            else:
                self.cfg.radio_map.auto_update = True
                if self.radio_map is not None:
                    # Show the existing radio map
                    rm_struct.set_enabled(True)
                else:
                    # Compute and show the radio map
                    self.set_radio_map(self.compute_radio_map(), show=True)

        if self.slice_plane is not None:
            # S: toggle slice plane
            if psim.IsKeyPressed(psim.ImGuiKey(ps.get_key_code("S")), repeat=False):
                self.set_slice_plane_active(not self.slice_plane.get_active())

            # Alt + left click drag: move slice plane along its normal.
            # TODO: avoid using middle click, since it's not easy to do on trackpads.
            if self.slice_plane.get_active() and imgui_io.KeyAlt and has_mouse_drag:
                ps.set_do_default_mouse_interaction(False)
                center = self.slice_plane.get_center()
                normal = self.slice_plane.get_normal()
                self.slice_plane.set_pose(
                    center + 0.3 * self.ui_scale * imgui_io.MouseDelta[1] * normal,
                    normal,
                )
            else:
                ps.set_do_default_mouse_interaction(True)

        # Tab: toggle show GUI (ours)
        if not has_active_item and psim.IsKeyPressed(psim.ImGuiKey_Tab, repeat=False):
            self.cfg.gui_mode = GuiMode((self.cfg.gui_mode.value + 1) % len(GuiMode))

        # Esc: close help window or de-select
        if psim.IsKeyPressed(psim.ImGuiKey_Escape, repeat=False):
            if self.cfg.show_help_window:
                self.cfg.show_help_window = False
            elif self.selected_object is not None:
                self.clear_selection()

        # Ctrl + Q: exit
        if imgui_io.KeyCtrl and psim.IsKeyPressed(psim.ImGuiKey(ps.get_key_code("Q"))):
            ps.unshow()

        self.was_mouse_dragging = has_mouse_drag

    def process_pick_result(self, pick_result: ps.PickResult) -> bool:
        if not pick_result.is_hit or "index" not in pick_result.structure_data:
            self.clear_selection()
            return False

        picked_index = pick_result.structure_data["index"]
        if pick_result.structure_name in "Transmitters":
            self.selected_object = list(self.scene._transmitters.values())[picked_index]
            self.selected_type = SelectionType.Transmitter
            return True
        elif pick_result.structure_name in "Receivers":
            self.selected_object = list(self.scene._receivers.values())[picked_index]
            self.selected_type = SelectionType.Receiver
            return True
        elif ps.has_surface_mesh(pick_result.structure_name):
            # Check if it's one of our scene meshes
            objects = getattr(self.scene, "objects", {})
            if pick_result.structure_name in objects:
                self.selected_object = objects[pick_result.structure_name]
                self.selected_type = SelectionType.Mesh
                return True

        self.clear_selection()
        return False

    def clear_selection(self):
        self.selected_object = None
        self.selected_type = None
        if ps.has_point_cloud("Gizmo"):
            ps.get_point_cloud("Gizmo").remove()
        if ps.has_curve_network("Trajectory"):
            ps.get_curve_network("Trajectory").remove()

    def gui(self):
        # TODO: change GUI accent color to a non-default color.

        if self.cfg.gui_mode == GuiMode.HIDDEN:
            return

        # --- Selection window
        if self.selected_object is not None:
            selection_gui(self, self.selected_object, self.selected_type)

        # --- Help window
        if self.cfg.show_help_window:
            self.gui_help_window()

        # --- Colorbar window
        if (
            self.has_visible_radio_map()[0]
            and self.cfg.radio_map.show_colorbar
            and (self.rm_colorbar is not None)
            and (self.rm_colorbar_texture_id is not None)
            and hasattr(psim, "Image")
        ):
            window_resolution = ps.get_window_size()
            h, w = self.rm_colorbar.shape[:2]
            psim.SetNextWindowSize((w * self.ui_scale, h * self.ui_scale))
            psim.SetNextWindowPos(
                (0.5 * (window_resolution[0] - w * self.ui_scale), 5 * self.ui_scale)
            )
            psim.Begin(
                "Colorbar",
                open=True,
                flags=(
                    psim.ImGuiWindowFlags_NoTitleBar
                    | psim.ImGuiWindowFlags_NoDecoration
                    | psim.ImGuiWindowFlags_NoBackground
                ),
            )
            psim.SetCursorPosX(0)
            psim.SetCursorPosY(0)
            psim.Image(
                psim.ImTextureRef(self.rm_colorbar_texture_id),
                (self.rm_colorbar.shape[1], self.rm_colorbar.shape[0]),
            )
            psim.End()

        # --- Main GUI window
        psim.SetNextWindowSize(
            (430 * self.ui_scale, 800 * self.ui_scale), psim.ImGuiCond_FirstUseEver
        )
        psim.SetNextWindowPos(
            (10 * self.ui_scale, 10 * self.ui_scale), psim.ImGuiCond_FirstUseEver
        )
        psim.Begin("Sionna RT##sionna", open=True)

        psim.Text(f"Frame time: {1000 * psim.GetIO().DeltaTime:.2f} ms")

        psim.SameLine()
        bw = 20
        psim.SetCursorPosX(
            (psim.GetCursorPosX() + psim.GetContentRegionAvail()[0] - bw)
            * self.ui_scale
        )
        if psim.Button("?", size=(bw * self.ui_scale, 0)):
            self.cfg.show_help_window = not self.cfg.show_help_window

        if psim.CollapsingHeader("Scene", psim.ImGuiTreeNodeFlags_DefaultOpen):
            psim.Spacing()

            # Quick pick from built-in or recently-loaded scenes.
            psim.Text("Scene selection:")
            changed, combo_i = psim.Combo(
                "##scene_picker",
                self.current_scene_idx,
                self.known_scene_names,
            )
            if changed:
                self.load_scene_requested = self.known_scene_paths[combo_i]

            if self.cfg.scenario_filename:
                psim.Spacing()
                if psim.Button("Reload Scenario", size=(psim.GetContentRegionAvail()[0], 0)):
                    self.code_reload_requested = True

            psim.Spacing()

            changed, val = psim.SliderFloat(
                "Road lift (Z)",
                self.cfg.road_lift,
                v_min=0.0,
                v_max=1.0,
                format="%.3f",
            )
            if changed:
                self.cfg.road_lift = val
                self.load_scene_requested = True


        self._gui_materials()

        if psim.CollapsingHeader("Radio devices", psim.ImGuiTreeNodeFlags_DefaultOpen):
            psim.Spacing()
            # TODO: button to place radio devices: at random; or samples from a radio map
            antenna_array_gui(self)

            clicked = psim.Button("Clear all radio devices")
            if clicked:
                self.clear_radio_devices()

            psim.Spacing()

        if psim.CollapsingHeader("Radio map", psim.ImGuiTreeNodeFlags_DefaultOpen):
            psim.Spacing()
            needs_update = False
            needs_visual_update = False

            if psim.Button("Compute radio map"):
                self.set_radio_map(self.compute_radio_map(), show=True)

            psim.SameLine()
            psim.BeginDisabled(self.radio_map is None)
            if psim.Button("Remove##radio_map"):
                self.clear_radio_map()
            psim.EndDisabled()

            psim.SameLine()
            _, self.cfg.radio_map.auto_update = psim.Checkbox(
                "Automatic update##rm", self.cfg.radio_map.auto_update
            )

            # -- Radio map computation options
            psim.Spacing()

            changed, self.cfg.radio_map.cell_size = psim.InputFloat2(
                "Cell size",
                self.cfg.radio_map.cell_size,
                # v_min=0.1,
                # v_max=100,
                format="%.2f",
            )
            if changed:
                self.cfg.radio_map.cell_size = tuple(
                    min(max(v, 0.01), 100) for v in self.cfg.radio_map.cell_size
                )
            needs_update |= changed

            changed, self.cfg.radio_map.log_samples_per_it = psim.SliderFloat(
                "Samples / it (log 10)",
                self.cfg.radio_map.log_samples_per_it,
                v_min=0,
                v_max=9,
                format="10^%.1f",
            )
            needs_update |= changed

            changed, self.cfg.radio_map.max_depth = psim.SliderInt(
                "Max depth##rm",
                self.cfg.radio_map.max_depth,
                v_min=1,
                v_max=10,
            )
            needs_update |= changed

            # -- Checkboxes table
            needs_update |= self._gui_features_checkboxes(self.cfg.radio_map, "##rm")

            # -- Radio map display options
            if self.radio_map is not None:
                psim.Spacing()

                changed_p = False
                changed_n = False
                changed_vmin = False
                changed_vmax = False

                psim.Text("Accumulating samples:")

                psim.PushStyleColor(psim.ImGuiCol_PlotHistogram, NVIDIA_GREEN)
                psim.ProgressBar(
                    min(
                        self.rm_accumulated_samples
                        / self.cfg.radio_map.accumulate_max_samples_per_tx,
                        1.0,
                    ),
                    (psim.CalcItemWidth(), 0),
                )
                psim.PopStyleColor()

                struct = ps.get_surface_mesh("radio_map")
                changed, show_rm = psim.Checkbox("Show radio map", struct.is_enabled())
                if changed:
                    struct.set_enabled(show_rm)

                psim.SameLine()
                _, self.cfg.radio_map.show_colorbar = psim.Checkbox(
                    "Show color bar", self.cfg.radio_map.show_colorbar
                )

                changed_cmap, self.rm_color_map_index = psim.Combo(
                    "Colormap",
                    self.rm_color_map_index,
                    self.rm_color_map_options,
                )
                if changed_cmap:
                    self.cfg.radio_map.color_map = self.rm_color_map_options[
                        self.rm_color_map_index
                    ]

                # -- Metric selection
                changed_m, metric_i = psim.Combo(
                    "Metric",
                    ["path_gain", "rss", "sinr"].index(self.cfg.radio_map.metric),
                    ["Path gain", "RSS", "SINR"],
                )
                if changed_m:
                    self.cfg.radio_map.metric = ["path_gain", "rss", "sinr"][metric_i]
                needs_visual_update = (
                    changed_cmap or changed_m or changed_vmin or changed_vmax
                )

                if self.cfg.radio_map.metric in ["rss", "sinr"]:
                    changed_p, self.cfg.radio_map.tx_power_dbm = psim.InputFloat(
                        "TX Power (dBm)", self.cfg.radio_map.tx_power_dbm
                    )
                    needs_visual_update |= changed_p
                if self.cfg.radio_map.metric == "sinr":
                    changed_n, self.cfg.radio_map.noise_power_dbm = psim.InputFloat(
                        "Noise Power (dBm)", self.cfg.radio_map.noise_power_dbm
                    )
                    needs_visual_update |= changed_n

                changed_vmin, self.cfg.radio_map.vmin = psim.SliderFloat(
                    "vmin",
                    self.cfg.radio_map.vmin,
                    v_min=-200,
                    v_max=100,
                )
                changed_vmax, self.cfg.radio_map.vmax = psim.SliderFloat(
                    "vmax",
                    self.cfg.radio_map.vmax,
                    v_min=-200,
                    v_max=100,
                )
                self.cfg.radio_map.vmin = min(
                    self.cfg.radio_map.vmin, self.cfg.radio_map.vmax
                )
                self.cfg.radio_map.vmax = max(
                    self.cfg.radio_map.vmin, self.cfg.radio_map.vmax
                )

                if self.cfg.radio_map.center is not None:
                    # Allow adjusting the height (Z-offset) of the radio map
                    center_list = list(self.cfg.radio_map.center)
                    psim.SetNextItemWidth(150 * self.ui_scale)
                    changed_z_s, center_list[2] = psim.SliderFloat(
                        "##height_z_slider",
                        center_list[2],
                        v_min=-20.0,
                        v_max=100.0,
                    )
                    psim.SameLine()
                    psim.SetNextItemWidth(80 * self.ui_scale)
                    changed_z_i, center_list[2] = psim.InputFloat(
                        "Height (Z)",
                        center_list[2],
                    )
                    changed_z = changed_z_s or changed_z_i
                    if changed_z:
                        self.cfg.radio_map.center = tuple(center_list)
                        needs_update = True

                needs_visual_update |= changed_cmap or changed_vmin or changed_vmax

            if self.cfg.radio_map.auto_update and needs_update:
                self.set_radio_map(self.compute_radio_map(), show=False)
            if needs_update or needs_visual_update:
                self.update_radio_map_colorbar()
                add_radio_map_to_polyscope(
                    "radio_map",
                    self.radio_map,
                    self.ps_groups,
                    self.cfg.radio_map,
                    direct_update_from_device=self.cfg.radio_map.use_direct_update_from_device,
                    use_alpha=self.cfg.radio_map.use_alpha,
                )

            # TODO: plane / mesh picker (changes type of radio map)
            # TODO: some way to move the radio map (at least the plane z offset)

            psim.Spacing()

        if psim.CollapsingHeader("Paths", psim.ImGuiTreeNodeFlags_DefaultOpen):
            psim.Spacing()
            needs_update = False

            clicked = psim.Button("Compute paths")
            if clicked:
                self.update_paths(show=True)

            psim.SameLine()
            psim.BeginDisabled(self.paths is None)
            if psim.Button("Remove##paths"):
                self.clear_paths()
            psim.EndDisabled()

            psim.SameLine()
            _, self.cfg.paths.auto_update = psim.Checkbox(
                "Automatic update##paths", self.cfg.paths.auto_update
            )

            changed, self.cfg.paths.max_depth = psim.SliderInt(
                "Max depth##paths",
                self.cfg.paths.max_depth,
                v_min=1,
                v_max=10,
            )
            needs_update |= changed

            psim.SetNextItemWidth(200 * self.ui_scale)
            changed, self.cfg.paths.synthetic_array = psim.Checkbox(
                "Synthetic array##paths", self.cfg.paths.synthetic_array
            )
            needs_update |= changed

            needs_update |= self._gui_features_checkboxes(self.cfg.paths, "##paths")

            # TODO: rendering parameters (radius, transparency, etc)
            # TODO: show & configure segment color per type of interaction
            needs_visual_update = False

            if self.cfg.paths.auto_update and needs_update:
                self.update_paths(show=False)
            if needs_update or needs_visual_update:
                add_paths_to_polyscope(self, self.paths, self.ps_groups)

            psim.Spacing()

        if psim.CollapsingHeader("Animation"):
            animation_gui(self)
            psim.Spacing()

        if psim.CollapsingHeader("Rendering"):
            psim.Spacing()

            changed, combo_i = psim.Combo(
                "Rendering mode",
                self.cfg.rendering.mode.value,
                RENDERING_MODE_NAMES,
            )
            if changed:
                self.cfg.rendering.mode = RenderingMode(combo_i)
                self.set_rendering_mode(self.cfg.rendering.mode)

            if (self.cfg.rendering.mode == RenderingMode.RAY_TRACING) and (
                "cuda" in mi.variant()
            ):
                _, self.cfg.rendering.spp_per_frame = psim.SliderInt(
                    "SPP / frame",
                    self.cfg.rendering.spp_per_frame,
                    v_min=1,
                    v_max=1024,
                )
                self.cfg.rendering.spp_per_frame = max(
                    self.cfg.rendering.spp_per_frame, 1
                )
                _, self.cfg.rendering.max_accumulated_spp = psim.SliderInt(
                    "SPP max",
                    self.cfg.rendering.max_accumulated_spp,
                    v_min=1,
                    v_max=1024,
                )
                self.cfg.rendering.max_accumulated_spp = max(
                    self.cfg.rendering.max_accumulated_spp, 1
                )

                changed, self.cfg.rendering.relative_resolution = psim.SliderFloat(
                    "Rel. resolution",
                    self.cfg.rendering.relative_resolution,
                    v_min=0.1,
                    v_max=1.0,
                    format="%.2f",
                )
                self.cfg.rendering.relative_resolution = min(
                    max(self.cfg.rendering.relative_resolution, 0.1), 1.0
                )
                if changed:
                    self.clear_ray_traced_image()
                    # Need to re-create the denoiser for the new resolution
                    self.set_use_denoiser(self.cfg.rendering.use_denoiser)

                changed, self.cfg.rendering.envmap_rotation_deg = psim.SliderFloat(
                    "Lighting angle",
                    self.cfg.rendering.envmap_rotation_deg,
                    v_min=-180,
                    v_max=180,
                    format="%.0f",
                )
                self.cfg.rendering.envmap_rotation_deg = min(
                    max(self.cfg.rendering.envmap_rotation_deg, -180), 180
                )
                if changed:
                    set_envmap_rotation(
                        self.render_cache, self.cfg.rendering.envmap_rotation_deg
                    )
                    self.reset_accumulation_requested = True

                changed, self.cfg.rendering.envmap_factor = psim.SliderFloat(
                    "Lighting brightness",
                    self.cfg.rendering.envmap_factor,
                    v_min=0.0,
                    v_max=20.0,
                    format="%.1f",
                )
                if changed:
                    if not set_envmap_intensity(
                        self.render_cache, self.cfg.rendering.envmap_factor
                    ):
                        self.render_cache = None  # Fallback: Re-create scene
                    self.reset_accumulation_requested = True

                changed, self.cfg.rendering.use_denoiser = psim.Checkbox(
                    "Use OptiX denoiser", self.cfg.rendering.use_denoiser
                )
                if changed:
                    self.set_use_denoiser(self.cfg.rendering.use_denoiser)

            changed, self.cfg.use_vsync = psim.Checkbox("VSync", self.cfg.use_vsync)
            if changed:
                ps.set_enable_vsync(self.cfg.use_vsync)

            psim.SameLine()

            self.cfg.background_color = ps.get_background_color()
            changed, self.cfg.background_color = psim.ColorEdit4(
                "Background",
                self.cfg.background_color,
                psim.ImGuiColorEditFlags_NoInputs,
            )
            if changed:
                ps.set_background_color(self.cfg.background_color)

            changed, self.cfg.show_polyscope_gui = psim.Checkbox(
                "Show Polyscope UI", self.cfg.show_polyscope_gui
            )
            if changed:
                ps.set_build_default_gui_panels(self.cfg.show_polyscope_gui)

            psim.Spacing()

            if self.slice_plane is not None:
                psim.SeparatorText("Slice plane")
                changed, plane_active = psim.Checkbox(
                    "Active", self.slice_plane.get_active()
                )
                if changed:
                    self.set_slice_plane_active(plane_active)

                if plane_active:
                    psim.SameLine()
                    changed, draw_plane = psim.Checkbox(
                        "Show plane", self.slice_plane.get_draw_plane()
                    )
                    if changed:
                        self.slice_plane.set_draw_plane(draw_plane)

                    psim.SameLine()
                    changed, gizmo_active = psim.Checkbox(
                        "Show gizmo", self.slice_plane.get_draw_widget()
                    )
                    if changed:
                        self.slice_plane.set_draw_widget(gizmo_active)

        psim.End()  # End main Sionna RT window

    def _gui_features_checkboxes(
        self, cfg: RadioMapConfig | PathsConfig, suffix: str
    ) -> bool:
        any_changed = False

        psim.Columns(2, borders=False)
        psim.SetColumnWidth(0, 165 * self.ui_scale)

        changed, cfg.los = psim.Checkbox("Line of sight" + suffix, cfg.los)
        any_changed |= changed

        changed, cfg.diffuse_reflection = psim.Checkbox(
            "Diffuse reflection" + suffix, cfg.diffuse_reflection
        )
        any_changed |= changed

        changed, cfg.diffraction = psim.Checkbox(
            "Diffraction" + suffix, cfg.diffraction
        )
        any_changed |= changed

        psim.NextColumn()

        changed, cfg.specular_reflection = psim.Checkbox(
            "Specular reflection" + suffix, cfg.specular_reflection
        )
        any_changed |= changed

        changed, cfg.refraction = psim.Checkbox("Refraction" + suffix, cfg.refraction)
        any_changed |= changed

        if cfg.diffraction:
            changed, cfg.edge_diffraction = psim.Checkbox(
                "Edge" + suffix, cfg.edge_diffraction
            )
            any_changed |= changed

            psim.SameLine()
            changed, cfg.diffraction_lit_region = psim.Checkbox(
                "Lit region" + suffix, cfg.diffraction_lit_region
            )
            any_changed |= changed

        psim.Columns(1)

        return any_changed

    def _gui_materials(self):
        if self.scene is None:
            return

        radio_materials = getattr(self.scene, "radio_materials", {})
        if not radio_materials:
            return

        if psim.CollapsingHeader("Materials", psim.ImGuiTreeNodeFlags_DefaultOpen):
            psim.Spacing()
            for name, mat in radio_materials.items():
                if not hasattr(mat, "color"):
                    continue

                # Convert to list if it's a drjit array or similar
                color = list(mat.color)

                changed, new_color = psim.ColorEdit3(
                    f"##mat_color_{name}",
                    color,
                    psim.ImGuiColorEditFlags_NoInputs,
                )
                psim.SameLine()
                psim.Text(name)

                if changed:
                    self.update_material_color(mat, new_color)
            psim.Spacing()

    def gui_help_window(self):
        window_resolution = ps.get_window_size()
        w, h = 600, 600
        psim.SetNextWindowSize(
            (w * self.ui_scale, h * self.ui_scale), psim.ImGuiCond_FirstUseEver
        )
        psim.SetNextWindowPos(
            (
                0.5 * (window_resolution[0] - w * self.ui_scale),
                0.5 * (window_resolution[1] - h * self.ui_scale),
            ),
            psim.ImGuiCond_FirstUseEver,
        )

        _, self.cfg.show_help_window = psim.Begin(
            "Help",
            open=True,
            flags=psim.ImGuiWindowFlags_Modal
            | psim.ImGuiWindowFlags_NoFocusOnAppearing,
        )

        psim.Text(
            f"Sionna RT GUI v{GUI_VERSION[0]}.{GUI_VERSION[1]}.{GUI_VERSION[2]}.\n(c) NVIDIA Corporation 2025.\n\n"
            "Uses map data from OpenStreetMap (openstreetmap.org/copyright)."
        )

        psim.NewLine()

        for title, table in HELP_WINDOW_TABLES.items():
            # Centered text
            text_pos = (
                psim.GetCursorPosX()
                + psim.GetColumnWidth(0) * 0.5
                - psim.CalcTextSize(title)[0]
            )
            psim.SetCursorPosX(text_pos * self.ui_scale)
            psim.Text(title)

            psim.Separator()
            psim.Columns(2)
            psim.SetColumnWidth(0, 220 * self.ui_scale)
            psim.Text(os.linesep.join(table.keys()))
            psim.NextColumn()
            psim.Text(os.linesep.join(table.values()))
            psim.Columns(1)
            psim.Separator()

            psim.NewLine()

        psim.Columns(1)
        psim.End()
