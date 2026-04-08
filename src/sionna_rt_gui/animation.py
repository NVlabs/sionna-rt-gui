#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

from collections import defaultdict
from enum import Enum
from dataclasses import dataclass, field
import time
import numpy as np

import drjit as dr
import polyscope as ps
from polyscope import imgui as psim
from sionna import rt
from sionna.rt.utils.render import scene_scale

from .config import DEFAULT_SLICE_PLANE_NAME
from .sionna_utils import set_or_update_radio_devices_polyscope


class LoopingMode(Enum):
    NoLoop = 0
    Mirror = 1
    Repeat = 2


LOOPING_MODE_NAMES = ["None", "Mirror", "Repeat"]
assert len(LOOPING_MODE_NAMES) == len(LoopingMode)

SPEED_BUTTON_COLOR = np.array((0.269, 0.474, 0.377, 1.0))


@dataclass(kw_only=True)
class Trajectory:
    # Whether to enable the trajectory
    enabled: bool = False
    # Distance along the trajectory [m].
    distance: float = 0.0
    # Control points defining the polyline of the trajectory
    # TODO: consider supporting changes in orientation as well (requires fancier interpolation)
    points: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    # Movement velocity [m/s]. Default is set based on a walking speed of 4 km/h.
    velocity: float = 1.11
    # Looping mode index
    looping_mode_i: int = LoopingMode.Mirror.value
    # Whether the trajectory is currently playing in reverse, due e.g. to mirror looping mode.
    backward: bool = False

    # Cumulative distribution of distances along the trajectory, starting at zero.
    # Has width equal to the number of points.
    _cumulative_distances: list[float] = field(default_factory=list)

    @property
    def looping_mode(self) -> LoopingMode:
        return LoopingMode(self.looping_mode_i)

    def current_position_and_direction(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return the current world-space position based on the distance along the trajectory."""
        n_points = len(self.points)
        if n_points == 0:
            return None
        if n_points == 1:
            return self.points[0], np.zeros(3)

        safe_dist = np.clip(self.distance, 0.0, self.total_distance())
        end_idx = np.searchsorted(self._cumulative_distances, safe_dist, side="left")
        start_idx = max(end_idx - 1, 0)
        if start_idx == end_idx:
            direction = np.zeros(3)
            return self.points[start_idx], direction

        start_dist = self._cumulative_distances[start_idx]
        end_dist = self._cumulative_distances[end_idx]
        dist_diff = end_dist - start_dist
        if dist_diff == 0:
            return self.points[start_idx], np.zeros(3)
        t = (safe_dist - start_dist) / dist_diff

        direction = self.points[end_idx] - self.points[start_idx]
        pos = self.points[start_idx] + t * direction
        norm = np.linalg.norm(direction)
        if norm == 0:
            return pos, np.zeros(3)
        return pos, direction / norm

    def add_point(self, point: np.ndarray | list[float]):
        point = np.array(point)
        assert point.size == 3, "Point must be a 3D vector"

        if len(self.points) == 0:
            self.points = point.squeeze()[None, :]
            self._cumulative_distances = [0.0]
        else:
            self.points = np.concatenate(
                [self.points, point.squeeze()[None, :]], axis=0
            )
            dist_to_new = np.linalg.norm(self.points[-1] - self.points[-2])
            self._cumulative_distances.append(
                self._cumulative_distances[-1] + dist_to_new
            )

        # Snap to the latest added point
        self.distance = self._cumulative_distances[-1]

    def total_distance(self) -> float:
        if len(self._cumulative_distances) == 0:
            return 0.0
        return self._cumulative_distances[-1]

    def clear(self):
        self.points = np.array([], dtype=float)
        self.distance = 0.0
        self.backward = False
        self._cumulative_distances.clear()

    def __len__(self) -> int:
        return len(self.points)


@dataclass(kw_only=True)
class AnimationConfig:
    playing: bool = True
    speed_multiplier: float = 1.0

    # Time at which the animation started playing the first time (Unix timestamp)
    time_started: float | None = None

    trajectories: dict[str, Trajectory] = field(
        default_factory=lambda: defaultdict(Trajectory)
    )

    def clear(self):
        self.trajectories.clear()
        self.time_started = None


def animation_gui(gui: "SionnaRtGui"):
    """
    GUI for the main animation controls.
    """
    was_playing = gui.animation_config.playing
    toggled = psim.Button("Pause" if was_playing else "Play")
    if toggled:
        gui.animation_config.playing = not gui.animation_config.playing
        if gui.animation_config.playing:
            gui.animation_config.time_started = time.time()

    for speed in [0.5, 1.0, 2.0, 5.0, 10.0, 50.0]:
        is_current = gui.animation_config.speed_multiplier == speed
        color = SPEED_BUTTON_COLOR.copy()
        if not is_current:
            color[:3] *= 0.5

        psim.PushStyleColor(psim.ImGuiCol_Button, color)
        if psim.Button(f"{speed}x"):
            gui.animation_config.speed_multiplier = speed
        psim.PopStyleColor()

        psim.SameLine()
    psim.Text(f"Speed: {gui.animation_config.speed_multiplier:.1f}x")


def trajectory_gui(gui: "SionnaRtGui", object: rt.SceneObject):
    """
    GUI to edit the animation trajectory & velocity of a selected object.
    """
    traj = gui.animation_config.trajectories[object.name]

    # TODO: edit/remove/split existing trajectory points

    if psim.Button("Add current position"):
        # Prevent the trajectory from playing while we are editing,
        # otherwise the point will keep snapping back.
        traj.enabled = False
        traj.add_point(object.position.numpy())

    n_points = len(traj)
    has_points = n_points > 0
    if has_points:
        psim.SameLine()
        if psim.Button(
            f"Clear ({len(traj)} point{'s' if n_points > 1 else ''})##trajectory"
        ):
            traj.clear()
            has_points = False

    psim.BeginDisabled(not has_points)
    _, traj.enabled = psim.Checkbox("Enabled##trajectory", traj.enabled)

    # TODO: allow scrubbing along the trajectory (need to trigger all necessary updates)
    psim.BeginDisabled(True)
    _, traj.distance = psim.SliderFloat(
        "Position [m]", traj.distance, 0.0, traj.total_distance()
    )
    psim.EndDisabled()

    _, traj.velocity = psim.SliderFloat("Velocity [m/s]", traj.velocity, 0.1, 10.0)

    _, traj.looping_mode_i = psim.Combo(
        "Loop##trajectory", traj.looping_mode_i, LOOPING_MODE_NAMES
    )
    if traj.looping_mode != LoopingMode.Mirror:
        # Only the mirror mode can play in reverse
        traj.backward = False
    psim.EndDisabled()

    if has_points:
        # Draw a preview of the trajectory
        display_radius = max(0.0003 * scene_scale(gui.scene), 0.3)
        struct = ps.register_curve_network(
            "Trajectory",
            traj.points,
            edges="line",
            enabled=True,
            color=(0.35, 0.15, 0.25),
            transparency=0.7,
        )
        struct.set_radius(display_radius, relative=False)
        struct.set_ignore_slice_plane(DEFAULT_SLICE_PLANE_NAME, True)
    elif ps.has_curve_network("Trajectory"):
        ps.remove_curve_network("Trajectory")


def animation_tick(gui: "SionnaRtGui", time_delta: float, force: bool = False):
    """
    Tick the animation.
    """
    cfg = gui.animation_config
    if not cfg.playing and not force:
        return

    tx_changed = False
    rx_changed = False
    for obj_name, traj in gui.animation_config.trajectories.items():
        if not traj.enabled and not force:
            continue
        if len(traj) == 0:
            continue

        distance_delta = time_delta * cfg.speed_multiplier * traj.velocity
        total_distance = traj.total_distance()
        traj.distance += (-1 if traj.backward else 1) * distance_delta

        if traj.distance <= 0:
            match traj.looping_mode:
                case LoopingMode.NoLoop:
                    traj.distance = 0.0
                case LoopingMode.Mirror:
                    traj.backward = False
                case LoopingMode.Repeat:
                    traj.distance = total_distance
                case _:
                    raise ValueError(f"Invalid looping mode: {traj.looping_mode}")
        elif traj.distance > total_distance:
            match traj.looping_mode:
                case LoopingMode.NoLoop:
                    traj.distance = total_distance
                case LoopingMode.Mirror:
                    traj.backward = True
                case LoopingMode.Repeat:
                    traj.distance = 0.0
                case _:
                    raise ValueError(f"Invalid looping mode: {traj.looping_mode}")

        # Protection in case of large single-frame jumps
        traj.distance = np.clip(traj.distance, 0.0, total_distance)

        # Update the object position accordingly
        obj = gui.scene.get(obj_name)
        pos, direction = traj.current_position_and_direction()
        # Set positions and velocities as lists of floats to avoid DrJit graph growth
        # and ensure the setters are called correctly.
        obj.position = [float(p) for p in pos]
        obj.velocity = [float(v) for v in (direction * traj.velocity)]
        if isinstance(obj, rt.Transmitter):
            tx_changed = True
        elif isinstance(obj, rt.Receiver):
            rx_changed = True

    if tx_changed or rx_changed:
        if tx_changed:
            set_or_update_radio_devices_polyscope(
                gui.scene.transmitters,
                is_transmitter=True,
                gui=gui,
            )
            # Note: receivers don't affect radio maps.
            gui.reset_radio_map()
        if rx_changed:
            set_or_update_radio_devices_polyscope(
                gui.scene.receivers,
                is_transmitter=False,
                gui=gui,
            )

        if gui.cfg.paths.auto_update:
            gui.update_paths(show=True)
