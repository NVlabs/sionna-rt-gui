#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

from enum import Enum

import logging
import drjit as dr
import mitsuba as mi
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from sionna import rt
from sionna.rt.utils.geometry import rotation_matrix

from .animation import trajectory_gui
from .config import DEFAULT_SLICE_PLANE_NAME
from .sionna_utils import set_or_update_radio_devices_polyscope


logger = logging.getLogger(__name__)


class SelectionType(Enum):
    Transmitter = "Transmitter"
    Receiver = "Receiver"
    RadioMap = "Radio map"
    Path = "Path"
    Mesh = "Mesh"


GIZMO_SCALE = 40


def vec_str(vec: np.ndarray) -> str:
    vec = vec.squeeze()
    return f"({vec[0]:.2f}, {vec[1]:.2f}, {vec[2]:.2f})"


def selection_gui(
    gui: "SionnaRtGui",
    selected_object: rt.SceneObject | None,
    selected_type: SelectionType | None,
):
    if selected_object is None:
        return

    # Place window in the top-right corner of the screen
    window_resolution = ps.get_window_size()
    w, h = 375, 480
    psim.SetNextWindowSize(
        (w * gui.ui_scale, h * gui.ui_scale), psim.ImGuiCond_FirstUseEver
    )
    # Top right corner
    window_pos = (
        window_resolution[0] - (w + 10) * gui.ui_scale,
        10 * gui.ui_scale,
    )
    psim.SetNextWindowPos(window_pos, psim.ImGuiCond_FirstUseEver)

    _, keep_selection = psim.Begin("Selection##sionna", open=True)

    if not keep_selection:
        psim.End()
        gui.clear_selection()
        return

    rd_update_needed = False
    is_transmitter = selected_type == SelectionType.Transmitter
    if selected_type in (SelectionType.Transmitter, SelectionType.Receiver):
        rd = selected_object
        array = gui.scene.tx_array if is_transmitter else gui.scene.rx_array
        pattern = array.antenna_pattern

        changed, rd.color = psim.ColorEdit3(
            f"{selected_type.value} '{selected_object.name}'\n",
            rd.color,
            psim.ImGuiColorEditFlags_NoInputs,
        )
        rd_update_needed |= changed

        psim.SameLine()
        # "Remove" button (right-aligned)
        bw = 80
        psim.SetCursorPosX(
            psim.GetCursorPosX() + psim.GetContentRegionAvail()[0] - bw - 5
        )
        pressed_del = not psim.IsAnyItemActive() and psim.IsKeyPressed(
            psim.ImGuiKey_Delete, repeat=False
        )
        if psim.Button("Remove", (bw, 0)) or pressed_del:
            gui.remove_object(selected_object, selected_type)
            rd_update_needed = True
            gui.clear_selection()

        psim.NewLine()

        # TODO: do not trigger constant GPU -> CPU transfers
        if psim.TreeNodeEx(
            "Characteristics:##selection", psim.ImGuiTreeNodeFlags_DefaultOpen
        ):
            psim.Text(
                f"Position [m]: {vec_str(rd.position.numpy())}\n"
                f"Orientation (angles): {vec_str(rd.orientation.numpy())}\n"
                f"Velocity [m/s]: {vec_str(rd.velocity.numpy())}\n"
                + (f"Transmit power [W]: {rd.power[0]:.2f}\n" if is_transmitter else "")
            )
            psim.TreePop()

        psim.Spacing()
        if psim.TreeNodeEx(
            "Antenna array:##selection", psim.ImGuiTreeNodeFlags_DefaultOpen
        ):
            psim.Text(
                f"Type: {type(array).__name__}\n"
                f"Array size: {dr.width(array.normalized_positions)}\n"
                f"Pattern: {type(pattern).__name__}\n"
            )

            psim.TreePop()

        psim.Spacing()
        if psim.TreeNodeEx(
            "Animation:##selection", psim.ImGuiTreeNodeFlags_DefaultOpen
        ):
            trajectory_gui(gui, selected_object)
            psim.TreePop()

        # --- Transformation gizmo
        # Check that the selection wasn't cleared before updating the gizmo.
        if gui.selected_object is not None:
            # TODO: make RD's orientation visible while the gizmo is shown.
            if not ps.has_point_cloud("Gizmo"):
                struct = ps.register_point_cloud(
                    "Gizmo", rd.position.numpy().T, enabled=False
                )
                gizmo = struct.get_transformation_gizmo()
                gizmo.set_enabled(True)
                gizmo.set_allow_scaling(False)
                struct.set_ignore_slice_plane(DEFAULT_SLICE_PLANE_NAME, True)
            else:
                struct = ps.get_point_cloud("Gizmo")

            to_world = struct.get_transform()

            # Gizmo moved since the last frame
            if (gui.prev_gizmo_to_world is not None) and not np.allclose(
                gui.prev_gizmo_to_world, to_world
            ):
                # Apply transform to the selected object
                rd.position = mi.Point3f(to_world[:3, -1])
                target = to_world[:3, -1] + (
                    to_world[:3, 0] / np.linalg.norm(to_world[:3, 0])
                )
                rd.look_at(target)
                # For debugging:
                # ps.register_point_cloud("Gizmo target", target[None, :], color=(1, 0, 1))
                dr.make_opaque(rd.position, rd.orientation)
                rd_update_needed = True

            else:
                # Reset position & orientation of gizmo to match the selected object
                to_world = np.eye(4)
                to_world[:3, :3] = rotation_matrix(rd.orientation).numpy()[..., 0]
                to_world[:3, -1] = rd.position.numpy().squeeze()
                struct.set_transform(to_world)

            gui.prev_gizmo_to_world = to_world

    elif selected_type == SelectionType.Mesh:
        obj = selected_object
        psim.Text(f"Mesh: '{obj.name}'")
        if hasattr(obj, "radio_material"):
            mat = obj.radio_material
            if isinstance(mat, str):
                mat = gui.scene.radio_materials.get(mat, mat)

            mat_name = getattr(mat, "name", str(mat))
            psim.Text(f"Material: {mat_name}")

            if hasattr(mat, "color") and not isinstance(mat, str):
                # Convert to list if it's a drjit array or similar
                color = list(mat.color)

                changed, new_color = psim.ColorEdit3(
                    f"Color##mat_color_{obj.name}",
                    color,
                    psim.ImGuiColorEditFlags_NoInputs,
                )
                if changed:
                    gui.update_material_color(mat, new_color)

    psim.Spacing()

    if rd_update_needed:
        # TODO: auto-pause animation if animated object was moved?

        set_or_update_radio_devices_polyscope(
            gui.scene.transmitters if is_transmitter else gui.scene.receivers,
            is_transmitter=is_transmitter,
            gui=gui,
        )
        if is_transmitter:
            # Note: receivers don't affect radio maps.
            gui.reset_radio_map()

        if gui.cfg.paths.auto_update:
            gui.update_paths(clear_first=True, show=True)

    psim.End()
