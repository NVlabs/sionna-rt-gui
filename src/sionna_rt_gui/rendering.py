#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

from typing import Any

import drjit as dr
import mitsuba as mi
import numpy as np
import polyscope as ps
import polyscope_bindings as psb
from sionna import rt
from sionna.rt.renderer import visual_scene_from_wireless_scene

from .config import RenderingConfig, RenderingMode
from .ps_utils import supports_direct_update_from_device


def setup_scene_for_rendering(
    cfg: RenderingConfig,
    scene: rt.Scene,
    use_denoiser: bool = False,
) -> dict[str, Any]:
    from mitsuba.python.util import _RenderOp

    assert (
        "_rgb" in mi.variant()
    ), "This function is supposed to be called with an RGB variant."

    render_res = cfg.rendering_resolution
    fov_y_deg = ps.get_view_camera_parameters().get_fov_vertical_deg()
    sensor = mi.load_dict(
        {
            "type": "perspective",
            "fov": fov_y_deg,
            "fov_axis": "y",
            "near_clip": 1.0,
            "far_clip": max(10000, 10 * rt.utils.render.scene_scale(scene)),
            "film": {
                "type": "hdrfilm",
                "pixel_format": "rgb",
                "width": render_res[0],
                "height": render_res[1],
                "filter": {
                    "type": "box",
                },
            },
        }
    )

    # NOTE: this assumes that the scene does *not* change between renders!
    visual_scene_dict = visual_scene_from_wireless_scene(
        scene,
        sensor=sensor,
        max_depth=8,
        envmap=cfg.envmap,
        #  clip_at: float | None = None,
        #  clip_plane_orientation: tuple[float, float, float] = (0, 0, -1),
        #  envmap: str | None = None,
        lighting_scale=cfg.envmap_factor,
        #  exclude_mesh_ids: set[str] = None
    )
    # Envmap rotation
    envmap_base_transform = None
    if "emitter" in visual_scene_dict:
        emitter = visual_scene_dict["emitter"]
        envmap_base_transform = emitter.get("to_world", mi.ScalarTransform4f())
        emitter["to_world"] = (
            mi.ScalarTransform4f.rotate(axis=(0, 0, 1), angle=cfg.envmap_rotation_deg)
            @ envmap_base_transform
        )

    visual_scene = mi.load_dict(visual_scene_dict)

    # Hack for better sampling: zero-out the bottom half or so of the envmap,
    # since our scenes always have a ground plane.
    params = mi.traverse(visual_scene)
    k = "emitter.data"
    if k in params:
        # Height, width, channels
        h, *_ = dr.shape(params[k])
        # Reduce the hack: keep more of the horizon but still dim the bottom
        params[k][int(0.8 * h) :, :, :] = 0

    integrator = {
        "type": "path",
        "hide_emitters": True,
        "max_depth": 5,
    }
    if use_denoiser:
        integrator = {
            "type": "aov",
            "aovs": "albedo:albedo,normals:sh_normal",
            "nested": integrator,
        }
    integrator = mi.load_dict(integrator)

    depth_integrator = mi.load_dict({"type": "depth"})

    # Helper to convert between radial and perpendicular-style depth buffers.
    # Camera parameters from the sensor
    fy = render_res[1] / (2.0 * np.tan(np.radians(fov_y_deg) / 2.0))
    fx = fy  # Assume square pixels
    cx = 0.5 * render_res[0]
    cy = 0.5 * render_res[1]
    # Create pixel coordinate grids
    i_coords, j_coords = dr.meshgrid(
        dr.arange(mi.Float, render_res[0]),
        dr.arange(mi.Float, render_res[1]),
        indexing="xy",
    )
    x_norm = (i_coords - cx) / fx
    y_norm = (j_coords - cy) / fy
    # Calculate radial distance factor
    radial_factor = mi.TensorXf(
        dr.sqrt(1.0 + dr.square(x_norm) + dr.square(y_norm)),
        shape=(render_res[1], render_res[0]),
    )

    return {
        "sensor": sensor,
        "visual_scene": visual_scene,
        "integrator": integrator,
        "depth_integrator": depth_integrator,
        "render_op": _RenderOp(),
        "radial_factor": radial_factor,
        "envmap_base_transform": envmap_base_transform,
    }


def _render_scene(
    cfg: RenderingConfig,
    seed: int,
    camera_changed: bool,
    cache: dict[str, Any],
    use_denoiser: bool = False,
) -> tuple[mi.TensorXf, list[mi.TensorXf], dict[str, Any]]:
    """
    Note that even though this function does RGB rendering, we do *not*
    need to switch variant, because all the relevant objects were already
    created using the RGB variant above.
    """

    sensor = cache["sensor"]
    visual_scene = cache["visual_scene"]
    integrator = cache["integrator"]
    depth_integrator = cache["depth_integrator"]
    render_op = cache["render_op"]
    radial_factor = cache["radial_factor"]

    if camera_changed:
        # Camera to world transform
        view_pose = ps.get_camera_view_matrix()

        ps_to_world = np.linalg.inv(view_pose)
        del view_pose
        view_rotation = np.eye(4)
        view_rotation[:3, :3] = ps_to_world[:3, :3]
        view_translation = np.eye(4)
        view_translation[:3, -1] = ps_to_world[:3, -1]

        # Conversions:
        # - World space stays the same (Z up, Y forward), so the translation is left intact.
        # - However, the local camera space needs to be converted from Polyscope's
        #   left-handed (+y up, -z forward, +x right) to Mitsuba's right-handed
        #   system (y up, -z forward, +x left).
        # TODO: do this in a single go for efficiency.
        to_world = (
            mi.ScalarTransform4f(view_translation)
            @ mi.ScalarTransform4f(
                [
                    [-1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1],
                ]
            )
            @ mi.ScalarTransform4f(view_rotation)
            @ mi.ScalarTransform4f(
                [
                    [1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )
        )

        params = mi.traverse(sensor)
        params["to_world"] = to_world
        params.update()

    # Note: we can't directly use `mi.render()` because of a couple of asserts on the
    # types of integrator, sensor, etc.
    # TODO: use kernel freezing, if it helps
    img = render_op.eval(
        scene=visual_scene,
        sensor=sensor,
        _=None,
        params=None,
        integrator=integrator,
        seed=(seed, None),
        spp=(cfg.spp_per_frame, None),
    )
    rgb = img[..., :3]

    # Depth: render separately because we really want 1spp. Otherwise, depth values
    # get averaged with 0.f at the edge of objects, which is problematic for compositing.
    perpendicular_depth = render_op.eval(
        scene=visual_scene,
        sensor=sensor,
        _=None,
        params=None,
        integrator=depth_integrator,
        seed=(seed, None),
        spp=(1, None),
    )
    # Polyscope: "Depth values should be radial ray distance from the camera origin,
    # not perpendicular distance from the image plane."
    # Convert from perpendicular distance to radial ray distance
    depth = perpendicular_depth[:, :, 0] / radial_factor
    depth = dr.select(depth == 0, dr.inf, depth)

    # Add a tiny depth bias (0.1%) to push the rendered background slightly
    # further away, ensuring Polyscope 3D structures (dots, arrows, paths)
    # placed on surfaces remain visible and don't get occluded by Z-fighting.
    depth = depth * 1.001

    aovs = [depth]
    if use_denoiser:
        # Albedo
        aovs.append(img[..., 3:6])
        # Normals
        aovs.append(img[..., 6:9])

    return rgb, aovs, cache


def render_scene(
    cfg: RenderingConfig,
    scene: rt.Scene,
    seed: int,
    camera_changed: bool,
    cache: dict[str, Any] = None,
    use_denoiser: bool = False,
) -> tuple[mi.TensorXf, mi.TensorXf, dict[str, Any]]:
    assert cfg.mode == RenderingMode.RAY_TRACING

    if cache is None:
        with mi.scoped_set_variant("cuda_ad_rgb", "llvm_ad_rgb"):
            cache = setup_scene_for_rendering(cfg, scene, use_denoiser=use_denoiser)
        camera_changed = True

    return _render_scene(
        cfg, seed, camera_changed, cache=cache, use_denoiser=use_denoiser
    )


def set_envmap_rotation(
    cache: dict[str, Any],
    angle_deg: float,
    axis: tuple[float, float, float] = (0, 0, 1),
) -> bool:
    if cache is None or "visual_scene" not in cache:
        return False

    props = mi.traverse(cache["visual_scene"])
    if "emitter.to_world" not in props:
        return False

    props["emitter.to_world"] = (
        mi.ScalarTransform4f.rotate(axis=axis, angle=angle_deg)
        @ cache["envmap_base_transform"]
    )
    props.update()
    return True


def set_envmap_intensity(
    cache: dict[str, Any],
    intensity: float,
) -> bool:
    if cache is None or "visual_scene" not in cache:
        return False

    props = mi.traverse(cache["visual_scene"])
    if "emitter.scale" not in props:
        return False

    props["emitter.scale"] = intensity
    props.update()
    return True


def add_or_update_ray_traced_image_quantity(
    ray_traced_img: mi.TensorXf, ray_traced_depth: mi.TensorXf
) -> bool:
    exists, _ = psb.get_global_floating_quantity_structure().has_quantity_buffer_type(
        "ray_traced_img", "colors"
    )

    # TODO: we shouldn't need to add an alpha channel that we don't use :(
    ray_traced_img = dr.concat(
        [ray_traced_img, dr.ones_like(ray_traced_img[..., :1])], axis=-1
    )

    if supports_direct_update_from_device() and exists:
        ps.get_quantity_buffer("ray_traced_img", "colors").update_data_from_device(
            ray_traced_img
        )
        ps.get_quantity_buffer("ray_traced_img", "depths").update_data_from_device(
            ray_traced_depth
        )
    else:
        ps.add_raw_color_alpha_render_image_quantity(
            "ray_traced_img",
            depth_values=ray_traced_depth.numpy(),
            color_values=ray_traced_img.numpy(),
            enabled=True,
            image_origin="upper_left",
        )

    ps.request_redraw()
    return exists
