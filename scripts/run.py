#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import argparse
import logging
import os
import subprocess
import sys


def add_project_root_to_path():
    lib_path = os.path.join(os.path.dirname(__file__), "..", "src")
    if lib_path not in sys.path:
        sys.path.append(lib_path)


# --- Compute device selection ------------------------------------------------
#
# Sionna RT selects a CUDA (GPU) Mitsuba variant by default whenever one is
# installed. This fails with an unrecoverable error on GPUs that cannot run
# Dr.Jit's OptiX backend (compute capability < 7.0, e.g. the Maxwell-based
# GeForce MX130). We therefore pick the variant explicitly *before* importing
# Sionna, which would otherwise lock in the CUDA variant.

# Minimum NVIDIA compute capability supported by Dr.Jit's OptiX backend.
_MIN_COMPUTE_CAPABILITY = 7.0


def detect_optix_capable_gpu() -> bool:
    """
    Return True if an NVIDIA GPU able to run Dr.Jit's OptiX backend is present.

    Returns True (optimistic) when the GPU cannot be inspected, so that a
    working GPU setup is never disabled by a failed detection.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return True

    if result.returncode != 0:
        return True

    capabilities = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            capabilities.append(float(line))
        except ValueError:
            # Older nvidia-smi versions may not know `compute_cap`.
            return True

    if not capabilities:
        return True

    return max(capabilities) >= _MIN_COMPUTE_CAPABILITY


def set_mitsuba_variant(device: str) -> str:
    """
    Set and return the Mitsuba variant for the requested ``device``.

    ``device`` is one of "auto", "cpu" or "gpu". With "auto" the GPU is used
    only if it can run the OptiX backend, otherwise the CPU is used.
    """
    import mitsuba as mi  # pylint: disable=import-outside-toplevel

    use_gpu = device == "gpu" or (device == "auto" and detect_optix_capable_gpu())

    if use_gpu:
        try:
            mi.set_variant("cuda_ad_mono_polarized")
            return mi.variant()
        except ImportError:
            if device == "gpu":
                raise
            print(
                "[i] CUDA Mitsuba variant not available, falling back to CPU.",
                file=sys.stderr,
            )

    mi.set_variant("llvm_ad_mono_polarized")
    return mi.variant()


def main():
    add_project_root_to_path()

    parser = argparse.ArgumentParser(
        prog=os.path.basename(__file__), description="Interactive Sionna RT GUI"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to the GUI configuration file to use.",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        choices=("auto", "cpu", "gpu"),
        default=os.environ.get("SIONNA_RT_GUI_DEVICE", "auto"),
        help=(
            "Compute device to use: 'auto' (default) uses the GPU only if it "
            "can run OptiX, 'cpu' forces the CPU, 'gpu' forces the GPU. Can "
            "also be set with the SIONNA_RT_GUI_DEVICE environment variable."
        ),
    )
    parser.add_argument(
        "scene",
        type=str,
        nargs="?",
        default=None,
        help="Path to the Sionna RT scene to load (.xml file or name of a built-in scene).",
    )
    watch_group = parser.add_mutually_exclusive_group()
    watch_group.add_argument(
        "--watch", action="store_true", dest="watch", default=False
    )
    watch_group.add_argument("--no-watch", action="store_false", dest="watch")
    args = parser.parse_args()

    # The Mitsuba variant must be selected before Sionna is imported, as
    # Sionna locks in a CUDA variant on import if none is set.
    variant = set_mitsuba_variant(args.device)

    # Keep the Sionna imports local so that the Mitsuba variant above is
    # selected before Sionna imports and locks it in.
    # pylint: disable=import-outside-toplevel
    from sionna_rt_gui import AppHolder, DEFAULT_CONFIG_PATH
    from sionna_rt_gui.config import load_config

    config_path = args.config if args.config is not None else DEFAULT_CONFIG_PATH

    cfg_overrides = {
        "use_live_reload": args.watch,
    }
    cfg = load_config(config_path, scene_filename=args.scene)

    # Configure logging
    logging.basicConfig(
        level=cfg.log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.info("Using Mitsuba variant: %s", variant)

    # --- Initialization
    app = AppHolder(cfg, scene_filename=args.scene, overrides=cfg_overrides)

    # --- Running loop
    app.show()


if __name__ == "__main__":
    main()
