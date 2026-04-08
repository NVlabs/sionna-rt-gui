#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import argparse
import logging
import os
import sys


def add_project_root_to_path():
    lib_path = os.path.join(os.path.dirname(__file__), "..", "src")
    if lib_path not in sys.path:
        sys.path.append(lib_path)


def main():
    # Ensure the src directory is in sys.path so we can import sionna_rt_gui
    add_project_root_to_path()

    # Hardcode/compute default config path to avoid importing whole package before parsing args.
    # This keeps --help fast and prevents heavy dependencies (Mitsuba, Polyscope) from loading prematurely.
    default_config_path = os.path.join(
        os.path.dirname(__file__), "..", "src", "sionna_rt_gui", "data", "configs", "sionna_rt_gui", "base.yaml"
    )

    parser = argparse.ArgumentParser(
        prog=os.path.basename(__file__), description="Interactive Sionna RT GUI"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=default_config_path,
        help="Path to the GUI configuration file to use.",
    )
    parser.add_argument(
        "scene",
        type=str,
        nargs="?",
        default=None,
        help="Path to the Sionna RT scene to load (.xml file or name of a built-in scene).",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Path to the scenario YAML file to use.",
    )
    watch_group = parser.add_mutually_exclusive_group()
    watch_group.add_argument(
        "--watch", action="store_true", dest="watch", default=False
    )
    watch_group.add_argument("--no-watch", action="store_false", dest="watch")
    args = parser.parse_args()

    # Heavy imports happen here, ONLY IF arguments were successfully parsed.
    from sionna_rt_gui import AppHolder
    from sionna_rt_gui.config import load_config

    cfg_overrides = {
        "use_live_reload": args.watch,
    }
    cfg = load_config(
        args.config, scene_filename=args.scene, scenario_filename=args.scenario
    )

    # Configure logging
    logging.basicConfig(
        level=cfg.log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # --- Initialization
    app = AppHolder(
        cfg,
        scene_filename=args.scene,
        scenario_filename=args.scenario,
        overrides=cfg_overrides,
    )

    # --- Running loop
    app.show()


if __name__ == "__main__":
    main()
