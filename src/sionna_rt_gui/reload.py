#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

from collections.abc import Sequence
import gc
import importlib
import importlib.util
import io
import os
from os.path import realpath, basename, splitext, isfile, isdir, join
import sys
import time
import traceback
from types import ModuleType
from typing import Any

import polyscope as ps

# Note: we don't import anything project-related at the top level to avoid
# holding onto outdated module references after a code reload.


class AppHolder:
    """
    Helper class that allows reloading configs, snapshots and code after starting.
    """

    def __init__(
        self,
        cfg,
        scene_filename: str | None,
        scenario_filename: str | None = None,
        overrides: dict[str, Any] | None = None,
    ):
        import sionna_rt_gui

        self.module_watcher: ModuleWatcher = ModuleWatcher(
            module=sionna_rt_gui, module_path=sionna_rt_gui.SOURCE_DIR
        )
        self.config_watcher: FilesWatcher | None = None
        self.scene_filename: str | None = scene_filename
        self.scenario_filename: str | None = scenario_filename
        self.overrides: dict[str, Any] = overrides or {}
        self.app = None

        self.app_failed: bool = False
        self.last_valid_config: Any | None = None

        self.apply_overrides(cfg)
        self.create_app(cfg)

        ps.set_user_callback(self.tick)

    def create_app(self, cfg) -> None:
        MainApp = self.module_watcher.get("gui.SionnaRtGui")
        drjit_cleanup = self.module_watcher.get("drjit_util.drjit_cleanup")

        self.app = None
        self.app_failed = False
        drjit_cleanup()

        watch_paths = [cfg.config_path]
        if self.scene_filename and os.path.isfile(self.scene_filename):
            watch_paths.append(self.scene_filename)
        if self.scenario_filename and os.path.isfile(self.scenario_filename):
            watch_paths.append(self.scenario_filename)

        self.config_watcher = FilesWatcher(tuple(watch_paths))
        self.app = MainApp(cfg)
        self.last_valid_config = self.app.cfg

    def maybe_reload(self):
        if self.app is None:
            # Maybe the app failed during init, so we can no longer check if reloads
            # were requested from the app. But at least we can keep detecting code
            # or config changes.
            cfg = self.last_valid_config
            snapshot_load_requested = False
            code_reload_requested = False
        else:
            cfg = self.app.cfg
            snapshot_load_requested = self.app.snapshot_load_requested
            code_reload_requested = self.app.code_reload_requested

        if not (
            cfg.use_live_reload or snapshot_load_requested or code_reload_requested
        ):
            return

        old_config_path: str = cfg.config_path
        new_config_path: str | None = None

        if self.config_watcher.change_detected():
            # --- Configuration change detected
            # Note: we don't want to reload snapshots that we just wrote to, e.g.:
            # 1. Load snapshot `base.unerf`
            # 2. Train for a while
            # 3. Overwrite snapshot `base.unerf`
            # We don't want the file update in step (3) to trigger an automatic reload.
            new_config_path = old_config_path
            print(f"[i] Detected config change, reloading: {new_config_path}")

        elif code_reload_requested or self.module_watcher.change_detected():
            # --- Code change detected
            # If this reload fails, the same code change won't be detected again at the
            # next tick. That's a good thing, since we don't want to try reloading an
            # invalid source / module at every tick.
            if self.app is not None:
                self.app.code_reload_requested = False
            print("[i] Detected code change, reloading module and config")
            if self.module_watcher.reload():
                # Reload the current config from scratch, since the code
                # interpreting configs may have changed.
                new_config_path = old_config_path

        if new_config_path is not None:
            load_fn = self.module_watcher.get("config.load_config")

            try:
                new_config = load_fn(
                    new_config_path,
                    scene_filename=self.scene_filename,
                    scenario_filename=self.scenario_filename,
                )
            except Exception as e:
                # We don't want to keep trying to load an invalid config
                if self.app is not None:
                    self.app.snapshot_load_requested = False
                new_config = None

                print(
                    f"[!] Failed to load snapshot from path: {new_config_path}\n{e}",
                    file=sys.stderr,
                )
                print(traceback.format_exc(), file=sys.stderr)

            if new_config is not None:
                self.apply_overrides(new_config)
                try:
                    self.create_app(new_config)
                except Exception as e:
                    print(
                        f"[!] Failed to instantiate using updated code or config: {new_config_path}\n{e}",
                        file=sys.stderr,
                    )
                    print(traceback.format_exc(), file=sys.stderr)
                gc.collect()

    def tick(self) -> None:
        self.maybe_reload()
        if (self.app is not None) and not self.app_failed:
            try:
                self.app.tick()
            except Exception as e:
                self.app_failed = True
                print(f"[!] Exception thrown in app.tick(): {e}", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)

    def show(self):
        ps.show()

    def apply_overrides(self, cfg):
        def apply(full_key: str, value, cfg):
            # Follow the dotpath, if any
            parts = full_key.split(".")
            for sub_key in parts[:-1]:
                # Just throw an exception if the path is invalid.
                cfg = getattr(cfg, sub_key)
            key = parts[-1]

            if not hasattr(cfg, key):
                raise ValueError(
                    f"Cannot apply override {full_key}={value} because config object of type {type(cfg)} has no attribute {key}."
                )

            setattr(cfg, key, value)

        # TODO: use OmegaConf merge mechanism? Support nested overrides with dotpaths.
        for k, v in self.overrides.items():
            apply(k, v, cfg)


class FilesWatcher:
    def __init__(
        self,
        watch_paths: Sequence[str],
        watch_period: float = 1.0,
        check_file_contents: bool = False,
    ):
        self.check_file_contents: bool = check_file_contents
        self.watch_paths = [realpath(p) for p in watch_paths]
        # How often to check for changed files (in seconds)
        self.watch_check_period: float = watch_period

        self.files_metadata: dict = self.collect_changed_metadata({})
        self.watch_time_last_checked: float = 0.0

    def collect_changed_metadata(self, current: dict) -> dict:
        new_metadata = {}

        def collect(fname):
            if fname in current:
                last_mtime, last_contents = current[fname]
            else:
                last_mtime, last_contents = 0, None

            mtime = os.stat(fname).st_mtime
            if last_mtime >= mtime:
                return

            if self.check_file_contents:
                contents = io.open(fname, mode="r", encoding="utf-8").read()
                if contents == last_contents:
                    return
            else:
                contents = None

            new_metadata[fname] = (mtime, contents)

        # Collect any potential metadata update from the watched files & directories
        for path in self.watch_paths:
            if isfile(path):
                collect(path)
            elif isdir(path):
                for dirpath, _, filenames in os.walk(path):
                    for fname in filenames:
                        if not fname.endswith(".py"):
                            continue
                        collect(join(dirpath, fname))

        return new_metadata

    def change_detected(self) -> bool:
        """
        If:
        - Enough time has elapsed since the last check,
        - And a watched file has been modified,
        then reload the requested module.

        Returns true if a reload was attempted and succeeded.
        """
        elapsed = time.time() - self.watch_time_last_checked
        if elapsed < self.watch_check_period:
            return False
        self.watch_time_last_checked = time.time()

        new_metadata = self.collect_changed_metadata(self.files_metadata)
        if not new_metadata:
            return False

        self.files_metadata.update(new_metadata)
        return True


class ModuleWatcher(FilesWatcher):
    def __init__(self, module: ModuleType, module_path: str, watch_period: float = 1.0):
        assert isdir(module_path)
        super().__init__(watch_paths=(module_path,), watch_period=watch_period)

        self.module_path: str = realpath(module_path)
        self.last_reload_failed: bool = False

        self.module: ModuleType = module

    def reload(self):
        module_name = splitext(basename(self.module_path))[0]
        if self.module is None:
            spec = importlib.util.spec_from_file_location(module_name, self.module_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            self.module = module
        else:
            try:
                reload_module_recursive(self.module, allowed_root=self.module_path)
            except Exception as e:
                self.last_reload_failed = True
                print(
                    f"\n[!] Failed to reload module! {type(e)}:\n    {e}\n",
                    file=sys.stderr,
                )
                print(traceback.format_exc(), file=sys.stderr)
                return False

        self.last_reload_failed = False
        return True

    def get(self, path: str):
        """
        Get a class, object, submodule, etc from ``self.module``.
        """
        result = self.module
        parent_path = result.__name__
        for name in path.split("."):
            submodule_path = f"{parent_path}.{name}"
            if hasattr(result, name):
                result = getattr(result, name)
            else:
                # Assume it's a submodule and load it
                spec = importlib.util.find_spec(submodule_path, package=parent_path)
                if not spec:
                    raise ImportError(
                        f"ModuleWatcher.get(): could not import symbol '{path}' from module: {self.module}"
                    )
                result = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(result)
            parent_path = submodule_path

        return result


def reload_module_recursive(module, allowed_root, seen=None, disallow=None):
    """Recursively reload modules.
    Adapted from: https://stackoverflow.com/a/17194836
    """
    from types import ModuleType

    importlib.reload(module)

    if seen is None:
        seen = set((module.__name__,))
    if disallow is None:
        disallow = set(
            (
                "importlib",
                "_bootstrap",
                "numpy",
            )
        ).union(sys.builtin_module_names)

    for attribute_name in dir(module):
        if attribute_name in seen:
            continue
        attribute = getattr(module, attribute_name)
        if callable(attribute):
            attribute = sys.modules[attribute.__module__]

        if isinstance(attribute, ModuleType):
            module_name = attribute.__name__

            if (
                (module_name in disallow)
                or ("_" + module_name in disallow)
                or (module_name in seen)
            ):
                continue
            if not hasattr(attribute, "__file__"):
                continue
            if not realpath(attribute.__file__).startswith(allowed_root):
                continue

            seen.add(module_name)
            reload_module_recursive(
                attribute, allowed_root, seen=seen, disallow=disallow
            )

    importlib.reload(module)
