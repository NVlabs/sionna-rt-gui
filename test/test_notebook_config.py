#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import os
import json
import pytest
from sionna_rt_gui.config import extract_scenario_from_notebook, load_config, CONFIGS_DIR

def test_extract_scenario_from_notebook(tmp_path):
    # Create a mock notebook
    nb_content = {
        "cells": [
            {
                "cell_type": "code",
                "source": [
                    "from sionna.rt import load_scene, PlanarArray\n",
                    "scene = load_scene('dummy_scene.xml')\n",
                    "sites = [{'name': 'site_1', 'position': [10.0, 20.0, 30.0], 'azimuths': [0.0], 'downtilt': -5.0, 'power_dbm': 40.0}]\n",
                    "num_ue = 25\n",
                    "road_segments = [([0, 0, 1.5], [10, 10, 1.5])]\n",
                    "tx_array = PlanarArray(num_rows=4, num_cols=2, vertical_spacing=0.5, horizontal_spacing=0.5)\n"
                ]
            }
        ]
    }
    nb_path = tmp_path / "test_nb.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb_content, f)
    
    # Run extraction
    result = extract_scenario_from_notebook(str(nb_path))
    
    # Assert extracted values
    assert result["scene_filename"] == "dummy_scene.xml"
    assert len(result["sites"]) == 1
    assert result["sites"][0]["name"] == "site_1"
    assert result["num_ue"] == 25
    assert len(result["road_segments"]) == 1
    assert result["tx_array"]["num_rows"] == 4
    assert result["tx_array"]["num_cols"] == 2

def test_load_config_with_notebook(tmp_path):
    # Create a mock notebook
    nb_content = {
        "cells": [
            {
                "cell_type": "code",
                "source": [
                    "scene = load_scene('dummy_scene.xml')\n",
                    "num_ue = 50\n"
                ]
            }
        ]
    }
    nb_path = tmp_path / "test_nb_config.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb_content, f)
    
    # Use the default base.yaml for loading
    base_config = os.path.join(CONFIGS_DIR, "base.yaml")
    
    # Run load_config with notebook as scene_filename
    cfg = load_config(base_config, scene_filename=str(nb_path))
    
    # Assert configuration properties
    assert cfg.scene_filename == "dummy_scene.xml"
    assert cfg.scenario.num_ue == 50
    assert cfg.create_example_scenario == True
    assert cfg.scenario_filename == str(nb_path)

def test_extract_planar_array_positional(tmp_path):
    # Test that PlanarArray with positional arguments is also handled if possible
    # (The current implementation in config.py uses keyword arguments for PlanarArray extraction)
    # Let's verify keyword arguments first as they are most common.
    nb_content = {
        "cells": [
            {
                "cell_type": "code",
                "source": [
                    "tx_array = PlanarArray(8, 4, 0.6, 0.6)\n"
                ]
            }
        ]
    }
    nb_path = tmp_path / "test_nb_positional.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb_content, f)
    
    result = extract_scenario_from_notebook(str(nb_path))
    
    # The current implementation handles positional arguments for PlanarArray
    assert result["tx_array"]["num_rows"] == 8
    assert result["tx_array"]["num_cols"] == 4
    assert result["tx_array"]["vertical_spacing"] == 0.6
    assert result["tx_array"]["horizontal_spacing"] == 0.6

def test_set_envmap_intensity_no_cache():
    from sionna_rt_gui.rendering import set_envmap_intensity
    # Test that it returns False when cache is None (standard safety check)
    assert set_envmap_intensity(None, 2.0) == False
    assert set_envmap_intensity({}, 2.0) == False
