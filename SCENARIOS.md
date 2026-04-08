Scenario Configuration and Loading
================================

This document describes how to configure and load radio scenarios (transmitters, road segments, UEs) in `sionna-rt-gui`.

Introduction
------------

The GUI supports loading scenario definitions (transmitters, road segments, UEs) primarily from **Jupyter Notebooks**. This allows you to define your simulation parameters directly in your research code and visualize them immediately.

These sources allow you to define:
-   **Sites**: Fixed transmitter locations with specific azimuths, downtilts, and power levels.
-   **Road Segments**: Linear segments where receivers (UEs) can be randomly spawned and animated.
-   **Antenna Arrays**: Configuration for both transmitter and receiver antenna arrays.
-   **Carrier Frequency**: The frequency used for the radio simulation.

Scenario Loading via Jupyter Notebooks
--------------------------------------

The GUI can automatically gather all needed information directly from a Jupyter notebook. This is the recommended way to visualize a scenario without manual configuration.

Simply pass the notebook file as the positional argument (see [example_scenario.ipynb](src/sionna_rt_gui/data/scenes/example_scenario.ipynb) for a template):

```bash
python ./scripts/run.py example_scenario.ipynb
```

The GUI will automatically:
1.  **Extract the scene path** from the `load_scene()` call in the notebook.
2.  **Gather scenario parameters** by searching for assignments to `sites`, `road_segments`, `num_ue`, `carrier_frequency`, `tx_array`, and `rx_array` in the notebook's code cells.

### Extracted Variables

The following variables are recognized by the GUI's notebook parser:

- **`sites`**: A list of dictionaries, each containing:
    - `name`: String identifier.
    - `position`: `[x, y, z]` list.
    - `azimuths`: List of azimuth angles (degrees) for sectors.
    - `downtilt`: Downtilt angle (degrees).
    - `power_dbm`: Transmit power in dBm.
- **`road_segments`**: A list of segments, where each segment is a pair of points `[[start_x, start_y, start_z], [end_x, end_y, end_z]]`.
- **`num_ue`**: Integer number of receivers to spawn along the road segments.
- **`carrier_frequency`**: Float frequency in Hz.
- **`tx_array` / `rx_array`**: Dictionaries or `PlanarArray` calls defining the antenna configuration.

Example Notebook Structure
--------------------------

```python
from sionna.rt import load_scene, PlanarArray

# MANDATORY: The GUI finds the XML scene via this call
scene = load_scene("path/to/scene.xml")

# OPTIONAL: Scenario parameters
carrier_frequency = 3.5e9
num_ue = 10

tx_array = PlanarArray(num_rows=8, num_cols=4, vertical_spacing=0.5, horizontal_spacing=0.5)

sites = [
    {
        "name": "site_A",
        "position": [100, 200, 30],
        "azimuths": [0, 120, 240],
        "downtilt": -6.0,
        "power_dbm": 30
    }
]

road_segments = [
    ([0, 0, 1.5], [500, 500, 1.5])
]
```

Advanced Configuration
----------------------

While notebooks are the easiest way to load scenarios, you can still use YAML configuration files for the GUI's general settings (rendering quality, ray depth, etc.) using the `--config` flag.

```bash
python ./scripts/run.py --config path/to/config.yaml path/to/notebook.ipynb
```
