# Run a Rivelero survey batch in Unreal

Start Unreal with the project's Project AirSim build, then run this from
`src/Camera` using the existing Project AirSim Python environment with
`pyproj` installed.

```powershell
python hello_camera.py -number 3 -directory "C:\surveys\experiment_2"
```

Export the **Viewpoints table** for each survey. Put only those CSVs directly
in the input directory and use unique filenames such as `viewpoints_1.csv`,
`viewpoints_2.csv`, and `viewpoints_3.csv`. The count must equal `-number`.
Files are sorted by filename with numeric parts sorted numerically. Cameras
are processed in the original CSV row order.

| Export column | Batch interpretation |
| --- | --- |
| `viewpoint_id` | Required unique camera name, retained in telemetry. Unreal actors are named `Camera1`, `Camera2`, etc. |
| `x`, `y`, `crs` | Required easting and northing in a projected CRS in metres. Use the same map frame as the Unreal terrain. |
| `heading_deg` | Required heading clockwise from grid north. Unreal's local north axis must align with grid north. |
| `z` | Optional absolute camera elevation in metres, in the same vertical reference as the configured origin. |
| `roll_deg`, `pitch_deg` | Optional rotation angles, defaulting to zero. |
| `horizontal_fov_deg` | Optional horizontal field of view, defaulting to 120°. Simulator cameras require a value between 0° and 180°, exclusive. |

Set `--origin EAST NORTH UP` to the map coordinates of Unreal's local origin.
It defaults to `0 0 0` for an aligned, already centred terrain. The script
converts map coordinates to AirSim north, east, down. With a blank `z`,
`--default-z-m` specifies a fixed AirSim down coordinate and defaults to
`-4`, matching the example camera setup. A fixed altitude does not follow
terrain. Supply camera elevations in `z` for terrain-relative placement.

The runner checks all input files before connecting. It reloads the scene
before every survey, creates the new cameras, and reapplies plant segmentation
from `src/Camera/flightpath.csv` when present. It writes
`telemetry/survey_1.csv`, `telemetry/survey_2.csv`, and so on, under the input
directory. Each file contains the source filename, camera order, configured
pose and field of view, plus visible pixels, X-ray pixels, occlusion fraction,
and distance per listed plant. With no plant list it writes one row per
camera with blank plant fields. Existing outputs are protected from overwrites.

Running `python hello_camera.py` without the batch flags opens the original
interactive viewer.
