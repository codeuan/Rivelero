# Fixed-camera visibility validation

Run the Python stages in Rivelero's environment (editable installation with GDAL).
Use a north-up metre DEM with elevations in metres and no missing cells. The
full raster is scored. On the intended 400 × 400 raster at 5 m, the 10 × 10
candidate grid occupies the central 1 × 1 km. All bearings point towards the
landscape centre. Comparisons use a flat-earth model in both engines.

1. Prepare a fresh batch. Select the pilot range explicitly (150, 250 or 400 m).

   ```shell
   python -m rivelero.experiments.visibility_validation prepare terrain.tif pilot150 --distance 150 --cameras 20 --replicates 100 --seed 42
   ```

   The command exports positions and compass bearings in `viewpoints.csv`,
   selections in `layouts.csv`, a shared elevation grid, and full-raster
   Rivelero masks in `predicted/*.npy`. Random draws sample without replacement.
   Clustered draws use the nearest neighbours of a random seed viewpoint.
   Dispersed draws start randomly and repeatedly select a candidate farthest
   from its nearest selected neighbour. Layouts may recur between replicates.

2. Open the matching DEM landscape in the UE5 Editor with the Python Editor
   Script Plugin enabled. Load the whole terrain. Align world +X to north and
   +Y to east; use centimetres and the same vertical datum and scale. Configure
   landscape collision to match the rendered terrain and block the stock
   Visibility trace channel (TraceTypeQuery1). Other scene geometry should
   ignore that channel. Verify the north-west and south-east cell centres
   against known landmarks before tracing. A flat terrain alone cannot reveal
   a horizontal registration error. In the editor's Python console:

   ```python
   import sys
   sys.path.insert(0, r"C:\Rivelero\src\Camera")
   import trace_visibility
   trace_visibility.run(r"C:\experiments\pilot150", northwest_cm=(0, 0), datum_z_cm=0)
   ```

   Replace the example XY with the Unreal coordinates of the **centre of DEM
   cell [0,0]**. Supply the world Z corresponding to DEM elevation zero.
   Cameras are transient. Each run checks the camera component's position,
   bearing and FOV, and checks ground collision within 5 cm of the exported
   camera-site elevation. Traces end 0.5 m above the shared DEM at each cell
   centre. The terrain collision in Unreal independently determines blocking.
   The editor may be busy during a batch. Re-running on the same unchanged
   scene and registration resumes after the last complete layout. Start a new
   batch if terrain, registration or trace settings change.

3. Score all completed layouts with a prespecified threshold (0.95 shown as
   an example). Rejecting one or more layouts returns exit status 1.

   ```shell
   python -m rivelero.experiments.visibility_validation compare pilot150 --threshold 0.95
   ```

   Fidelity is visible-cell intersection over union (IoU). The run passes
   only if every camera and the combined layout strictly exceed the threshold.
   Empty/empty receives zero. Missing or malformed masks stop scoring.
   Inspect `camera_scores.csv` for directional disagreement counts and
   `run_scores.csv` for each decision. Disagreement can arise from GDAL grid
   sampling or UE collision geometry as well as from implementation errors.
   Inspect an individual UE mask in NumPy with:

   ```python
   import numpy as np, zlib
   from pathlib import Path
   mask = np.frombuffer(zlib.decompress(Path(mask_path).read_bytes()), np.uint8).reshape(height, width)
   ```

## Run from the desktop app

The **Experiments…** menu opens three tabs. Each tab has its own settings and
Go button. Choose a DEM and a results folder accessible to both applications.
The app and Unreal may use different paths to the same folder.

Start the experiment server once in the Python console of the Unreal Editor
containing the corresponding landscape. For example, after adding the Camera
directory to sys.path as above:

```python
import experiment_server
experiment_server.start(r"C:\experiments", northwest_cm=(0, 0), datum_z_cm=0,
                        othonna_mesh="/Game/Plants/SM_Othonna.SM_Othonna")
```

Set registration to the actual landscape values. The default server URL is
`http://127.0.0.1:8765`. For a separate UE machine on a trusted network,
supply its LAN address as the server's `host` argument and enter that address
in the app. The bridge uses unauthenticated HTTP and a shared results folder.
Use `experiment_server.stop()` before changing levels or registration.

Press **Go**. Rivelero prepares the predictions in a new job folder, waits for
the server if necessary, and submits the batch automatically. The editor
performs collision traces on its main thread. Cancel stops the desktop task
and signals the editor to clean up at the next camera boundary. Each job
records the settings and seed, the viewpoints and layout IDs, fidelity scores,
and results for accepted layouts. Rejected layouts remain in run_scores.csv.

- **Viewpoint selection** compares random orderings with descending individual
  coverage and greedy marginal coverage. Each row reports central-domain and
  outside-domain coverage after adding a camera. Random orderings are
  replicated; deterministic rankings run once. Run the tab separately for
  flat and rugged DEMs with the corresponding Unreal landscape loaded.
- **Observation gaps** selects and validates one fixed layout. Each replicate
  draws a new two-dimensional gradient Perlin field and samples distinct
  central-domain cells with weights proportional to exp(contrast × noise).
  Patch scale is in metres. Contrast zero gives uniform sampling. plants.csv
  records every target's coordinates and detection; results.csv counts
  detections and misses. Detection is perfect in cells visible to any camera.
- **Spatial redundancy** compares random, dispersed and clustered layouts at
  the same camera count and reports blind, singly visible and repeatedly
  visible cells, plus mean and maximum exposure inside the central domain.

These experiments use geometric line traces. The interactive hello_camera.py
viewer is for Project AirSim image streams and is unnecessary for these runs.
The experiment endpoint is provided by the editor bridge, separately from the
Project AirSim service ports.


All three tabs accept camera height above local ground in metres (default
1.75). The same setting drives GDAL's observer height and the Unreal camera
component position. The CLI equivalent is `prepare --camera-height-m 2.5`.

For Experiment 2, replace the example asset path with your actual Othonna
Static Mesh object path from the Content Browser. Use a mesh with its origin at
the plant base. Restart the Unreal Editor and start the updated bridge after
updating the scripts. The other experiments do not require a plant asset.

Before submitting the job, Rivelero writes planting_list.csv with one entry per
plant per replicate. After the terrain-only camera traces, Unreal loads one
replicate at a time, spawns from raster row/column and reads back each
labelled mesh component's world position. Expected XYZ comes from the exported
easting, northing and DEM elevation. plant_validation.csv records both
positions and the 3D error in centimetres. A missing or duplicate expected
actor, incorrect mesh, displacement over 5 cm, or missing ground collision
within 5 cm stops the job. Actors are cleaned up between replicates. Detection
still uses the fixed camera visibility mask and the 0.5 m target height.
