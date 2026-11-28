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
