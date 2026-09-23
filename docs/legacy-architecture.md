# The removed legacy architecture (historical note)

Before the canonical rewrite, Rivelero proposed survey locations with a
*weighted observability-potential* workflow. That code was removed in the C1
cleanup; it remains in git history at commit `24c31e9` (and earlier). This
note records what it did, so future work (for example environmental
obstacles) can learn from it without inheriting its APIs.

## Pipeline

```text
DEM + target region ──► visibility field  V(x, y)   (visibility/field.py)
NDVI (Sentinel-2)   ──► suitability field N(x, y)   (suitability/botanical.py)
OSM geometries      ──► obstacle field    O(x, y)   (visibility/obstacles.py)
                              │
                              ▼
      potential  F = 0.5·V + 0.3·N − 0.2·O          (observability/potential_field.py)
                              │
                              ▼
      candidate viewpoints from high-F regions      (design/candidates.py, viewpoint.py)
      score-concentration pruning                   (design/redundancy.py)
      grid downsampling of candidate points         (design/selection.py)
```

| Module | Algorithm |
|---|---|
| `visibility/field.py` | For each possible **observer cell**, the fraction of sampled target-region cells visible from it (GDAL viewsheds from a regular sample of targets, every `target_spacing_cells`). A candidate-location surface, not the per-Viewpoint visibility of a survey. Its GDAL configuration survives in `visibility/viewshed.py`. |
| `visibility/obstacles.py` | Rasterised OpenStreetMap buildings/barriers (buffered points and lines, polygons) on the DEM grid, then the **obstacle-covered fraction of a square neighbourhood** around each cell, computed in O(H·W) with integral images (box sums). An occlusion-*risk proxy*: no heights, no line-of-sight. |
| `io/osm.py` | Fetched those geometries with OSMnx for a projected extent. |
| `suitability/botanical.py` | Sentinel-2 NDVI (or an Unreal-Engine EXR capture) aligned to the DEM and mapped linearly between an NDVI floor and ceiling to a 0–1 suitability. |
| `observability/potential_field.py` | The weighted sum above; a relative score, not a calibrated quantity. |
| `design/candidates.py`, `design/viewpoint.py` | Candidate cells visited from highest to lowest F; a peak was kept if enough of its neighbourhood was within a tolerance of the peak value ("support fraction") and it was not within a suppression radius of an accepted candidate. |
| `design/redundancy.py` | Kept the shortest highest-scoring prefix of candidates whose cumulative score reached a retention share (e.g. 95 %). |
| `design/selection.py` | Merged candidate points falling in the same coarse grid cell. |

## Why it was replaced

- It conflated **target suitability** (NDVI) with **observation opportunity**;
  the current framework keeps them separate and reconstructs observability
  only from the survey and the environment.
- Candidate selection ranked cells by score and never accounted for **overlap
  between viewpoints**: two adjacent candidates seeing the same area both
  scored highly. The current Analysis & Design framework works on the Survey
  Observability Field, where contribution, unique and repeated coverage and
  scenario comparison are overlap-aware by construction.
- Obstacles were a local density proxy rather than an occlusion along sight
  lines.

## Notes for future obstacle support

Future environmental obstacles should occlude sight lines inside the
canonical single-viewpoint engine (`visibility/engine.py`), driven by
`Environment.layers` / `VisibilityConfiguration.use_environment_obstacles`,
and be reflected in the cache fingerprint. The removed module's rasterisation
choices (buffering points and lines, `all_touched` rasterisation) and the
integral-image neighbourhood sum are reusable techniques; its output was not
a visibility quantity.
