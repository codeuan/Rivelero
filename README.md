<p align="center">
  <img src="Rivelero Logo.png" alt="Rivelero" width="520">
</p>

<p align="center">
  <strong>Reconstructing and analysing the spatial observability of surveys.</strong>
</p>

<p align="center">
  An open-source spatial framework for understanding what an observation system
  <em>could have observed</em>, where its blind spots lie, and how alternative
  survey designs change observation opportunity.
</p>

---

> [!IMPORTANT]
> **Rivelero is under active development** (version `0.1.0.dev0`).
>
> The scientific model, GUI, file formats, and public Python API are still evolving.
> Results should be independently validated before use in operational or
> decision-critical applications.

## Contents

- [What is Rivelero?](#what-is-rivelero)
- [Installation](#installation)
- [Running Rivelero](#running-rivelero)
- [Quick start](#quick-start)
- [Workflow in detail](#workflow-in-detail): [Survey](#1-survey) · [World](#2-world) · [Observability](#3-observability) · [Analysis & Design](#4-analysis--design) · [Output](#5-output)
- [Input file reference](#input-file-reference)
- [The Survey Observability Field](#the-survey-observability-field)
- [Visibility cache](#visibility-cache)
- [Project files](#project-files)
- [Scientific interpretation](#scientific-interpretation)
- [Current limitations](#current-limitations)
- [Roadmap](#roadmap)
- [Architecture and repository structure](#architecture)
- [Development](#development)
- [Citation, licence and acknowledgements](#citation)

---

## Documentation

- [Scientific and Technical Architecture](docs/Rivelero_Scientific_Technical_Architecture.md) — canonical description of Rivelero's scientific model, engine architecture, implementation status, and development roadmap.
- [Legacy Architecture Notes](docs/legacy-architecture.md) — historical implementation notes retained for reference only.


## What is Rivelero?

Spatial datasets rarely sample the world uniformly.

An observation may be absent from a location because the target was genuinely
absent — but it may also be absent because that location was difficult or
impossible to observe from the available sampling positions.

**Rivelero reconstructs this observation opportunity spatially.**

Given a survey, its sensing geometry, and a representation of the environment,
Rivelero estimates which parts of an analysis area could have been observed
from each sampling unit and combines them into a **Survey Observability Field
(SOF)**.

This makes it possible to distinguish between:

- areas that were observable;
- spatial blind spots;
- areas outside the analysis domain;
- invalid terrain or missing environmental data;
- areas observed from exactly one sampling unit;
- areas receiving repeated observation opportunity.

Rivelero is intended particularly for **opportunistic and heterogeneous spatial
datasets**, where sampling locations and acquisition conditions were not
necessarily designed as part of a controlled survey.

Potential applications include street-level and crowdsourced imagery, UAV and
mobile surveys, camera networks, biodiversity and ecological observations,
historical or legacy surveys, survey quality assessment, and prospective survey
design.

### Why observability matters

A suitable location is not necessarily an observable location.

Terrain, viewing geometry, sensor characteristics, orientation, distance, and
missing acquisition metadata can all affect whether a target could have been
observed. Rivelero therefore treats **observability as a property of the
interaction between the survey and the environment**, rather than assuming that
the presence of data implies uniform sampling effort.

```text
Survey                         World
  ├── Viewpoints                 ├── Environment (DEM / DTM)
  ├── ObservationEvents          ├── AnalysisGrid
  └── Sensors                    └── AnalysisDomain
            │                          │
            └──────────┬───────────────┘
                       ▼
       Visibility model (VisibilityConfiguration)
                       ▼
     Individual visibility per sampling unit
                       ▼
          Survey Observability Field
   observable space · exposure · blind spots · repeated coverage
```

A central design principle is that **source observations and modelling
assumptions remain separate**. If the heading of an imported camera is unknown,
Rivelero keeps `heading_deg = None` rather than silently replacing it with an
assumed value. How a missing heading is interpreted is defined explicitly by
the `VisibilityConfiguration`, and recorded in the provenance of every result.
This matters for reproducibility and for future probabilistic,
uncertainty-aware observability models.

---

# Installation

Rivelero requires **Python ≥ 3.11** and is developed and tested with Python
3.11 in a Conda environment. GDAL's Python bindings (`osgeo`, used for the
viewshed computation) are most reliably installed from conda-forge.

```bash
git clone https://github.com/MaxwellML/VISTA.git
cd VISTA
conda env create -f environment.yml
conda activate vista
pip install -e ".[gui,test]"
```

`pip install -e .` installs the package from `src/` in editable mode and adds
the `rivelero` command. (The repository and Conda environment are still named
`VISTA` / `vista` for historical reasons; the package is `rivelero`.)

Dependencies (`pyproject.toml` is authoritative; `environment.yml` and
`requirements.txt` mirror it):

| Group | Packages |
|---|---|
| Scientific core | numpy, rasterio, affine, shapely, pyproj, matplotlib, requests, and GDAL's Python bindings (`osgeo`, from conda-forge) |
| `gui` extra | PySide6 (PyQt6 is accepted as a fallback) |
| `test` extra | pytest, pillow |

Without Conda, install GDAL's Python bindings for your platform first (they are
not listed in `requirements.txt`), then `pip install -e ".[gui,test]"`.

The scientific packages (`rivelero.core`, `visibility`, `observability`,
`analysis`, `export`, `project`) need no display and can be used headlessly
from scripts and notebooks.

### Optional: OpenTopography API key

To download elevation models from OpenTopography inside the application, set
the environment variable `OPENTOPO_API_KEY` to your
[OpenTopography API key](https://opentopography.org/) before starting
Rivelero. There is no field for the key in the GUI; without it the download
button stays disabled. (See the limitation on downloaded DEMs under
[World](#2-world).)

### Network use

Rivelero works fully offline. It only goes online when you ask it to: for an
OpenTopography download, or when you tick **OpenStreetMap** on a map to show
background tiles.

---

# Running Rivelero

```bash
rivelero                              # start with an empty project
rivelero "Sicily Survey.rivelero"     # open a saved project
python -m rivelero                    # equivalent, without the installed command
rivelero --version
rivelero --help
```

For development without installing the package, `python tests/launch_new_gui.py`
starts the same application from the source tree.

## The application window

The sidebar leads through five pages: **1 Survey**, **2 World**,
**3 Observability**, **4 Analysis & Design**, **5 Output**. The window title
shows the project name, with `*` when there are unsaved changes; you are asked
before unsaved work is discarded.

The **File** menu:

| Action | Shortcut |
|---|---|
| New Project | Ctrl+N |
| Open Project… | Ctrl+O |
| Save | Ctrl+S |
| Save As… | Ctrl+Shift+S |
| Export Data… (opens the Output page) | — |

Every map has a Matplotlib navigation toolbar (pan, zoom, home, save image) or
a Reset button, and an optional **OpenStreetMap** background (see
[Maps](#maps-and-the-openstreetmap-background)).

<!-- Screenshots: Survey · World · Observability · Analysis & Design · Output -->

---

# Quick start

A synthetic 1 km × 1 km test world ships with the repository in
[`rivelero_synthetic_observability/`](rivelero_synthetic_observability/) and is
a good first project:

1. **Survey** → *Import Viewpoints*. Select
   `viewpoints/viewpoints.csv` (and optionally `sensors.csv` and
   `observation_events.csv` from the same folder). The source CRS
   (EPSG:32633) is read from the file's `crs` column. Validate and import.
2. **World** → import `environments/dem_flat.tif` (or `dem_ridge.tif`) as a
   local DEM, then choose an analysis area (e.g. *Entire terrain extent*) and
   apply it. Check the World QC messages.
3. **Observability** → review the visibility settings (maximum distance,
   heights, direction, missing-metadata policies) and click build. Inspect
   *Analysis state*, *Exposure* and *Blind spots*; select a Viewpoint to see
   its individual visibility.
4. **Analysis & Design** → read the coverage summary, analyse sampling-unit
   contributions, and try a what-if scenario: deactivate units or place a
   candidate Viewpoint on the map; save the scenario and compare it with the
   baseline.
5. **Save** the project (Ctrl+S) and, on **Output**, export GeoTIFF/CSV data,
   figures, or an offline HTML report.

For your own data: prepare a Viewpoints CSV (see the
[input file reference](#input-file-reference)) and a projected DEM in metres
covering the survey.

---

# Workflow in detail

## 1. Survey

Define the observation system.

- A **Viewpoint** is an observation location and configuration (position,
  heights, orientation, field of view, sensor). Several Viewpoints can share a
  location; coordinates do not define identity.
- An **ObservationEvent** is an occurrence at a Viewpoint, preserving time,
  order, repeated visits and event-level acquisition metadata that can override
  the Viewpoint's values.
- A **Sensor** describes the instrument (modality, image size, focal length,
  native field of view, spectral bands…).

### Importing

*Import Viewpoints* opens a four-step dialog:

1. **Files**: a Viewpoints CSV (required) and optional Sensors and
   ObservationEvents CSVs, with a preview of the first rows.
2. **Coordinates**: the source CRS (default EPSG:4326, or taken from the
   file's `crs` column when all its values agree) and, optionally, a target
   CRS to transform the coordinates into on import.
3. **Field mapping**: map your Viewpoint columns to Rivelero fields. Common
   names (e.g. `lon`, `easting`, `bearing`, `hfov`) are suggested
   automatically; pitch, roll, vertical FOV, `z` and uncertainties are under
   *Advanced fields*.
4. **Validate**: set the survey ID and name, run validation and review
   counts, warnings and errors.

Rows that fail validation are skipped and listed; you confirm before importing
a partial file. Missing values stay missing (no defaults are applied at
import). Unmapped columns are kept as `extra_metadata`. Importing **replaces**
the current survey and its Sensors.

CRS safeguards: X/Y are never assumed to be longitude/latitude; with a
geographic source CRS, values outside ±180° / ±90° are rejected as probably
projected coordinates; a `crs` column that is empty, invalid or mixes several
CRSs is reported; a file CRS that disagrees with the one you entered produces a
warning. The exact column names are in the
[input file reference](#input-file-reference).

### Survey page

- summary of the survey and a searchable **Viewpoint table** (ID, X, Y,
  heading, FOV, height, sensor, source, metadata status);
- **add, edit and delete** Viewpoints manually (once a survey exists), and
  manage Sensors and ObservationEvents. Renaming a Viewpoint updates its
  events; a Viewpoint referenced by events, or a Sensor referenced by
  Viewpoints, cannot be deleted;
- **metadata completeness** of coordinates, heading, horizontal FOV, observer
  height and sensor;
- **survey map** coloured by: all Viewpoints; heading / FOV / observer-height
  availability; sensor, source or platform; or the observation sequence of
  events. Options for Viewpoint IDs, heading arrows and an OpenStreetMap
  background;
- **spatial checks**: a single consistent CRS (the map refuses to draw mixed
  CRSs), repeated locations and missing-metadata counts.

*Acquire from platform* (direct import from imagery providers) is not
implemented yet; unsupported example adapters live in [`contrib/`](contrib/).

---

## 2. World

Define the physical environment and analysis area.

### Terrain

- **Local DEM / DTM**: a GeoTIFF or any raster rasterio can read. It must
  have a CRS, and the visibility engine requires a **projected CRS in metres**.
  Only metadata (CRS, size, resolution, NoData, driver) is read at import.
- **OpenTopography download**: Copernicus 30 m (`COP30`) or SRTM GL1
  (`SRTMGL1`) for the survey extent plus a buffer (default 500 m). Requires a
  loaded Survey and `OPENTOPO_API_KEY`.

> [!WARNING]
> OpenTopography returns geographic (WGS84) rasters and Rivelero does not
> reproject them, so a **downloaded DEM cannot yet be used to compute
> visibility**. Reproject it to a projected CRS (e.g. the survey's UTM zone,
> with `gdalwarp`) and import the result as a local DEM.

DSM support is represented in the architecture but is disabled ("coming soon").

### Analysis area (AnalysisDomain)

Choose one of:

- **Entire terrain extent** (default);
- **Survey extent + buffer** (default 250 m around the Viewpoints);
- **Draw on map**: click the polygon vertices on the terrain map;
- **Import polygon**: a GeoJSON file (FeatureCollection, Feature or
  geometry; multiple parts are merged), with its source CRS stated explicitly
  and an optional filter on `properties.role`.

Every domain is clipped to the terrain grid, and terrain NoData cells become
**INVALID**, never blind spots. Rivelero distinguishes four cell states, which
are never interchangeable:

```text
OUTSIDE DOMAIN   not part of the analysis
INVALID          inside the domain, but no valid terrain
BLIND SPOT       analysable, observed by no sampling unit
OBSERVABLE       analysable, observed by at least one sampling unit
```

### World QC

Before building, the World page checks: an empty survey; Viewpoints with
unknown or mixed CRSs; survey and terrain CRSs differing (Viewpoints are
transformed for display and computation, never silently rewritten);
transformation failures; Viewpoints outside the terrain; a domain in another
CRS than the grid; and a domain wholly or partly outside the terrain.
Viewpoints outside the domain are reported as information only, since
observers may legitimately stand outside the area they observe.

---

## 3. Observability

Configure and reconstruct observation opportunity.

### Visibility configuration

| Setting | Default | Notes |
|---|---|---|
| Maximum distance | 500 m | must be finite and > 0 |
| Default observer height | 1.75 m | above the terrain |
| Target height | 0 m | one value for the whole analysis |
| Directionality | on | when off, every unit is omnidirectional |
| Default heading | none | compass degrees (0 = north, clockwise, relative to grid north) |
| Default horizontal FOV | 360° | range (0, 360]; 360° = omnidirectional |
| Curvature coefficient | 0.85714 | Earth curvature with standard refraction (1 − 1/7), under *Advanced* |
| Sampling unit | Viewpoint | or ObservationEvent (one contribution per event) |
| Extra parameters | `{}` | free JSON, recorded in provenance |

Vertical FOV / pitch and environmental obstacle layers appear in the
configuration but are not yet modelled.

### Missing metadata

Rivelero never invents missing acquisition metadata. Each parameter has an
explicit policy:

| Parameter | Default policy | Allowed policies | Resolved from |
|---|---|---|---|
| Heading | treat as omnidirectional | use configured default · omnidirectional · exclude unit · stop with error | Event → Viewpoint |
| Horizontal FOV | use configured default | use configured default · omnidirectional · exclude unit · stop with error | Event → Viewpoint → Sensor → configuration |
| Observer height | use configured default | use configured default · exclude unit · stop with error | Event → Viewpoint → configuration |

Units excluded by a policy are counted in the build report and in provenance.

### Build

For each active sampling unit Rivelero runs a GDAL viewshed
(`ViewshedGenerate`) on the terrain within the maximum distance, then applies
the directional field of view as a separate mask (a cell is kept when its
bearing lies within ±FOV/2 of the heading; the observer's own cell is always
kept), and restricts the result to valid cells in the domain. The analysis grid
is the terrain grid itself (no resampling). Masks are computed lazily, cached,
and combined incrementally into the Survey Observability Field; long builds run
in the background with progress and cancellation.

### Observability maps

*Analysis state*, *Observable space*, *Exposure*, *Normalized exposure*,
*Blind spots* and *Selected unit visibility* (select a Viewpoint on the map or
in the table). Terrain can be shown beneath cells without a value.

---

## 4. Analysis & Design

Once observability has been reconstructed, the Analysis & Design page has four
tabs.

### Overview: coverage

- observable, blind-spot, unique-coverage and repeated-coverage shares;
- mean, median (also over observable cells only), maximum and total exposure;
- analysable, invalid and outside-domain cell counts and active units;
- exposure-distribution and coverage-composition charts;
- maps of unique / repeated coverage, exposure, blind spots and analysis state.

All shares use **analysable cells** as the denominator.

### Contribution: per sampling unit

For each unit: visible cells, uniquely covered cells, repeatedly covered cells,
**coverage lost if it were removed**, and unique / repeated shares of its own
visibility, with a map of where its unique and repeated contributions lie.
These are derived from the existing field and cached masks; the survey is
**not** rebuilt for every unit.

### Scenario: non-destructive what-if design

- deactivate and reactivate existing sampling units;
- add temporary **candidate Viewpoints** by dialog or by clicking the map;
  include, exclude or remove them;
- compute candidate visibility and see each candidate's **gain vs the current
  scenario**;
- a Baseline / Scenario / Difference table (units, coverage share, observable,
  blind, unique, repeated, mean and maximum exposure) and maps of where
  coverage is gained or lost and of scenario exposure;
- **Reset scenario** restores the original survey exactly; **Save scenario for
  comparison** keeps a named snapshot in the project.

Scenario changes never modify the Survey. Candidates are not available with
ObservationEvent sampling (a candidate would also need an event); applying a
scenario back to the Survey is not implemented yet.

### Compare

Compare any two of the baseline, the current scenario and saved snapshots.
Differences are always **right − left** (with a ⇄ swap button): active
existing units, candidates, sampling units, observable cells, coverage share
(percentage points), blind, unique and repeated cells, mean and maximum
exposure, plus coverage-change and exposure-difference maps. There is
deliberately no single "best survey" score. Snapshots can be renamed,
described, deleted or loaded back into the Scenario tab; snapshots made
against an earlier build are marked *out of date* and cannot be compared.

---

## 5. Output

Projects are saved as `.rivelero` files (File › Save), the authoritative
record of an analysis (see [Project files](#project-files)). The **Output**
page has three tabs.

### Data

Machine-readable scientific data for GIS, Python/R, spreadsheets and
archiving. File names are `<project>_<name>`:

| Group | Files |
|---|---|
| Survey | `viewpoints.csv`, `sensors.csv`, `observation_events.csv` (importer column names, so they can be re-imported) |
| Observability | `exposure_count.tif`, `normalized_exposure.tif`, `observability_state.tif`, `blindspot_mask.tif`, `observable_mask.tif`, `analysis_mask.tif`, `valid_mask.tif` |
| Analysis | `coverage_class.tif`, `sampling_unit_contributions.csv` |
| Scenario | `scenario_exposure.tif`, `scenario_change_baseline_to_scenario.tif`, `scenario_summary.csv` |
| Comparison | `comparison_<left>_vs_<right>_summary.csv`, `…_exposure_difference_right_minus_left.tif`, `…_change_left_to_right.tif` |

Conventions:

- every raster is on the exact grid of the observability field (same CRS,
  transform and shape);
- blind spots are 0, never NoData. NoData marks only cells outside the
  AnalysisDomain or with invalid terrain (255 in the blind-spot and observable
  masks). Categorical rasters keep every code;
- every raster carries `RIVELERO_*` GeoTIFF tags **and** a `<file>.json`
  sidecar; tables have a sidecar. Sidecars record meaning, units, NoData,
  codes, direction and the SOF, AnalysisDomain and VisibilityConfiguration
  the file comes from;
- CSVs are UTF-8 (RFC 4180) with empty cells for missing values;
- existing files are never replaced unless explicitly allowed.

### Figures

Standalone PNG (default 300 dpi), SVG or PDF figures, each with a JSON
sidecar, optional title and caption, and optional Viewpoint overlay:
analysis-state, exposure, normalized-exposure, blind-spot and coverage-class
maps; coverage-composition, exposure-distribution and contribution-distribution
charts; scenario change and exposure maps; and comparison coverage-change and
exposure-difference (right − left) maps. They use exactly the colours,
whole-field scales, masks and legends of the application maps.

### Report

A self-contained report folder that opens offline in any browser (no
JavaScript or external resources):

```text
<project>_report/
├── report.html
├── provenance.json
├── provenance.md
└── figures/
```

Sections cover the project, survey, world, visibility assumptions,
observability results, coverage and exposure, sampling-unit contribution,
design scenario, scenario comparison, and provenance and limitations. Sections
for results that do not exist yet are omitted, so a Survey-only report is
possible.

The provenance manifest records the software version and git commit, the
Survey, the terrain file and its SHA-256 checksum, the AnalysisDomain, every
VisibilityConfiguration parameter, the observability build and results, and the
available design analyses. It keeps what the source data *contained* (e.g.
Viewpoints without a heading) separate from what the configuration *assumed*
(e.g. the missing-heading policy, and how many units it was applied to).

---

## Maps and the OpenStreetMap background

Every map (Survey, World, Observability, and the Analysis & Design maps) has
an **OpenStreetMap** checkbox that draws OpenStreetMap beneath the map, so you
can check that a survey, terrain or result lies where it should in the real
world.

- Tiles are reprojected into the map's own CRS for display only; no survey,
  terrain or result is transformed, and zoom and extent behave as without the
  background.
- Raster layers become partly transparent so the background shows through;
  on observability maps it replaces the grey terrain underlay.
- Tiles load in the background after each pan or zoom and are cached in
  `basemap_tiles/osm` under the Rivelero cache directory (see
  [Visibility cache](#visibility-cache) for its location) for 30 days.
- The background is off by default and needs an internet connection; it is
  unavailable for maps without a known CRS. If tiles cannot be loaded, a short
  message appears on the map.
- Map data © [OpenStreetMap](https://www.openstreetmap.org/copyright)
  contributors, shown on the map. Use follows the
  [OSM tile usage policy](https://operations.osmfoundation.org/policies/tiles/);
  the number of tiles per view is limited.

Exported figures do not include the background.

---

# Input file reference

All survey files are **CSV**: UTF-8 (a BOM is tolerated), comma-delimited,
with a header row. Angles are in degrees, lengths in metres. Examples:
[`rivelero_synthetic_observability/viewpoints/`](rivelero_synthetic_observability/viewpoints/).

### Viewpoints (required)

| Column | Required | Meaning |
|---|---|---|
| `viewpoint_id` | yes | unique ID |
| `x`, `y` | yes, or longitude/latitude | coordinates in the source CRS (X/Y win if both pairs are present) |
| `longitude`, `latitude` | alternative to x/y | |
| `crs` | no | CRS of the coordinates, e.g. `EPSG:32633`; used as source CRS when consistent |
| `z` | no | absolute elevation |
| `observer_height_m` | no | height above terrain, ≥ 0 |
| `heading_deg` | no | compass bearing, 0 = north, 90 = east (normalised to [0, 360)) |
| `horizontal_fov_deg` | no | (0, 360]; 360 = omnidirectional |
| `pitch_deg`, `roll_deg`, `vertical_fov_deg` | no | stored; not yet used by the visibility model |
| `sensor_id` | no | reference to a Sensor |
| `platform`, `source`, `source_id` | no | acquisition platform, data source, ID in the source dataset |
| `position_uncertainty_m`, `orientation_uncertainty_deg` | no | ≥ 0; stored for future uncertainty models |

Any other column is kept as `extra_metadata`. Column names can be remapped in
the import dialog.

### Sensors (optional)

Exact column names are required.

| Column | Meaning |
|---|---|
| `sensor_id` | unique ID (required) |
| `modality` | `rgb` (default), `multispectral`, `hyperspectral`, `thermal`, `infrared`, `night_vision`, `depth`, `lidar`, `other` |
| `model`, `manufacturer`, `source`, `name` | descriptive |
| `image_width_px`, `image_height_px` | integers |
| `focal_length_mm`, `sensor_width_mm`, `sensor_height_mm` | |
| `horizontal_fov_deg`, `vertical_fov_deg` | native field of view, (0, 360] |
| `spectral_bands` | separated by `;` or `,` |
| `wavelength_range_nm` | e.g. `400-700` |
| `spatial_resolution` | |

A Viewpoint referring to a Sensor that is not imported produces a warning.

### ObservationEvents (optional)

Exact column names are required; row order is preserved.

| Column | Meaning |
|---|---|
| `event_id` | unique ID (required) |
| `viewpoint_id` | an imported Viewpoint (required) |
| `timestamp` | ISO 8601; a trailing `Z` means UTC |
| `sequence_id`, `sequence_index` | sequence and position (integer ≥ 0) |
| `image_id`, `source` | |
| `observer_height_m`, `heading_deg`, `pitch_deg`, `roll_deg`, `horizontal_fov_deg`, `vertical_fov_deg` | per-event overrides of the Viewpoint's values |

### Analysis-area polygon (optional)

A GeoJSON FeatureCollection, Feature or geometry (Polygon/MultiPolygon). Its
CRS is chosen in the import dialog.

---

# The Survey Observability Field

The **Survey Observability Field (SOF)** is the central aggregate
representation. For each valid cell in the AnalysisDomain it records the
accumulated observation opportunity (**exposure**: the number of active
sampling units that could observe the cell). **Normalized exposure** is
exposure divided by the number of active units.

```text
Exposure = 0    →  blind spot
Exposure = 1    →  unique coverage
Exposure ≥ 2    →  repeated coverage
```

All coverage statistics are calculated relative to **valid analysable cells**,
not the complete raster extent, so cells outside the AnalysisDomain or with
missing terrain do not artificially reduce estimated coverage.

---

# Visibility cache

Large opportunistic surveys may contain tens of thousands of Viewpoints, so
Rivelero does **not** keep every viewshed in memory:

```text
Viewpoints / ObservationEvents ── lightweight metadata in RAM
            │ visibility requested
            ▼
     compute lazily ──► disk cache (.npz) ◄──► LRU RAM cache
                                  │
                                  ▼
                     Survey Observability Field
```

- **Location**: `RIVELERO_CACHE_DIR` if set; otherwise
  `%LOCALAPPDATA%\Rivelero\visibility_cache` on Windows and
  `$XDG_CACHE_HOME/rivelero/visibility_cache` (default
  `~/.cache/rivelero/visibility_cache`) elsewhere. It can be changed under
  *Advanced computation and storage* on the Observability page.
- **Format**: one (optionally compressed) NumPy `.npz` file per mask.
- **Memory**: the most recently used masks are kept in RAM (64 by default in
  the application, roughly 256 MB for a 2000 × 2000 grid).
- **Management**: count cached masks, clear the memory cache, or delete cached
  masks from the Observability page.
- **Safety**: cache keys fingerprint every scientific input (terrain file
  identity, grid, domain and validity masks, configuration, and each unit's
  Viewpoint, Event and Sensor geometry), so changed inputs never reuse
  incompatible viewsheds. Keys also include the environment, domain and
  configuration IDs, so re-importing the same terrain or re-creating the same
  domain computes fresh masks.

The cache is never required: deleting it only costs recomputation.

---

# Project files

A `.rivelero` project is a ZIP archive of UTF-8 JSON documents and NumPy
arrays (schema version 1; older schemas are migrated, newer ones refused).

- **Saved**: the Survey (Viewpoints, ObservationEvents, Sensors), the
  Environment, AnalysisGrid and AnalysisDomain, the VisibilityConfiguration
  and cache settings, the build report, the Survey Observability Field, the
  live scenario, saved scenario snapshots and the selected comparison.
- **Not saved**: selections, map views, tabs, the visibility cache and
  contribution analyses (recomputed on demand).
- **Terrain is linked, not copied**: the project records the terrain's
  absolute path, its path relative to the project, size, SHA-256 and raster
  properties. On opening, a terrain found at the relative path (e.g. after
  moving the project and terrain together) is used; a changed checksum or a
  missing file is reported. The saved field is restored only if its inputs and
  the terrain are unchanged.
- Saving is atomic (write, verify, replace).

---

# Scientific interpretation

Rivelero reconstructs **observation opportunity**, not detection probability.

*Observable* means that the current geometric/environmental model indicates
that a target at that location could have been observed by at least one
sampling unit. It does **not** necessarily mean that a target was present,
would certainly have been detected, that the imagery was of sufficient quality
for a particular classifier, or that an observer would recognise the target.
These processes may be modelled separately.

This distinction is particularly important in ecological and biodiversity
applications, where habitat suitability and observability are related but
different processes. The original Rivelero research motivation explicitly
highlighted the risk of conflating target suitability with the opportunity to
observe it.

---

# Current limitations

- only terrain (DEM/DTM) occludes sight lines: vegetation, buildings and other
  obstacle layers are not modelled; DSM-based visibility is future work;
- vertical field of view and pitch are not modelled;
- the terrain must be in a projected CRS in metres; OpenTopography downloads
  (geographic WGS84) are not reprojected and must be reprojected externally;
- the analysis grid is the terrain grid (no resampling);
- target height is a single value for the whole analysis;
- sight lines crossing DEM NoData regions may be unreliable under the GDAL
  backend;
- observability is deterministic modelled observation opportunity, not a
  detection probability;
- analysis-area polygons can only be imported from GeoJSON;
- survey import reads CSV only; direct acquisition from imagery platforms is
  not implemented;
- candidate Viewpoints cannot be added under ObservationEvent sampling, and
  scenarios cannot yet be committed back to the Survey;
- comparison covers scenarios of one baseline, not different visibility
  configurations; saved scenarios become *out of date* after the field is
  rebuilt, even with identical inputs;
- automated survey optimization, candidate generation and Bayesian /
  probabilistic observability are not implemented.

---

# Roadmap

- [x] Canonical Survey model
- [x] Survey import and QC
- [x] Environment / World workflow
- [x] Interactive AnalysisDomain definition
- [x] Lazy visibility storage
- [x] Survey Observability Field reconstruction
- [x] Exposure and blind-spot analysis
- [x] Sampling-unit contribution analysis
- [x] Non-destructive survey-design scenarios
- [x] Scenario comparison
- [x] Project save/load
- [x] Raster and table data export
- [x] Figure and report export
- [x] Provenance and reproducibility manifests
- [x] OpenStreetMap map background
- [ ] Reprojection of downloaded DEMs
- [ ] Comparison of visibility configurations
- [ ] Candidate-pool generation
- [ ] User-defined survey optimization (measurable objective components exist in `rivelero.analysis.objectives`)
- [ ] DSM support
- [ ] Environmental obstacle integration
- [ ] Vertical FOV support
- [ ] Probabilistic / Bayesian observability
- [ ] Uncertainty propagation

## Probabilistic observability

A particularly important future research direction is moving from
deterministic statements such as *this cell is observable* towards quantities
such as:

```text
P(cell observable | survey metadata, environment, modelling assumptions)
```

This could propagate uncertainty in Viewpoint position, orientation, sensor
parameters, missing metadata, environmental information, and visibility
itself. The architecture deliberately keeps missing metadata, provenance, and
modelling assumptions separate to support this direction.

---

# Architecture

```text
┌─────────────────────────────────────────────┐
│                    GUI                      │
│ Survey · World · Observability · Analysis   │
│ · Output                                    │
└──────────────────────┬──────────────────────┘
                ApplicationState
┌──────────────────────▼──────────────────────┐
│    Analysis · Export · Project services     │
│ coverage · contribution · scenarios · I/O   │
└──────────────────────┬──────────────────────┘
┌──────────────────────▼──────────────────────┐
│       Observability / Visibility engine     │
│ builder · storage · exposure · masks        │
└──────────────────────┬──────────────────────┘
┌──────────────────────▼──────────────────────┐
│              Canonical models               │
│ Survey · Sensor · Environment · Domain      │
└─────────────────────────────────────────────┘
```

Scientific objects are independent of the GUI, and the GUI operates on the
canonical models rather than keeping its own copies.

## Repository structure

```text
src/rivelero/
├── core/            canonical models: Viewpoint, Sensor, ObservationEvent,
│                    ViewpointConfiguration, Environment, AnalysisDomain
├── visibility/      VisibilityConfiguration, GDAL viewshed, directional FOV, engine
├── observability/   builder, VisibilityStore cache, masks, SurveyObservabilityField
├── analysis/        coverage, contribution, scenarios, comparison, objectives
├── project/         .rivelero project files (schema, codec, linked resources)
├── export/          GeoTIFF/CSV data, figures, provenance, HTML report
├── visualization/   Qt-free map layers, colours, legends, plotting, OSM basemap
├── io/              DEM download/reading, SOF raster writers
└── gui/             PySide6 application (ApplicationState, pages, maps, TaskController)

tests/               pytest suite (markers in tests/conftest.py), shared fixtures,
                     golden project files in tests/data, development launchers
rivelero_synthetic_observability/
                     synthetic 1 km test world (EPSG:32633): DEMs, domain,
                     survey CSVs and scenario definitions used by tests and demos
contrib/             unsupported data-source adapters (Street View, Sentinel-2);
                     not installed with the package
docs/                legacy-architecture.md (the removed earlier application)
CHANGELOG.md         notable changes
pyproject.toml       package metadata and dependencies (authoritative)
environment.yml      Conda environment (`vista`)
requirements.txt     pip mirror of the dependencies (without GDAL)
```

The earlier generation of the application (a previous GUI and a weighted
"observability potential" design workflow) was removed; see `CHANGELOG.md`
and `docs/legacy-architecture.md`.

---

# Development

## Tests

```bash
pytest                   # complete suite: the authoritative check
pytest -m "not slow"     # fast development loop
pytest -m unit           # Qt-free unit tests only
pytest -m "not gui"      # everything that needs no Qt widgets
```

Tests are classified in `tests/conftest.py` with the markers `unit`,
`integration`, `gui` and `slow` (registered in `pyproject.toml`; an
unclassified test module is an error). GUI tests run offscreen by default
(`QT_QPA_PLATFORM=offscreen`); set the variable yourself to watch them. Tests
never touch the real cache (`RIVELERO_CACHE_DIR` points to a temporary
directory) or the network.

> [!NOTE]
> Some test modules still refer to the synthetic data by an absolute
> Windows path (`C:\Users\zool2620\VISTA\…`) and fail elsewhere until they are
> changed to paths relative to the repository.

Prefer tests of scientific and state logic over pixel comparisons; figure
tests inspect the Matplotlib objects (source arrays, norms, legends).
`python tests/visualize_synthetic_observability.py` regenerates the visual
validation images of the synthetic world.

## Architecture for contributors

- **Canonical models** (`rivelero.core`) are the only representation of
  Survey, Sensors, Environment and AnalysisDomain; the GUI never keeps copies.
- **Scientific services** (`visibility`, `observability`, `analysis`,
  `export`, `project`) are Qt-free functions and dataclasses; they can run in
  a worker thread or a notebook.
- **`ApplicationState`** (`gui/application_state.py`, Qt-free) owns the
  current project: inputs, the current Survey Observability Field, derived
  analyses, selection and view state. Changing an input invalidates every
  derived result; results computed for superseded inputs are refused
  (`StaleObservabilityResultError`). Selection, view and task changes never
  mark the project modified.
- **`TaskController`** (`gui/task_controller.py`) runs one long operation at a
  time on a thread pool, with progress, cancellation and error signals.
  Results are installed on the GUI thread only if still current.
- **Pages** (`gui/*_page.py`) read ApplicationState in `refresh_from_state()`
  and call services; expensive Analysis panels are built when first opened.
  Every Matplotlib canvas is `gui.canvas.SafeFigureCanvas`.
- **`visualization.layers.map_layer`** defines each map layer's colours,
  scales and legend once, for both the GUI maps and exported figures.
- **`gui.basemap.BasemapLayer`** adds the optional OpenStreetMap background
  to any Matplotlib map without changing its limits.

## Versioning

The version is defined once, in `rivelero/__init__.py` (`__version__`), read by
`pyproject.toml` and recorded in projects, exports and provenance. Rivelero
uses development versions (`0.x.devN`) until a first public release.

## Contributing

Contributions are welcome, particularly in uncertainty propagation and
Bayesian observability, additional visibility backends, DSM and 3D
environments, environmental occlusion, sensor models, survey import adapters,
candidate generation, survey optimization, geospatial export, performance, and
documentation and examples.

When contributing scientific functionality, preserve the separation between
source observations, modelling assumptions, derived observability, and
analysis/design decisions, and include tests for new scientific behaviour.

---

# Citation

A formal software citation and associated publication are in preparation.
Until then, if you use Rivelero in research, please cite the repository and
record the version or commit hash used (both are written into every project,
export and report):

```text
Dean-Pijuan, C. and Fenollosa, E. Rivelero: spatial observability
reconstruction and survey-design framework (version 0.1.0.dev0).
Software under active development. https://github.com/MaxwellML/VISTA
```

# Licence

No licence has been chosen yet and the repository contains no `LICENSE` file.
Until one is added, please contact the authors before reusing or
redistributing the code.

# Acknowledgements

Rivelero is being developed as a research framework for understanding and
designing spatial observation systems. It builds on the open-source geospatial
Python ecosystem, including GDAL, Rasterio, Shapely, PyProj, NumPy, Matplotlib
and Qt. Elevation data can be obtained from
[OpenTopography](https://opentopography.org/); map backgrounds are ©
[OpenStreetMap](https://www.openstreetmap.org/copyright) contributors.

---

<p align="center">
  <img src="Rivelero Icon.png" alt="Rivelero icon" width="72">
</p>

<p align="center">
  <strong>Observe the survey, not only the observations.</strong>
</p>
