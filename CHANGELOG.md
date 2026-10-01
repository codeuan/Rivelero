# Changelog

All notable changes to Rivelero. Versions follow `rivelero.__version__`.

## Unreleased (0.1.0.dev0)

A rewrite of Rivelero around canonical scientific models and a new desktop
application. Highlights:

### Added
- **Canonical Survey and World model**: Viewpoints, Sensors, ObservationEvents,
  ViewpointConfiguration, Environment, AnalysisGrid and AnalysisDomain, with
  survey import, QC and interactive domain definition.
- **Observability reconstruction**: per-sampling-unit visibility from an
  explicit VisibilityConfiguration (missing-metadata policies kept separate
  from source metadata) combined into a Survey Observability Field with
  analysis states (outside domain / invalid / blind spot / observable) and
  exposure.
- **Hybrid visibility cache**: fingerprinted on-disk visibility masks with an
  in-memory LRU, so large surveys are rebuilt incrementally and never reuse
  incompatible viewsheds.
- **Analysis & Design**: coverage and exposure summaries, sampling-unit
  contribution, non-destructive what-if scenarios, saved scenario snapshots
  and right − left comparison; objective components for future design search.
- **Project persistence**: `.rivelero` project files (schema version 1) with
  linked, checksummed terrain.
- **Scientific export and reporting**: GeoTIFF/CSV data with metadata
  sidecars, standalone figures (PNG/SVG/PDF), a provenance manifest and an
  offline HTML report.
- Package metadata (`pyproject.toml`), the `rivelero` command and
  `python -m rivelero`.
- **OpenStreetMap background** for the Survey, World and observability maps:
  an "OpenStreetMap" toggle draws OSM tiles beneath the map, reprojected into
  the map's CRS for display only, so a survey or terrain can be checked
  against its real-world location. Tiles load in the background, are cached
  per user and carry the OpenStreetMap attribution; off by default.
- **OpenTopography download usable for visibility**: the OpenTopography tab
  of *Add terrain* has an API key field (32 hexadecimal characters, checked as
  typed, masked, optionally remembered; `OPENTOPO_API_KEY` still works), seven
  datasets listed with their native resolution (COP30, COP90, NASADEM,
  SRTMGL1, SRTMGL3, AW3D30, EU_DTM), an area of interest (survey + buffer or a
  custom WGS84 box, with area and OpenTopography's size limit), and an
  explicit terrain CRS and cell size. The geographic download is reprojected
  to square metre cells (bilinear by default) so the visibility engine can
  use it; the original is kept, and dataset, resolutions and bounding box are
  recorded in provenance.

### Fixed
- OpenTopography errors no longer include the request URL (which contained
  the API key), and report a rejected key, a used-up request limit, missing
  data or a connection failure explicitly.
- The Survey map's *Reset* button now fits the Viewpoints again after zooming.

### Removed
- The earlier generation of the application, superseded by the above and not
  used by it (recoverable from git history before this cleanup, commit
  `24c31e9`):
  - the previous PySide6 GUI (`src/GUI`) and its process monitor (`src/admin`);
  - the flat prototype package `src/soe` and the drone simulation scripts
    (`src/Drone`, `rivelero/applications`);
  - the weighted observability-potential design workflow:
    `rivelero.design`, `rivelero.observability.potential_field`,
    `rivelero.suitability`, `rivelero.visibility.field`,
    `rivelero.visibility.obstacles`, `rivelero.io.osm` and the legacy
    `ViewpointRegion` / `ViewpointOPFResult` structures. Its candidate
    ranking combined visibility, NDVI and obstacle scores without accounting
    for overlap between viewpoints; the current Analysis & Design framework
    works on the Survey Observability Field instead. See
    `docs/legacy-architecture.md` for what those algorithms did;
  - unused modules `rivelero.core.config`, `rivelero.metrics`,
    `rivelero.visualization.survey` and `rivelero.visualization.visibility`;
  - cached OpenStreetMap responses (`cache/`, `src/cache`).
- Dependencies required only by the removed code: geopandas, osmnx, OpenEXR,
  psutil, python-dateutil; and never-imported lark, scikit-learn, pandas.

### Moved
- `rivelero.io.gsv` (Street View) and `rivelero.io.sentinel` (Sentinel-2 NDVI)
  to `contrib/data_sources/`: unsupported adapters kept for future
  data-source integrations, no longer part of the installed package.
