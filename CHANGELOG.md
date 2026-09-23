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
