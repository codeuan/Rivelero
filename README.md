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
> **Rivelero is under active development.**
>
> The scientific model, GUI, file formats, and public Python API are still evolving.
> Results should be independently validated before use in operational or
> decision-critical applications.

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

Potential applications include:

- street-level imagery;
- crowdsourced geospatial imagery;
- UAV and mobile surveys;
- camera networks;
- biodiversity and ecological observations;
- historical or legacy surveys;
- survey quality assessment;
- prospective survey design.

---

## Why observability matters

A suitable location is not necessarily an observable location.

Terrain, viewing geometry, sensor characteristics, orientation, distance, and
missing acquisition metadata can all affect whether a target could have been
observed.

Rivelero therefore treats **observability as a property of the interaction
between the survey and the environment**, rather than assuming that the
presence of data implies uniform sampling effort.

Conceptually:

```text
Survey
  │
  ├── Viewpoints
  ├── ObservationEvents
  └── Sensors
  │
  ▼
World
  │
  ├── Environment
  ├── AnalysisGrid
  └── AnalysisDomain
  │
  ▼
Visibility model
  │
  ▼
Individual visibility
  │
  ▼
Survey Observability Field
  │
  ├── observable space
  ├── exposure
  ├── blind spots
  └── repeated coverage
```

A central design principle is that **source observations and modelling
assumptions remain separate**.

For example, if the heading of an imported camera is unknown, Rivelero retains:

```python
heading_deg = None
```

rather than silently replacing it with an assumed value.

How missing heading should be interpreted is instead defined explicitly by the
`VisibilityConfiguration`.

This separation is important for reproducibility and for future probabilistic
and uncertainty-aware observability models.

---

# Current workflow

The Rivelero desktop application is organised around five stages:

```text
SURVEY
   ↓
WORLD
   ↓
OBSERVABILITY
   ↓
ANALYSIS & DESIGN
   ↓
OUTPUT
```

## 1. Survey

Define the observation system.

Rivelero currently supports:

- CSV survey import;
- Viewpoints;
- Sensors;
- ObservationEvents;
- repeated visits to the same location;
- missing metadata;
- CRS handling;
- manual Viewpoint editing;
- Sensor and event management;
- survey maps;
- metadata completeness checks;
- spatial quality control.

A **Viewpoint** represents an observation location/configuration.

An **ObservationEvent** represents an occurrence at that Viewpoint, allowing
Rivelero to preserve time, order, repeated visits, and event-level acquisition
metadata.

---

## 2. World

Define the physical environment and analysis area.

Current functionality includes:

- local DEM/DTM import;
- OpenTopography DEM acquisition;
- raster metadata inspection;
- interactive terrain maps;
- elevation colourbars;
- pan and zoom;
- Survey Viewpoint overlays;
- spatial compatibility checks;
- AnalysisDomain creation from:
  - the complete terrain extent;
  - buffered survey extent;
  - an interactively drawn polygon;
  - an imported polygon.

Rivelero distinguishes between:

```text
OUTSIDE DOMAIN
INVALID
BLIND SPOT
OBSERVABLE
```

These states are not interchangeable.

**DSM support is represented in the architecture but is not yet enabled as a
fully supported visibility workflow.**

---

## 3. Observability

Configure and reconstruct observation opportunity.

Visibility can currently account for parameters including:

- maximum observation distance;
- observer height;
- target height;
- heading;
- horizontal field of view;
- missing metadata policies;
- Viewpoint- or ObservationEvent-based sampling.

Individual visibility masks are computed lazily and combined into the
**Survey Observability Field (SOF)**.

The resulting field supports:

- observable space;
- raw exposure;
- normalised exposure;
- blind spots;
- analysis-state maps;
- individual sampling-unit visibility.

### Missing metadata

Rivelero does not silently invent missing acquisition metadata.

Instead, configurable policies determine what happens when values such as
heading, FOV, or observer height are unavailable.

Depending on the parameter, policies can include behaviours such as:

- use a configured default;
- treat direction as omnidirectional;
- exclude the sampling unit;
- stop with an error.

---

## 4. Analysis & Design

Once observability has been reconstructed, Rivelero can analyse how observation
opportunity is distributed across the survey.

Current tools include:

### Coverage analysis

- observable share;
- blind-spot share;
- unique coverage;
- repeated coverage;
- mean, median, and maximum exposure;
- exposure distributions;
- spatial coverage maps.

### Sampling-unit contribution

For each sampling unit, Rivelero can calculate:

- visible cells;
- uniquely covered cells;
- repeatedly covered cells;
- coverage that would be lost if that unit were removed;
- the spatial location of unique and repeated contributions.

These quantities are derived from the existing SOF and cached visibility masks;
the complete survey does **not** need to be rebuilt for every unit.

### What-if scenarios

Rivelero also supports temporary, non-destructive survey-design scenarios.

Users can:

- temporarily deactivate existing sampling units;
- reactivate them;
- add temporary candidate Viewpoints;
- compute candidate visibility;
- inspect marginal coverage gains;
- see where coverage is gained or lost;
- compare scenario exposure with the baseline;
- reset exactly to the original survey.

Scenario changes do **not** modify the original Survey.

### Scenario comparison

Scenario comparison is currently under development.

The intended workflow supports comparing alternative survey designs using
explicit quantities such as coverage, blind spots, exposure, and number of
sampling units rather than assuming a universal definition of the "best"
survey.

---

## 5. Output

Projects are saved as `.rivelero` files (File › Save), the authoritative
record of an analysis.

The **Output** page (also File › Export Data…) exports machine-readable
scientific data for GIS, Python/R, spreadsheets and archiving:

- Survey tables (Viewpoints, Sensors, ObservationEvents) as CSV, using the
  importer's column names so they can be re-imported;
- Survey Observability Field GeoTIFFs: exposure count, normalized exposure,
  categorical observability state, blind-spot, observable, AnalysisDomain and
  validity masks;
- coverage-class rasters and sampling-unit contribution tables;
- the current what-if scenario (exposure and baseline → scenario change),
  a scenario summary table, and the selected comparison (table, exposure
  difference **right − left**, and left → right change classes).

Every raster is written on the exact grid of the observability field (same CRS,
transform and shape). Blind spots are 0, never NoData; NoData marks only cells
outside the AnalysisDomain or with invalid terrain; categorical rasters keep
every code. Each file has `RIVELERO_*` GeoTIFF tags or a JSON sidecar
(`<file>.json`) recording its meaning, units, NoData, codes, direction and the
SOF, AnalysisDomain and VisibilityConfiguration it comes from. Existing files
are never replaced unless explicitly allowed.

The **Figures** tab exports standalone PNG (default 300 dpi), SVG or PDF
figures: analysis-state, exposure, normalized exposure, blind-spot and
coverage-class maps, the coverage-composition and exposure-distribution
charts, the sampling-unit contribution distribution, scenario change and
exposure maps, and the comparison coverage-change and exposure-difference
(right − left) maps. They use exactly the colours, whole-field scales, masks
and legends of the application maps.

The **Report** tab writes a self-contained report folder that opens offline in
any browser:

```
<project>_report/
├── report.html
├── provenance.json
├── provenance.md
└── figures/
```

The provenance manifest records the software version and git commit, the
Survey, the terrain file and its SHA-256 checksum, the AnalysisDomain, every
VisibilityConfiguration parameter, the observability build and results, and
the available design analyses. It keeps what the source data *contained*
(e.g. Viewpoints without a heading) separate from what the configuration
*assumed* (e.g. the missing-heading policy, and how many sampling units it was
applied to). Warnings and limitations are listed only when they apply.
Sections for results that do not exist yet are omitted, so a Survey-only
report is possible.

---

# The Survey Observability Field

The **Survey Observability Field (SOF)** is the central aggregate representation
in the current Rivelero architecture.

For each valid cell in the AnalysisDomain, the field records the accumulated
observation opportunity produced by active sampling units.

A simplified interpretation of exposure is:

```text
Exposure = 0
    ↓
Blind spot

Exposure = 1
    ↓
Unique coverage

Exposure >= 2
    ↓
Repeated coverage
```

All coverage statistics are calculated relative to **valid analysable cells**,
not the complete raster extent.

Therefore cells outside the AnalysisDomain or cells invalidated by missing
terrain data do not artificially reduce estimated coverage.

---

# Scalable visibility storage

Large opportunistic surveys may contain tens of thousands of Viewpoints.

Rivelero therefore does **not** compute and retain every viewshed in memory.

Instead it uses a hybrid architecture:

```text
Viewpoints / ObservationEvents
            │
            │ lightweight metadata
            ▼
           RAM

Visibility requested
            │
            ▼
      compute lazily
            │
            ▼
       disk cache
            ↕
       LRU RAM cache
            │
            ▼
Survey Observability Field
```

This allows individual visibility masks to be reused without requiring the
entire survey's visibility data to remain in RAM.

Cache entries are fingerprinted using the scientific inputs that affect
visibility so that changed Viewpoints, Sensors, terrain, domains, or visibility
settings do not silently reuse incompatible viewsheds.

---

# Installation

> [!NOTE]
> Installation instructions are still being consolidated while Rivelero is
> under active development.

Rivelero is currently developed and tested using a Conda environment.

Clone the repository:

```bash
git clone https://github.com/codeuan/Rivelero.git
cd Rivelero
```

Create the environment:

```bash
conda env create -f environment.yml
```

Activate it:

```bash
conda activate vista
```

Alternatively, dependencies are listed in:

```text
requirements.txt
```

The current stack includes geospatial/scientific packages such as NumPy,
Rasterio, GDAL, GeoPandas, Shapely, PyProj, Matplotlib, and PySide6. Earlier
repository validation also identified these as core dependencies. :contentReference[oaicite:1]{index=1}

---

# Running Rivelero

During the current development stage, launch the new GUI from the repository
root with:

```bash
python tests/launch_new_gui.py
```

The permanent application entry point will be documented here once packaging
is finalised.

---

# Quick start

A typical Rivelero analysis follows this sequence:

### 1. Import a Survey

Load:

```text
Viewpoints
Sensors
ObservationEvents
```

and inspect the Survey table and map.

### 2. Define the World

Load or download an elevation model and define the AnalysisDomain.

Check that Survey and terrain are spatially compatible.

### 3. Configure visibility

Define observation distance, observer/target height, directional behaviour,
and policies for missing metadata.

### 4. Build observability

Rivelero computes or retrieves individual visibility masks and incrementally
constructs the Survey Observability Field.

### 5. Inspect the result

Explore:

```text
observability
exposure
blind spots
analysis states
individual visibility
```

### 6. Analyse the Survey

Inspect:

```text
coverage
unique coverage
repeated coverage
sampling-unit contribution
```

### 7. Explore alternative designs

Temporarily remove observations or add candidate Viewpoints and inspect how
the survey's observability changes.

---

# Architecture

Rivelero follows a layered architecture.

```text
┌─────────────────────────────────────────────┐
│                    GUI                      │
│ Survey · World · Observability · Analysis   │
└──────────────────────┬──────────────────────┘
                       │
                ApplicationState
                       │
┌──────────────────────▼──────────────────────┐
│             Analysis / Services             │
│ coverage · contribution · scenarios         │
└──────────────────────┬──────────────────────┘
                       │
┌──────────────────────▼──────────────────────┐
│       Observability / Visibility Engine     │
│ builder · storage · exposure · masks        │
└──────────────────────┬──────────────────────┘
                       │
┌──────────────────────▼──────────────────────┐
│              Canonical models               │
│ Survey · Sensor · Environment · Domain      │
└─────────────────────────────────────────────┘
```

Scientific objects are intentionally independent of the GUI.

The GUI operates on canonical Rivelero models rather than maintaining
GUI-specific copies of Viewpoints, Sensors, Environments, or visibility
configurations.

---

# Repository structure

The current source tree is organised approximately as:

```text
src/rivelero/
├── core/
│   ├── viewpoint.py
│   ├── sensor.py
│   ├── observation.py
│   ├── configuration.py
│   ├── environment.py
│   └── domain.py
│
├── visibility/
│   ├── configuration.py
│   ├── directional.py
│   └── engine.py
│
├── observability/
│   ├── builder.py
│   ├── storage.py
│   ├── survey_field.py
│   ├── exposure.py
│   └── masks.py
│
├── analysis/
│   ├── coverage.py
│   ├── contribution.py
│   └── scenario.py
│
├── visualization/
│
├── io/
│
└── gui/
```

The repository still contains some legacy modules from earlier Rivelero
architectures. These are being progressively replaced or isolated rather than
removed before their remaining dependencies are understood.

---

# Scientific interpretation

Rivelero reconstructs **observation opportunity**, not detection probability
itself.

For example:

```text
Observable
```

means that the current geometric/environmental model indicates that a target
at that location could have been observed by at least one sampling unit.

It does **not** necessarily mean that:

- a target was present;
- a target would certainly have been detected;
- the imagery was of sufficient quality for a particular classifier;
- the observer would recognise the target.

These additional processes may be modelled separately.

This distinction is particularly important in ecological and biodiversity
applications, where habitat suitability and observability are related but
different processes. The original Rivelero research motivation explicitly
highlighted the risk of conflating target suitability with the opportunity to
observe it. :contentReference[oaicite:2]{index=2}

---

# Current limitations

Rivelero is actively evolving.

Important current limitations include:

- vertical field-of-view filtering is not yet supported by the current
  visibility engine;
- environmental obstacle layers are not yet integrated into the new visibility
  engine;
- DSM-based visibility remains future work;
- sight lines crossing DEM NoData regions may be unreliable depending on GDAL
  behaviour;
- candidate addition is currently limited under ObservationEvent-based
  sampling;
- scenario changes are non-destructive and cannot yet be committed back to the
  canonical Survey;
- automated survey optimization is not yet implemented;

---

# Roadmap

Near-term development priorities include:

- [x] Canonical Survey model
- [x] Survey import and QC
- [x] Environment / World workflow
- [x] Interactive AnalysisDomain definition
- [x] Lazy visibility storage
- [x] Survey Observability Field reconstruction
- [x] Exposure and blind-spot analysis
- [x] Sampling-unit contribution analysis
- [x] Non-destructive survey-design scenarios
- [x] Scenario/configuration comparison
- [x] Project save/load
- [x] Raster and table data export
- [x] Figure and report export
- [x] Provenance and reproducibility manifests
- [ ] Candidate-pool generation
- [ ] User-defined survey optimization
- [ ] DSM support
- [ ] Environmental obstacle integration
- [ ] Vertical FOV support
- [ ] Probabilistic / Bayesian observability
- [ ] Uncertainty propagation

## Probabilistic observability

A particularly important future research direction is moving from deterministic
statements such as:

```text
this cell is observable
```

towards quantities such as:

```text
P(cell observable | survey metadata,
                    environment,
                    modelling assumptions)
```

This could propagate uncertainty in:

- Viewpoint position;
- orientation;
- sensor parameters;
- missing metadata;
- environmental information;
- visibility itself.

The current architecture deliberately preserves missing metadata, provenance,
and modelling assumptions separately to support this direction.

---

# Development

Run the complete test suite from the repository root:

```bash
python -m pytest tests -v
```

The project has an expanding automated suite covering:

- canonical scientific models;
- survey import;
- visibility;
- observability;
- caching;
- World/domain construction;
- GUI state transitions;
- asynchronous computation;
- coverage analysis;
- contribution analysis;
- design scenarios.

When adding functionality, prefer tests of scientific/state logic over
pixel-perfect GUI screenshot tests.

---

# Contributing

Rivelero is under active research development, and contributions are welcome.

Areas particularly suitable for future contributions include:

- uncertainty propagation;
- Bayesian observability;
- additional visibility backends;
- DSM and 3D environments;
- environmental occlusion;
- sensor models;
- additional survey import adapters;
- candidate generation;
- survey optimization;
- geospatial export;
- performance and scalability;
- documentation and examples.

When contributing scientific functionality, please preserve the separation
between:

```text
source observations
modelling assumptions
derived observability
analysis/design decisions
```

and include tests for new scientific behaviour.

---

# Citation

A formal software citation and associated publication are in preparation.

Until then, if you use Rivelero in research, please cite the repository and
record the version or commit hash used for the analysis.

```text
Rivelero contributors: Connor Dean-Pijuan and Erola Fenollosa
Rivelero: spatial observability reconstruction and survey-design framework.
Software under active development.
```

This section will be replaced with the formal citation and DOI when available.

---

# Licence

Please see the repository's `LICENSE` file for the current licence terms.

---

# Acknowledgements

Rivelero is being developed as a research framework for understanding and
designing spatial observation systems.

The project builds on the wider open-source geospatial Python ecosystem,
including GDAL, Rasterio, GeoPandas, Shapely, PyProj, NumPy, Matplotlib and Qt.

OpenTopography is supported as a source of elevation data.

---

<p align="center">
  <img src="Rivelero Icon.png" alt="Rivelero icon" width="72">
</p>

<p align="center">
  <strong>Observe the survey, not only the observations.</strong>
</p>