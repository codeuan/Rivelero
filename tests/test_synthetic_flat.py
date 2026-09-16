from pathlib import Path

import numpy as np
import rasterio
from shapely.geometry import box

from rivelero.core.configuration import (
    ConfigurationType,
    ViewpointConfiguration,
)
from rivelero.core.domain import (
    AnalysisDomain,
    AnalysisGrid,
)
from rivelero.core.environment import (
    ElevationModel,
    Environment,
)
from rivelero.core.viewpoint import Viewpoint
from rivelero.observability.builder import (
    build_survey_observability_field,
)
from rivelero.observability.storage import VisibilityStore
from rivelero.visibility.configuration import (
    MissingMetadataPolicy,
    VisibilityConfiguration,
)


# ---------------------------------------------------------
# Synthetic data location
# ---------------------------------------------------------

PROJECT_ROOT = Path(
    r"C:\Users\zool2620\VISTA"
)

DATA_ROOT = (
    PROJECT_ROOT
    / "rivelero_synthetic_observability"
)

DEM_PATH = (
    DATA_ROOT
    / "environments"
    / "dem_flat.tif"
)


def main():

    print(
        "\n--- RIVELERO SYNTHETIC TEST: FLAT SINGLE ---\n"
    )

    # -----------------------------------------------------
    # 1. Load DEM metadata
    # -----------------------------------------------------

    with rasterio.open(DEM_PATH) as src:

        transform = src.transform
        crs = src.crs
        width = src.width
        height = src.height
        bounds = src.bounds

        dem = src.read(1)

        if src.nodata is not None:
            valid_mask = dem != src.nodata
        else:
            valid_mask = np.ones(
                (height, width),
                dtype=bool,
            )

    print("DEM loaded")
    print("DEM:", DEM_PATH)
    print("Shape:", (height, width))
    print("CRS:", crs)
    print("Bounds:", bounds)

    # -----------------------------------------------------
    # 2. Environment
    # -----------------------------------------------------

    elevation = ElevationModel(
        source=DEM_PATH,
        model_type="dem",
        crs=crs,
        resolution_m=10.0,
        source_name="Rivelero synthetic reference world",
    )

    environment = Environment(
        environment_id="synthetic_flat",
        name="Synthetic flat environment",
        elevation_model=elevation,
    )

    # -----------------------------------------------------
    # 3. Analysis grid
    # -----------------------------------------------------

    grid = AnalysisGrid(
        crs=crs,
        transform=transform,
        width=width,
        height=height,
    )

    geometry = box(
        bounds.left,
        bounds.bottom,
        bounds.right,
        bounds.top,
    )

    analysis_mask = np.ones(
        (height, width),
        dtype=bool,
    )

    domain = AnalysisDomain(
        domain_id="synthetic_domain",
        name="Synthetic analysis domain",
        geometry=geometry,
        crs=crs,
        grid=grid,
        analysis_mask=analysis_mask,
        valid_mask=valid_mask,
        creation_method="synthetic_reference_world",
    )

    # -----------------------------------------------------
    # 4. Viewpoint
    # -----------------------------------------------------

    viewpoint = Viewpoint(
        viewpoint_id="vp_center_360",
        x=500500.0,
        y=4099500.0,
        crs=crs,
        observer_height_m=1.75,

        # Deliberately unknown.
        heading_deg=None,

        horizontal_fov_deg=360.0,

        platform="synthetic",
        source="synthetic",
    )

    viewpoint_configuration = ViewpointConfiguration(
        configuration_id="flat_single",
        name="Flat single-viewpoint test",
        viewpoints=[viewpoint],
        configuration_type=ConfigurationType.SIMULATED,
    )

    # -----------------------------------------------------
    # 5. Visibility configuration
    # -----------------------------------------------------

    visibility_configuration = VisibilityConfiguration(
        configuration_id="flat_single_visibility",
        name="Flat single visibility",

        max_distance_m=200.0,

        default_observer_height_m=1.75,
        default_target_height_m=0.0,

        use_direction=True,

        missing_heading_policy=(
            MissingMetadataPolicy.OMNIDIRECTIONAL
        ),

        default_horizontal_fov_deg=360.0,

        curvature_coefficient=0.85714,
    )

    # -----------------------------------------------------
    # 6. Visibility cache
    # -----------------------------------------------------

    cache_directory = (
        DATA_ROOT
        / "runtime_cache"
        / "flat_single"
    )

    store = VisibilityStore(
        cache_directory=cache_directory,
        max_memory_items=4,
    )

    # -----------------------------------------------------
    # 7. Build SOF
    # -----------------------------------------------------

    print(
        "\nBuilding Survey Observability Field..."
    )

    result = build_survey_observability_field(
        sof_id="sof_flat_single",
        viewpoint_configuration=viewpoint_configuration,
        environment=environment,
        domain=domain,
        visibility_configuration=visibility_configuration,
        store=store,
    )

    sof = result.sof
    report = result.report

    # -----------------------------------------------------
    # 8. Results
    # -----------------------------------------------------

    print("\n--- BUILD REPORT ---")

    print("Requested:", report.requested_units)
    print("Added:", report.added_units)
    print("Excluded:", report.excluded_units)
    print("Failed:", report.failed_units)
    print("Computed:", report.computed_units)
    print("Cache hits:", report.cache_hits)

    print("\n--- SURVEY OBSERVABILITY FIELD ---")

    print(
        "Analysable cells:",
        sof.n_analysable_cells,
    )

    print(
        "Observable cells:",
        sof.n_observable_cells,
    )

    print(
        "Blind-spot cells:",
        sof.n_blindspot_cells,
    )

    print(
        "Maximum exposure:",
        int(sof.exposure_count.max()),
    )

    print(
        "Active sampling units:",
        sof.n_active_units,
    )

    # -----------------------------------------------------
    # 9. Assertions
    # -----------------------------------------------------

    assert report.requested_units == 1
    assert report.added_units == 1
    assert report.failed_units == 0

    assert sof.n_active_units == 1

    assert sof.exposure_count.shape == (
        height,
        width,
    )

    assert sof.exposure_count.max() == 1

    assert sof.n_observable_cells > 0

    assert sof.n_blindspot_cells > 0

    assert (
        sof.n_observable_cells
        + sof.n_blindspot_cells
        == sof.n_analysable_cells
    )

    print(
        "\nSUCCESS: flat_single synthetic test passed."
    )


if __name__ == "__main__":
    main()