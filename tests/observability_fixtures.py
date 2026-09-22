"""Shared synthetic fixtures for the Observability GUI tests (O2-O4)."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

from rivelero.gui.application_state import ApplicationState
from rivelero.gui.domain_service import (
    build_domain_from_grid_extent,
    build_domain_from_survey_extent,
    elevation_valid_mask,
)
from rivelero.gui.environment_import import import_elevation_environment
from rivelero.gui.observability_service import create_visibility_store
from rivelero.gui.survey_import import SurveyImportOptions, import_survey_csv
from rivelero.visibility.configuration import (
    SamplingUnit,
    VisibilityConfiguration,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "rivelero_synthetic_observability"
DEM_FLAT = DATA_ROOT / "environments" / "dem_flat.tif"
DEM_NODATA = DATA_ROOT / "environments" / "dem_nodata.tif"
VIEWPOINTS_PATH = DATA_ROOT / "viewpoints" / "viewpoints.csv"
SENSORS_PATH = DATA_ROOT / "viewpoints" / "sensors.csv"
EVENTS_PATH = DATA_ROOT / "viewpoints" / "observation_events.csv"

DEFAULT_VIEWPOINTS = ("vp_center_360", "vp_overlap_west", "vp_overlap_east")


def visibility_configuration(
    *,
    configuration_id: str = "visibility-1",
    max_distance_m: float = 150.0,
    sampling_unit: SamplingUnit = SamplingUnit.VIEWPOINT,
    **overrides,
) -> VisibilityConfiguration:
    return VisibilityConfiguration(
        configuration_id=configuration_id,
        name="Test visibility",
        max_distance_m=max_distance_m,
        sampling_unit=sampling_unit,
        **overrides,
    )


def make_ready_state(
    cache_directory: Path,
    *,
    viewpoint_ids: tuple[str, ...] | None = DEFAULT_VIEWPOINTS,
    configuration: VisibilityConfiguration | None = None,
    with_store: bool = True,
    dem: Path = DEM_FLAT,
    survey_buffer_m: float | None = None,
) -> ApplicationState:
    """Survey + World + VisibilityConfiguration + VisibilityStore."""

    imported = import_survey_csv(
        viewpoints_path=VIEWPOINTS_PATH,
        sensors_path=SENSORS_PATH,
        observation_events_path=EVENTS_PATH,
        configuration_id="synthetic-survey",
        configuration_name="Synthetic survey",
        options=SurveyImportOptions(
            source_crs="EPSG:32633",
            target_crs="EPSG:32633",
            strict=True,
            allow_duplicate_coordinates=True,
        ),
    )

    survey = imported.viewpoint_configuration
    if viewpoint_ids is not None:
        survey = replace(
            survey,
            viewpoints=[
                viewpoint
                for viewpoint in survey.viewpoints
                if viewpoint.viewpoint_id in viewpoint_ids
            ],
            observation_events=[
                event
                for event in survey.observation_events
                if event.viewpoint_id in viewpoint_ids
            ],
        )

    state = ApplicationState()
    state.set_viewpoint_configuration(survey)
    state.set_sensors(imported.sensors)

    environment = import_elevation_environment(
        dem,
        environment_id=f"{dem.stem}-environment",
        name="Synthetic terrain",
    )
    state.set_environment(environment.environment, grid=environment.grid)
    valid_mask = elevation_valid_mask(dem, environment.grid)
    if survey_buffer_m is None:
        domain = build_domain_from_grid_extent(
            grid=environment.grid,
            domain_id=f"{dem.stem}-domain",
            name="Entire terrain",
            valid_mask=valid_mask,
        ).domain
    else:
        domain = build_domain_from_survey_extent(
            viewpoints=survey.viewpoints,
            grid=environment.grid,
            domain_id=f"{dem.stem}-survey-domain",
            name="Buffered survey extent",
            buffer_m=survey_buffer_m,
            valid_mask=valid_mask,
        ).domain
    state.set_analysis_domain(domain)

    state.set_visibility_configuration(
        configuration or visibility_configuration()
    )

    if with_store:
        state.set_visibility_store(create_visibility_store(cache_directory))

    return state


def wait_for(predicate, *, timeout: float = 60.0) -> None:
    """Pump the Qt event loop until ``predicate`` is true."""

    from PySide6.QtWidgets import QApplication

    deadline = time.monotonic() + timeout
    while not predicate():
        QApplication.processEvents()
        if time.monotonic() > deadline:
            raise AssertionError("Timed out waiting for condition.")
        time.sleep(0.005)
    QApplication.processEvents()
