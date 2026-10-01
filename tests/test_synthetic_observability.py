from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
import rasterio
from shapely.geometry import box

from rivelero.core.configuration import ConfigurationType, ViewpointConfiguration
from rivelero.core.domain import AnalysisDomain, AnalysisGrid
from rivelero.core.environment import ElevationModel, Environment
from rivelero.core.observation import ObservationEvent
from rivelero.core.sensor import Sensor
from rivelero.core.viewpoint import Viewpoint
from rivelero.observability.builder import (
    build_survey_observability_field,
    make_visibility_key,
)
from rivelero.observability.storage import VisibilityStore
from rivelero.observability.survey_field import SurveyObservabilityField
from rivelero.visibility.configuration import (
    MissingMetadataPolicy,
    SamplingUnit,
    VisibilityConfiguration,
)
from rivelero.visibility.engine import (
    compute_viewpoint_visibility,
    resolve_visibility_parameters,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "rivelero_synthetic_observability"
VIEWPOINTS_PATH = DATA_ROOT / "viewpoints" / "viewpoints.csv"
SENSORS_PATH = DATA_ROOT / "viewpoints" / "sensors.csv"
EVENTS_PATH = DATA_ROOT / "viewpoints" / "observation_events.csv"


def _optional_float(value: str) -> float | None:
    return None if value == "" else float(value)


@pytest.fixture(scope="module")
def viewpoints() -> dict[str, Viewpoint]:
    with VIEWPOINTS_PATH.open(newline="", encoding="utf-8") as handle:
        return {
            row["viewpoint_id"]: Viewpoint(
                viewpoint_id=row["viewpoint_id"],
                x=float(row["x"]),
                y=float(row["y"]),
                crs=row["crs"],
                observer_height_m=_optional_float(row["observer_height_m"]),
                heading_deg=_optional_float(row["heading_deg"]),
                horizontal_fov_deg=_optional_float(row["horizontal_fov_deg"]),
                sensor_id=row["sensor_id"] or None,
                platform=row["platform"] or None,
                source=row["source"] or None,
            )
            for row in csv.DictReader(handle)
        }


@pytest.fixture(scope="module")
def sensors() -> dict[str, Sensor]:
    with SENSORS_PATH.open(newline="", encoding="utf-8") as handle:
        return {
            row["sensor_id"]: Sensor(
                sensor_id=row["sensor_id"],
                modality=row["modality"],
                horizontal_fov_deg=_optional_float(row["horizontal_fov_deg"]),
                model=row["description"] or None,
                source="synthetic",
            )
            for row in csv.DictReader(handle)
        }


@pytest.fixture(scope="module")
def events() -> dict[str, ObservationEvent]:
    with EVENTS_PATH.open(newline="", encoding="utf-8") as handle:
        return {
            row["event_id"]: ObservationEvent(
                event_id=row["event_id"],
                viewpoint_id=row["viewpoint_id"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                sequence_id=row["sequence_id"] or None,
                sequence_index=int(row["sequence_index"]),
                image_id=row["image_id"] or None,
                heading_deg=_optional_float(row["heading_deg"]),
                horizontal_fov_deg=_optional_float(row["horizontal_fov_deg"]),
            )
            for row in csv.DictReader(handle)
        }


@dataclass(frozen=True)
class AnalysisContext:
    dem_path: Path
    environment: Environment
    domain: AnalysisDomain
    crs: Any
    transform: Any
    shape: tuple[int, int]


@pytest.fixture
def make_context(tmp_path: Path) -> Callable[..., AnalysisContext]:
    def factory(dem_name: str, environment_id: str | None = None) -> AnalysisContext:
        dem_path = DATA_ROOT / "environments" / dem_name
        with rasterio.open(dem_path) as source:
            dem = source.read(1)
            nodata = source.nodata
            if nodata is None:
                valid_mask = np.ones(dem.shape, dtype=bool)
            elif np.isnan(nodata):
                valid_mask = ~np.isnan(dem)
            else:
                valid_mask = dem != nodata
            crs = source.crs
            transform = source.transform
            bounds = source.bounds
            width, height = source.width, source.height

        elevation = ElevationModel(
            source=dem_path,
            model_type="dem",
            crs=crs,
            resolution_m=10.0,
            nodata_value=nodata,
            source_name="Rivelero synthetic reference world",
        )
        environment = Environment(
            environment_id=environment_id or dem_path.stem,
            name=f"Synthetic {dem_path.stem} environment",
            elevation_model=elevation,
        )
        grid = AnalysisGrid(
            crs=crs,
            transform=transform,
            width=width,
            height=height,
        )
        domain = AnalysisDomain(
            domain_id="synthetic_domain",
            name="Synthetic analysis domain",
            geometry=box(bounds.left, bounds.bottom, bounds.right, bounds.top),
            crs=crs,
            grid=grid,
            analysis_mask=np.ones((height, width), dtype=bool),
            valid_mask=valid_mask,
            creation_method="synthetic_reference_world",
        )
        return AnalysisContext(
            dem_path=dem_path,
            environment=environment,
            domain=domain,
            crs=crs,
            transform=transform,
            shape=(height, width),
        )

    return factory


def _viewpoint_configuration(
    selected: list[Viewpoint],
    observation_events: list[ObservationEvent] | None = None,
    configuration_id: str = "synthetic_viewpoints",
) -> ViewpointConfiguration:
    return ViewpointConfiguration(
        configuration_id=configuration_id,
        name="Synthetic viewpoint configuration",
        viewpoints=selected,
        observation_events=observation_events or [],
        configuration_type=ConfigurationType.SIMULATED,
    )


def _visibility_configuration(
    configuration_id: str,
    max_distance_m: float,
    *,
    sampling_unit: SamplingUnit = SamplingUnit.VIEWPOINT,
    **overrides: Any,
) -> VisibilityConfiguration:
    parameters = {
        "default_observer_height_m": 1.75,
        "default_target_height_m": 0.0,
        "default_horizontal_fov_deg": 360.0,
        "sampling_unit": sampling_unit,
        **overrides,
    }
    return VisibilityConfiguration(
        configuration_id=configuration_id,
        name=f"Synthetic visibility {configuration_id}",
        max_distance_m=max_distance_m,
        **parameters,
    )


def _build(
    *,
    context: AnalysisContext,
    viewpoints: dict[str, Viewpoint],
    sensors: dict[str, Sensor],
    tmp_path: Path,
    ids: list[str],
    max_distance_m: float,
    configuration_id: str,
    visibility_configuration: VisibilityConfiguration | None = None,
    observation_events: list[ObservationEvent] | None = None,
    store: VisibilityStore | None = None,
) -> tuple[SurveyObservabilityField, Any, VisibilityStore, VisibilityConfiguration]:
    viewpoint_configuration = _viewpoint_configuration(
        [viewpoints[viewpoint_id] for viewpoint_id in ids],
        observation_events,
        configuration_id=f"{configuration_id}_viewpoints",
    )
    visibility_configuration = visibility_configuration or _visibility_configuration(
        configuration_id,
        max_distance_m,
    )
    store = store or VisibilityStore(tmp_path / configuration_id)
    result = build_survey_observability_field(
        sof_id=f"sof_{configuration_id}",
        viewpoint_configuration=viewpoint_configuration,
        environment=context.environment,
        domain=context.domain,
        visibility_configuration=visibility_configuration,
        store=store,
        sensors=sensors,
    )
    return result.sof, result.report, store, visibility_configuration


def _direct_visibility(
    context: AnalysisContext,
    viewpoint: Viewpoint,
    configuration: VisibilityConfiguration,
    sensors: dict[str, Sensor],
    event: ObservationEvent | None = None,
):
    sensor = sensors.get(viewpoint.sensor_id) if viewpoint.sensor_id else None
    return compute_viewpoint_visibility(
        viewpoint=viewpoint,
        event=event,
        sensor=sensor,
        environment=context.environment,
        domain=context.domain,
        configuration=configuration,
    )


def _x_grid(context: AnalysisContext) -> np.ndarray:
    rows, columns = np.indices(context.shape)
    return context.transform.c + (columns + 0.5) * context.transform.a


def test_flat_single(make_context, viewpoints, sensors, tmp_path):
    context = make_context("dem_flat.tif")
    sof, report, _, _ = _build(
        context=context,
        viewpoints=viewpoints,
        sensors=sensors,
        tmp_path=tmp_path,
        ids=["vp_center_360"],
        max_distance_m=200,
        configuration_id="flat_single",
    )

    assert report.added_units == 1
    assert report.failed_units == 0
    assert sof.exposure_count.max() == 1
    assert sof.n_observable_cells > 0
    assert sof.n_blindspot_cells > 0
    assert sof.n_observable_cells + sof.n_blindspot_cells == sof.n_analysable_cells


def test_directional_east(make_context, viewpoints, sensors, tmp_path):
    context = make_context("dem_flat.tif")
    configuration = _visibility_configuration(
        "directional_east", 250,
        default_horizontal_fov_deg=360.0,
    )
    effective = _direct_visibility(
        context, viewpoints["vp_center_east"], configuration, sensors
    )
    omnidirectional = _direct_visibility(
        context,
        viewpoints["vp_center_360"],
        _visibility_configuration("directional_reference", 250),
        sensors,
    )

    assert np.all(~effective.visibility_mask | omnidirectional.visibility_mask)
    assert np.count_nonzero(effective.visibility_mask) < np.count_nonzero(
        omnidirectional.visibility_mask
    )
    east_of_observer = _x_grid(context) > viewpoints["vp_center_east"].x
    visible_east = np.count_nonzero(effective.visibility_mask & east_of_observer)
    assert visible_east / np.count_nonzero(effective.visibility_mask) > 0.75


def test_overlap(make_context, viewpoints, sensors, tmp_path):
    sof, _, _, _ = _build(
        context=make_context("dem_flat.tif"),
        viewpoints=viewpoints,
        sensors=sensors,
        tmp_path=tmp_path,
        ids=["vp_overlap_west", "vp_overlap_east"],
        max_distance_m=300,
        configuration_id="overlap",
    )

    assert sof.n_active_units == 2
    assert sof.exposure_count.max() == 2
    assert np.count_nonzero(sof.exposure_count == 1) > 0
    assert np.count_nonzero(sof.exposure_count == 2) > 0


def test_ridge_occlusion(make_context, viewpoints, sensors):
    flat = make_context("dem_flat.tif", "flat_for_ridge")
    ridge = make_context("dem_ridge.tif", "ridge_for_ridge")
    configuration = _visibility_configuration("ridge", 600)
    flat_visibility = _direct_visibility(
        flat, viewpoints["vp_west_ridge"], configuration, sensors
    )
    ridge_visibility = _direct_visibility(
        ridge, viewpoints["vp_west_ridge"], configuration, sensors
    )
    behind_ridge = _x_grid(flat) > viewpoints["vp_west_ridge"].x + 100
    flat_behind = flat_visibility.visibility_mask & behind_ridge
    ridge_behind = ridge_visibility.visibility_mask & behind_ridge

    assert np.count_nonzero(flat_behind & ~ridge_behind) > 0
    assert np.count_nonzero(ridge_behind) < np.count_nonzero(flat_behind)


def test_nodata_states(make_context, viewpoints, sensors, tmp_path):
    context = make_context("dem_nodata.tif")
    sof, _, _, _ = _build(
        context=context,
        viewpoints=viewpoints,
        sensors=sensors,
        tmp_path=tmp_path,
        ids=["vp_center_360"],
        max_distance_m=300,
        configuration_id="nodata_states",
    )
    invalid = sof.analysis_mask & ~sof.valid_mask

    assert np.count_nonzero(invalid) > 0
    assert np.all(sof.observability_state[invalid] == 1)
    assert np.count_nonzero(sof.observability_state == 2) > 0


def test_metadata_fallback(make_context, viewpoints, sensors):
    context = make_context("dem_flat.tif")
    missing_heading = viewpoints["vp_missing_heading"]
    heading_before = missing_heading.heading_deg
    omni_configuration = _visibility_configuration(
        "metadata_omni", 250,
        missing_heading_policy=MissingMetadataPolicy.OMNIDIRECTIONAL,
    )
    resolved_omni = resolve_visibility_parameters(
        viewpoint=missing_heading,
        sensor=sensors[missing_heading.sensor_id],
        configuration=omni_configuration,
    )
    resolved_sensor_fov = resolve_visibility_parameters(
        viewpoint=missing_heading,
        sensor=sensors[missing_heading.sensor_id],
        configuration=_visibility_configuration(
            "metadata_sensor", 250,
            missing_heading_policy=MissingMetadataPolicy.USE_DEFAULT,
            default_heading_deg=90,
        ),
    )
    resolved_height = resolve_visibility_parameters(
        viewpoint=viewpoints["vp_missing_height"],
        sensor=sensors["sensor_rgb_90"],
        configuration=_visibility_configuration("metadata_height", 250),
    )

    assert resolved_omni.omnidirectional
    assert resolved_omni.heading_deg is None
    assert resolved_omni.horizontal_fov_deg == 70
    assert missing_heading.heading_deg == heading_before
    assert resolved_sensor_fov.heading_deg == 90
    assert resolved_sensor_fov.horizontal_fov_deg == 70
    assert resolved_sensor_fov.used_default_heading
    assert resolved_height.observer_height_m == 1.75
    assert resolved_height.used_default_observer_height


def test_temporal_events(make_context, viewpoints, sensors, events, tmp_path):
    selected_events = [events[event_id] for event_id in ("event_001", "event_002", "event_003", "event_004")]
    sof, report, store, configuration = _build(
        context=make_context("dem_flat.tif"),
        viewpoints=viewpoints,
        sensors=sensors,
        tmp_path=tmp_path,
        ids=["vp_redundant_a", "vp_overlap_east", "vp_complementary"],
        max_distance_m=300,
        configuration_id="temporal_events",
        visibility_configuration=_visibility_configuration(
            "temporal_events", 300,
            sampling_unit=SamplingUnit.OBSERVATION_EVENT,
        ),
        observation_events=selected_events,
    )

    assert report.requested_units == 4
    assert report.added_units == 4
    assert sof.active_sampling_unit_ids == ("event_001", "event_002", "event_003", "event_004")
    assert sof.active_viewpoint_ids.count("vp_redundant_a") == 2
    event_003 = events["event_003"]
    expected = _direct_visibility(
        make_context("dem_flat.tif"),
        viewpoints[event_003.viewpoint_id],
        configuration,
        sensors,
        event_003,
    )
    key = make_visibility_key(
        viewpoint=viewpoints[event_003.viewpoint_id],
        sensor=sensors[viewpoints[event_003.viewpoint_id].sensor_id],
        event=event_003,
        environment=make_context("dem_flat.tif").environment,
        domain=make_context("dem_flat.tif").domain,
        visibility_configuration=configuration,
    )
    stored = store.get(key)
    assert stored is not None
    assert np.array_equal(stored.visibility_mask, expected.visibility_mask)
    assert expected.resolved_parameters.heading_deg == 180
    assert expected.resolved_parameters.horizontal_fov_deg == 90


def test_redundancy_and_marginal_gain(make_context, viewpoints, sensors, tmp_path):
    context = make_context("dem_flat.tif")
    configuration = _visibility_configuration("redundancy", 300)
    base, _, store, _ = _build(
        context=context,
        viewpoints=viewpoints,
        sensors=sensors,
        tmp_path=tmp_path,
        ids=["vp_redundant_a", "vp_redundant_b"],
        max_distance_m=300,
        configuration_id="redundancy",
        visibility_configuration=configuration,
    )
    all_sof, _, _, _ = _build(
        context=context,
        viewpoints=viewpoints,
        sensors=sensors,
        tmp_path=tmp_path,
        ids=["vp_redundant_a", "vp_redundant_b", "vp_complementary"],
        max_distance_m=300,
        configuration_id="redundancy_all",
        visibility_configuration=configuration,
        store=store,
    )
    keys = {
        viewpoint_id: make_visibility_key(
            viewpoint=viewpoints[viewpoint_id],
            sensor=sensors[viewpoints[viewpoint_id].sensor_id],
            environment=context.environment,
            domain=context.domain,
            visibility_configuration=configuration,
        )
        for viewpoint_id in ("vp_redundant_a", "vp_redundant_b", "vp_complementary")
    }
    visibility = {viewpoint_id: store.get(key) for viewpoint_id, key in keys.items()}
    assert all(value is not None for value in visibility.values())
    mask_a = visibility["vp_redundant_a"].visibility_mask
    mask_b = visibility["vp_redundant_b"].visibility_mask
    overlap = np.count_nonzero(mask_a & mask_b) / min(np.count_nonzero(mask_a), np.count_nonzero(mask_b))
    assert overlap > 0.5

    original = base.exposure_count.copy()
    added = base.exposure_if_added(visibility["vp_complementary"])
    removed = base.exposure_if_removed(visibility["vp_redundant_a"])
    assert np.array_equal(base.exposure_count, original)
    assert np.count_nonzero(added > original) > 0
    assert np.array_equal(base.exposure_count, original)
    assert np.all(removed <= original)
    assert all_sof.n_observable_cells > base.n_observable_cells


def test_visibility_store_disk_cache(make_context, viewpoints, sensors, tmp_path):
    context = make_context("dem_flat.tif")
    configuration = _visibility_configuration("disk_cache", 200)
    viewpoint = viewpoints["vp_center_360"]
    key = make_visibility_key(
        viewpoint=viewpoint,
        sensor=sensors[viewpoint.sensor_id],
        environment=context.environment,
        domain=context.domain,
        visibility_configuration=configuration,
    )
    store = VisibilityStore(tmp_path / "disk_cache")
    compute = lambda: _direct_visibility(context, viewpoint, configuration, sensors)
    store.get_or_compute(key, compute)

    assert store.path_for(key).is_file()
    assert store.disk_item_count == 1
    store.clear_memory()
    cached = store.get_or_compute(key, lambda: pytest.fail("disk cache was not reused"))
    assert cached.key == key
    assert store.memory_item_count == 1


def test_visibility_store_lru(make_context, viewpoints, sensors, tmp_path):
    context = make_context("dem_flat.tif")
    configuration = _visibility_configuration("lru", 300)
    store = VisibilityStore(tmp_path / "lru", max_memory_items=2)
    keys = []
    for viewpoint_id in ("vp_center_360", "vp_overlap_west", "vp_overlap_east"):
        viewpoint = viewpoints[viewpoint_id]
        key = make_visibility_key(
            viewpoint=viewpoint,
            sensor=sensors[viewpoint.sensor_id],
            environment=context.environment,
            domain=context.domain,
            visibility_configuration=configuration,
        )
        keys.append(key)
        store.get_or_compute(
            key,
            lambda viewpoint=viewpoint: _direct_visibility(
                context, viewpoint, configuration, sensors
            ),
        )

    assert store.memory_item_count == 2
    assert store.disk_item_count == 3
    assert store.get(keys[0]) is not None
    assert store.memory_item_count == 2