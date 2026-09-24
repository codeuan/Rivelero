"""Common single-observer visibility engine for Rivelero.

This module connects Rivelero's canonical data model to the existing 2.5D
visibility implementation.

The engine resolves source metadata and analysis assumptions, computes
terrain-based geometric visibility, applies horizontal viewing-direction
constraints, and returns a standardized visibility result for one sampling
unit.

It deliberately does NOT aggregate multiple viewpoints, construct a
SurveyObservabilityField, or decide how viewpoint-cell relationships are
stored. Those responsibilities belong to the observability layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.warp import transform as transform_coordinates

from rivelero.core.domain import AnalysisDomain
from rivelero.core.environment import Environment
from rivelero.core.observation import ObservationEvent
from rivelero.core.sensor import Sensor
from rivelero.core.viewpoint import Viewpoint, geographic_coordinate_problem
from rivelero.visibility.configuration import (
    MissingMetadataPolicy,
    VisibilityBackend,
    VisibilityConfiguration,
)
from rivelero.visibility.directional import apply_horizontal_fov
from rivelero.visibility.viewshed import (
    accumulate_viewshed,
    open_viewshed_dem,
    run_viewshed,
)


@dataclass(frozen=True, slots=True)
class ResolvedVisibilityParameters:
    """Effective parameters used for one visibility calculation."""

    observer_height_m: float
    target_height_m: float

    heading_deg: float | None
    horizontal_fov_deg: float | None

    pitch_deg: float | None
    vertical_fov_deg: float | None

    max_distance_m: float
    curvature_coefficient: float

    used_default_observer_height: bool = False
    used_default_heading: bool = False
    used_default_horizontal_fov: bool = False

    omnidirectional: bool = False


@dataclass(slots=True)
class SingleViewpointVisibility:
    """Standard visibility result for one Viewpoint or ObservationEvent."""

    viewpoint_id: str

    visibility_mask: np.ndarray
    geometric_visibility_mask: np.ndarray
    valid_mask: np.ndarray

    transform: Affine
    crs: CRS

    resolved_parameters: ResolvedVisibilityParameters

    event_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "visibility_mask",
            "geometric_visibility_mask",
            "valid_mask",
        ):
            value = getattr(self, name)

            if not isinstance(value, np.ndarray):
                raise TypeError(f"{name} must be a numpy.ndarray.")

            if value.ndim != 2:
                raise ValueError(f"{name} must be two-dimensional.")

        shape = self.visibility_mask.shape

        if self.geometric_visibility_mask.shape != shape:
            raise ValueError(
                "geometric_visibility_mask shape does not match "
                "visibility_mask."
            )

        if self.valid_mask.shape != shape:
            raise ValueError(
                "valid_mask shape does not match visibility_mask."
            )

        self.visibility_mask = self.visibility_mask.astype(bool, copy=False)
        self.geometric_visibility_mask = (
            self.geometric_visibility_mask.astype(bool, copy=False)
        )
        self.valid_mask = self.valid_mask.astype(bool, copy=False)

    @property
    def visible_cell_count(self) -> int:
        return int(np.count_nonzero(self.visibility_mask))


class ViewpointExcludedError(RuntimeError):
    """Raised when missing-metadata policy excludes a sampling unit."""


def compute_viewpoint_visibility(
    *,
    viewpoint: Viewpoint,
    environment: Environment,
    domain: AnalysisDomain,
    configuration: VisibilityConfiguration,
    sensor: Sensor | None = None,
    event: ObservationEvent | None = None,
) -> SingleViewpointVisibility:
    """Compute effective visibility for one standardized observation unit."""

    _validate_relationships(
        viewpoint=viewpoint,
        sensor=sensor,
        event=event,
    )

    if configuration.backend != VisibilityBackend.GDAL:
        raise NotImplementedError(
            "The common Rivelero visibility engine currently supports "
            "only the GDAL backend."
        )

    if configuration.use_environment_obstacles:
        raise NotImplementedError(
            "Environment obstacle layers are represented by the Rivelero "
            "architecture but are not yet integrated into the common "
            "single-viewpoint visibility engine."
        )

    if configuration.use_vertical_fov:
        raise NotImplementedError(
            "Vertical FOV filtering is not yet implemented."
        )

    resolved = resolve_visibility_parameters(
        viewpoint=viewpoint,
        configuration=configuration,
        sensor=sensor,
        event=event,
    )

    observer_x, observer_y = _viewpoint_in_analysis_crs(
        viewpoint,
        domain.crs,
    )

    geometric_visibility = _compute_geometric_visibility(
        observer_x=observer_x,
        observer_y=observer_y,
        environment=environment,
        domain=domain,
        resolved=resolved,
    )

    analysable_mask = (
        domain.effective_analysis_mask
        & domain.effective_valid_mask
    )
    
    geometric_visibility &= analysable_mask

    if configuration.use_direction and not resolved.omnidirectional:
        if resolved.heading_deg is None:
            raise ValueError(
                "Directional visibility requires an effective heading."
            )

        if resolved.horizontal_fov_deg is None:
            raise ValueError(
                "Directional visibility requires an effective horizontal FOV."
            )

        effective_visibility = apply_horizontal_fov(
            geometric_visibility,
            observer_x=observer_x,
            observer_y=observer_y,
            transform=domain.grid.transform,
            heading_deg=resolved.heading_deg,
            horizontal_fov_deg=resolved.horizontal_fov_deg,
            valid_mask=analysable_mask,
        )
    else:
        effective_visibility = geometric_visibility.copy()

    return SingleViewpointVisibility(
        viewpoint_id=viewpoint.viewpoint_id,
        event_id=None if event is None else event.event_id,
        visibility_mask=effective_visibility,
        geometric_visibility_mask=geometric_visibility,
        valid_mask=analysable_mask,
        transform=domain.grid.transform,
        crs=domain.grid.crs,
        resolved_parameters=resolved,
        metadata={
            "environment_id": environment.environment_id,
            "analysis_domain_id": domain.domain_id,
            "visibility_configuration_id": configuration.configuration_id,
            "sensor_id": None if sensor is None else sensor.sensor_id,
        },
    )


def resolve_visibility_parameters(
    *,
    viewpoint: Viewpoint,
    configuration: VisibilityConfiguration,
    sensor: Sensor | None = None,
    event: ObservationEvent | None = None,
) -> ResolvedVisibilityParameters:
    """Resolve Event -> Viewpoint -> Sensor -> Configuration metadata."""

    # Observer height
    observer_height = (
        event.observer_height_m
        if event is not None and event.observer_height_m is not None
        else viewpoint.observer_height_m
    )

    used_default_height = False

    if observer_height is None:
        policy = configuration.missing_observer_height_policy

        if policy == MissingMetadataPolicy.USE_DEFAULT:
            observer_height = configuration.default_observer_height_m
            used_default_height = True
        elif policy == MissingMetadataPolicy.EXCLUDE:
            raise ViewpointExcludedError(
                f"{viewpoint.viewpoint_id!r}: observer height missing."
            )
        elif policy == MissingMetadataPolicy.ERROR:
            raise ValueError(
                f"{viewpoint.viewpoint_id!r}: observer height missing."
            )
        else:
            raise ValueError(
                "OMNIDIRECTIONAL is invalid for observer-height metadata."
            )

    if observer_height is None:
        raise ValueError("No effective observer height could be resolved.")

    # Heading
    heading = (
        event.heading_deg
        if event is not None and event.heading_deg is not None
        else viewpoint.heading_deg
    )

    used_default_heading = False
    omnidirectional = not configuration.use_direction

    if configuration.use_direction and heading is None:
        policy = configuration.missing_heading_policy

        if policy == MissingMetadataPolicy.USE_DEFAULT:
            heading = configuration.default_heading_deg
            used_default_heading = True
        elif policy == MissingMetadataPolicy.OMNIDIRECTIONAL:
            omnidirectional = True
        elif policy == MissingMetadataPolicy.EXCLUDE:
            raise ViewpointExcludedError(
                f"{viewpoint.viewpoint_id!r}: heading missing."
            )
        elif policy == MissingMetadataPolicy.ERROR:
            raise ValueError(
                f"{viewpoint.viewpoint_id!r}: heading missing."
            )

    if heading is not None:
        heading = float(heading) % 360.0

    # Horizontal FOV
    if (
        event is not None
        and event.horizontal_fov_deg is not None
    ):
        horizontal_fov = event.horizontal_fov_deg
    elif viewpoint.horizontal_fov_deg is not None:
        horizontal_fov = viewpoint.horizontal_fov_deg
    elif sensor is not None and sensor.horizontal_fov_deg is not None:
        horizontal_fov = sensor.horizontal_fov_deg
    else:
        horizontal_fov = None

    used_default_fov = False

    if (
        configuration.use_direction
        and not omnidirectional
        and horizontal_fov is None
    ):
        policy = configuration.missing_fov_policy

        if policy == MissingMetadataPolicy.USE_DEFAULT:
            horizontal_fov = configuration.default_horizontal_fov_deg
            used_default_fov = True
        elif policy == MissingMetadataPolicy.OMNIDIRECTIONAL:
            horizontal_fov = 360.0
            omnidirectional = True
        elif policy == MissingMetadataPolicy.EXCLUDE:
            raise ViewpointExcludedError(
                f"{viewpoint.viewpoint_id!r}: horizontal FOV missing."
            )
        elif policy == MissingMetadataPolicy.ERROR:
            raise ValueError(
                f"{viewpoint.viewpoint_id!r}: horizontal FOV missing."
            )

    if horizontal_fov is not None:
        horizontal_fov = float(horizontal_fov)

        if not 0.0 < horizontal_fov <= 360.0:
            raise ValueError(
                "Resolved horizontal FOV must be within (0, 360]."
            )

        if np.isclose(horizontal_fov, 360.0):
            omnidirectional = True

    # Pitch
    pitch = (
        event.pitch_deg
        if event is not None and event.pitch_deg is not None
        else viewpoint.pitch_deg
    )

    if pitch is None:
        pitch = configuration.default_pitch_deg

    # Vertical FOV
    if (
        event is not None
        and event.vertical_fov_deg is not None
    ):
        vertical_fov = event.vertical_fov_deg
    elif viewpoint.vertical_fov_deg is not None:
        vertical_fov = viewpoint.vertical_fov_deg
    elif sensor is not None and sensor.vertical_fov_deg is not None:
        vertical_fov = sensor.vertical_fov_deg
    else:
        vertical_fov = configuration.default_vertical_fov_deg

    return ResolvedVisibilityParameters(
        observer_height_m=float(observer_height),
        target_height_m=float(configuration.default_target_height_m),
        heading_deg=heading,
        horizontal_fov_deg=horizontal_fov,
        pitch_deg=None if pitch is None else float(pitch),
        vertical_fov_deg=(
            None if vertical_fov is None else float(vertical_fov)
        ),
        max_distance_m=float(configuration.max_distance_m),
        curvature_coefficient=float(
            configuration.curvature_coefficient
        ),
        used_default_observer_height=used_default_height,
        used_default_heading=used_default_heading,
        used_default_horizontal_fov=used_default_fov,
        omnidirectional=omnidirectional,
    )


def _compute_geometric_visibility(
    *,
    observer_x: float,
    observer_y: float,
    environment: Environment,
    domain: AnalysisDomain,
    resolved: ResolvedVisibilityParameters,
) -> np.ndarray:
    """Run one GDAL viewshed and place it on the full AnalysisGrid."""

    elevation_source = environment.elevation_model.source

    if not isinstance(elevation_source, (str, Path)):
        raise TypeError(
            "The GDAL backend currently requires a local elevation raster."
        )

    dem_path = Path(elevation_source)

    if not dem_path.is_file():
        raise FileNotFoundError(f"DEM does not exist: {dem_path}")

    with rasterio.open(dem_path) as source:
        if source.count < 1:
            raise ValueError("The elevation raster contains no bands.")

        coordinate_mode = (
            source.tags().get("coordinate_mode", "").strip().lower()
        )

        if coordinate_mode != "unreal_local":
            if source.crs is None:
                raise ValueError("The elevation raster has no CRS.")

            if not source.crs.is_projected:
                raise ValueError(
                    "The current 2.5D visibility engine requires a "
                    "projected elevation raster."
                )

        _validate_grid_matches_dem(source=source, domain=domain)

        dem_transform = source.transform
        dem_shape = (source.height, source.width)

        effective_curvature = (
            0.0
            if coordinate_mode == "unreal_local"
            else resolved.curvature_coefficient
        )

    dataset = open_viewshed_dem(
        dem_path,
        strip_crs=(coordinate_mode == "unreal_local"),
    )

    try:
        band = dataset.GetRasterBand(1)

        if band is None:
            raise RuntimeError("GDAL could not access DEM band 1.")

        viewshed = run_viewshed(
            band=band,
            observer_x=observer_x,
            observer_y=observer_y,
            observer_height_m=resolved.observer_height_m,
            target_height_m=resolved.target_height_m,
            max_distance_m=resolved.max_distance_m,
            curvature_coefficient=effective_curvature,
        )

        try:
            full_visibility = np.zeros(
                dem_shape,
                dtype=np.uint8,
            )

            accumulate_viewshed(
                accumulator=full_visibility,
                viewshed_dataset=viewshed,
                dem_transform=dem_transform,
            )
        finally:
            viewshed = None

    finally:
        dataset = None

    return full_visibility > 0


def _validate_grid_matches_dem(
    *,
    source: rasterio.io.DatasetReader,
    domain: AnalysisDomain,
) -> None:
    """Require the current AnalysisGrid to match the DEM exactly."""

    if source.crs != domain.grid.crs:
        raise ValueError(
            "AnalysisGrid CRS does not match the elevation raster."
        )

    if source.transform != domain.grid.transform:
        raise ValueError(
            "AnalysisGrid transform does not match the elevation raster."
        )

    if source.width != domain.grid.width:
        raise ValueError(
            "AnalysisGrid width does not match the elevation raster."
        )

    if source.height != domain.grid.height:
        raise ValueError(
            "AnalysisGrid height does not match the elevation raster."
        )


def _viewpoint_in_analysis_crs(
    viewpoint: Viewpoint,
    target_crs: CRS,
) -> tuple[float, float]:
    if viewpoint.crs == target_crs:
        return viewpoint.x, viewpoint.y

    advice = (
        "The Survey was probably imported with the wrong source CRS; "
        "re-import it with the CRS its coordinates were recorded in."
    )
    problem = geographic_coordinate_problem(viewpoint.x, viewpoint.y, viewpoint.crs)
    if problem is not None:
        raise ValueError(
            f"Viewpoint {viewpoint.viewpoint_id!r} cannot be placed on the terrain: "
            f"{problem} {advice}"
        )

    try:
        xs, ys = transform_coordinates(
            viewpoint.crs,
            target_crs,
            [viewpoint.x],
            [viewpoint.y],
        )
    except Exception as error:
        raise ValueError(
            f"Viewpoint {viewpoint.viewpoint_id!r}: coordinates ({viewpoint.x:g}, "
            f"{viewpoint.y:g}) in {viewpoint.crs.to_string()} could not be "
            f"transformed to {target_crs.to_string()} ({error}). {advice}"
        ) from error

    x, y = float(xs[0]), float(ys[0])
    if not (np.isfinite(x) and np.isfinite(y)):
        raise ValueError(
            f"Viewpoint {viewpoint.viewpoint_id!r}: coordinates ({viewpoint.x:g}, "
            f"{viewpoint.y:g}) in {viewpoint.crs.to_string()} have no position in "
            f"{target_crs.to_string()}. {advice}"
        )
    return x, y


def _validate_relationships(
    *,
    viewpoint: Viewpoint,
    sensor: Sensor | None,
    event: ObservationEvent | None,
) -> None:
    if sensor is not None:
        if not isinstance(sensor, Sensor):
            raise TypeError("sensor must be a Sensor or None.")

        if (
            viewpoint.sensor_id is not None
            and viewpoint.sensor_id != sensor.sensor_id
        ):
            raise ValueError(
                "The supplied Sensor does not match viewpoint.sensor_id."
            )

    if event is not None:
        if not isinstance(event, ObservationEvent):
            raise TypeError(
                "event must be an ObservationEvent or None."
            )

        if event.viewpoint_id != viewpoint.viewpoint_id:
            raise ValueError(
                "The ObservationEvent references a different Viewpoint."
            )