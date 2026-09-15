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
from rivelero.core.viewpoint import Viewpoint
from rivelero.visibility.configuration import (
    MissingMetadataPolicy,
    VisibilityConfiguration,
)
from rivelero.visibility.directional import apply_horizontal_fov
from rivelero.visibility.viewshed import (
    open_viewshed_dem,
    run_viewshed,
)


@dataclass(frozen=True, slots=True)
class ResolvedVisibilityParameters:
    """Effective parameters used for one visibility calculation.

    These values may originate from source metadata or from explicit defaults
    in VisibilityConfiguration. Keeping the resolved values in the result
    makes the assumptions used for each viewpoint auditable.
    """

    observer_height_m: float
    target_height_m: float

    heading_deg: float | None
    horizontal_fov_deg: float | None

    pitch_deg: float | None
    vertical_fov_deg: float | None

    max_distance_m: float | None

    used_default_observer_height: bool = False
    used_default_heading: bool = False
    used_default_horizontal_fov: bool = False

    omnidirectional: bool = False


@dataclass(slots=True)
class SingleViewpointVisibility:
    """Standard visibility result for one Rivelero sampling unit.

    Parameters
    ----------
    viewpoint_id
        Identifier of the Viewpoint used for the calculation.

    visibility_mask
        Final effective visibility mask after terrain, domain, and directional
        constraints have been applied.

    geometric_visibility_mask
        Terrain-based 2.5D visibility before directional filtering.

    valid_mask
        Cells that were valid and analysable for this calculation.

    transform, crs
        Spatial definition of the result raster.

    resolved_parameters
        Effective visibility parameters used by the engine.

    event_id
        ObservationEvent identifier when the calculation represents an event.

    metadata
        Additional processing/provenance information.
    """

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
        """Validate result arrays."""

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
                "geometric_visibility_mask must have the same shape as "
                "visibility_mask."
            )

        if self.valid_mask.shape != shape:
            raise ValueError(
                "valid_mask must have the same shape as visibility_mask."
            )

        self.visibility_mask = self.visibility_mask.astype(
            bool,
            copy=False,
        )
        self.geometric_visibility_mask = (
            self.geometric_visibility_mask.astype(bool, copy=False)
        )
        self.valid_mask = self.valid_mask.astype(bool, copy=False)

    @property
    def visible_cell_count(self) -> int:
        """Number of effectively visible cells."""

        return int(np.count_nonzero(self.visibility_mask))

    @property
    def geometric_visible_cell_count(self) -> int:
        """Number of cells visible before directional filtering."""

        return int(
            np.count_nonzero(self.geometric_visibility_mask)
        )


class ViewpointExcludedError(RuntimeError):
    """Raised when analysis policy excludes a Viewpoint from processing."""


def compute_viewpoint_visibility(
    *,
    viewpoint: Viewpoint,
    environment: Environment,
    domain: AnalysisDomain,
    configuration: VisibilityConfiguration,
    sensor: Sensor | None = None,
    event: ObservationEvent | None = None,
) -> SingleViewpointVisibility:
    """Compute effective 2.5D visibility for one Viewpoint or event.

    Parameters
    ----------
    viewpoint
        Spatial observer configuration.

    environment
        Physical Environment containing the elevation model.

    domain
        AnalysisDomain defining the output grid and analysable space.

    configuration
        Scientific assumptions controlling visibility.

    sensor
        Optional Sensor associated with the Viewpoint. When supplied, its
        metadata may provide fallback camera characteristics such as native
        field of view.

    event
        Optional ObservationEvent associated with the Viewpoint.

    Returns
    -------
    SingleViewpointVisibility
        Standardized visibility result suitable for later ingestion by a
        SurveyObservabilityField.

    Raises
    ------
    ViewpointExcludedError
        When a missing-metadata policy explicitly excludes the viewpoint.

    ValueError
        When required metadata cannot be resolved or spatial inputs are
        incompatible.
    """

    _validate_relationships(
        viewpoint=viewpoint,
        sensor=sensor,
        event=event,
    )

    _validate_domain_environment(
        environment=environment,
        domain=domain,
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
        configuration=configuration,
        resolved=resolved,
    )

    analysable_mask = domain.analysable_mask

    if analysable_mask is None:
        analysable_mask = np.ones(
            domain.grid.shape,
            dtype=bool,
        )
    else:
        analysable_mask = analysable_mask.astype(
            bool,
            copy=False,
        )

    geometric_visibility &= analysable_mask

    if (
        configuration.use_direction
        and not resolved.omnidirectional
    ):
        if resolved.heading_deg is None:
            raise ValueError(
                "Directional visibility was requested but no effective "
                "heading was resolved."
            )

        if resolved.horizontal_fov_deg is None:
            raise ValueError(
                "Directional visibility was requested but no effective "
                "horizontal FOV was resolved."
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

    if configuration.use_vertical_fov:
        raise NotImplementedError(
            "Vertical FOV filtering is represented in "
            "VisibilityConfiguration but is not yet implemented in the "
            "Rivelero 2.5D visibility engine."
        )

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
            "visibility_configuration_id": (
                configuration.configuration_id
            ),
            "sensor_id": (
                None if sensor is None else sensor.sensor_id
            ),
        },
    )


def resolve_visibility_parameters(
    *,
    viewpoint: Viewpoint,
    configuration: VisibilityConfiguration,
    sensor: Sensor | None = None,
    event: ObservationEvent | None = None,
) -> ResolvedVisibilityParameters:
    """Resolve source metadata and analysis assumptions for one calculation.

    The intended precedence is:

        ObservationEvent
            -> Viewpoint
            -> Sensor
            -> VisibilityConfiguration
            -> missing-metadata policy

    The current ObservationEvent model does not yet contain standardized
    orientation/FOV fields. Event-specific overrides can therefore be supplied
    through ``extra_metadata`` when explicitly present.

    No resolved default is written back into the original source objects.
    """

    event_metadata = (
        {}
        if event is None
        else event.extra_metadata
    )

    # --------------------------------------------------------------
    # Observer height
    # --------------------------------------------------------------

    event_height = _optional_numeric_metadata(
        event_metadata,
        "observer_height_m",
    )

    observer_height = (
        event_height
        if event_height is not None
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
                f"Viewpoint {viewpoint.viewpoint_id!r} excluded because "
                "observer height is missing."
            )

        elif policy == MissingMetadataPolicy.ERROR:
            raise ValueError(
                f"Viewpoint {viewpoint.viewpoint_id!r} has no observer "
                "height and the visibility policy is 'error'."
            )

        else:
            raise ValueError(
                "OMNIDIRECTIONAL is not a valid missing-observer-height "
                "policy."
            )

    if observer_height is None:
        raise ValueError(
            "No effective observer height could be resolved."
        )

    # --------------------------------------------------------------
    # Heading
    # --------------------------------------------------------------

    event_heading = _optional_numeric_metadata(
        event_metadata,
        "heading_deg",
    )

    heading = (
        event_heading
        if event_heading is not None
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
                f"Viewpoint {viewpoint.viewpoint_id!r} excluded because "
                "heading is missing."
            )

        elif policy == MissingMetadataPolicy.ERROR:
            raise ValueError(
                f"Viewpoint {viewpoint.viewpoint_id!r} has no heading "
                "and the visibility policy is 'error'."
            )

    if heading is not None:
        heading = float(heading) % 360.0

    # --------------------------------------------------------------
    # Horizontal FOV
    # --------------------------------------------------------------

    event_fov = _optional_numeric_metadata(
        event_metadata,
        "horizontal_fov_deg",
    )

    if event_fov is not None:
        horizontal_fov = event_fov
    elif viewpoint.horizontal_fov_deg is not None:
        horizontal_fov = viewpoint.horizontal_fov_deg
    elif (
        sensor is not None
        and sensor.horizontal_fov_deg is not None
    ):
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
            horizontal_fov = (
                configuration.default_horizontal_fov_deg
            )
            used_default_fov = True

        elif policy == MissingMetadataPolicy.OMNIDIRECTIONAL:
            omnidirectional = True
            horizontal_fov = 360.0

        elif policy == MissingMetadataPolicy.EXCLUDE:
            raise ViewpointExcludedError(
                f"Viewpoint {viewpoint.viewpoint_id!r} excluded because "
                "horizontal FOV is missing."
            )

        elif policy == MissingMetadataPolicy.ERROR:
            raise ValueError(
                f"Viewpoint {viewpoint.viewpoint_id!r} has no horizontal "
                "FOV and the visibility policy is 'error'."
            )

    if horizontal_fov is not None:
        horizontal_fov = float(horizontal_fov)

        if not 0.0 < horizontal_fov <= 360.0:
            raise ValueError(
                "Resolved horizontal FOV must be within (0, 360] degrees."
            )

        if np.isclose(horizontal_fov, 360.0):
            omnidirectional = True

    # --------------------------------------------------------------
    # Pitch / vertical FOV
    # --------------------------------------------------------------

    event_pitch = _optional_numeric_metadata(
        event_metadata,
        "pitch_deg",
    )

    pitch = (
        event_pitch
        if event_pitch is not None
        else viewpoint.pitch_deg
    )

    if pitch is None:
        pitch = configuration.default_pitch_deg

    event_vertical_fov = _optional_numeric_metadata(
        event_metadata,
        "vertical_fov_deg",
    )

    if event_vertical_fov is not None:
        vertical_fov = event_vertical_fov
    elif viewpoint.vertical_fov_deg is not None:
        vertical_fov = viewpoint.vertical_fov_deg
    elif (
        sensor is not None
        and sensor.vertical_fov_deg is not None
    ):
        vertical_fov = sensor.vertical_fov_deg
    else:
        vertical_fov = configuration.default_vertical_fov_deg

    return ResolvedVisibilityParameters(
        observer_height_m=float(observer_height),
        target_height_m=float(
            configuration.default_target_height_m
        ),
        heading_deg=heading,
        horizontal_fov_deg=horizontal_fov,
        pitch_deg=None if pitch is None else float(pitch),
        vertical_fov_deg=(
            None
            if vertical_fov is None
            else float(vertical_fov)
        ),
        max_distance_m=configuration.max_distance_m,
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
    configuration: VisibilityConfiguration,
    resolved: ResolvedVisibilityParameters,
) -> np.ndarray:
    """Run the existing Rivelero/GDAL 2.5D visibility calculation.

    This function is intentionally the only adapter between the new common
    engine and the existing low-level viewshed API. If the exact signature of
    ``run_viewshed`` changes, adapt this function rather than coupling the
    rest of the engine to GDAL details.
    """

    elevation_source = environment.elevation_model.source

    if not isinstance(elevation_source, (str, Path)):
        raise TypeError(
            "The current GDAL visibility backend requires the Environment "
            "elevation model to reference a local raster path."
        )

    elevation_path = Path(elevation_source)

    if not elevation_path.exists():
        raise FileNotFoundError(
            f"Elevation raster does not exist: {elevation_path}"
        )

    # Open the elevation raster through Rivelero's common low-level API.
    dem = open_viewshed_dem(elevation_path)

    try:
        viewshed = run_viewshed(
            dem,
            observer_x=observer_x,
            observer_y=observer_y,
            observer_height_m=resolved.observer_height_m,
            target_height_m=resolved.target_height_m,
            max_distance_m=resolved.max_distance_m,
            use_earth_curvature=configuration.use_earth_curvature,
            refraction_coefficient=configuration.refraction_coefficient,
        )
    finally:
        close_method = getattr(dem, "close", None)
        if callable(close_method):
            close_method()

    visibility = _extract_visibility_array(viewshed)

    if visibility.shape != domain.grid.shape:
        visibility = _align_visibility_to_domain(
            visibility=visibility,
            source=viewshed,
            domain=domain,
        )

    return visibility.astype(bool, copy=False)


def _extract_visibility_array(result: Any) -> np.ndarray:
    """Extract a 2D visibility array from the low-level backend result."""

    if isinstance(result, np.ndarray):
        array = result
    elif hasattr(result, "ReadAsArray"):
        array = result.ReadAsArray()
    elif hasattr(result, "read"):
        try:
            array = result.read(1)
        except TypeError:
            array = result.read()
    elif hasattr(result, "array"):
        array = result.array
    else:
        raise TypeError(
            "Unsupported viewshed result type. The low-level visibility "
            "backend must return an array or raster-like object."
        )

    array = np.asarray(array)

    if array.ndim != 2:
        raise ValueError(
            "The low-level viewshed result must be two-dimensional."
        )

    return array > 0


def _align_visibility_to_domain(
    *,
    visibility: np.ndarray,
    source: Any,
    domain: AnalysisDomain,
) -> np.ndarray:
    """Handle visibility rasters that do not match the AnalysisGrid.

    The current common engine requires the low-level viewshed to be aligned
    with the AnalysisDomain grid. Automatic reprojection is deliberately not
    performed here because silently resampling a binary visibility product
    would introduce an analytical decision.

    This function exists as an explicit boundary for future grid-alignment
    support.
    """

    raise ValueError(
        "Viewshed output shape does not match the AnalysisDomain grid. "
        f"Viewshed shape={visibility.shape}, "
        f"domain shape={domain.grid.shape}. "
        "The visibility backend and AnalysisDomain must currently use the "
        "same raster grid."
    )


def _viewpoint_in_analysis_crs(
    viewpoint: Viewpoint,
    target_crs: CRS,
) -> tuple[float, float]:
    """Return Viewpoint coordinates in the AnalysisDomain CRS."""

    if viewpoint.crs == target_crs:
        return viewpoint.x, viewpoint.y

    xs, ys = transform_coordinates(
        viewpoint.crs,
        target_crs,
        [viewpoint.x],
        [viewpoint.y],
    )

    return float(xs[0]), float(ys[0])


def _validate_relationships(
    *,
    viewpoint: Viewpoint,
    sensor: Sensor | None,
    event: ObservationEvent | None,
) -> None:
    """Validate relationships among the observation-side objects."""

    if sensor is not None:
        if not isinstance(sensor, Sensor):
            raise TypeError("sensor must be a Sensor or None.")

        if (
            viewpoint.sensor_id is not None
            and viewpoint.sensor_id != sensor.sensor_id
        ):
            raise ValueError(
                f"Viewpoint {viewpoint.viewpoint_id!r} references Sensor "
                f"{viewpoint.sensor_id!r}, but Sensor {sensor.sensor_id!r} "
                "was supplied to the visibility engine."
            )

    if event is not None:
        if not isinstance(event, ObservationEvent):
            raise TypeError(
                "event must be an ObservationEvent or None."
            )

        if event.viewpoint_id != viewpoint.viewpoint_id:
            raise ValueError(
                f"ObservationEvent {event.event_id!r} references Viewpoint "
                f"{event.viewpoint_id!r}, but Viewpoint "
                f"{viewpoint.viewpoint_id!r} was supplied."
            )


def _validate_domain_environment(
    *,
    environment: Environment,
    domain: AnalysisDomain,
) -> None:
    """Validate basic spatial compatibility before processing."""

    if not isinstance(environment, Environment):
        raise TypeError("environment must be an Environment.")

    if not isinstance(domain, AnalysisDomain):
        raise TypeError("domain must be an AnalysisDomain.")

    elevation_crs = environment.elevation_model.crs

    if (
        elevation_crs is not None
        and elevation_crs != domain.grid.crs
    ):
        raise ValueError(
            "The current visibility engine requires the Environment "
            "elevation model and AnalysisGrid to use the same CRS. "
            "Reprojection should occur during environment/domain "
            "preparation rather than silently inside the visibility engine."
        )


def _optional_numeric_metadata(
    metadata: dict[str, Any],
    key: str,
) -> float | None:
    """Read an optional numeric event-level metadata override."""

    if key not in metadata or metadata[key] is None:
        return None

    value = metadata[key]

    if not isinstance(value, (int, float)):
        raise TypeError(
            f"ObservationEvent metadata {key!r} must be numeric when "
            "provided."
        )

    result = float(value)

    if not np.isfinite(result):
        raise ValueError(
            f"ObservationEvent metadata {key!r} must be finite."
        )

    return result