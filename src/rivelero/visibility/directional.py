"""Directional visibility utilities for Rivelero.

This module applies viewing-direction constraints to geometric visibility
rasters.

The terrain/viewshed calculation determines whether a line of sight exists
between an observer and a spatial cell. Directional filtering determines
whether that visible cell also lies within the observer's viewing direction
and horizontal field of view.

The module is deliberately independent of Viewpoint, Sensor, and
VisibilityConfiguration. Metadata resolution and missing-value policies are
handled by the higher-level visibility engine.
"""

from __future__ import annotations

import numpy as np
from affine import Affine


def apply_horizontal_fov(
    visibility_mask: np.ndarray,
    *,
    observer_x: float,
    observer_y: float,
    transform: Affine,
    heading_deg: float,
    horizontal_fov_deg: float,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Restrict a geometric visibility mask to a horizontal viewing cone.

    Parameters
    ----------
    visibility_mask
        Two-dimensional boolean or binary raster describing geometric
        visibility before directional filtering.

    observer_x, observer_y
        Observer coordinates expressed in the same coordinate reference
        system as ``transform``.

    transform
        Affine transform of the visibility raster.

    heading_deg
        Horizontal viewing direction in compass degrees:

        - 0 degrees = north
        - 90 degrees = east
        - 180 degrees = south
        - 270 degrees = west

        Values outside [0, 360) are normalized automatically.

    horizontal_fov_deg
        Full horizontal field of view in degrees. For example, a 90-degree
        FOV includes locations within 45 degrees on either side of the
        heading.

        A value of 360 degrees represents omnidirectional horizontal
        visibility.

    valid_mask
        Optional boolean mask identifying cells that are valid for analysis.
        Invalid cells are excluded from the result.

    Returns
    -------
    numpy.ndarray
        Boolean raster with the same shape as ``visibility_mask``. A cell is
        True only when it is geometrically visible, falls inside the
        horizontal viewing cone, and, when supplied, is valid.

    Notes
    -----
    This function does not calculate terrain visibility. It only filters an
    existing geometric visibility result.

    The observer cell itself has undefined bearing. If it is marked visible
    in the input mask, it is retained.
    """

    visibility = _as_boolean_mask(
        "visibility_mask",
        visibility_mask,
    )

    _validate_transform(transform)

    observer_x = _finite_float("observer_x", observer_x)
    observer_y = _finite_float("observer_y", observer_y)

    heading = normalize_heading(heading_deg)
    fov = validate_horizontal_fov(horizontal_fov_deg)

    if valid_mask is not None:
        valid = _as_boolean_mask("valid_mask", valid_mask)

        if valid.shape != visibility.shape:
            raise ValueError(
                "valid_mask and visibility_mask must have the same shape."
            )
    else:
        valid = None

    # A 360-degree FOV does not require any angular calculations.
    if np.isclose(fov, 360.0):
        result = visibility.copy()

        if valid is not None:
            result &= valid

        return result

    bearings, observer_cell_mask = bearing_grid(
        shape=visibility.shape,
        transform=transform,
        observer_x=observer_x,
        observer_y=observer_y,
    )

    angular_difference = smallest_angular_difference(
        bearings,
        heading,
    )

    half_fov = fov / 2.0

    directional_mask = angular_difference <= half_fov

    # Bearing at the exact observer position is undefined. Retain that cell
    # when the geometric viewshed itself marks it as visible.
    directional_mask |= observer_cell_mask

    result = visibility & directional_mask

    if valid is not None:
        result &= valid

    return result


def horizontal_fov_mask(
    shape: tuple[int, int],
    *,
    observer_x: float,
    observer_y: float,
    transform: Affine,
    heading_deg: float,
    horizontal_fov_deg: float,
) -> np.ndarray:
    """Create a horizontal viewing-cone mask independently of a viewshed.

    Parameters
    ----------
    shape
        Raster shape as ``(rows, columns)``.

    observer_x, observer_y
        Observer coordinates in the raster CRS.

    transform
        Affine transform describing the raster grid.

    heading_deg
        Viewing direction using compass convention.

    horizontal_fov_deg
        Full horizontal field of view in degrees.

    Returns
    -------
    numpy.ndarray
        Boolean raster whose True cells lie inside the viewing cone.

    Notes
    -----
    This function is useful for visualization and testing. Effective
    visibility should normally be produced with ``apply_horizontal_fov`` so
    that the directional mask is combined with terrain visibility.
    """

    _validate_shape(shape)
    _validate_transform(transform)

    observer_x = _finite_float("observer_x", observer_x)
    observer_y = _finite_float("observer_y", observer_y)

    heading = normalize_heading(heading_deg)
    fov = validate_horizontal_fov(horizontal_fov_deg)

    if np.isclose(fov, 360.0):
        return np.ones(shape, dtype=bool)

    bearings, observer_cell_mask = bearing_grid(
        shape=shape,
        transform=transform,
        observer_x=observer_x,
        observer_y=observer_y,
    )

    angular_difference = smallest_angular_difference(
        bearings,
        heading,
    )

    mask = angular_difference <= (fov / 2.0)

    # Include the observer location for consistency with apply_horizontal_fov.
    mask |= observer_cell_mask

    return mask


def bearing_grid(
    *,
    shape: tuple[int, int],
    transform: Affine,
    observer_x: float,
    observer_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate observer-to-cell bearings for an entire raster grid.

    Parameters
    ----------
    shape
        Raster shape as ``(rows, columns)``.

    transform
        Affine transform of the raster.

    observer_x, observer_y
        Observer coordinates in the same CRS as the raster.

    Returns
    -------
    bearings
        Array of compass bearings in degrees within [0, 360).

    observer_cell_mask
        Boolean array identifying cells whose centres coincide with the
        observer coordinates within numerical tolerance.

    Notes
    -----
    Bearings use compass convention rather than mathematical-angle
    convention. North is zero degrees and angles increase clockwise.
    """

    _validate_shape(shape)
    _validate_transform(transform)

    observer_x = _finite_float("observer_x", observer_x)
    observer_y = _finite_float("observer_y", observer_y)

    rows, cols = np.indices(shape)

    # Cell centres are used rather than cell corners.
    col_centres = cols.astype(np.float64) + 0.5
    row_centres = rows.astype(np.float64) + 0.5

    x = (
        transform.a * col_centres
        + transform.b * row_centres
        + transform.c
    )
    y = (
        transform.d * col_centres
        + transform.e * row_centres
        + transform.f
    )

    dx = x - observer_x
    dy = y - observer_y

    # atan2(dx, dy) rather than atan2(dy, dx) converts directly to compass
    # bearings: north=0, east=90, south=180, west=270.
    bearings = (
        np.degrees(np.arctan2(dx, dy)) + 360.0
    ) % 360.0

    tolerance = max(
        abs(float(transform.a)),
        abs(float(transform.b)),
        abs(float(transform.d)),
        abs(float(transform.e)),
        1.0,
    ) * 1e-12

    observer_cell_mask = (
        np.abs(dx) <= tolerance
    ) & (
        np.abs(dy) <= tolerance
    )

    return bearings, observer_cell_mask


def smallest_angular_difference(
    angle_deg: np.ndarray | float,
    reference_deg: float,
) -> np.ndarray:
    """Return the smallest absolute angular difference in degrees.

    The result lies within [0, 180].

    Examples
    --------
    350 degrees and 10 degrees differ by 20 degrees, not 340 degrees.
    """

    reference = normalize_heading(reference_deg)

    angles = np.asarray(angle_deg, dtype=np.float64)

    if not np.all(np.isfinite(angles)):
        raise ValueError("angle_deg must contain only finite values.")

    return np.abs(
        (angles - reference + 180.0) % 360.0 - 180.0
    )


def normalize_heading(heading_deg: float) -> float:
    """Normalize a compass heading to the interval [0, 360)."""

    heading = _finite_float("heading_deg", heading_deg)
    return heading % 360.0


def validate_horizontal_fov(horizontal_fov_deg: float) -> float:
    """Validate and return a horizontal field of view in degrees."""

    fov = _finite_float(
        "horizontal_fov_deg",
        horizontal_fov_deg,
    )

    if not 0.0 < fov <= 360.0:
        raise ValueError(
            "horizontal_fov_deg must be within (0, 360] degrees."
        )

    return fov


def _as_boolean_mask(
    name: str,
    mask: np.ndarray,
) -> np.ndarray:
    """Validate a two-dimensional raster mask and return boolean form."""

    if not isinstance(mask, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray.")

    if mask.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array.")

    return mask.astype(bool, copy=False)


def _validate_shape(shape: tuple[int, int]) -> None:
    """Validate a raster shape."""

    if (
        not isinstance(shape, tuple)
        or len(shape) != 2
        or not all(isinstance(value, int) for value in shape)
    ):
        raise TypeError(
            "shape must be a tuple of two integers: (rows, columns)."
        )

    if shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("Raster dimensions must be greater than zero.")


def _validate_transform(transform: Affine) -> None:
    """Validate an affine raster transform."""

    if not isinstance(transform, Affine):
        raise TypeError("transform must be an affine.Affine object.")


def _finite_float(name: str, value: float) -> float:
    """Validate and convert a numeric value to a finite float."""

    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")

    result = float(value)

    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")

    return result