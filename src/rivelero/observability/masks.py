"""Spatial observability masks and state classification for Rivelero.

This module contains reusable operations for classifying spatial cells
according to their relationship with the analysis domain and reconstructed
visual observation opportunity.

The fundamental distinction is:

    outside domain
        !=
    invalid / unanalysable
        !=
    analysable but unobservable
        !=
    observable

A blind spot is therefore not simply a cell with zero exposure. It is a cell
that belongs to the valid analysable domain but has zero reconstructed
observation opportunity.

The functions in this module operate only on NumPy arrays. They are
independent of SurveyObservabilityField, VisibilityStore, and the visibility
engine, allowing them to be reused for visualization, testing, metrics, and
future uncertainty analyses.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np


class ObservabilityState(IntEnum):
    """Categorical state assigned to each spatial cell."""

    OUTSIDE_DOMAIN = 0
    INVALID = 1
    BLIND_SPOT = 2
    OBSERVABLE = 3


# ---------------------------------------------------------------------------
# Fundamental masks
# ---------------------------------------------------------------------------


def calculate_analysable_mask(
    analysis_mask: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Return cells that belong to the domain and are valid for analysis.

    The calculation is:

        analysable = analysis_mask AND valid_mask

    Parameters
    ----------
    analysis_mask
        Boolean or binary raster identifying cells belonging to the spatial
        AnalysisDomain.

    valid_mask
        Boolean or binary raster identifying cells for which sufficient
        environmental/spatial information exists to perform the requested
        analysis.

    Returns
    -------
    numpy.ndarray
        Boolean raster identifying analysable cells.
    """

    analysis = _as_boolean_mask(
        "analysis_mask",
        analysis_mask,
    )

    valid = _as_boolean_mask(
        "valid_mask",
        valid_mask,
    )

    _require_same_shape(
        analysis,
        valid,
        names=("analysis_mask", "valid_mask"),
    )

    return analysis & valid


def observable_mask(
    exposure_count: np.ndarray,
    *,
    analysable: np.ndarray,
) -> np.ndarray:
    """Return analysable cells visible from at least one sampling unit.

    The calculation is:

        observable = analysable AND exposure_count > 0

    Parameters
    ----------
    exposure_count
        Non-negative integer raster containing cumulative visual exposure.

    analysable
        Boolean raster identifying cells that legitimately participate in
        the analysis.

    Returns
    -------
    numpy.ndarray
        Boolean observable-space mask.
    """

    exposure = _validate_exposure_count(
        exposure_count
    )

    valid = _as_boolean_mask(
        "analysable",
        analysable,
    )

    _require_same_shape(
        exposure,
        valid,
        names=("exposure_count", "analysable"),
    )

    return valid & (exposure > 0)


def blindspot_mask(
    exposure_count: np.ndarray,
    *,
    analysable: np.ndarray,
) -> np.ndarray:
    """Return analysable cells with no reconstructed observation opportunity.

    The calculation is:

        blind_spot = analysable AND exposure_count == 0

    Cells outside the AnalysisDomain and cells lacking valid environmental
    information are explicitly excluded.

    Parameters
    ----------
    exposure_count
        Non-negative integer raster containing cumulative exposure.

    analysable
        Boolean raster defining valid analysable space.

    Returns
    -------
    numpy.ndarray
        Boolean blind-spot mask.
    """

    exposure = _validate_exposure_count(
        exposure_count
    )

    valid = _as_boolean_mask(
        "analysable",
        analysable,
    )

    _require_same_shape(
        exposure,
        valid,
        names=("exposure_count", "analysable"),
    )

    return valid & (exposure == 0)


# ---------------------------------------------------------------------------
# Domain-state masks
# ---------------------------------------------------------------------------


def outside_domain_mask(
    analysis_mask: np.ndarray,
) -> np.ndarray:
    """Return cells outside the defined AnalysisDomain."""

    analysis = _as_boolean_mask(
        "analysis_mask",
        analysis_mask,
    )

    return ~analysis


def invalid_mask(
    analysis_mask: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Return in-domain cells that cannot legitimately be analysed.

    The calculation is:

        invalid = analysis_mask AND NOT valid_mask

    This distinguishes missing/invalid spatial information from genuine
    observability blind spots.
    """

    analysis = _as_boolean_mask(
        "analysis_mask",
        analysis_mask,
    )

    valid = _as_boolean_mask(
        "valid_mask",
        valid_mask,
    )

    _require_same_shape(
        analysis,
        valid,
        names=("analysis_mask", "valid_mask"),
    )

    return analysis & ~valid


# ---------------------------------------------------------------------------
# Categorical observability state
# ---------------------------------------------------------------------------


def observability_state(
    *,
    analysis_mask: np.ndarray,
    valid_mask: np.ndarray,
    exposure_count: np.ndarray,
    dtype: np.dtype | type = np.uint8,
) -> np.ndarray:
    """Classify every raster cell into a mutually exclusive state.

    States
    ------
    0
        OUTSIDE_DOMAIN

    1
        INVALID -- inside the AnalysisDomain but not valid for analysis.

    2
        BLIND_SPOT -- analysable but exposure is zero.

    3
        OBSERVABLE -- analysable and exposure is greater than zero.

    Parameters
    ----------
    analysis_mask
        Cells belonging to the AnalysisDomain.

    valid_mask
        Cells containing sufficient information for analysis.

    exposure_count
        Cumulative exposure raster.

    dtype
        Integer dtype used for the categorical raster.

    Returns
    -------
    numpy.ndarray
        Categorical observability-state raster.

    Notes
    -----
    Every cell receives exactly one state. This raster is particularly useful
    for visualization and export because it preserves distinctions that would
    be lost in a simple visible/not-visible mask.
    """

    analysis = _as_boolean_mask(
        "analysis_mask",
        analysis_mask,
    )

    valid = _as_boolean_mask(
        "valid_mask",
        valid_mask,
    )

    exposure = _validate_exposure_count(
        exposure_count
    )

    _require_same_shape(
        analysis,
        valid,
        exposure,
        names=(
            "analysis_mask",
            "valid_mask",
            "exposure_count",
        ),
    )

    output_dtype = np.dtype(dtype)

    if not np.issubdtype(
        output_dtype,
        np.integer,
    ):
        raise TypeError(
            "Observability-state dtype must be an integer dtype."
        )

    maximum_state = max(
        state.value
        for state in ObservabilityState
    )

    if np.iinfo(output_dtype).max < maximum_state:
        raise ValueError(
            "Selected dtype cannot represent all ObservabilityState values."
        )

    result = np.full(
        analysis.shape,
        ObservabilityState.OUTSIDE_DOMAIN,
        dtype=output_dtype,
    )

    invalid = analysis & ~valid
    analysable = analysis & valid

    blind = analysable & (exposure == 0)
    observable = analysable & (exposure > 0)

    result[invalid] = ObservabilityState.INVALID
    result[blind] = ObservabilityState.BLIND_SPOT
    result[observable] = ObservabilityState.OBSERVABLE

    return result


# ---------------------------------------------------------------------------
# State extraction
# ---------------------------------------------------------------------------


def mask_for_state(
    state_raster: np.ndarray,
    state: ObservabilityState | int,
) -> np.ndarray:
    """Extract a boolean mask for one categorical observability state."""

    if not isinstance(
        state_raster,
        np.ndarray,
    ):
        raise TypeError(
            "state_raster must be a numpy.ndarray."
        )

    if state_raster.ndim != 2:
        raise ValueError(
            "state_raster must be two-dimensional."
        )

    if not np.issubdtype(
        state_raster.dtype,
        np.integer,
    ):
        raise TypeError(
            "state_raster must use an integer dtype."
        )

    try:
        requested = ObservabilityState(
            int(state)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Unknown observability state: {state!r}."
        ) from exc

    return state_raster == requested.value


# ---------------------------------------------------------------------------
# Mask combination utilities
# ---------------------------------------------------------------------------


def combine_valid_masks(
    masks: list[np.ndarray] | tuple[np.ndarray, ...],
) -> np.ndarray:
    """Combine multiple validity masks using logical AND.

    This is useful when an analysis requires several environmental inputs to
    be valid simultaneously.

    For example:

        DEM valid
            AND
        target layer valid
            AND
        additional required layer valid

        = final valid analysis space
    """

    if not isinstance(
        masks,
        (list, tuple),
    ):
        raise TypeError(
            "masks must be a list or tuple of numpy arrays."
        )

    if not masks:
        raise ValueError(
            "masks must contain at least one mask."
        )

    first = _as_boolean_mask(
        "mask",
        masks[0],
    )

    result = first.copy()

    for mask in masks[1:]:
        current = _as_boolean_mask(
            "mask",
            mask,
        )

        if current.shape != result.shape:
            raise ValueError(
                "All validity masks must have the same shape."
            )

        result &= current

    return result


# ---------------------------------------------------------------------------
# Validation / consistency checks
# ---------------------------------------------------------------------------


def validate_observability_masks(
    *,
    analysis_mask: np.ndarray,
    valid_mask: np.ndarray,
    exposure_count: np.ndarray,
) -> None:
    """Validate consistency among domain and observability rasters.

    Raises
    ------
    ValueError
        If shapes differ or exposure occurs outside analysable space.

    Notes
    -----
    This function is useful when loading or reconstructing SOF products from
    disk. It ensures that exposure has not accidentally been assigned to
    cells that Rivelero considers outside the valid analysis domain.
    """

    analysis = _as_boolean_mask(
        "analysis_mask",
        analysis_mask,
    )

    valid = _as_boolean_mask(
        "valid_mask",
        valid_mask,
    )

    exposure = _validate_exposure_count(
        exposure_count
    )

    _require_same_shape(
        analysis,
        valid,
        exposure,
        names=(
            "analysis_mask",
            "valid_mask",
            "exposure_count",
        ),
    )

    analysable = analysis & valid

    if np.any(
        exposure[~analysable] > 0
    ):
        raise ValueError(
            "Exposure exists outside the analysable domain."
        )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _as_boolean_mask(
    name: str,
    mask: np.ndarray,
) -> np.ndarray:
    """Validate a two-dimensional mask and return boolean form."""

    if not isinstance(mask, np.ndarray):
        raise TypeError(
            f"{name} must be a numpy.ndarray."
        )

    if mask.ndim != 2:
        raise ValueError(
            f"{name} must be two-dimensional."
        )

    return mask.astype(
        bool,
        copy=False,
    )


def _validate_exposure_count(
    exposure_count: np.ndarray,
) -> np.ndarray:
    """Validate a non-negative integer exposure raster."""

    if not isinstance(
        exposure_count,
        np.ndarray,
    ):
        raise TypeError(
            "exposure_count must be a numpy.ndarray."
        )

    if exposure_count.ndim != 2:
        raise ValueError(
            "exposure_count must be two-dimensional."
        )

    if not np.issubdtype(
        exposure_count.dtype,
        np.integer,
    ):
        raise TypeError(
            "exposure_count must use an integer dtype."
        )

    if np.any(exposure_count < 0):
        raise ValueError(
            "exposure_count cannot contain negative values."
        )

    return exposure_count


def _require_same_shape(
    *arrays: np.ndarray,
    names: tuple[str, ...],
) -> None:
    """Require all supplied raster arrays to have the same shape."""

    if len(arrays) != len(names):
        raise ValueError(
            "Internal validation error: arrays and names differ in length."
        )

    if not arrays:
        return

    expected = arrays[0].shape

    for array, name in zip(
        arrays[1:],
        names[1:],
    ):
        if array.shape != expected:
            raise ValueError(
                f"{name} shape {array.shape} does not match "
                f"{names[0]} shape {expected}."
            )