"""Descriptive coverage and exposure analysis (Analysis & Design A1).

All quantities are derived from the aggregate exposure raster and the domain
masks of a SurveyObservabilityField. No individual visibility mask is read
and no visibility is recomputed.

Denominator
-----------
Every share reported here is relative to **analysable cells**: cells inside
the AnalysisDomain *and* valid for analysis (``analysis_mask & valid_mask``,
the SOF's ``analysable_mask``). Cells outside the domain and invalid cells
(for example terrain NoData) are counted separately and never enter a
denominator, so they can neither inflate nor deflate coverage.

Within analysable space the classes partition the cells exactly::

    blind spot        exposure == 0
    unique coverage   exposure == 1
    repeated coverage exposure >= 2

    observable = unique + repeated
    analysable = blind + observable

"Repeated coverage" describes repeated observation opportunity. It does not
imply that the additional observations are without value.

The functions accept plain arrays as well as an SOF so that future design
scenarios (temporary exposure rasters) and comparisons can reuse them. The
summary deliberately describes a distribution of per-cell exposure rather
than assuming a binary field, leaving room for expected-exposure products
from future probabilistic observability.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from rivelero.observability.masks import (
    ObservabilityState,
    calculate_analysable_mask,
    invalid_mask,
    outside_domain_mask,
)
from rivelero.observability.survey_field import SurveyObservabilityField


class CoverageClass(IntEnum):
    """Categorical coverage class of a cell.

    The first three values coincide with ObservabilityState; OBSERVABLE is
    split into unique and repeated coverage.
    """

    OUTSIDE_DOMAIN = int(ObservabilityState.OUTSIDE_DOMAIN)
    INVALID = int(ObservabilityState.INVALID)
    BLIND_SPOT = int(ObservabilityState.BLIND_SPOT)
    UNIQUE = 3
    REPEATED = 4


@dataclass(frozen=True, slots=True)
class CoverageMasks:
    """Boolean masks partitioning analysable space by exposure.

    ``blind | unique | repeated == analysable`` and the three are disjoint.
    """

    analysable: np.ndarray
    blind: np.ndarray
    unique: np.ndarray
    repeated: np.ndarray

    @property
    def observable(self) -> np.ndarray:
        return self.unique | self.repeated


def coverage_masks(
    exposure_count: np.ndarray,
    *,
    analysis_mask: np.ndarray,
    valid_mask: np.ndarray,
) -> CoverageMasks:
    """Return the canonical blind/unique/repeated masks of an exposure field.

    This is the single definition of those classes shared by descriptive
    (A1) and contribution (A2) analysis.
    """

    exposure = np.asarray(exposure_count)
    analysable = calculate_analysable_mask(analysis_mask, valid_mask)

    if analysable.shape != exposure.shape:
        raise ValueError("Masks and exposure_count must have the same shape.")

    return CoverageMasks(
        analysable=analysable,
        blind=analysable & (exposure == 0),
        unique=analysable & (exposure == 1),
        repeated=analysable & (exposure >= 2),
    )


# ---------------------------------------------------------------------------
# Exposure distribution
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExposureBin:
    """Cells whose exposure lies in ``[lower, upper]`` (inclusive)."""

    lower: int
    upper: int
    cells: int
    fraction: float | None

    @property
    def label(self) -> str:
        if self.lower == self.upper:
            return str(self.lower)
        return f"{self.lower}–{self.upper}"


@dataclass(frozen=True, slots=True, eq=False)
class ExposureDistribution:
    """Number of analysable cells at each exposure level.

    ``cell_counts[k]`` is the number of analysable cells observable from
    exactly ``k`` active sampling units. Fractions use ``analysable_cells``
    as denominator.
    """

    cell_counts: np.ndarray
    analysable_cells: int

    def __eq__(self, other: object) -> bool:
        # Value equality (the default dataclass __eq__ is ambiguous for
        # arrays), so summaries of scenarios and baselines can be compared.
        if not isinstance(other, ExposureDistribution):
            return NotImplemented
        return (
            self.analysable_cells == other.analysable_cells
            and np.array_equal(self.cell_counts, other.cell_counts)
        )

    __hash__ = None

    @property
    def levels(self) -> np.ndarray:
        return np.arange(len(self.cell_counts))

    @property
    def fractions(self) -> np.ndarray | None:
        if self.analysable_cells == 0:
            return None
        return self.cell_counts / float(self.analysable_cells)

    def cells_at(self, level: int) -> int:
        if 0 <= level < len(self.cell_counts):
            return int(self.cell_counts[level])
        return 0

    def binned(self, max_bins: int = 20) -> list[ExposureBin]:
        """Group levels into at most ``max_bins`` bins for display.

        Levels 0 (blind spot) and 1 (unique coverage) always keep their own
        bins because they have distinct scientific meaning; higher levels are
        grouped into equal-width integer ranges when necessary.
        """

        if max_bins < 3:
            raise ValueError("max_bins must be at least 3.")

        maximum = len(self.cell_counts) - 1
        edges: list[tuple[int, int]] = [(0, 0)]
        if maximum >= 1:
            edges.append((1, 1))
        if maximum >= 2:
            remaining = maximum - 1
            width = int(np.ceil(remaining / (max_bins - 2)))
            lower = 2
            while lower <= maximum:
                upper = min(maximum, lower + width - 1)
                edges.append((lower, upper))
                lower = upper + 1

        bins = []
        for lower, upper in edges:
            cells = int(self.cell_counts[lower:upper + 1].sum())
            bins.append(
                ExposureBin(
                    lower=lower,
                    upper=upper,
                    cells=cells,
                    fraction=(
                        None
                        if self.analysable_cells == 0
                        else cells / self.analysable_cells
                    ),
                )
            )
        return bins


# ---------------------------------------------------------------------------
# Coverage summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CoverageSummary:
    """Descriptive coverage and exposure statistics of one exposure field.

    Cell counts
    -----------
    ``total_cells = outside_domain_cells + invalid_cells + analysable_cells``
    and ``analysable_cells = blind_cells + unique_cells + repeated_cells``.

    Exposure statistics
    -------------------
    ``mean_exposure``/``median_exposure`` are taken over all analysable cells
    (blind spots contribute zero). ``mean_exposure_observable``/
    ``median_exposure_observable`` are conditional on the cell being
    observable. Statistics are ``None`` when their population is empty.
    """

    total_cells: int
    outside_domain_cells: int
    invalid_cells: int
    analysable_cells: int

    blind_cells: int
    unique_cells: int
    repeated_cells: int

    active_units: int

    maximum_exposure: int
    total_exposure: int

    mean_exposure: float | None
    median_exposure: float | None
    mean_exposure_observable: float | None
    median_exposure_observable: float | None

    distribution: ExposureDistribution

    @property
    def observable_cells(self) -> int:
        return self.unique_cells + self.repeated_cells

    def _share(self, cells: int) -> float | None:
        if self.analysable_cells == 0:
            return None
        return cells / self.analysable_cells

    @property
    def observable_fraction(self) -> float | None:
        """Coverage: observable / analysable."""
        return self._share(self.observable_cells)

    @property
    def blind_fraction(self) -> float | None:
        return self._share(self.blind_cells)

    @property
    def unique_fraction(self) -> float | None:
        return self._share(self.unique_cells)

    @property
    def repeated_fraction(self) -> float | None:
        return self._share(self.repeated_cells)

    @property
    def repeated_share_of_observable(self) -> float | None:
        """Share of observable cells seen by two or more sampling units."""
        if self.observable_cells == 0:
            return None
        return self.repeated_cells / self.observable_cells


def summarize_exposure(
    exposure_count: np.ndarray,
    *,
    analysis_mask: np.ndarray,
    valid_mask: np.ndarray,
    active_units: int,
) -> CoverageSummary:
    """Summarise an integer exposure raster over analysable space."""

    exposure = np.asarray(exposure_count)

    if exposure.ndim != 2:
        raise ValueError("exposure_count must be two-dimensional.")

    if not np.issubdtype(exposure.dtype, np.integer):
        raise TypeError("exposure_count must use an integer dtype.")

    if np.any(exposure < 0):
        raise ValueError("exposure_count cannot contain negative values.")

    if not isinstance(active_units, (int, np.integer)) or active_units < 0:
        raise ValueError("active_units must be a non-negative integer.")

    analysable = calculate_analysable_mask(analysis_mask, valid_mask)

    if analysable.shape != exposure.shape:
        raise ValueError("Masks and exposure_count must have the same shape.")

    values = exposure[analysable].astype(np.int64, copy=False)

    # A histogram over integer levels gives counts, mean and medians without
    # sorting, which matters for grids with tens of millions of cells.
    counts = (
        np.bincount(values)
        if values.size
        else np.zeros(1, dtype=np.int64)
    )
    if counts.size == 0:
        counts = np.zeros(1, dtype=np.int64)

    analysable_cells = int(values.size)
    blind = int(counts[0])
    unique = int(counts[1]) if counts.size > 1 else 0
    repeated = int(counts[2:].sum()) if counts.size > 2 else 0

    total = int(values.sum(dtype=np.int64))
    maximum = int(counts.size - 1) if analysable_cells else 0

    if maximum > active_units:
        raise ValueError(
            "exposure_count exceeds the number of active sampling units."
        )

    observable_counts = counts.copy()
    observable_counts[0] = 0

    return CoverageSummary(
        total_cells=int(exposure.size),
        outside_domain_cells=int(np.count_nonzero(outside_domain_mask(analysis_mask))),
        invalid_cells=int(np.count_nonzero(invalid_mask(analysis_mask, valid_mask))),
        analysable_cells=analysable_cells,
        blind_cells=blind,
        unique_cells=unique,
        repeated_cells=repeated,
        active_units=int(active_units),
        maximum_exposure=maximum,
        total_exposure=total,
        mean_exposure=(total / analysable_cells if analysable_cells else None),
        median_exposure=_median_from_counts(counts),
        mean_exposure_observable=(
            total / (unique + repeated) if unique + repeated else None
        ),
        median_exposure_observable=_median_from_counts(observable_counts),
        distribution=ExposureDistribution(
            cell_counts=counts.astype(np.int64, copy=False),
            analysable_cells=analysable_cells,
        ),
    )


def summarize_coverage(sof: SurveyObservabilityField) -> CoverageSummary:
    """Summarise coverage and exposure of a SurveyObservabilityField."""

    if not isinstance(sof, SurveyObservabilityField):
        raise TypeError("sof must be a SurveyObservabilityField.")

    return summarize_exposure(
        sof.exposure_count,
        analysis_mask=sof.analysis_mask,
        valid_mask=sof.valid_mask,
        active_units=sof.n_active_units,
    )


def coverage_class_raster(
    exposure_count: np.ndarray,
    *,
    analysis_mask: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Return a CoverageClass raster (uint8) for mapping."""

    exposure = np.asarray(exposure_count)
    analysis = np.asarray(analysis_mask, dtype=bool)
    masks = coverage_masks(
        exposure, analysis_mask=analysis_mask, valid_mask=valid_mask
    )

    classes = np.full(exposure.shape, CoverageClass.OUTSIDE_DOMAIN, dtype=np.uint8)
    classes[invalid_mask(analysis_mask, valid_mask)] = CoverageClass.INVALID
    classes[masks.blind] = CoverageClass.BLIND_SPOT
    classes[masks.unique] = CoverageClass.UNIQUE
    classes[masks.repeated] = CoverageClass.REPEATED

    # Outside-domain takes precedence, matching ObservabilityState.
    classes[~analysis] = CoverageClass.OUTSIDE_DOMAIN
    return classes


def _median_from_counts(counts: np.ndarray) -> float | None:
    """Median of the multiset described by ``counts[level]``."""

    n = int(counts.sum())

    if n == 0:
        return None

    cumulative = np.cumsum(counts)
    lower = int(np.searchsorted(cumulative, (n - 1) // 2, side="right"))
    upper = int(np.searchsorted(cumulative, n // 2, side="right"))
    return (lower + upper) / 2.0
