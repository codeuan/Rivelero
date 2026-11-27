"""Visibility diagnostics across and beyond an AnalysisDomain."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from affine import Affine
from pyproj import CRS as PyprojCRS
from rasterio.crs import CRS
from shapely.geometry import Polygon

from rivelero.core.domain import AnalysisDomain
from rivelero.core.environment import Environment
from rivelero.core.observation import ObservationEvent
from rivelero.core.sensor import Sensor
from rivelero.core.viewpoint import Viewpoint
from rivelero.gui.domain_service import elevation_valid_mask
from rivelero.visibility.configuration import VisibilityConfiguration
from rivelero.visibility.engine import (
    SingleViewpointVisibility,
    compute_viewpoint_visibility,
)


@dataclass(slots=True)
class ExtendedVisibilityDiagnostic:
    """Full-grid visibility and its in-domain/out-of-domain partition."""

    full_visibility: SingleViewpointVisibility
    in_domain_visibility_mask: np.ndarray
    out_of_domain_visibility_mask: np.ndarray
    analysis_domain_mask: np.ndarray

    @property
    def full_visible_cells(self) -> int:
        """Number of valid visible cells across the full terrain grid."""

        return (
            self.in_domain_visible_cells
            + self.out_of_domain_visible_cells
        )

    @property
    def cell_area_m2(self) -> float:
        """Area of one raster cell in square metres."""

        return _cell_area_m2(
            self.full_visibility.transform,
            self.full_visibility.crs,
        )

    @property
    def full_visible_area_m2(self) -> float:
        """Area of all valid visible cells across the terrain grid."""

        return self.full_visible_cells * self.cell_area_m2

    @property
    def in_domain_visible_area_m2(self) -> float:
        """Area of valid visible cells inside the AnalysisDomain."""

        return self.in_domain_visible_cells * self.cell_area_m2

    @property
    def out_of_domain_visible_area_m2(self) -> float:
        """Area of valid visible cells outside the AnalysisDomain."""

        return self.out_of_domain_visible_cells * self.cell_area_m2

    @property
    def in_domain_visible_cells(self) -> int:
        """Number of visible valid cells inside the AnalysisDomain."""

        return int(np.count_nonzero(self.in_domain_visibility_mask))

    @property
    def out_of_domain_visible_cells(self) -> int:
        """Number of visible valid cells outside the AnalysisDomain."""

        return int(np.count_nonzero(self.out_of_domain_visibility_mask))

    @property
    def in_domain_footprint_share(self) -> float | None:
        """Fraction of visible cells inside the domain, or None if empty."""

        return _share(
            self.in_domain_visible_cells,
            self.full_visible_cells,
        )

    @property
    def out_of_domain_footprint_share(self) -> float | None:
        """Fraction of visible cells outside the domain, or None if empty."""

        return _share(
            self.out_of_domain_visible_cells,
            self.full_visible_cells,
        )

    @property
    def in_domain_coverage_share(self) -> float | None:
        """Fraction of valid in-domain cells visible, or None if empty."""

        analysable_domain_cells = int(
            np.count_nonzero(
                self.analysis_domain_mask & self.full_visibility.valid_mask
            )
        )
        return _share(
            self.in_domain_visible_cells,
            analysable_domain_cells,
        )


def partition_visibility_by_domain(
    visibility_mask: np.ndarray,
    domain: AnalysisDomain,
) -> tuple[np.ndarray, np.ndarray]:
    """Partition a full-grid visibility mask using a domain's masks.

    The input must represent visibility computed over the full terrain grid,
    rather than a mask already clipped to the AnalysisDomain.
    """

    if not isinstance(visibility_mask, np.ndarray):
        raise TypeError("visibility_mask must be a numpy.ndarray.")

    if visibility_mask.ndim != 2:
        raise ValueError("visibility_mask must be two-dimensional.")

    if visibility_mask.shape != domain.grid.shape:
        raise ValueError(
            "visibility_mask shape does not match the AnalysisDomain grid."
        )

    visible = visibility_mask.astype(bool, copy=False)
    domain_mask = domain.effective_analysis_mask
    valid_mask = domain.effective_valid_mask

    in_domain = visible & domain_mask & valid_mask
    out_of_domain = visible & ~domain_mask & valid_mask
    return in_domain, out_of_domain


def compute_extended_visibility(
    *,
    viewpoint: Viewpoint,
    environment: Environment,
    domain: AnalysisDomain,
    configuration: VisibilityConfiguration,
    sensor: Sensor | None = None,
    event: ObservationEvent | None = None,
) -> ExtendedVisibilityDiagnostic:
    """Compute full-grid visibility, then partition it by the given domain.

    Visibility parameters and raster processing are delegated to the
    canonical single-viewpoint engine. A temporary domain exposes the full
    grid while preserving the original terrain-validity mask; the caller's
    AnalysisDomain is not modified.
    """

    terrain_valid_mask = elevation_valid_mask(
        environment.elevation_model.source,
        domain.grid,
        nodata_value=environment.elevation_model.nodata_value,
    )
    valid_mask = domain.effective_valid_mask & terrain_valid_mask
    full_grid_domain = _full_grid_domain(domain, valid_mask)
    full_visibility = compute_viewpoint_visibility(
        viewpoint=viewpoint,
        environment=environment,
        domain=full_grid_domain,
        configuration=configuration,
        sensor=sensor,
        event=event,
    )
    in_domain, out_of_domain = partition_visibility_by_domain(
        full_visibility.visibility_mask,
        domain,
    )

    return ExtendedVisibilityDiagnostic(
        full_visibility=full_visibility,
        in_domain_visibility_mask=in_domain,
        out_of_domain_visibility_mask=out_of_domain,
        analysis_domain_mask=domain.effective_analysis_mask,
    )


def _full_grid_domain(
    domain: AnalysisDomain,
    valid_mask: np.ndarray,
) -> AnalysisDomain:
    grid = domain.grid
    corners = (
        grid.transform @ (0, 0),
        grid.transform @ (grid.width, 0),
        grid.transform @ (grid.width, grid.height),
        grid.transform @ (0, grid.height),
    )
    return AnalysisDomain(
        domain_id=f"{domain.domain_id}__extended_visibility",
        name=f"{domain.name} (extended visibility)",
        geometry=Polygon(corners),
        crs=domain.crs,
        grid=grid,
        analysis_mask=np.ones(grid.shape, dtype=bool),
        valid_mask=valid_mask,
        creation_method="extended_visibility_grid",
        provenance={"source_domain_id": domain.domain_id},
    )


def _share(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _cell_area_m2(transform: Affine, crs: CRS | str) -> float:
    projected_crs = PyprojCRS.from_user_input(crs)
    if not projected_crs.is_projected or len(projected_crs.axis_info) < 2:
        raise ValueError(
            "Visible-cell areas in square metres require a projected CRS."
        )

    factor_x = projected_crs.axis_info[0].unit_conversion_factor
    factor_y = projected_crs.axis_info[1].unit_conversion_factor
    if factor_x is None or factor_y is None:
        raise ValueError(
            "The projected CRS does not define valid linear units."
        )
    if (
        not np.isfinite(factor_x)
        or not np.isfinite(factor_y)
        or factor_x <= 0
        or factor_y <= 0
    ):
        raise ValueError(
            "The projected CRS does not define valid linear units."
        )

    area_in_crs_units = abs(
        transform.a * transform.e - transform.b * transform.d
    )
    if not np.isfinite(area_in_crs_units) or area_in_crs_units <= 0:
        raise ValueError(
            "The AnalysisGrid affine transform must define positive cell area."
        )
    return float(area_in_crs_units * factor_x * factor_y)
