"""Spatial analysis-domain data model for Rivelero.

An AnalysisDomain defines where Rivelero evaluates observability.

It describes the spatial region of interest, optional target and observer
regions, the analysis grid, and masks defining which locations are valid for
analysis.

The AnalysisDomain is distinct from the Environment. The Environment
describes the physical spatial world, whereas the AnalysisDomain defines the
part of that world about which an observability question is being asked.

Importantly, locations outside the AnalysisDomain or locations lacking valid
environmental data must not be interpreted as blind spots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
from affine import Affine
from rasterio.crs import CRS
from shapely.geometry.base import BaseGeometry


@dataclass(slots=True)
class AnalysisGrid:
    """Raster grid on which Rivelero evaluates observability.

    Parameters
    ----------
    crs
        Coordinate reference system of the analysis grid.

    transform
        Affine transform mapping raster indices to spatial coordinates.

    width, height
        Number of raster columns and rows.

    resolution_x, resolution_y
        Horizontal grid resolution. Values are stored as positive magnitudes.

    Notes
    -----
    The grid describes spatial alignment only. It does not itself determine
    which cells belong to the AnalysisDomain or are valid for analysis.
    """

    crs: CRS | str
    transform: Affine
    width: int
    height: int

    resolution_x: float | None = None
    resolution_y: float | None = None

    def __post_init__(self) -> None:
        """Validate and normalize grid metadata."""

        try:
            self.crs = CRS.from_user_input(self.crs)
        except Exception as exc:
            raise ValueError(
                f"Invalid AnalysisGrid CRS: {self.crs!r}"
            ) from exc

        if not isinstance(self.transform, Affine):
            raise TypeError("transform must be an affine.Affine object.")

        if not isinstance(self.width, int) or self.width <= 0:
            raise ValueError("width must be a positive integer.")

        if not isinstance(self.height, int) or self.height <= 0:
            raise ValueError("height must be a positive integer.")

        # If resolution was not supplied explicitly, derive it from the
        # affine transform. Positive magnitudes are stored even though
        # north-up rasters normally have a negative y pixel size.
        if self.resolution_x is None:
            self.resolution_x = abs(float(self.transform.a))
        else:
            self.resolution_x = self._positive_float(
                "resolution_x",
                self.resolution_x,
            )

        if self.resolution_y is None:
            self.resolution_y = abs(float(self.transform.e))
        else:
            self.resolution_y = self._positive_float(
                "resolution_y",
                self.resolution_y,
            )

        if self.resolution_x <= 0 or self.resolution_y <= 0:
            raise ValueError(
                "AnalysisGrid resolution must be greater than zero."
            )

    @property
    def shape(self) -> tuple[int, int]:
        """Return raster shape as ``(height, width)``."""

        return self.height, self.width

    @property
    def cell_area(self) -> float:
        """Return nominal raster-cell area in CRS units squared.

        This is appropriate for projected grids whose horizontal CRS units
        have meaningful linear dimensions. Geographic latitude/longitude
        grids require geodesic treatment for physical area calculations.
        """

        return self.resolution_x * self.resolution_y

    @staticmethod
    def _positive_float(name: str, value: float) -> float:
        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric.")

        result = float(value)

        if not np.isfinite(result) or result <= 0:
            raise ValueError(f"{name} must be finite and greater than zero.")

        return result


@dataclass(slots=True)
class AnalysisDomain:
    """Spatial domain over which Rivelero evaluates observability.

    Parameters
    ----------
    domain_id
        Unique identifier for the analysis domain.

    name
        Human-readable name.

    geometry
        Main Area of Interest (AOI). Locations outside this geometry are not
        part of the observability analysis.

    crs
        Coordinate reference system in which ``geometry`` is expressed.

    grid
        Raster grid used for observability calculations.

    target_region
        Optional geometry defining the region whose observability is of
        particular interest. If None, the main AOI may be treated as the
        target region by downstream workflows.

    observer_region
        Optional geometry constraining where observer/viewpoint locations may
        occur. This is especially useful for prospective survey design.

    analysis_mask
        Optional boolean raster mask identifying cells belonging to the
        AnalysisDomain.

    valid_mask
        Optional boolean raster mask identifying cells for which sufficient
        environmental data exist to perform the requested analysis.

    creation_method
        Description of how the domain was constructed, for example
        ``user_polygon``, ``viewpoint_convex_hull``, ``buffered_hull``, or
        ``environment_extent``.

    provenance
        Information describing the source and construction of the domain.

    created_at
        Time at which the AnalysisDomain representation was created.

    extra_metadata
        Extensible project- or application-specific metadata.

    Notes
    -----
    ``analysis_mask`` and ``valid_mask`` have different meanings:

    - analysis_mask: Is this cell part of the spatial question?
    - valid_mask: Can Rivelero legitimately evaluate this cell?

    A blind spot may only be identified within cells satisfying both
    conditions.
    """

    domain_id: str
    name: str

    geometry: BaseGeometry
    crs: CRS | str

    grid: AnalysisGrid

    target_region: BaseGeometry | None = None
    observer_region: BaseGeometry | None = None

    analysis_mask: np.ndarray | None = None
    valid_mask: np.ndarray | None = None

    creation_method: str | None = None

    provenance: dict[str, Any] = field(default_factory=dict)

    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize the AnalysisDomain."""

        self.domain_id = self._required_string(
            "domain_id",
            self.domain_id,
        )
        self.name = self._required_string("name", self.name)

        self._validate_geometry("geometry", self.geometry)

        try:
            self.crs = CRS.from_user_input(self.crs)
        except Exception as exc:
            raise ValueError(
                f"Invalid AnalysisDomain CRS: {self.crs!r}"
            ) from exc

        if not isinstance(self.grid, AnalysisGrid):
            raise TypeError("grid must be an AnalysisGrid.")

        if self.grid.crs != self.crs:
            raise ValueError(
                "AnalysisDomain CRS and AnalysisGrid CRS must match."
            )

        if self.target_region is not None:
            self._validate_geometry(
                "target_region",
                self.target_region,
            )

        if self.observer_region is not None:
            self._validate_geometry(
                "observer_region",
                self.observer_region,
            )

        self.analysis_mask = self._validate_mask(
            "analysis_mask",
            self.analysis_mask,
        )
        self.valid_mask = self._validate_mask(
            "valid_mask",
            self.valid_mask,
        )

        self.creation_method = self._optional_string(
            "creation_method",
            self.creation_method,
        )

        if not isinstance(self.provenance, dict):
            raise TypeError("provenance must be a dictionary.")

        if not isinstance(self.extra_metadata, dict):
            raise TypeError("extra_metadata must be a dictionary.")

        if not isinstance(self.created_at, datetime):
            raise TypeError("created_at must be a datetime object.")

    # ------------------------------------------------------------------
    # Masks
    # ------------------------------------------------------------------

    def _validate_mask(
        self,
        name: str,
        mask: np.ndarray | None,
    ) -> np.ndarray | None:
        """Validate a raster mask against the analysis grid."""

        if mask is None:
            return None

        if not isinstance(mask, np.ndarray):
            raise TypeError(f"{name} must be a numpy.ndarray or None.")

        if mask.shape != self.grid.shape:
            raise ValueError(
                f"{name} shape {mask.shape} does not match "
                f"AnalysisGrid shape {self.grid.shape}."
            )

        # Normalize to a boolean representation without changing shape.
        return mask.astype(bool, copy=False)

    @property
    def analysable_mask(self) -> np.ndarray | None:
        """Return cells that are both inside the domain and valid for analysis.

        Returns None if neither raster mask has been materialized.

        If only one mask exists, the missing component is treated as unrestricted:
        - missing analysis_mask -> all grid cells are considered in-domain
        - missing valid_mask -> all in-domain cells are considered valid

        This fallback is explicit and should primarily support domains whose
        raster masks have not yet been fully materialized.
        """

        if self.analysis_mask is None and self.valid_mask is None:
            return None

        if self.analysis_mask is None:
            analysis = np.ones(
                self.grid.shape,
                dtype=bool,
            )
        else:
            analysis = self.analysis_mask

        if self.valid_mask is None:
            valid = np.ones(
                self.grid.shape,
                dtype=bool,
            )
        else:
            valid = self.valid_mask

        return analysis & valid

    @property
    def effective_analysis_mask(self) -> np.ndarray:
        """Return the rasterized analysis-domain mask.

        If no explicit analysis mask has been materialized, all AnalysisGrid
        cells are provisionally treated as belonging to the domain.
        """

        if self.analysis_mask is None:
            return np.ones(
                self.grid.shape,
                dtype=bool,
            )

        return self.analysis_mask.copy()


    @property
    def effective_valid_mask(self) -> np.ndarray:
        """Return the effective environmental-validity mask.

        If no explicit validity mask has been materialized, all grid cells are
        provisionally treated as valid.
        """

        if self.valid_mask is None:
            return np.ones(
                self.grid.shape,
                dtype=bool,
            )

        return self.valid_mask.copy()

    @property
    def n_analysable_cells(self) -> int | None:
        """Return the number of currently analysable cells."""

        mask = self.analysable_mask

        if mask is None:
            return None

        return int(np.count_nonzero(mask))

    # ------------------------------------------------------------------
    # Target and observer regions
    # ------------------------------------------------------------------

    @property
    def effective_target_region(self) -> BaseGeometry:
        """Return target region, defaulting explicitly to the main AOI."""

        if self.target_region is not None:
            return self.target_region

        return self.geometry

    @property
    def effective_observer_region(self) -> BaseGeometry:
        """Return observer region, defaulting explicitly to the main AOI.

        This default is primarily useful for prospective analyses. Existing
        ViewpointConfigurations may contain observers outside the AOI if a
        workflow explicitly permits them.
        """

        if self.observer_region is not None:
            return self.observer_region

        return self.geometry

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        """Return the analysis-grid shape."""

        return self.grid.shape

    @property
    def transform(self) -> Affine:
        """Return the analysis-grid affine transform."""

        return self.grid.transform

    @property
    def resolution(self) -> tuple[float, float]:
        """Return grid resolution as ``(x_resolution, y_resolution)``."""

        return (
            self.grid.resolution_x,
            self.grid.resolution_y,
        )

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_geometry(
        name: str,
        geometry: BaseGeometry,
    ) -> None:
        """Validate a Shapely geometry."""

        if not isinstance(geometry, BaseGeometry):
            raise TypeError(
                f"{name} must be a Shapely geometry."
            )

        if geometry.is_empty:
            raise ValueError(f"{name} cannot be empty.")

        if not geometry.is_valid:
            raise ValueError(f"{name} must be a valid geometry.")

    @staticmethod
    def _required_string(name: str, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string.")

        result = value.strip()

        if not result:
            raise ValueError(f"{name} must be a non-empty string.")

        return result

    @staticmethod
    def _optional_string(
        name: str,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string or None.")

        result = value.strip()

        if not result:
            raise ValueError(
                f"{name} cannot be an empty string when provided."
            )

        return result