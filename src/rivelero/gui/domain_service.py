"""AnalysisDomain construction services for the Rivelero GUI.

This module converts user-facing spatial choices into canonical Rivelero
AnalysisDomain objects.

Supported construction methods
------------------------------
- complete elevation-raster extent;
- buffered survey extent;
- user-drawn polygon;
- imported polygon geometry.

The service is deliberately independent of Qt. GUI map tools and dialogs
collect user choices; this module validates, transforms, clips and rasterizes
those choices into canonical scientific objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import json
import numpy as np
from pyproj import CRS as PyprojCRS
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.features import geometry_mask
from shapely.geometry import (
    Polygon,
    box,
    mapping,
    shape,
)
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union

from rivelero.core.domain import (
    AnalysisDomain,
    AnalysisGrid,
)


class DomainConstructionError(ValueError):
    """Raised when an AnalysisDomain cannot be constructed safely."""


@dataclass(frozen=True, slots=True)
class DomainConstructionResult:
    """Result of constructing an AnalysisDomain."""

    domain: AnalysisDomain

    original_geometry: BaseGeometry

    clipped_to_grid: bool

    source_crs: CRS

    domain_crs: CRS

    @property
    def analysis_mask(self) -> np.ndarray:
        """Rasterized domain mask."""

        return self.domain.effective_analysis_mask


# ---------------------------------------------------------------------------
# Grid geometry
# ---------------------------------------------------------------------------


def grid_bounds(
    grid: AnalysisGrid,
) -> tuple[float, float, float, float]:
    """Return grid bounds as left, bottom, right, top."""

    if not isinstance(grid, AnalysisGrid):
        raise TypeError(
            "grid must be an AnalysisGrid."
        )

    transform = grid.transform

    corners = (
        transform * (0, 0),
        transform * (grid.width, 0),
        transform * (0, grid.height),
        transform * (grid.width, grid.height),
    )

    xs = [
        float(point[0])
        for point in corners
    ]

    ys = [
        float(point[1])
        for point in corners
    ]

    return (
        min(xs),
        min(ys),
        max(xs),
        max(ys),
    )


def grid_extent_geometry(
    grid: AnalysisGrid,
) -> Polygon:
    """Return a polygon covering the complete AnalysisGrid."""

    left, bottom, right, top = grid_bounds(
        grid
    )

    return box(
        left,
        bottom,
        right,
        top,
    )


# ---------------------------------------------------------------------------
# Geometry validation / transformation
# ---------------------------------------------------------------------------


def validate_domain_geometry(
    geometry: BaseGeometry,
) -> BaseGeometry:
    """Validate geometry for use as an AnalysisDomain."""

    if not isinstance(
        geometry,
        BaseGeometry,
    ):
        raise TypeError(
            "geometry must be a Shapely geometry."
        )

    if geometry.is_empty:
        raise DomainConstructionError(
            "Analysis geometry cannot be empty."
        )

    if not geometry.is_valid:
        raise DomainConstructionError(
            "Analysis geometry is not valid."
        )

    if geometry.area <= 0:
        raise DomainConstructionError(
            "Analysis geometry must have positive area."
        )

    return geometry


def transform_geometry(
    geometry: BaseGeometry,
    *,
    source_crs: CRS | str,
    target_crs: CRS | str,
) -> BaseGeometry:
    """Transform a Shapely geometry explicitly between CRSs."""

    validate_domain_geometry(
        geometry
    )

    source = CRS.from_user_input(
        source_crs
    )

    target = CRS.from_user_input(
        target_crs
    )

    if source == target:
        return geometry

    transformer = Transformer.from_crs(
        PyprojCRS.from_user_input(
            source.to_string()
        ),
        PyprojCRS.from_user_input(
            target.to_string()
        ),
        always_xy=True,
    )

    transformed = shapely_transform(
        transformer.transform,
        geometry,
    )

    return validate_domain_geometry(
        transformed
    )


def clip_geometry_to_grid(
    geometry: BaseGeometry,
    grid: AnalysisGrid,
) -> tuple[BaseGeometry, bool]:
    """Clip geometry to the analysis grid.

    Raises
    ------
    DomainConstructionError
        If the geometry does not overlap the grid.
    """

    validate_domain_geometry(
        geometry
    )

    extent = grid_extent_geometry(
        grid
    )

    clipped = geometry.intersection(
        extent
    )

    if clipped.is_empty:
        raise DomainConstructionError(
            "Analysis geometry does not overlap the elevation grid."
        )

    clipped = validate_domain_geometry(
        clipped
    )

    changed = not clipped.equals(
        geometry
    )

    return (
        clipped,
        changed,
    )


# ---------------------------------------------------------------------------
# Rasterization
# ---------------------------------------------------------------------------


def rasterize_domain_geometry(
    geometry: BaseGeometry,
    grid: AnalysisGrid,
) -> np.ndarray:
    """Rasterize domain geometry onto an AnalysisGrid."""

    validate_domain_geometry(
        geometry
    )

    if not isinstance(
        grid,
        AnalysisGrid,
    ):
        raise TypeError(
            "grid must be an AnalysisGrid."
        )

    # geometry_mask returns False inside supplied shapes by default.
    # invert=True therefore gives exactly the semantics required by
    # AnalysisDomain.analysis_mask: True = belongs to the analysis domain.
    mask = geometry_mask(
        [mapping(geometry)],
        out_shape=grid.shape,
        transform=grid.transform,
        invert=True,
        all_touched=False,
    )

    return mask.astype(
        bool,
        copy=False,
    )


# ---------------------------------------------------------------------------
# Generic constructor
# ---------------------------------------------------------------------------


def build_domain_from_geometry(
    *,
    geometry: BaseGeometry,
    geometry_crs: CRS | str,
    grid: AnalysisGrid,
    domain_id: str,
    name: str,
    creation_method: str,
    target_region: BaseGeometry | None = None,
    target_region_crs: CRS | str | None = None,
    observer_region: BaseGeometry | None = None,
    observer_region_crs: CRS | str | None = None,
    valid_mask: np.ndarray | None = None,
    clip_to_grid: bool = True,
    provenance: dict[str, Any] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> DomainConstructionResult:
    """Build a canonical AnalysisDomain from arbitrary geometry."""

    if not isinstance(
        grid,
        AnalysisGrid,
    ):
        raise TypeError(
            "grid must be an AnalysisGrid."
        )

    source_crs = CRS.from_user_input(
        geometry_crs
    )

    domain_geometry = transform_geometry(
        geometry,
        source_crs=source_crs,
        target_crs=grid.crs,
    )

    original_geometry = domain_geometry

    clipped = False

    if clip_to_grid:
        domain_geometry, clipped = (
            clip_geometry_to_grid(
                domain_geometry,
                grid,
            )
        )

    elif not grid_extent_geometry(
        grid
    ).contains(
        domain_geometry
    ):
        raise DomainConstructionError(
            "Analysis geometry extends outside the elevation grid. "
            "Enable clipping or choose a geometry within the grid."
        )

    target = _prepare_optional_region(
        target_region,
        region_crs=target_region_crs,
        fallback_crs=source_crs,
        grid=grid,
        clip_to_grid=clip_to_grid,
        name="target_region",
    )

    observer = _prepare_optional_region(
        observer_region,
        region_crs=observer_region_crs,
        fallback_crs=source_crs,
        grid=grid,
        clip_to_grid=clip_to_grid,
        name="observer_region",
    )

    analysis_mask = rasterize_domain_geometry(
        domain_geometry,
        grid,
    )

    if not np.any(
        analysis_mask
    ):
        raise DomainConstructionError(
            "Analysis geometry contains no grid-cell centres."
        )

    if valid_mask is not None:

        if not isinstance(
            valid_mask,
            np.ndarray,
        ):
            raise TypeError(
                "valid_mask must be a numpy.ndarray or None."
            )

        if valid_mask.shape != grid.shape:
            raise ValueError(
                f"valid_mask shape {valid_mask.shape} does not match "
                f"grid shape {grid.shape}."
            )

        valid_mask = valid_mask.astype(
            bool,
            copy=False,
        )

    domain = AnalysisDomain(
        domain_id=domain_id,
        name=name,
        geometry=domain_geometry,
        crs=grid.crs,
        grid=grid,
        target_region=target,
        observer_region=observer,
        analysis_mask=analysis_mask,
        valid_mask=valid_mask,
        creation_method=creation_method,
        provenance=dict(
            provenance
            or {}
        ),
        extra_metadata={
            "geometry_was_clipped_to_grid": clipped,
            **dict(
                extra_metadata
                or {}
            ),
        },
    )

    return DomainConstructionResult(
        domain=domain,
        original_geometry=original_geometry,
        clipped_to_grid=clipped,
        source_crs=source_crs,
        domain_crs=grid.crs,
    )


def _prepare_optional_region(
    geometry: BaseGeometry | None,
    *,
    region_crs: CRS | str | None,
    fallback_crs: CRS,
    grid: AnalysisGrid,
    clip_to_grid: bool,
    name: str,
) -> BaseGeometry | None:

    if geometry is None:
        return None

    source_crs = (
        fallback_crs
        if region_crs is None
        else CRS.from_user_input(
            region_crs
        )
    )

    result = transform_geometry(
        geometry,
        source_crs=source_crs,
        target_crs=grid.crs,
    )

    if clip_to_grid:
        result, _ = clip_geometry_to_grid(
            result,
            grid,
        )

    elif not grid_extent_geometry(
        grid
    ).contains(
        result
    ):
        raise DomainConstructionError(
            f"{name} extends outside the elevation grid."
        )

    return result


# ---------------------------------------------------------------------------
# Construction method 1 — complete terrain
# ---------------------------------------------------------------------------


def build_domain_from_grid_extent(
    *,
    grid: AnalysisGrid,
    domain_id: str,
    name: str,
    valid_mask: np.ndarray | None = None,
    provenance: dict[str, Any] | None = None,
) -> DomainConstructionResult:
    """Use the complete elevation-grid extent as AnalysisDomain."""

    geometry = grid_extent_geometry(
        grid
    )

    return build_domain_from_geometry(
        geometry=geometry,
        geometry_crs=grid.crs,
        grid=grid,
        domain_id=domain_id,
        name=name,
        creation_method="environment_extent",
        valid_mask=valid_mask,
        clip_to_grid=False,
        provenance={
            "construction": "complete_elevation_grid",
            **dict(
                provenance
                or {}
            ),
        },
    )


# ---------------------------------------------------------------------------
# Construction method 2 — buffered survey
# ---------------------------------------------------------------------------


def build_domain_from_survey_extent(
    *,
    viewpoints: Iterable[Any],
    grid: AnalysisGrid,
    domain_id: str,
    name: str,
    buffer_m: float,
    valid_mask: np.ndarray | None = None,
    clip_to_grid: bool = True,
    provenance: dict[str, Any] | None = None,
) -> DomainConstructionResult:
    """Build AnalysisDomain from Viewpoint extent plus metric buffer.

    Viewpoints may be expressed in another CRS. They are explicitly
    transformed to the grid CRS before the envelope is constructed.

    ``buffer_m`` requires a projected grid with metre-like linear units.
    """

    items = list(
        viewpoints
    )

    if not items:
        raise DomainConstructionError(
            "At least one Viewpoint is required to build a survey domain."
        )

    buffer_m = float(
        buffer_m
    )

    if (
        not np.isfinite(
            buffer_m
        )
        or buffer_m < 0
    ):
        raise ValueError(
            "buffer_m must be finite and non-negative."
        )

    if not grid.crs.is_projected:
        raise DomainConstructionError(
            "Survey buffering requires a projected AnalysisGrid. "
            "Geographic degrees cannot be treated as metres."
        )

    units = str(
        getattr(
            grid.crs,
            "linear_units",
            "",
        )
    ).lower()

    if units not in {
        "metre",
        "meter",
        "metres",
        "meters",
        "m",
    }:
        raise DomainConstructionError(
            "Survey buffering currently requires grid units in metres."
        )

    points = []

    transformer_cache: dict[
        str,
        Transformer,
    ] = {}

    for viewpoint in items:

        try:
            x = float(
                viewpoint.x
            )
            y = float(
                viewpoint.y
            )
            viewpoint_crs = CRS.from_user_input(
                viewpoint.crs
            )

        except (
            AttributeError,
            TypeError,
            ValueError,
        ) as exc:
            raise DomainConstructionError(
                "Every Viewpoint must provide valid x, y and crs."
            ) from exc

        if not (
            np.isfinite(x)
            and np.isfinite(y)
        ):
            raise DomainConstructionError(
                "Viewpoint coordinates must be finite."
            )

        if viewpoint_crs != grid.crs:

            key = viewpoint_crs.to_string()

            transformer = transformer_cache.get(
                key
            )

            if transformer is None:

                transformer = Transformer.from_crs(
                    PyprojCRS.from_user_input(
                        viewpoint_crs.to_string()
                    ),
                    PyprojCRS.from_user_input(
                        grid.crs.to_string()
                    ),
                    always_xy=True,
                )

                transformer_cache[
                    key
                ] = transformer

            x, y = transformer.transform(
                x,
                y,
            )

        points.append(
            (
                float(x),
                float(y),
            )
        )

    xs = [
        point[0]
        for point in points
    ]
    ys = [
        point[1]
        for point in points
    ]

    # A single Viewpoint or a perfectly vertical/horizontal configuration
    # has a zero-area envelope. Buffering makes it a valid area whenever
    # buffer_m > 0.
    envelope = box(
        min(xs),
        min(ys),
        max(xs),
        max(ys),
    )

    if envelope.area <= 0:

        from shapely.geometry import MultiPoint

        geometry = MultiPoint(
            points
        ).convex_hull

    else:
        geometry = envelope

    if buffer_m > 0:
        geometry = geometry.buffer(
            buffer_m
        )

    if geometry.area <= 0:
        raise DomainConstructionError(
            "The survey extent has zero area. "
            "Use a positive buffer for a single or collinear survey."
        )

    return build_domain_from_geometry(
        geometry=geometry,
        geometry_crs=grid.crs,
        grid=grid,
        domain_id=domain_id,
        name=name,
        creation_method="buffered_survey_extent",
        valid_mask=valid_mask,
        clip_to_grid=clip_to_grid,
        provenance={
            "construction": "survey_extent_buffer",
            "buffer_m": buffer_m,
            "viewpoint_count": len(items),
            **dict(
                provenance
                or {}
            ),
        },
    )


# ---------------------------------------------------------------------------
# Construction method 3 — map-drawn polygon
# ---------------------------------------------------------------------------


def polygon_from_vertices(
    vertices: Iterable[
        tuple[float, float]
    ],
) -> Polygon:
    """Create a valid Polygon from map-coordinate vertices.

    This is the bridge required by Connor's existing PolygonSelector tool.
    """

    points = [
        (
            float(x),
            float(y),
        )
        for x, y
        in vertices
    ]

    if len(points) < 3:
        raise DomainConstructionError(
            "A drawn domain requires at least three vertices."
        )

    if not all(
        np.isfinite(value)
        for point in points
        for value in point
    ):
        raise DomainConstructionError(
            "Drawn polygon vertices must be finite."
        )

    polygon = Polygon(
        points
    )

    return validate_domain_geometry(
        polygon
    )


def build_domain_from_drawn_polygon(
    *,
    vertices: Iterable[
        tuple[float, float]
    ],
    map_crs: CRS | str,
    grid: AnalysisGrid,
    domain_id: str,
    name: str,
    valid_mask: np.ndarray | None = None,
    clip_to_grid: bool = True,
) -> DomainConstructionResult:
    """Build AnalysisDomain from map-drawn polygon vertices."""

    polygon = polygon_from_vertices(
        vertices
    )

    return build_domain_from_geometry(
        geometry=polygon,
        geometry_crs=map_crs,
        grid=grid,
        domain_id=domain_id,
        name=name,
        creation_method="user_drawn_polygon",
        valid_mask=valid_mask,
        clip_to_grid=clip_to_grid,
        provenance={
            "construction": "map_polygon_selector",
        },
    )


# ---------------------------------------------------------------------------
# Construction method 4 — imported polygon
# ---------------------------------------------------------------------------


def read_geojson_geometry(
    path: str | Path,
    *,
    role: str | None = None,
) -> BaseGeometry:
    """Read polygonal geometry from GeoJSON.

    Parameters
    ----------
    path
        GeoJSON file.

    role
        Optional feature-property role used to select geometry, for example
        ``analysis_domain`` in the synthetic fixture.

    Notes
    -----
    GeoJSON CRS is not inferred here. The caller must explicitly provide the
    source CRS when constructing the AnalysisDomain.
    """

    source = Path(
        path
    ).expanduser().resolve()

    if not source.exists():
        raise FileNotFoundError(
            f"Domain file does not exist: {source}"
        )

    with source.open(
        "r",
        encoding="utf-8",
    ) as file:
        document = json.load(
            file
        )

    geometries = []

    if document.get(
        "type"
    ) == "FeatureCollection":

        for feature in document.get(
            "features",
            []
        ):
            if role is not None:
                properties = (
                    feature.get(
                        "properties"
                    )
                    or {}
                )

                if properties.get(
                    "role"
                ) != role:
                    continue

            geometry_mapping = feature.get(
                "geometry"
            )

            if geometry_mapping is not None:
                geometries.append(
                    shape(
                        geometry_mapping
                    )
                )

    elif document.get(
        "type"
    ) == "Feature":

        if role is not None:
            properties = (
                document.get(
                    "properties"
                )
                or {}
            )

            if properties.get(
                "role"
            ) != role:
                raise DomainConstructionError(
                    f"GeoJSON feature does not have role {role!r}."
                )

        geometry_mapping = document.get(
            "geometry"
        )

        if geometry_mapping is not None:
            geometries.append(
                shape(
                    geometry_mapping
                )
            )

    else:
        geometries.append(
            shape(
                document
            )
        )

    if not geometries:
        raise DomainConstructionError(
            "No matching geometry was found in the GeoJSON."
        )

    geometry = unary_union(
        geometries
    )

    return validate_domain_geometry(
        geometry
    )


def build_domain_from_geojson(
    *,
    path: str | Path,
    source_crs: CRS | str,
    grid: AnalysisGrid,
    domain_id: str,
    name: str,
    role: str | None = None,
    valid_mask: np.ndarray | None = None,
    clip_to_grid: bool = True,
) -> DomainConstructionResult:
    """Build AnalysisDomain from a GeoJSON polygon source."""

    geometry = read_geojson_geometry(
        path,
        role=role,
    )

    return build_domain_from_geometry(
        geometry=geometry,
        geometry_crs=source_crs,
        grid=grid,
        domain_id=domain_id,
        name=name,
        creation_method="imported_polygon",
        valid_mask=valid_mask,
        clip_to_grid=clip_to_grid,
        provenance={
            "construction": "imported_geojson",
            "source_path": str(
                Path(
                    path
                ).expanduser()
            ),
            "source_crs": (
                CRS.from_user_input(
                    source_crs
                ).to_string()
            ),
            "role": role,
        },
    )