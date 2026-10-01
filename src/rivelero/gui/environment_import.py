"""Environment import services for the Rivelero GUI.

This module converts elevation rasters into canonical Rivelero Environment
objects without loading the complete raster into memory.

It is intentionally independent of Qt so the same service can be used by
dialogs, tests, scripts, and future project persistence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from rasterio.coords import BoundingBox
from rasterio.crs import CRS

from rivelero.core.domain import AnalysisGrid
from rivelero.core.environment import (
    ElevationModel,
    ElevationModelType,
    Environment,
)


class EnvironmentImportError(ValueError):
    """Raised when an elevation source cannot form a valid Environment."""


@dataclass(frozen=True, slots=True)
class RasterMetadata:
    """Lightweight metadata describing one elevation raster."""

    path: Path

    crs: CRS
    transform: Affine

    width: int
    height: int

    bounds: BoundingBox

    resolution_x: float
    resolution_y: float

    # Canonical scalar metric resolution when it can be stated honestly.
    # Geographic-degree rasters therefore receive None here.
    resolution_m: float | None

    nodata_value: float | None

    dtype: str
    band_count: int
    driver: str

    is_projected: bool
    linear_units: str | None

    @property
    def shape(self) -> tuple[int, int]:
        """Raster shape as rows, columns."""

        return (
            self.height,
            self.width,
        )

    @property
    def analysis_grid(self) -> AnalysisGrid:
        """Return an AnalysisGrid aligned exactly with the raster."""

        return AnalysisGrid(
            crs=self.crs,
            transform=self.transform,
            width=self.width,
            height=self.height,
            resolution_x=self.resolution_x,
            resolution_y=self.resolution_y,
        )


@dataclass(frozen=True, slots=True)
class EnvironmentImportResult:
    """Result of importing an elevation raster."""

    environment: Environment

    raster: RasterMetadata

    grid: AnalysisGrid


def inspect_elevation_raster(
    path: str | Path,
) -> RasterMetadata:
    """Inspect an elevation raster without reading its pixel array.

    Parameters
    ----------
    path
        Local raster path.

    Returns
    -------
    RasterMetadata
        Spatial and storage metadata required by the World workflow.

    Notes
    -----
    The full DEM is deliberately not read into RAM here. Later World-map
    rendering can therefore reuse Rivelero's dynamic/windowed raster
    infrastructure.
    """

    source = Path(
        path
    ).expanduser().resolve()

    if not source.exists():
        raise FileNotFoundError(
            f"Elevation raster does not exist: {source}"
        )

    if not source.is_file():
        raise EnvironmentImportError(
            f"Elevation source is not a file: {source}"
        )

    try:
        with rasterio.open(
            source
        ) as dataset:

            if dataset.count < 1:
                raise EnvironmentImportError(
                    "Elevation raster contains no bands."
                )

            if dataset.crs is None:
                raise EnvironmentImportError(
                    "Elevation raster has no CRS. "
                    "Assign a CRS before using it in Rivelero."
                )

            crs = CRS.from_user_input(
                dataset.crs
            )

            transform = dataset.transform

            resolution_x = float(
                abs(
                    transform.a
                )
            )

            resolution_y = float(
                abs(
                    transform.e
                )
            )

            if (
                not np.isfinite(
                    resolution_x
                )
                or not np.isfinite(
                    resolution_y
                )
                or resolution_x <= 0
                or resolution_y <= 0
            ):
                raise EnvironmentImportError(
                    "Elevation raster has invalid grid resolution."
                )

            linear_units = None
            resolution_m = None

            if crs.is_projected:

                try:
                    linear_units = str(
                        crs.linear_units
                    )

                except AttributeError:
                    linear_units = None

                if (
                    linear_units
                    and linear_units.lower()
                    in {
                        "metre",
                        "meter",
                        "metres",
                        "meters",
                        "m",
                    }
                ):
                    # ElevationModel currently stores one nominal scalar
                    # resolution. The exact X/Y values remain available
                    # through RasterMetadata and AnalysisGrid.
                    resolution_m = (
                        resolution_x
                        + resolution_y
                    ) / 2.0

            return RasterMetadata(
                path=source,
                crs=crs,
                transform=transform,
                width=int(
                    dataset.width
                ),
                height=int(
                    dataset.height
                ),
                bounds=dataset.bounds,
                resolution_x=resolution_x,
                resolution_y=resolution_y,
                resolution_m=resolution_m,
                nodata_value=dataset.nodata,
                dtype=str(
                    dataset.dtypes[0]
                ),
                band_count=int(
                    dataset.count
                ),
                driver=str(
                    dataset.driver
                ),
                is_projected=bool(
                    crs.is_projected
                ),
                linear_units=linear_units,
            )

    except EnvironmentImportError:
        raise

    except rasterio.errors.RasterioError as exc:
        raise EnvironmentImportError(
            f"Unable to open elevation raster {source}: {exc}"
        ) from exc


def import_elevation_environment(
    path: str | Path,
    *,
    environment_id: str,
    name: str,
    model_type: ElevationModelType | str = ElevationModelType.DEM,
    source_name: str | None = None,
    description: str | None = None,
    vertical_datum: str | None = None,
    vertical_accuracy_m: float | None = None,
    provenance: dict[str, Any] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> EnvironmentImportResult:
    """Create a canonical Environment from a local elevation raster.

    DSM is accepted by the canonical Environment model. The GUI may still
    expose DSM as a future feature until the current visibility workflow has
    been explicitly validated for that use.
    """

    metadata = inspect_elevation_raster(
        path
    )

    elevation_provenance = dict(
        provenance
        or {}
    )

    elevation_provenance.setdefault(
        "import_method",
        "local_raster",
    )

    elevation_provenance.setdefault(
        "source_path",
        str(
            metadata.path
        ),
    )

    elevation = ElevationModel(
        source=metadata.path,
        model_type=model_type,
        crs=metadata.crs,
        resolution_m=metadata.resolution_m,
        nodata_value=metadata.nodata_value,
        vertical_datum=vertical_datum,
        vertical_accuracy_m=vertical_accuracy_m,
        source_name=source_name,
        provenance=elevation_provenance,
        extra_metadata={
            "width": metadata.width,
            "height": metadata.height,
            "resolution_x": metadata.resolution_x,
            "resolution_y": metadata.resolution_y,
            "bounds": list(
                metadata.bounds
            ),
            "dtype": metadata.dtype,
            "band_count": metadata.band_count,
            "driver": metadata.driver,
            **dict(
                extra_metadata
                or {}
            ),
        },
    )

    environment = Environment(
        environment_id=environment_id,
        name=name,
        elevation_model=elevation,
        description=description,
        provenance={
            "elevation_source": str(
                metadata.path
            ),
        },
    )

    return EnvironmentImportResult(
        environment=environment,
        raster=metadata,
        grid=metadata.analysis_grid,
    )