"""
Low-level terrain line-of-sight engine for Rivelero.

Owns GDAL viewshed execution only. Pass 1 preserves the GDAL configuration
and raster-placement behaviour formerly embedded in visibility_field.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from affine import Affine

try:
    from osgeo import gdal
except ImportError as error:
    gdal = None
    _GDAL_IMPORT_ERROR = error
else:
    _GDAL_IMPORT_ERROR = None
    # The viewshed code checks GDAL return values itself; state that mode
    # explicitly (GDAL 4 will otherwise switch the default to exceptions).
    gdal.DontUseExceptions()

__all__ = ["open_viewshed_dem", "run_viewshed", "accumulate_viewshed"]

def open_viewshed_dem(dem_path: str | Path, *, strip_crs: bool = False) -> Any:
    _require_gdal()
    return _open_gdal_viewshed_dataset(Path(dem_path), strip_crs=strip_crs)

def run_viewshed(
    *, band: Any, observer_x: float, observer_y: float,
    observer_height_m: float, target_height_m: float,
    max_distance_m: float, curvature_coefficient: float,
) -> Any:
    _require_gdal()
    return _run_gdal_viewshed(
        band=band, observer_x=observer_x, observer_y=observer_y,
        temporary_observer_height=observer_height_m,
        temporary_target_height=target_height_m,
        max_distance_m=max_distance_m,
        curvature_coefficient=curvature_coefficient,
    )

def accumulate_viewshed(
    *, accumulator: np.ndarray, viewshed_dataset: Any, dem_transform: Affine,
) -> None:
    _accumulate_viewshed(
        accumulator=accumulator,
        viewshed_dataset=viewshed_dataset,
        dem_transform=dem_transform,
    )

def _open_gdal_viewshed_dataset(
    dem_path: Path,
    *,
    strip_crs: bool,
) -> Any:
    """
    Open a DEM for GDAL ViewshedGenerate.

    Normal GIS DEMs are returned directly. For an Unreal-local DEM, a temporary
    GDAL MEM copy is created with the same raster values and GeoTransform but
    with no CRS. This prevents GDAL/PROJ from trying to obtain an Earth
    ellipsoid from an engineering/local Unreal CRS.
    """
    source_dataset = gdal.Open(str(dem_path), gdal.GA_ReadOnly)

    if source_dataset is None:
        raise RuntimeError(f"GDAL could not open DEM: {dem_path}")

    if not strip_crs:
        return source_dataset

    source_band = source_dataset.GetRasterBand(1)

    if source_band is None:
        source_dataset = None
        raise RuntimeError("GDAL could not access DEM band 1.")

    memory_driver = gdal.GetDriverByName("MEM")

    if memory_driver is None:
        source_dataset = None
        raise RuntimeError("GDAL MEM driver is unavailable.")

    local_dataset = memory_driver.Create(
        "",
        source_dataset.RasterXSize,
        source_dataset.RasterYSize,
        1,
        source_band.DataType,
    )

    if local_dataset is None:
        source_dataset = None
        raise RuntimeError(
            "GDAL could not create the temporary Unreal-local DEM."
        )

    # Preserve the local affine coordinates, but deliberately do NOT copy
    # source_dataset.GetSpatialRef().
    local_dataset.SetGeoTransform(source_dataset.GetGeoTransform())

    local_band = local_dataset.GetRasterBand(1)

    if local_band is None:
        source_dataset = None
        local_dataset = None
        raise RuntimeError(
            "GDAL could not access the temporary Unreal-local DEM band."
        )

    source_array = source_band.ReadAsArray()

    if source_array is None:
        source_dataset = None
        local_dataset = None
        raise RuntimeError("GDAL could not read the Unreal-local DEM.")

    local_band.WriteArray(source_array)

    nodata = source_band.GetNoDataValue()
    if nodata is not None:
        local_band.SetNoDataValue(nodata)

    local_band.FlushCache()
    local_dataset.FlushCache()

    # The in-memory dataset is now independent of the file-backed source.
    source_dataset = None

    return local_dataset


def _run_gdal_viewshed(
    *,
    band: Any,
    observer_x: float,
    observer_y: float,
    temporary_observer_height: float,
    temporary_target_height: float,
    max_distance_m: float,
    curvature_coefficient: float,
) -> Any:
    """
    Run one binary GDAL viewshed and return its in-memory dataset.

    ``ViewshedGenerate`` added an extra-options argument in newer GDAL
    versions, so the function tries the modern call and then the older
    compatible signature.
    """
    mode = gdal.GVM_Edge
    output_type = gdal.GVOT_NORMAL

    common_arguments = (
        band,
        "MEM",
        "",
        [],
        float(observer_x),
        float(observer_y),
        float(temporary_observer_height),
        float(temporary_target_height),
        1.0,   # visible
        0.0,   # invisible
        0.0,   # outside maximum range
        0.0,   # output NoData value
        float(curvature_coefficient),
        mode,
        float(max_distance_m),
        None,  # progress callback
        None,  # callback data
        output_type,
    )

    try:
        viewshed = gdal.ViewshedGenerate(
            *common_arguments,
            [],  # modern GDAL extra options
        )
    except TypeError:
        viewshed = gdal.ViewshedGenerate(*common_arguments)

    if viewshed is None:
        raise RuntimeError(
            "GDAL ViewshedGenerate returned no output dataset."
        )

    return viewshed


def _accumulate_viewshed(
    *,
    accumulator: np.ndarray,
    viewshed_dataset: Any,
    dem_transform: Affine,
) -> None:
    """
    Insert GDAL's possibly cropped viewshed into the full DEM-sized accumulator.
    """
    viewshed_array = viewshed_dataset.GetRasterBand(1).ReadAsArray()

    if viewshed_array is None:
        raise RuntimeError("GDAL produced a viewshed with no raster data.")

    viewshed_array = np.asarray(viewshed_array)
    viewshed_transform = Affine.from_gdal(
        *viewshed_dataset.GetGeoTransform()
    )

    # Convert the output raster's upper-left corner into DEM pixel coordinates.
    column_float, row_float = (~dem_transform) * (
        viewshed_transform.c,
        viewshed_transform.f,
    )

    destination_column0 = int(round(column_float))
    destination_row0 = int(round(row_float))

    source_height, source_width = viewshed_array.shape

    destination_row1 = destination_row0 + source_height
    destination_column1 = destination_column0 + source_width

    clipped_row0 = max(0, destination_row0)
    clipped_column0 = max(0, destination_column0)
    clipped_row1 = min(accumulator.shape[0], destination_row1)
    clipped_column1 = min(accumulator.shape[1], destination_column1)

    if clipped_row0 >= clipped_row1 or clipped_column0 >= clipped_column1:
        raise RuntimeError(
            "The GDAL viewshed does not overlap the source DEM grid."
        )

    source_row0 = clipped_row0 - destination_row0
    source_column0 = clipped_column0 - destination_column0
    source_row1 = source_row0 + (clipped_row1 - clipped_row0)
    source_column1 = source_column0 + (
        clipped_column1 - clipped_column0
    )

    visible = (
        viewshed_array[
            source_row0:source_row1,
            source_column0:source_column1,
        ]
        == 1
    )

    accumulator[
        clipped_row0:clipped_row1,
        clipped_column0:clipped_column1,
    ] += visible



def _require_gdal() -> None:
    if gdal is None:
        raise ImportError(
            "The GDAL Python bindings are required. Install GDAL so that "
            "'from osgeo import gdal' works in the same Python environment."
        ) from _GDAL_IMPORT_ERROR