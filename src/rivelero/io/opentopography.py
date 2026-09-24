"""OpenTopography terrain ready for the Rivelero visibility engine.

OpenTopography's global datasets are delivered on geographic (WGS84) grids,
while the visibility engine needs a projected grid in metres. This module
downloads a dataset for an area of interest and warps it to a projected CRS
with an explicit, square cell size, keeping the original download alongside.

It also describes each dataset (native resolution, kind of surface,
coverage, per-request area limit) and checks the format of API keys. The
module does not depend on Qt; the API key is never stored in any output.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import requests
from pyproj import CRS as PyprojCRS

from rivelero.io.dem import (
    OPENTOPO_GLOBALDEM_URL,
    OpenTopographyError,
    _validate_bbox,
    download_dem_from_opentopo,
    normalize_opentopo_api_key,
)

OPENTOPO_API_KEY_ENVIRONMENT_VARIABLE = "OPENTOPO_API_KEY"
# Where users request a free key (My OpenTopo › Request API key).
OPENTOPO_API_KEY_URL = "https://portal.opentopography.org/myopentopo"

# OpenTopography keys are 32 hexadecimal characters.
_API_KEY_PATTERN = re.compile(r"^[0-9a-fA-F]{32}$")
# Plausible keys of another format are tried with a warning rather than
# refused, so a valid key is never blocked by this check.
_API_KEY_PLAUSIBLE = re.compile(r"^[A-Za-z0-9_-]{8,128}$")

# Metres per degree of latitude (mean).
METRES_PER_DEGREE = 111_320.0
# Above this many output cells, visibility computation becomes slow.
LARGE_GRID_CELLS = 25_000_000

RESAMPLING_METHODS = ("bilinear", "cubic", "nearest")


@dataclass(frozen=True)
class OpenTopographyDataset:
    """One OpenTopography global elevation dataset."""

    code: str
    label: str
    # Native grid spacing in arc-seconds (1 arc-second ≈ 30.9 m of latitude).
    native_arcsec: float
    nominal_resolution_m: float
    surface: str
    coverage: str
    # Rivelero ElevationModelType value recorded for the terrain.
    model_type: str = "dem"
    # OpenTopography's largest area per request.
    max_area_km2: float = 450_000.0

    @property
    def native_resolution_text(self) -> str:
        unit = "arc-second" if self.native_arcsec == 1 else "arc-seconds"
        return f"{self.native_arcsec:g} {unit} (≈ {self.nominal_resolution_m:g} m)"


_SURFACE_MODEL = "surface model: includes buildings and forest canopy"

OPENTOPO_DATASETS: dict[str, OpenTopographyDataset] = {
    dataset.code: dataset
    for dataset in (
        OpenTopographyDataset("COP30", "Copernicus GLO-30", 1.0, 30.0, _SURFACE_MODEL, "global"),
        OpenTopographyDataset("COP90", "Copernicus GLO-90", 3.0, 90.0, _SURFACE_MODEL, "global",
                              max_area_km2=4_050_000.0),
        OpenTopographyDataset("NASADEM", "NASADEM", 1.0, 30.0,
                              "radar surface model (reprocessed SRTM)", "60°N – 56°S"),
        OpenTopographyDataset("SRTMGL1", "SRTM GL1", 1.0, 30.0,
                              "radar surface model", "60°N – 56°S"),
        OpenTopographyDataset("SRTMGL3", "SRTM GL3", 3.0, 90.0,
                              "radar surface model", "60°N – 56°S", max_area_km2=4_050_000.0),
        OpenTopographyDataset("AW3D30", "ALOS World 3D", 1.0, 30.0,
                              "photogrammetric surface model", "82°N – 82°S"),
        OpenTopographyDataset("EU_DTM", "EU DTM", 1.0, 30.0,
                              "bare-earth terrain model (buildings and vegetation removed)",
                              "Europe only", model_type="dtm"),
    )
}


# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ApiKeyCheck:
    """Result of checking an API key's format (not whether it is active)."""

    status: str  # "empty", "valid", "unusual" or "invalid"
    key: str
    message: str

    @property
    def usable(self) -> bool:
        return self.status in {"valid", "unusual"}


def check_opentopo_api_key(text: str | None) -> ApiKeyCheck:
    """Check the format of an OpenTopography API key.

    Only the format is checked; OpenTopography decides whether the key is
    active when the download is requested.
    """

    key = normalize_opentopo_api_key(text)
    if not key:
        return ApiKeyCheck("empty", "", "Enter your OpenTopography API key.")
    if _API_KEY_PATTERN.match(key):
        return ApiKeyCheck("valid", key, "Key format is valid (32 hexadecimal characters).")
    if any(character.isspace() for character in key):
        return ApiKeyCheck("invalid", key,
                           "The key cannot contain spaces. Paste only the key itself.")
    if _API_KEY_PLAUSIBLE.match(key):
        return ApiKeyCheck(
            "unusual", key,
            "OpenTopography keys are normally 32 hexadecimal characters; this one "
            f"has {len(key)} characters. It will be tried as entered.",
        )
    return ApiKeyCheck(
        "invalid", key,
        "The key may only contain letters, digits, '-' and '_' "
        "(normally 32 hexadecimal characters: 0–9 and a–f).",
    )


# ---------------------------------------------------------------------------
# Area, resolution and CRS
# ---------------------------------------------------------------------------


def bbox_area_km2(bbox: tuple[float, float, float, float]) -> float:
    """Area of a WGS84 ``(south, north, west, east)`` box on a sphere."""

    south, north, west, east = _validate_bbox(*bbox)
    radius_km = 6371.0088
    return (
        radius_km ** 2
        * math.radians(east - west)
        * abs(math.sin(math.radians(north)) - math.sin(math.radians(south)))
    )


def bbox_size_m(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    """Approximate (east-west, north-south) size of a WGS84 box in metres."""

    south, north, west, east = _validate_bbox(*bbox)
    centre = math.radians((south + north) / 2.0)
    return (
        (east - west) * METRES_PER_DEGREE * math.cos(centre),
        (north - south) * METRES_PER_DEGREE,
    )


def native_cell_size_m(
    dataset: OpenTopographyDataset | str, latitude: float
) -> tuple[float, float]:
    """Native (east-west, north-south) cell size in metres at ``latitude``.

    The datasets are on geographic grids, so the east-west spacing shrinks
    with the cosine of the latitude.
    """

    if isinstance(dataset, str):
        dataset = OPENTOPO_DATASETS[dataset]
    north_south = dataset.native_arcsec / 3600.0 * METRES_PER_DEGREE
    return north_south * math.cos(math.radians(float(latitude))), north_south


def estimated_grid_shape(
    bbox: tuple[float, float, float, float], resolution_m: float
) -> tuple[int, int]:
    """Approximate (rows, columns) of the projected terrain."""

    width, height = bbox_size_m(bbox)
    return max(1, math.ceil(height / resolution_m)), max(1, math.ceil(width / resolution_m))


def utm_crs_for(longitude: float, latitude: float) -> PyprojCRS:
    """WGS84 / UTM zone CRS containing ``(longitude, latitude)``."""

    zone = int((float(longitude) + 180.0) // 6.0) % 60 + 1
    return PyprojCRS.from_epsg((32600 if latitude >= 0 else 32700) + zone)


def _is_metric(crs: PyprojCRS) -> bool:
    try:
        return all(
            (axis.unit_name or "").lower() in {"metre", "meter", "metres", "meters", "m"}
            for axis in crs.axis_info[:2]
        )
    except Exception:  # noqa: BLE001
        return False


def validate_projected_output_crs(crs: Any) -> PyprojCRS:
    """Return ``crs`` if it is projected in metres, else raise ValueError."""

    try:
        resolved = PyprojCRS.from_user_input(crs)
    except Exception:  # noqa: BLE001
        raise ValueError(f"Unknown CRS: {crs!r}.") from None
    if not resolved.is_projected or not _is_metric(resolved):
        raise ValueError(
            "The terrain CRS must be projected in metres (e.g. a UTM zone); "
            "the visibility engine cannot use geographic coordinates."
        )
    return resolved


def default_output_crs(
    viewpoints: Iterable[Any], bbox: tuple[float, float, float, float]
) -> PyprojCRS:
    """Projected CRS for downloaded terrain.

    The survey's CRS when all Viewpoints share one projected CRS in metres,
    so survey and terrain need no transformation; otherwise the UTM zone of
    the area's centre.
    """

    names = set()
    for viewpoint in viewpoints:
        try:
            names.add(PyprojCRS.from_user_input(viewpoint.crs).to_string())
        except Exception:  # noqa: BLE001 - unusable CRS: fall back to UTM
            names.add(None)
    if len(names) == 1 and None not in names:
        try:
            return validate_projected_output_crs(next(iter(names)))
        except ValueError:
            pass
    south, north, west, east = bbox
    return utm_crs_for((west + east) / 2.0, (south + north) / 2.0)


# ---------------------------------------------------------------------------
# Reprojection and download
# ---------------------------------------------------------------------------


def reproject_dem(
    source_path: str | Path,
    output_path: str | Path,
    *,
    crs: Any,
    resolution_m: float,
    resampling: str = "bilinear",
) -> Path:
    """Warp an elevation raster to ``crs`` with square ``resolution_m`` cells.

    Elevation is continuous, so bilinear resampling is the default. The
    output is a tiled, compressed float32 GeoTIFF with NoData preserved.
    """

    import numpy as np
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    resolution_m = float(resolution_m)
    if not math.isfinite(resolution_m) or resolution_m <= 0:
        raise ValueError("The cell size must be finite and greater than zero.")
    if resampling not in RESAMPLING_METHODS:
        raise ValueError(f"resampling must be one of {', '.join(RESAMPLING_METHODS)}.")
    target = CRS.from_wkt(validate_projected_output_crs(crs).to_wkt())

    source_path = Path(source_path)
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(source_path) as source:
        if source.crs is None:
            raise ValueError("The downloaded raster has no CRS.")
        transform, width, height = calculate_default_transform(
            source.crs, target, source.width, source.height, *source.bounds,
            resolution=(resolution_m, resolution_m),
        )
        nodata = float(source.nodata) if source.nodata is not None else -9999.0
        destination = np.full((height, width), nodata, dtype=np.float32)
        reproject(
            source=rasterio.band(source, 1),
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=source.nodata,
            dst_transform=transform,
            dst_crs=target,
            dst_nodata=nodata,
            resampling=Resampling[resampling],
        )

    profile = {
        "driver": "GTiff", "crs": target, "transform": transform,
        "width": width, "height": height, "count": 1, "dtype": "float32",
        "nodata": nodata, "compress": "deflate", "predictor": 3, "tiled": True,
        "blockxsize": 256, "blockysize": 256, "BIGTIFF": "IF_SAFER",
    }
    temporary = output_path.with_name(output_path.stem + ".part.tif")
    with rasterio.open(temporary, "w", **profile) as output:
        output.write(destination, 1)
        output.update_tags(
            RIVELERO_SOURCE=source_path.name,
            RIVELERO_RESOLUTION_M=f"{resolution_m:g}",
            RIVELERO_RESAMPLING=resampling,
        )
    temporary.replace(output_path)
    return output_path


@dataclass(frozen=True)
class OpenTopographyDownload:
    """A downloaded dataset and the projected terrain made from it."""

    dataset: OpenTopographyDataset
    bbox: tuple[float, float, float, float]
    area_km2: float
    raw_path: Path
    raw_cell_size_deg: tuple[float, float]
    output_path: Path
    output_crs: str
    resolution_m: float
    resampling: str
    shape: tuple[int, int]

    def provenance(self) -> dict[str, Any]:
        """Acquisition record for the Environment (never the API key)."""

        south, north, west, east = self.bbox
        return {
            "acquisition": "OpenTopography",
            "api": OPENTOPO_GLOBALDEM_URL,
            "dataset": self.dataset.code,
            "dataset_name": self.dataset.label,
            "dataset_surface": self.dataset.surface,
            "native_resolution": self.dataset.native_resolution_text,
            "native_resolution_arcsec": self.dataset.native_arcsec,
            "downloaded_cell_size_deg": list(self.raw_cell_size_deg),
            "bbox_wgs84": {"south": south, "north": north, "west": west, "east": east},
            "area_km2": round(self.area_km2, 3),
            "downloaded_file": str(self.raw_path),
            "output_crs": self.output_crs,
            "output_resolution_m": self.resolution_m,
            "resampling": self.resampling,
            "output_shape": list(self.shape),
        }


def raw_download_path(output_path: str | Path, dataset: str) -> Path:
    """Where the original (geographic) download is kept."""

    output_path = Path(output_path).expanduser().resolve()
    return output_path.with_name(f"{output_path.stem}_{dataset}_wgs84.tif")


def download_projected_dem(
    bbox: tuple[float, float, float, float],
    *,
    dataset: str,
    api_key: str,
    output_path: str | Path,
    crs: Any,
    resolution_m: float,
    resampling: str = "bilinear",
    session: requests.Session | None = None,
    timeout: float = 300.0,
) -> OpenTopographyDownload:
    """Download ``dataset`` for ``bbox`` and warp it for the visibility engine.

    ``bbox`` is ``(south, north, west, east)`` in WGS84 degrees. The original
    geographic GeoTIFF is kept as :func:`raw_download_path`; ``output_path``
    becomes the projected terrain with square ``resolution_m`` cells in
    ``crs``.
    """

    import rasterio

    if dataset not in OPENTOPO_DATASETS:
        raise ValueError(f"Unknown OpenTopography dataset: {dataset}.")
    info = OPENTOPO_DATASETS[dataset]
    south, north, west, east = _validate_bbox(*bbox)
    area = bbox_area_km2((south, north, west, east))
    if area > info.max_area_km2:
        raise ValueError(
            f"The area ({area:,.0f} km²) exceeds OpenTopography's limit of "
            f"{info.max_area_km2:,.0f} km² per {info.label} request."
        )
    check = check_opentopo_api_key(api_key)
    if not check.usable:
        raise OpenTopographyError(check.message)
    target = validate_projected_output_crs(crs)

    output_path = Path(output_path).expanduser().resolve()
    raw_path = raw_download_path(output_path, info.code)
    download_dem_from_opentopo(
        south, north, west, east, info.code,
        api_key=check.key, output_path=raw_path, timeout=timeout, session=session,
    )
    with rasterio.open(raw_path) as raw:
        raw_cell = (abs(raw.transform.a), abs(raw.transform.e))

    reproject_dem(raw_path, output_path, crs=target, resolution_m=resolution_m,
                  resampling=resampling)
    with rasterio.open(output_path) as output:
        shape = (output.height, output.width)

    return OpenTopographyDownload(
        dataset=info,
        bbox=(south, north, west, east),
        area_km2=area,
        raw_path=raw_path,
        raw_cell_size_deg=raw_cell,
        output_path=output_path,
        output_crs=target.to_string(),
        resolution_m=float(resolution_m),
        resampling=resampling,
        shape=shape,
    )
