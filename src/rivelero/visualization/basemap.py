"""OpenStreetMap background for Rivelero maps.

``render_basemap(extent, crs, size)`` returns an RGBA image of OpenStreetMap
tiles covering ``extent`` (left, right, bottom, top) in the map's own CRS, so
it can be drawn beneath survey, terrain and observability layers without
transforming any scientific data. Tiles are Web Mercator (EPSG:3857); they
are mosaicked and warped into ``crs`` for display only.

The OpenStreetMap tile usage policy is respected: requests carry an
identifying User-Agent, tiles are cached on disk and reused, the number of
tiles per view is bounded, and the attribution must be shown with the map
(:data:`OSM_ATTRIBUTION`).

The module does not depend on Qt.
"""

from __future__ import annotations

import io
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from matplotlib import image as mpl_image
from rasterio.crs import CRS
from rasterio.transform import from_bounds, from_origin
from rasterio.warp import Resampling, reproject, transform_bounds

from rivelero import __version__

OSM_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
OSM_ATTRIBUTION = "© OpenStreetMap contributors"
USER_AGENT = (
    f"Rivelero/{__version__} (spatial observability of field surveys; "
    "desktop application)"
)

TILE_SIZE = 256
MAX_ZOOM = 19
# Upper bound on tiles fetched for one view; the zoom is lowered to stay
# within it, which also keeps first loads quick.
MAX_TILES = 48
# Cached tiles are reused for this long before being downloaded again.
CACHE_MAX_AGE_SECONDS = 30 * 24 * 3600
REQUEST_TIMEOUT_SECONDS = 10.0

# Half the side of the Web Mercator square, in metres.
_MERCATOR_HALF = 20037508.342789244
_MERCATOR_LATITUDE_LIMIT = 85.05112878
_WEB_MERCATOR = CRS.from_epsg(3857)
_WGS84 = CRS.from_epsg(4326)

# Returns the PNG bytes of tile (z, x, y); raises on failure.
TileFetcher = Callable[[int, int, int], bytes]


class BasemapUnavailableError(RuntimeError):
    """Raised when no background tile could be obtained for a view."""


@dataclass(frozen=True)
class BasemapImage:
    """RGBA image (rows, columns, 4; uint8) covering ``extent`` in ``crs``."""

    rgba: np.ndarray
    extent: tuple[float, float, float, float]
    crs: CRS
    zoom: int


# ---------------------------------------------------------------------------
# Tile cache and download
# ---------------------------------------------------------------------------


def default_tile_cache_directory() -> Path:
    """Per-user directory for cached tiles (``RIVELERO_CACHE_DIR`` overrides)."""

    override = os.environ.get("RIVELERO_CACHE_DIR")
    if override:
        root = Path(override)
    elif sys.platform == "win32" and os.environ.get("LOCALAPPDATA"):
        root = Path(os.environ["LOCALAPPDATA"]) / "Rivelero"
    else:
        root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "rivelero"
    return (root / "basemap_tiles" / "osm").expanduser()


class CachedTileFetcher:
    """Fetch OpenStreetMap tiles over HTTP, keeping a disk cache."""

    def __init__(
        self,
        cache_directory: Path | str | None = None,
        *,
        url_template: str = OSM_TILE_URL,
        session: Any | None = None,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
        max_age: float = CACHE_MAX_AGE_SECONDS,
    ) -> None:
        self.cache_directory = (
            default_tile_cache_directory()
            if cache_directory is None else Path(cache_directory)
        )
        self.url_template = url_template
        self.timeout = float(timeout)
        self.max_age = float(max_age)
        self._session = session

    def _client(self):
        if self._session is None:
            import requests

            self._session = requests.Session()
            self._session.headers["User-Agent"] = USER_AGENT
        return self._session

    def __call__(self, z: int, x: int, y: int) -> bytes:
        path = self.cache_directory / str(z) / str(x) / f"{y}.png"
        try:
            if time.time() - path.stat().st_mtime < self.max_age:
                return path.read_bytes()
        except OSError:
            pass

        response = self._client().get(
            self.url_template.format(z=z, x=x, y=y), timeout=self.timeout
        )
        response.raise_for_status()
        content = response.content

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".part")
            temporary.write_bytes(content)
            temporary.replace(path)
        except OSError:
            pass  # An unwritable cache only costs a later download.
        return content


_default_fetcher: CachedTileFetcher | None = None


def default_tile_fetcher() -> CachedTileFetcher:
    global _default_fetcher
    if _default_fetcher is None:
        _default_fetcher = CachedTileFetcher()
    return _default_fetcher


# ---------------------------------------------------------------------------
# Tile geometry
# ---------------------------------------------------------------------------


def tile_resolution(zoom: int) -> float:
    """Metres per tile pixel (at the equator) of Web Mercator ``zoom``."""

    return 2 * _MERCATOR_HALF / (TILE_SIZE * 2**zoom)


def tile_range(
    mercator_bounds: tuple[float, float, float, float], zoom: int
) -> tuple[int, int, int, int]:
    """Inclusive tile indices (x0, x1, y0, y1) covering ``mercator_bounds``.

    ``mercator_bounds`` is (left, bottom, right, top) in EPSG:3857.
    """

    left, bottom, right, top = mercator_bounds
    span = tile_resolution(zoom) * TILE_SIZE
    last = 2**zoom - 1

    def index(value: float) -> int:
        return min(last, max(0, int(math.floor(value / span))))

    x0 = index(left + _MERCATOR_HALF)
    x1 = index(right + _MERCATOR_HALF - 1e-6)
    y0 = index(_MERCATOR_HALF - top)
    y1 = index(_MERCATOR_HALF - bottom - 1e-6)
    return x0, x1, y0, y1


def choose_zoom(
    mercator_bounds: tuple[float, float, float, float],
    width_px: int,
    *,
    max_tiles: int = MAX_TILES,
) -> int:
    """Zoom whose resolution matches ``width_px`` pixels across the view.

    The zoom is lowered until at most ``max_tiles`` tiles are needed.
    """

    left, _bottom, right, _top = mercator_bounds
    width = max(right - left, 1e-3)
    wanted = width / max(int(width_px), 1)
    zoom = int(math.ceil(math.log2(tile_resolution(0) / wanted)))
    zoom = min(MAX_ZOOM, max(0, zoom))
    while zoom > 0:
        x0, x1, y0, y1 = tile_range(mercator_bounds, zoom)
        if (x1 - x0 + 1) * (y1 - y0 + 1) <= max_tiles:
            break
        zoom -= 1
    return zoom


def _mercator_bounds(
    extent: tuple[float, float, float, float], crs: CRS
) -> tuple[float, float, float, float]:
    left, right, bottom, top = extent
    # Clip to the latitudes Web Mercator can represent before projecting.
    west, south, east, north = transform_bounds(crs, _WGS84, left, bottom, right, top, densify_pts=21)
    south = max(south, -_MERCATOR_LATITUDE_LIMIT)
    north = min(north, _MERCATOR_LATITUDE_LIMIT)
    west = max(west, -180.0)
    east = min(east, 180.0)
    if not (west < east and south < north):
        raise BasemapUnavailableError("The map extent lies outside the area OpenStreetMap covers.")
    return transform_bounds(_WGS84, _WEB_MERCATOR, west, south, east, north, densify_pts=21)


def _decode_tile(content: bytes) -> np.ndarray:
    array = mpl_image.imread(io.BytesIO(content), format="png")
    if array.dtype != np.uint8:
        array = np.clip(np.round(array * 255.0), 0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.shape[2] == 3:
        alpha = np.full(array.shape[:2] + (1,), 255, dtype=np.uint8)
        array = np.concatenate([array, alpha], axis=2)
    return array[:TILE_SIZE, :TILE_SIZE, :4]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_basemap(
    extent: tuple[float, float, float, float],
    crs: Any,
    size: tuple[int, int],
    *,
    fetch_tile: TileFetcher | None = None,
    max_tiles: int = MAX_TILES,
) -> BasemapImage:
    """Return OpenStreetMap covering ``extent`` in ``crs`` at ``size`` pixels.

    ``extent`` is (left, right, bottom, top) as used by ``imshow``; ``size``
    is (width, height) of the output image. Missing tiles are transparent;
    :class:`BasemapUnavailableError` is raised if none could be fetched.
    """

    target_crs = CRS.from_user_input(crs)
    left, right, bottom, top = (float(value) for value in extent)
    if not (left < right and bottom < top):
        raise ValueError("extent bounds must be increasing.")
    width, height = (max(1, int(value)) for value in size)
    fetch = default_tile_fetcher() if fetch_tile is None else fetch_tile

    mercator = _mercator_bounds((left, right, bottom, top), target_crs)
    zoom = choose_zoom(mercator, width, max_tiles=max_tiles)
    x0, x1, y0, y1 = tile_range(mercator, zoom)

    mosaic = np.zeros(((y1 - y0 + 1) * TILE_SIZE, (x1 - x0 + 1) * TILE_SIZE, 4), dtype=np.uint8)
    fetched = 0
    last_error: Exception | None = None
    for ty in range(y0, y1 + 1):
        for tx in range(x0, x1 + 1):
            try:
                tile = _decode_tile(fetch(zoom, tx, ty))
            except Exception as exc:  # noqa: BLE001 - one bad tile is not fatal
                last_error = exc
                continue
            row = (ty - y0) * TILE_SIZE
            column = (tx - x0) * TILE_SIZE
            mosaic[row:row + tile.shape[0], column:column + tile.shape[1]] = tile
            fetched += 1

    if fetched == 0:
        reason = f": {last_error}" if last_error is not None else ""
        raise BasemapUnavailableError(f"No OpenStreetMap tiles could be loaded{reason}")

    resolution = tile_resolution(zoom)
    source_transform = from_origin(
        -_MERCATOR_HALF + x0 * TILE_SIZE * resolution,
        _MERCATOR_HALF - y0 * TILE_SIZE * resolution,
        resolution,
        resolution,
    )
    destination_transform = from_bounds(left, bottom, right, top, width, height)
    output = np.zeros((4, height, width), dtype=np.uint8)
    reproject(
        source=np.moveaxis(mosaic, 2, 0),
        destination=output,
        src_transform=source_transform,
        src_crs=_WEB_MERCATOR,
        dst_transform=destination_transform,
        dst_crs=target_crs,
        resampling=Resampling.bilinear,
        src_nodata=None,
        dst_nodata=0,
    )
    return BasemapImage(np.moveaxis(output, 0, 2).copy(), (left, right, bottom, top), target_crs, zoom)
