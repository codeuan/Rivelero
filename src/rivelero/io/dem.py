"""DEM acquisition utilities for Rivelero.

OpenTopography acquisition belongs to the Rivelero I/O layer.

GUI code should provide canonical survey/domain information to these
functions rather than implementing provider-specific HTTP requests itself.
"""

from __future__ import annotations

import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Iterable,
    Mapping,
    Sequence,
)

import requests
from pyproj import (
    CRS as PyprojCRS,
    Transformer,
)


OPENTOPO_API_KEY = os.getenv(
    "OPENTOPO_API_KEY"
)

OPENTOPO_GLOBALDEM_URL = (
    "https://portal.opentopography.org/API/globaldem"
)


# ---------------------------------------------------------------------------
# Bounding boxes
# ---------------------------------------------------------------------------


def _validate_bbox(
    south: float,
    north: float,
    west: float,
    east: float,
) -> tuple[
    float,
    float,
    float,
    float,
]:
    """Validate a WGS84 bounding box."""

    values = tuple(
        float(value)
        for value in (
            south,
            north,
            west,
            east,
        )
    )

    if not all(
        math.isfinite(
            value
        )
        for value in values
    ):
        raise ValueError(
            "OpenTopography bounds must be finite."
        )

    south, north, west, east = values

    if not (
        -90.0
        <= south
        < north
        <= 90.0
    ):
        raise ValueError(
            "Latitude bounds must satisfy "
            "-90 <= south < north <= 90."
        )

    if not (
        -180.0
        <= west
        < east
        <= 180.0
    ):
        raise ValueError(
            "Longitude bounds must satisfy "
            "-180 <= west < east <= 180."
        )

    return values


def buffered_wgs84_bbox(
    lon_lat_points: Iterable[
        tuple[
            float,
            float,
        ]
    ],
    *,
    buffer_m: float = 0.0,
) -> tuple[
    float,
    float,
    float,
    float,
]:
    """Return a buffered WGS84 bounding box.

    Parameters
    ----------
    lon_lat_points
        Longitude/latitude pairs.

    buffer_m
        Approximate geographic buffer in metres.

    Returns
    -------
    tuple
        ``(south, north, west, east)``.
    """

    points = [
        (
            float(lon),
            float(lat),
        )
        for lon, lat
        in lon_lat_points
    ]

    if not points:
        raise ValueError(
            "At least one point is required."
        )

    buffer_m = float(
        buffer_m
    )

    if (
        not math.isfinite(
            buffer_m
        )
        or buffer_m < 0
    ):
        raise ValueError(
            "buffer_m must be finite and non-negative."
        )

    if not all(
        math.isfinite(value)
        for point in points
        for value in point
    ):
        raise ValueError(
            "Point coordinates must be finite."
        )

    if not all(
        -180.0 <= lon <= 180.0
        and -90.0 <= lat <= 90.0
        for lon, lat in points
    ):
        raise ValueError(
            "WGS84 point coordinates are outside valid "
            "longitude/latitude ranges."
        )

    lons = [
        point[0]
        for point in points
    ]

    lats = [
        point[1]
        for point in points
    ]

    center_lat = (
        sum(lats)
        / len(lats)
    )

    lat_buffer_deg = (
        buffer_m
        / 111_320.0
    )

    lon_buffer_deg = (
        buffer_m
        / (
            111_320.0
            * max(
                0.1,
                math.cos(
                    math.radians(
                        center_lat
                    )
                ),
            )
        )
    )

    return _validate_bbox(
        max(
            -90.0,
            min(lats)
            - lat_buffer_deg,
        ),
        min(
            90.0,
            max(lats)
            + lat_buffer_deg,
        ),
        max(
            -180.0,
            min(lons)
            - lon_buffer_deg,
        ),
        min(
            180.0,
            max(lons)
            + lon_buffer_deg,
        ),
    )


def _bbox_from_samples(
    sample_metadata: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
    buffer_m: float,
) -> tuple[
    float,
    float,
    float,
    float,
]:
    """Backward-compatible helper for lon/lat dictionaries."""

    if not sample_metadata:
        raise ValueError(
            "sample_metadata is empty."
        )

    return buffered_wgs84_bbox(
        [
            (
                float(
                    sample["lon"]
                ),
                float(
                    sample["lat"]
                ),
            )
            for sample
            in sample_metadata
        ],
        buffer_m=buffer_m,
    )


def wgs84_bbox_from_viewpoints(
    viewpoints: Iterable[Any],
    *,
    buffer_m: float = 0.0,
) -> tuple[
    float,
    float,
    float,
    float,
]:
    """Return OpenTopography bounds around canonical Viewpoints.

    Each Viewpoint is explicitly transformed from its own CRS to WGS84.
    Numeric coordinates are never guessed to be longitude/latitude.
    """

    viewpoints = list(
        viewpoints
    )

    if not viewpoints:
        raise ValueError(
            "At least one Viewpoint is required."
        )

    wgs84 = PyprojCRS.from_epsg(
        4326
    )

    transformers: dict[
        str,
        Transformer,
    ] = {}

    lon_lat = []

    for viewpoint in viewpoints:

        try:
            x = float(
                viewpoint.x
            )

            y = float(
                viewpoint.y
            )

            source_crs = (
                PyprojCRS.from_user_input(
                    viewpoint.crs
                )
            )

        except (
            AttributeError,
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                "Every Viewpoint must provide valid x, y and crs."
            ) from exc

        key = source_crs.to_string()

        transformer = transformers.get(
            key
        )

        if transformer is None:

            transformer = (
                Transformer.from_crs(
                    source_crs,
                    wgs84,
                    always_xy=True,
                )
            )

            transformers[
                key
            ] = transformer

        lon, lat = transformer.transform(
            x,
            y,
        )

        lon_lat.append(
            (
                float(lon),
                float(lat),
            )
        )

    return buffered_wgs84_bbox(
        lon_lat,
        buffer_m=buffer_m,
    )


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def _download_binary_to_tempfile(
    content: bytes,
    suffix: str,
) -> str:
    """Write binary content to a temporary file."""

    temporary = (
        tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        )
    )

    try:
        temporary.write(
            content
        )

    finally:
        temporary.close()

    return temporary.name


def download_dem_from_opentopo(
    south: float,
    north: float,
    west: float,
    east: float,
    demtype: str = "COP30",
    *,
    api_key: str | None = None,
    output_path: str | Path | None = None,
    timeout: float = 120.0,
    session: requests.Session | None = None,
) -> str:
    """Download an OpenTopography global DEM GeoTIFF.

    Existing callers remain compatible. New GUI code may supply an explicit
    destination and API key.
    """

    south, north, west, east = (
        _validate_bbox(
            south,
            north,
            west,
            east,
        )
    )

    key = (
        normalize_opentopo_api_key(api_key or "")
        or os.getenv(
            "OPENTOPO_API_KEY"
        )
        or OPENTOPO_API_KEY
    )

    if not key:
        raise OpenTopographyError(
            "No OpenTopography API key: enter one or set OPENTOPO_API_KEY."
        )

    demtype = str(
        demtype
    ).strip()

    if not demtype:
        raise ValueError(
            "demtype must be non-empty."
        )

    timeout = float(
        timeout
    )

    if (
        not math.isfinite(
            timeout
        )
        or timeout <= 0
    ):
        raise ValueError(
            "timeout must be finite and greater than zero."
        )

    client = (
        requests
        if session is None
        else session
    )

    try:
        response = client.get(
            OPENTOPO_GLOBALDEM_URL,
            params={
                "demtype": demtype,
                "south": south,
                "north": north,
                "west": west,
                "east": east,
                "outputFormat": "GTiff",
                "API_Key": key,
            },
            timeout=timeout,
        )
    except requests.RequestException as exc:
        # Messages of requests exceptions can contain the request URL, and
        # therefore the API key: report only the kind of failure.
        raise OpenTopographyError(
            "Could not reach OpenTopography "
            f"({type(exc).__name__}). Check the internet connection."
        ) from None

    _raise_for_opentopo_response(response)

    content = response.content

    if output_path is None:
        return _download_binary_to_tempfile(
            content,
            suffix=".tif",
        )

    destination = Path(
        output_path
    ).expanduser().resolve()

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination.write_bytes(
        content
    )

    return str(
        destination
    )


class OpenTopographyError(RuntimeError):
    """OpenTopography refused or failed a request (message is key-free)."""


_API_KEY_PREFIX = re.compile(
    r"^(?:export\s+|set\s+)?(?:OPENTOPO_API_KEY|API_Key)\s*[=:]\s*", re.I
)


def normalize_opentopo_api_key(text: str | None) -> str:
    """Return the key from pasted text.

    Surrounding whitespace and quotes are removed, and a pasted assignment
    such as ``OPENTOPO_API_KEY=…`` or ``API_Key=…`` is reduced to its value.
    """

    value = str(text or "").strip()
    value = _API_KEY_PREFIX.sub("", value).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1].strip()
    return value


def _opentopo_message(response) -> str:
    """Server-provided error text, without markup."""

    try:
        text = response.text or ""
    except Exception:  # noqa: BLE001 - binary or undecodable body
        return ""
    match = re.search(r"<error>(.*?)</error>", text, flags=re.S | re.I)
    if match:
        text = match.group(1)
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(text.split())[:300]


def _raise_for_opentopo_response(response) -> None:
    status = int(getattr(response, "status_code", 200) or 200)
    message = _opentopo_message(response) if status >= 400 else ""
    detail = f" OpenTopography says: {message}" if message else ""

    if status in (401, 403, 429) and "rate limit" in message.lower():
        # OpenTopography answers 401 when a key's daily quota is used up.
        raise OpenTopographyError(
            f"OpenTopography's request limit for this API key was reached "
            f"(HTTP {status}); try again later." + detail
        )
    if status in (401, 403):
        raise OpenTopographyError(
            "OpenTopography rejected the API key "
            f"(HTTP {status}). Check that the key is complete and active in "
            "your OpenTopography account." + detail
        )
    if status == 204:
        raise OpenTopographyError(
            "OpenTopography has no data for this dataset in the requested area."
        )
    if status >= 400:
        raise OpenTopographyError(
            f"OpenTopography request failed (HTTP {status})." + detail
        )

    content_type = (
        response.headers
        .get(
            "Content-Type",
            "",
        )
        .lower()
    )

    if (
        "html" in content_type
        or "json" in content_type
        or "xml" in content_type
        or "text" in content_type
    ):
        message = _opentopo_message(response)
        raise OpenTopographyError(
            "OpenTopography did not return a GeoTIFF. "
            "Check dataset, area and API key."
            + (f" OpenTopography says: {message}" if message else "")
        )

    if not response.content:
        raise OpenTopographyError(
            "OpenTopography returned an empty response."
        )


def download_dem_for_samples(
    sample_metadata: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
    max_distance_m: float,
    demtype: str = "COP30",
    **download_kwargs: Any,
) -> str:
    """Backward-compatible download around WGS84 sample dictionaries."""

    south, north, west, east = (
        _bbox_from_samples(
            sample_metadata,
            max_distance_m,
        )
    )

    return download_dem_from_opentopo(
        south=south,
        north=north,
        west=west,
        east=east,
        demtype=demtype,
        **download_kwargs,
    )


def download_dem_for_viewpoints(
    viewpoints: Iterable[Any],
    *,
    buffer_m: float,
    demtype: str = "COP30",
    **download_kwargs: Any,
) -> str:
    """Download an OpenTopography DEM covering canonical Viewpoints."""

    south, north, west, east = (
        wgs84_bbox_from_viewpoints(
            viewpoints,
            buffer_m=buffer_m,
        )
    )

    return download_dem_from_opentopo(
        south=south,
        north=north,
        west=west,
        east=east,
        demtype=demtype,
        **download_kwargs,
    )