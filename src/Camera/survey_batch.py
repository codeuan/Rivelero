"""Read Rivelero viewpoint exports and define the batch telemetry CSV."""

import csv
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
import re


@dataclass(frozen=True)
class CameraSetup:
    name: str
    north_m: float
    east_m: float
    down_m: float
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    horizontal_fov_deg: float


TELEMETRY_COLUMNS = (
    "source_file", "camera_index", "camera_name", "simulator_actor",
    "north_m", "east_m", "down_m",
    "roll_deg", "pitch_deg", "yaw_deg", "horizontal_fov_deg", "plant_name",
    "visible_pixels", "xray_pixels", "occlusion_fraction", "distance_m",
)


def load_surveys(directory, number, origin, default_z):
    """Load one CSV per survey, in filename order, before starting the simulator."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Survey directory does not exist: {directory}")
    if number < 1:
        raise ValueError("-number must be at least 1")
    if not all(isfinite(value) for value in (*origin, default_z)):
        raise ValueError("The map origin and fallback camera Z must be finite.")

    files = sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".csv"),
        key=lambda path: [
            (0, int(part)) if part.isdecimal() else (1, part.casefold(), part)
            for part in re.split(r"(\d+)", path.name)
        ],
    )
    if len(files) != number:
        raise ValueError(f"Requested {number} surveys but found {len(files)} CSV files in {directory}.")
    return [(path, read_cameras(path, origin, default_z)) for path in files]


def read_cameras(path, origin, default_z):
    from pyproj import CRS
    from pyproj.exceptions import CRSError

    origin_east, origin_north, origin_up = origin
    cameras = []
    names = set()
    survey_crs = None
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        missing = {"viewpoint_id", "x", "y", "crs", "heading_deg"} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path.name} is missing columns: {', '.join(sorted(missing))}")
        for line, row in enumerate(reader, start=2):
            try:
                name = (row["viewpoint_id"] or "").strip()
                if not name or name in names:
                    raise ValueError(f"blank or duplicate viewpoint ID: {name!r}")
                crs = CRS.from_user_input(row["crs"])
                if not crs.is_projected or any(
                    axis.unit_conversion_factor != 1 for axis in crs.axis_info[:2]
                ):
                    raise ValueError("CRS must be projected in metres")
                if survey_crs is not None and survey_crs != crs:
                    raise ValueError("all cameras in a survey must use the same CRS")
                survey_crs = crs
                north = float(row["y"]) - origin_north
                east = float(row["x"]) - origin_east
                down = origin_up - float(row["z"]) if row.get("z") else default_z
                roll = float(row.get("roll_deg") or 0)
                pitch = float(row.get("pitch_deg") or 0)
                yaw = float(row["heading_deg"]) % 360
                fov = float(row.get("horizontal_fov_deg") or 120)
                if not all(isfinite(v) for v in (north, east, down, roll, pitch, yaw, fov)):
                    raise ValueError("camera coordinates and angles must be finite")
                if not 0 < fov < 180:
                    raise ValueError("horizontal field of view must be between 0 and 180 degrees")
            except (CRSError, TypeError, ValueError) as exc:
                raise ValueError(f"{path.name}, line {line}: {exc}") from exc
            names.add(name)
            cameras.append(CameraSetup(name, north, east, down, roll, pitch, yaw, fov))
    if not cameras:
        raise ValueError(f"{path.name} contains no camera rows.")
    return cameras
