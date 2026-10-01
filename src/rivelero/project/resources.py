"""External (linked) resources such as terrain rasters.

Large inputs are referenced, not copied: a project records where a resource
was, a path relative to the project file when practical, and a SHA-256
content checksum. On opening, the resource is resolved to one of:

    FOUND     at its recorded location, content unchanged
    MOVED     at the project-relative location, content unchanged
    CHANGED   present, but its content differs from the saved checksum
    MISSING   not found at either location

A CHANGED resource can still be used as an input, but derived products
computed from the old content (such as a saved SOF) are never trusted.
The schema reserves ``storage`` for a future "embedded" policy.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import rasterio

from rivelero.project.schema import ProjectFormatError


class ResourceStatus(str, Enum):
    FOUND = "found"
    MOVED = "moved"
    CHANGED = "changed"
    MISSING = "missing"


@dataclass(frozen=True, slots=True)
class ResourceReference:
    """Saved description of one linked external file."""

    role: str
    absolute_path: str
    relative_path: str | None
    size_bytes: int
    sha256: str
    raster: dict[str, Any] | None = None
    storage: str = "linked"

    def to_json(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "storage": self.storage,
            "absolute_path": self.absolute_path,
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "raster": self.raster,
        }

    @classmethod
    def from_json(cls, data: Any) -> "ResourceReference":
        try:
            if data.get("storage", "linked") != "linked":
                raise ProjectFormatError(
                    f"Resource storage {data.get('storage')!r} is not supported."
                )
            return cls(
                role=str(data["role"]),
                absolute_path=str(data["absolute_path"]),
                relative_path=(
                    None if data.get("relative_path") is None else str(data["relative_path"])
                ),
                size_bytes=int(data["size_bytes"]),
                sha256=str(data["sha256"]),
                raster=data.get("raster"),
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ProjectFormatError(f"Invalid resource reference: {exc}") from None


@dataclass(frozen=True, slots=True)
class ResolvedResource:
    reference: ResourceReference
    status: ResourceStatus
    path: Path | None
    message: str

    @property
    def usable(self) -> bool:
        return self.status in (ResourceStatus.FOUND, ResourceStatus.MOVED, ResourceStatus.CHANGED)

    @property
    def unchanged(self) -> bool:
        return self.status in (ResourceStatus.FOUND, ResourceStatus.MOVED)


def file_sha256(path: Path, *, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_resource(path: Path | str, *, role: str, project_file: Path) -> ResourceReference:
    """Describe a local file for linking from ``project_file``."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Resource not found: {source}")

    try:
        relative = os.path.relpath(source, Path(project_file).resolve().parent)
    except ValueError:  # different drive on Windows
        relative = None

    raster = None
    try:
        with rasterio.open(source) as dataset:
            raster = {
                "crs": None if dataset.crs is None else dataset.crs.to_string(),
                "width": dataset.width,
                "height": dataset.height,
                "transform": [float(v) for v in tuple(dataset.transform)[:6]],
                "driver": dataset.driver,
            }
    except rasterio.errors.RasterioError:
        raster = None

    return ResourceReference(
        role=role,
        absolute_path=str(source),
        relative_path=None if relative is None else Path(relative).as_posix(),
        size_bytes=source.stat().st_size,
        sha256=file_sha256(source),
        raster=raster,
    )


def resolve_resource(reference: ResourceReference, *, project_file: Path) -> ResolvedResource:
    """Locate a linked resource and check that its content is unchanged."""

    candidates: list[tuple[Path, ResourceStatus]] = []
    absolute = Path(reference.absolute_path)
    candidates.append((absolute, ResourceStatus.FOUND))
    if reference.relative_path is not None:
        relative = (Path(project_file).resolve().parent / reference.relative_path).resolve()
        if relative != absolute:
            candidates.append((relative, ResourceStatus.MOVED))

    changed: Path | None = None
    for candidate, status in candidates:
        if not candidate.is_file():
            continue
        if candidate.stat().st_size == reference.size_bytes and file_sha256(candidate) == reference.sha256:
            message = (
                f"{reference.role} found at {candidate}."
                if status == ResourceStatus.FOUND
                else f"{reference.role} was moved; using {candidate}."
            )
            return ResolvedResource(reference, status, candidate, message)
        changed = changed or candidate

    if changed is not None:
        return ResolvedResource(
            reference,
            ResourceStatus.CHANGED,
            changed,
            f"{reference.role} at {changed} differs from the file used when the "
            "project was saved. Results derived from the old file are not reused.",
        )
    return ResolvedResource(
        reference,
        ResourceStatus.MISSING,
        None,
        f"{reference.role} not found (expected {reference.absolute_path}"
        + (f" or {reference.relative_path} next to the project" if reference.relative_path else "")
        + ").",
    )
