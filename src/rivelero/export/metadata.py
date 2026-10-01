"""Metadata carried by every exported file.

Each export gets one metadata record describing what the file *means*:
product, meaning, units, NoData semantics, categorical codes, comparison
direction and the scientific identities it derives from (SOF, AnalysisDomain,
VisibilityConfiguration, project and software). The same record is written

* into GeoTIFF tags (flattened to ``RIVELERO_*`` strings), so the meaning
  survives when a raster is copied on its own, and
* as a JSON sidecar ``<file>.json`` next to every raster and table.

P3 will add a richer provenance manifest; this record is deliberately
compact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rivelero.export.files import atomic_output
from rivelero.project.schema import SCHEMA_VERSION, software_metadata

EXPORT_FORMAT = "rivelero-export"
EXPORT_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class ExportProvenance:
    """Identities shared by all files of one export."""

    project_name: str | None = None
    project_path: str | None = None
    project_saved: bool | None = None
    sof_id: str | None = None
    viewpoint_configuration_id: str | None = None
    environment_id: str | None = None
    analysis_domain_id: str | None = None
    visibility_configuration_id: str | None = None
    visibility_configuration_name: str | None = None
    sampling_unit: str | None = None
    exported_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


def product_metadata(
    provenance: ExportProvenance,
    *,
    product: str,
    meaning: str,
    units: str | None = None,
    nodata: str | None = None,
    codes: dict[int, str] | None = None,
    columns: dict[str, str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the metadata record of one exported file."""

    record: dict[str, Any] = {
        "format": EXPORT_FORMAT,
        "format_version": EXPORT_FORMAT_VERSION,
        "product": product,
        "meaning": meaning,
        "units": units,
        "nodata": nodata,
        "software": software_metadata(),
        "project_schema_version": SCHEMA_VERSION,
        "provenance": {
            name: getattr(provenance, name)
            for name in provenance.__dataclass_fields__
        },
    }
    if codes is not None:
        record["codes"] = {str(code): label for code, label in codes.items()}
    if columns is not None:
        record["columns"] = dict(columns)
    if extra:
        record.update(extra)
    return record


def geotiff_tags(record: dict[str, Any]) -> dict[str, str]:
    """Flatten a metadata record into GeoTIFF tag strings."""

    provenance = record.get("provenance", {})
    software = record.get("software", {})
    tags = {
        "RIVELERO_FORMAT": f"{record['format']}/{record['format_version']}",
        "RIVELERO_PRODUCT": record["product"],
        "RIVELERO_MEANING": record["meaning"],
        "RIVELERO_UNITS": record.get("units") or "",
        "RIVELERO_NODATA": record.get("nodata") or "none",
        "RIVELERO_SOFTWARE_VERSION": str(software.get("version")),
        "RIVELERO_GIT_COMMIT": str(software.get("git_commit")),
        "RIVELERO_EXPORTED_AT": str(provenance.get("exported_at")),
    }
    for name in (
        "project_name", "sof_id", "analysis_domain_id",
        "visibility_configuration_id", "sampling_unit",
    ):
        if provenance.get(name) is not None:
            tags[f"RIVELERO_{name.upper()}"] = str(provenance[name])
    for code, label in (record.get("codes") or {}).items():
        tags[f"RIVELERO_CODE_{code}"] = label
    for name in ("direction", "left", "right", "scenario"):
        if name in record:
            value = record[name]
            tags[f"RIVELERO_{name.upper()}"] = (
                value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            )
    return tags


def sidecar_path(path: Path) -> Path:
    return path.with_name(path.name + ".json")


def write_sidecar(path: Path, record: dict[str, Any], *, overwrite: bool) -> Path:
    """Write ``record`` as ``<path>.json`` atomically."""

    target = sidecar_path(Path(path))
    with atomic_output(target, overwrite=overwrite) as temporary:
        temporary.write_text(
            json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
    return target
