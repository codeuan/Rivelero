"""Saving and opening ``.rivelero`` project files.

Layout (a ZIP container; documents are UTF-8 JSON, arrays are ``.npz``
written and read with ``allow_pickle=False``)::

    project.json        manifest: format, schema_version, software, project
                        metadata and linked resources
    survey.json         source data: ViewpointConfiguration and Sensors
    world.json          source data: Environment, AnalysisGrid, AnalysisDomain
    world_arrays.npz    AnalysisDomain masks
    observability.json  assumptions (VisibilityConfiguration), cache settings
                        and optional derived products (BuildReport, SOF)
    field.npz           SOF arrays (only when the SOF is saved)
    analysis.json       saved ScenarioSnapshots and the live scenario
    analysis_arrays.npz small arrays of saved summaries

Saving is atomic: the project is written to a temporary file next to the
target, read back and validated, and only then moved over the target with
``os.replace``; a failed save leaves any previous project untouched.

Opening never modifies application state: it returns a validated
:class:`ProjectData`, which the caller installs as a whole.

A saved SOF is restored only if its *inputs digest* - computed over the
saved survey and world documents, the domain masks, the visibility
configuration and the terrain checksum - matches on opening and the terrain
file is unchanged. A missing visibility cache never affects opening.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from rivelero.analysis.comparison import ScenarioSnapshot
from rivelero.core.configuration import ViewpointConfiguration
from rivelero.core.domain import AnalysisDomain, AnalysisGrid
from rivelero.core.environment import Environment
from rivelero.core.sensor import Sensor
from rivelero.observability.builder import SOFBuildReport
from rivelero.observability.survey_field import SurveyObservabilityField
from rivelero.project.codec import ArraySink, decode, encode
from rivelero.project.resources import (
    ResolvedResource,
    ResourceReference,
    describe_resource,
    resolve_resource,
)
from rivelero.project.schema import (
    PROJECT_FORMAT,
    SCHEMA_VERSION,
    ProjectFormatError,
    check_manifest,
    migrate,
    software_metadata,
)
from rivelero.visibility.configuration import VisibilityConfiguration


DOCUMENTS = ("survey.json", "world.json", "observability.json", "analysis.json")
ARRAY_FILES = ("world_arrays.npz", "field.npz", "analysis_arrays.npz")
MAX_DOCUMENT_BYTES = 512 * 1024 * 1024
ELEVATION_ROLE = "Elevation model"
_RESOURCE_MARKER = {"$resource": "elevation"}


class ProjectSaveError(RuntimeError):
    """Saving failed; any previous project file is unchanged."""


@dataclass(slots=True)
class ProjectData:
    """Qt-free scientific content of a project."""

    name: str = "Untitled Rivelero project"
    description: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    # Source data
    viewpoint_configuration: ViewpointConfiguration | None = None
    sensors: dict[str, Sensor] = field(default_factory=dict)
    environment: Environment | None = None
    grid: AnalysisGrid | None = None
    domain: AnalysisDomain | None = None

    # Modelling assumptions and computational settings
    visibility_configuration: VisibilityConfiguration | None = None
    store_settings: dict[str, Any] | None = None

    # Derived products (optional)
    sof: SurveyObservabilityField | None = None
    build_report: SOFBuildReport | None = None

    # Design work
    snapshots: tuple[ScenarioSnapshot, ...] = ()
    live_scenario: ScenarioSnapshot | None = None
    comparison: dict[str, Any] = field(default_factory=dict)

    # Filled in when opening
    elevation: ResolvedResource | None = None
    field_status: str = "not saved"
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_project(data: ProjectData, path: Path | str, *, include_field: bool = True) -> Path:
    """Write ``data`` to ``path`` atomically and return the resolved path."""

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{uuid4().hex}")

    try:
        members = _serialize(data, target, include_field=include_field)
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, payload in members.items():
                compress = zipfile.ZIP_STORED if name.endswith(".npz") else zipfile.ZIP_DEFLATED
                archive.writestr(name, payload, compress_type=compress)
        # Validate the written file before it replaces anything.
        _read_members(temporary)
        os.replace(temporary, target)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        if isinstance(exc, ProjectSaveError):
            raise
        raise ProjectSaveError(f"The project could not be saved: {exc}") from exc
    return target


def _serialize(data: ProjectData, target: Path, *, include_field: bool) -> dict[str, bytes]:
    resources: dict[str, Any] = {}
    elevation_sha = None
    if data.environment is not None:
        reference = describe_resource(
            data.environment.elevation_model.source, role=ELEVATION_ROLE, project_file=target
        )
        resources["elevation"] = reference.to_json()
        elevation_sha = reference.sha256

    survey = _dump({
        "viewpoint_configuration": encode(data.viewpoint_configuration, path="viewpoint_configuration"),
        "sensors": encode(dict(data.sensors), path="sensors"),
    })

    world_arrays = ArraySink("world_")
    environment = encode(data.environment, arrays=world_arrays, path="environment")
    if environment is not None:
        # The terrain is a linked resource: its location lives in the
        # manifest, so moving project + terrain together keeps the document
        # (and therefore the inputs digest) unchanged.
        environment["fields"]["elevation_model"]["fields"]["source"] = dict(_RESOURCE_MARKER)
    world = _dump({
        "environment": environment,
        "grid": encode(data.grid, arrays=world_arrays, path="grid"),
        "domain": encode(data.domain, arrays=world_arrays, path="domain"),
    })
    world_npz = _npz(world_arrays.arrays)

    configuration = encode(data.visibility_configuration, path="visibility_configuration")

    field_arrays = ArraySink("field_")
    field_record = None
    if include_field and data.sof is not None:
        field_record = {
            "sof": encode(data.sof, arrays=field_arrays, path="sof"),
            "inputs_digest": _inputs_digest(survey, world, world_npz, configuration, elevation_sha),
        }
    observability = _dump({
        "visibility_configuration": configuration,
        "store": data.store_settings,
        "build_report": encode(data.build_report, path="build_report"),
        "field": field_record,
    })

    analysis_arrays = ArraySink("analysis_")
    analysis = _dump({
        "snapshots": encode(list(data.snapshots), arrays=analysis_arrays, path="snapshots"),
        "live_scenario": encode(data.live_scenario, arrays=analysis_arrays, path="live_scenario"),
        "comparison": encode(dict(data.comparison), path="comparison"),
    })

    manifest = _dump({
        "format": PROJECT_FORMAT,
        "schema_version": SCHEMA_VERSION,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "software": software_metadata(),
        "project": {
            "name": data.name,
            "description": data.description,
            "created_at": data.created_at.isoformat(),
            "metadata": encode(dict(data.metadata), path="metadata"),
        },
        "resources": resources,
        "contents": {
            "field": field_record is not None,
            "snapshots": len(data.snapshots),
            "live_scenario": data.live_scenario is not None,
        },
    })

    members = {
        "project.json": manifest,
        "survey.json": survey,
        "world.json": world,
        "world_arrays.npz": world_npz,
        "observability.json": observability,
        "analysis.json": analysis,
        "analysis_arrays.npz": _npz(analysis_arrays.arrays),
    }
    if field_record is not None:
        members["field.npz"] = _npz(field_arrays.arrays)
    return members


# ---------------------------------------------------------------------------
# Open
# ---------------------------------------------------------------------------


def load_project(path: Path | str) -> ProjectData:
    """Read and validate a project; never modifies application state."""

    source = Path(path).expanduser().resolve()
    members = _read_members(source)

    manifest = members["project.json"]
    version = check_manifest(manifest)
    documents = migrate({name: members[name] for name in DOCUMENTS}, version)

    world_arrays = members.get("world_arrays.npz", {})
    analysis_arrays = members.get("analysis_arrays.npz", {})

    project = manifest.get("project") or {}
    data = ProjectData(
        name=str(project.get("name") or source.stem),
        description=project.get("description"),
        created_at=_datetime(project.get("created_at")),
        metadata=decode(project.get("metadata") or {}, path="metadata"),
    )

    # Survey -----------------------------------------------------------------
    survey = documents["survey.json"]
    data.viewpoint_configuration = _typed(
        decode(survey.get("viewpoint_configuration"), path="viewpoint_configuration"),
        ViewpointConfiguration, "viewpoint_configuration",
    )
    sensors = decode(survey.get("sensors") or {}, path="sensors")
    if not isinstance(sensors, dict) or not all(isinstance(s, Sensor) for s in sensors.values()):
        raise ProjectFormatError("sensors must map sensor IDs to Sensors.")
    data.sensors = sensors

    # World ------------------------------------------------------------------
    world = documents["world.json"]
    environment_record = world.get("environment")
    elevation_reference = (manifest.get("resources") or {}).get("elevation")
    if environment_record is not None:
        if elevation_reference is None:
            raise ProjectFormatError("The project references terrain but lists no resource.")
        data.elevation = resolve_resource(
            ResourceReference.from_json(elevation_reference), project_file=source
        )
        if data.elevation.usable:
            _set_elevation_source(environment_record, data.elevation.path)
            data.environment = _typed(
                decode(environment_record, arrays=world_arrays, path="environment"),
                Environment, "environment",
            )
            data.grid = _typed(decode(world.get("grid"), path="grid"), AnalysisGrid, "grid")
            data.domain = _typed(
                decode(world.get("domain"), arrays=world_arrays, path="domain"),
                AnalysisDomain, "domain",
            )
            if data.elevation.status.value != "found":
                data.warnings.append(data.elevation.message)
        else:
            data.warnings.append(
                data.elevation.message
                + " Terrain, AnalysisGrid and AnalysisDomain were not restored; "
                "load the terrain again in World."
            )

    # Observability ------------------------------------------------------------
    observability = documents["observability.json"]
    data.visibility_configuration = _typed(
        decode(observability.get("visibility_configuration"), path="visibility_configuration"),
        VisibilityConfiguration, "visibility_configuration",
    )
    store = observability.get("store")
    data.store_settings = store if isinstance(store, dict) else None
    data.build_report = _typed(
        decode(observability.get("build_report"), path="build_report"),
        SOFBuildReport, "build_report",
    )
    _restore_field(data, members, documents, observability.get("field"))

    # Analysis -----------------------------------------------------------------
    analysis = documents["analysis.json"]
    snapshots = decode(analysis.get("snapshots") or [], arrays=analysis_arrays, path="snapshots")
    if not all(isinstance(s, ScenarioSnapshot) for s in snapshots):
        raise ProjectFormatError("snapshots must contain ScenarioSnapshots.")
    data.snapshots = tuple(snapshots)
    data.live_scenario = _typed(
        decode(analysis.get("live_scenario"), arrays=analysis_arrays, path="live_scenario"),
        ScenarioSnapshot, "live_scenario",
    )
    comparison = decode(analysis.get("comparison") or {}, path="comparison")
    data.comparison = comparison if isinstance(comparison, dict) else {}
    return data


def _restore_field(data, members, documents, record) -> None:
    """Restore the saved SOF only when its inputs are provably unchanged."""

    if record is None:
        data.field_status = "not saved"
        return
    if data.environment is None or data.elevation is None or not data.elevation.unchanged:
        data.field_status = "rejected"
        data.warnings.append(
            "The saved observability field was not restored because the terrain "
            "is missing or has changed; rebuild it in Observability."
        )
        return

    expected = _inputs_digest(
        members["_raw"]["survey.json"],
        members["_raw"]["world.json"],
        members["_raw"].get("world_arrays.npz", b""),
        documents["observability.json"].get("visibility_configuration"),
        data.elevation.reference.sha256,
    )
    if record.get("inputs_digest") != expected:
        data.field_status = "rejected"
        data.warnings.append(
            "The saved observability field does not match the saved inputs and "
            "was not restored; rebuild it in Observability."
        )
        return

    if "_field_error" in members:
        data.field_status = "rejected"
        data.warnings.append(
            f"The saved observability field was rejected ({members['_field_error']}); "
            "rebuild it in Observability."
        )
        return

    try:
        field_arrays = members["field.npz"]
        sof = _typed(decode(record.get("sof"), arrays=field_arrays, path="sof"),
                     SurveyObservabilityField, "sof")
    except (KeyError, ProjectFormatError) as exc:
        data.field_status = "rejected"
        data.warnings.append(
            f"The saved observability field is unreadable ({exc}); rebuild it."
        )
        return
    data.sof = sof
    data.field_status = "restored"


# ---------------------------------------------------------------------------
# Container helpers
# ---------------------------------------------------------------------------


def _read_members(path: Path) -> dict[str, Any]:
    """Read and parse every known member; unknown members are ignored."""

    try:
        archive = zipfile.ZipFile(path)
    except FileNotFoundError:
        raise
    except (zipfile.BadZipFile, OSError) as exc:
        raise ProjectFormatError(f"{path.name} is not a readable Rivelero project: {exc}") from None

    with archive:
        names = set(archive.namelist())
        required = {"project.json", *DOCUMENTS}
        missing = sorted(required - names)
        if missing:
            raise ProjectFormatError(f"The project is incomplete; missing {', '.join(missing)}.")

        parsed: dict[str, Any] = {"_raw": {}}
        for name in ["project.json", *DOCUMENTS]:
            raw = _read_member(archive, name)
            parsed["_raw"][name] = raw
            try:
                parsed[name] = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ProjectFormatError(f"{name} is not valid JSON: {exc}") from None
        for name in ARRAY_FILES:
            if name in names:
                raw = _read_member(archive, name)
                parsed["_raw"][name] = raw
                try:
                    parsed[name] = _load_npz(raw, name)
                except ProjectFormatError as exc:
                    if name != "field.npz":
                        raise
                    # The SOF is an optional derived product: an unreadable
                    # or unsafe field file rejects the SOF, not the project.
                    parsed["_field_error"] = str(exc)
    return parsed


def _read_member(archive: zipfile.ZipFile, name: str) -> bytes:
    info = archive.getinfo(name)
    if info.file_size > MAX_DOCUMENT_BYTES and not name.endswith(".npz"):
        raise ProjectFormatError(f"{name} is unexpectedly large.")
    try:
        return archive.read(name)
    except (zipfile.BadZipFile, OSError, RuntimeError) as exc:
        raise ProjectFormatError(f"{name} is corrupted: {exc}") from None


def _load_npz(raw: bytes, name: str) -> dict[str, np.ndarray]:
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
    except (ValueError, OSError, zipfile.BadZipFile, EOFError) as exc:
        raise ProjectFormatError(f"{name} is corrupted or unsafe: {exc}") from None
    for key, array in arrays.items():
        if array.dtype.kind not in "biuf":
            raise ProjectFormatError(f"{name}:{key} has unsupported dtype {array.dtype}.")
    return arrays


def _npz(arrays: dict[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def _dump(document: dict[str, Any]) -> bytes:
    return json.dumps(document, ensure_ascii=False, indent=1, allow_nan=False).encode("utf-8")


def _inputs_digest(survey: bytes, world: bytes, world_npz: bytes, configuration, elevation_sha) -> str:
    digest = hashlib.sha256()
    digest.update(b"rivelero-inputs-v1\0")
    for part in (
        survey,
        world,
        world_npz,
        json.dumps(configuration, sort_keys=True).encode("utf-8"),
        str(elevation_sha).encode("utf-8"),
    ):
        digest.update(hashlib.sha256(part).digest())
    return digest.hexdigest()


def _set_elevation_source(record: dict[str, Any], path: Path) -> None:
    try:
        model = record["fields"]["elevation_model"]["fields"]
    except (KeyError, TypeError):
        raise ProjectFormatError("The saved Environment has no elevation model.") from None
    if model.get("source") != _RESOURCE_MARKER:
        raise ProjectFormatError("The saved elevation model does not reference its resource.")
    model["source"] = {"$type": "path", "value": str(path)}


def _typed(value, cls, name):
    if value is not None and not isinstance(value, cls):
        raise ProjectFormatError(f"{name} must be a {cls.__name__}.")
    return value


def _datetime(value) -> datetime:
    if not isinstance(value, str):
        return datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        raise ProjectFormatError("Invalid project created_at.") from None
