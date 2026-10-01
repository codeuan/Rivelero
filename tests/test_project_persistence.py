"""P1 tests: project format, round trips, resources and transactions (Qt-free)."""

from __future__ import annotations

import io
import json
import math
import shutil
import zipfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import rasterio

from observability_fixtures import DEM_FLAT, DEM_NODATA, make_ready_state, visibility_configuration
from rivelero.analysis.comparison import ScenarioWorkspace
from rivelero.analysis.scenario import SurveyDesignScenario
from rivelero.core.configuration import ViewpointConfiguration
from rivelero.core.observation import ObservationEvent
from rivelero.core.viewpoint import Viewpoint
from rivelero.gui.application_state import ApplicationState
from rivelero.gui.observability_service import (
    VisibilityKeyResolver,
    install_build_result,
    prepare_build,
)
from rivelero.gui.project_service import open_state, save_state
from rivelero.observability.builder import (
    build_survey_observability_field,
    visibility_context_fingerprint,
)
from rivelero.project.codec import ArraySink, decode, encode
from rivelero.project.io import ProjectData, ProjectSaveError, load_project, save_project
from rivelero.project.resources import ResourceStatus
from rivelero.project.schema import (
    ProjectFormatError,
    ProjectSerializationError,
    SCHEMA_VERSION,
    UnsupportedSchemaError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _local_dem(tmp_path, source=DEM_NODATA, name="terrain.tif"):
    folder = tmp_path / "data"
    folder.mkdir(exist_ok=True)
    target = folder / name
    shutil.copy2(source, target)
    return target


def _built(tmp_path, *, dem=None, **kwargs):
    dem = dem or _local_dem(tmp_path)
    kwargs.setdefault("viewpoint_ids", None)
    kwargs.setdefault("configuration", visibility_configuration(max_distance_m=250.0))
    state = make_ready_state(tmp_path / "cache", dem=dem, survey_buffer_m=300.0, **kwargs)
    request = prepare_build(state)
    install_build_result(state, request, build_survey_observability_field(**request.build_kwargs()))
    return state, dem


def _with_design(state):
    sof = state.analysis.survey_observability_field
    store = state.analysis.visibility_store
    resolver = VisibilityKeyResolver()
    live = SurveyDesignScenario(sof)
    key = resolver.resolve(state, viewpoint_id="vp_center_east").key
    live.deactivate({key: store.get(key).visibility_mask})
    workspace = ScenarioWorkspace()
    snapshot = workspace.save(live, name="Without centre east", description="Ωmega ✓")
    state.analysis.scenario_workspace = workspace
    state.set_design_scenario(live)
    return snapshot


def _rewrite(project, **replacements):
    """Copy a project zip, replacing members (bytes) or dropping them (None)."""
    source = zipfile.ZipFile(project)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as target:
        for name in source.namelist():
            if name in replacements:
                if replacements[name] is not None:
                    target.writestr(name, replacements[name])
            else:
                target.writestr(name, source.read(name))
        for name, payload in replacements.items():
            if name not in source.namelist() and payload is not None:
                target.writestr(name, payload)
    source.close()
    Path(project).write_bytes(buffer.getvalue())


def _json_member(project, name):
    with zipfile.ZipFile(project) as archive:
        return json.loads(archive.read(name))


# ---------------------------------------------------------------------------
# Codec
# ---------------------------------------------------------------------------


def test_codec_round_trips_non_json_types():
    from affine import Affine
    from rasterio.crs import CRS
    from shapely.geometry import Polygon

    value = {
        "when": datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
        "naive": datetime(2026, 1, 1, 10, 0),
        "crs": CRS.from_epsg(32633),
        "transform": Affine(10, 0, 500000, 0, -10, 4100000),
        "shape": Polygon([(0, 0), (1, 0), (1, 1)]),
        "path": Path("a/b.tif"),
        "tuple": (1, "two", 3.5),
        "nan": math.nan, "inf": math.inf,
        "numpy": np.float32(1.5), "count": np.int64(7),
        "unicode": "Sicília · 城市 · ✓",
        "$dangerous": 1, 3: "int key",
        "nested": {"list": [None, True, {"deep": (1, 2)}]},
    }
    sink = ArraySink("a")
    array = np.arange(6, dtype=np.uint16).reshape(2, 3)
    encoded = encode({"data": value, "array": array}, arrays=sink)
    text = json.dumps(encoded, allow_nan=False)  # strictly valid JSON
    decoded = decode(json.loads(text), arrays=sink.arrays)

    got = decoded["data"]
    assert got["when"] == value["when"] and got["when"].tzinfo is not None
    assert got["naive"] == value["naive"] and got["naive"].tzinfo is None
    assert got["crs"] == value["crs"]
    assert got["transform"] == value["transform"]
    assert got["shape"].equals(value["shape"])
    assert got["path"] == value["path"]
    assert got["tuple"] == value["tuple"]
    assert math.isnan(got["nan"]) and got["inf"] == math.inf
    assert got["numpy"] == 1.5 and got["count"] == 7
    assert got["unicode"] == value["unicode"]
    assert got["$dangerous"] == 1 and got[3] == "int key"
    assert got["nested"] == value["nested"]
    assert decoded["array"].dtype == np.uint16 and np.array_equal(decoded["array"], array)


def test_codec_rejects_unrepresentable_values_with_path():
    with pytest.raises(ProjectSerializationError, match=r"\$\.extra\.handle"):
        encode({"extra": {"handle": object()}})
    with pytest.raises(ProjectSerializationError):
        encode(np.array([object()]), arrays=ArraySink("a"))


def test_decoder_refuses_unknown_classes_and_fields():
    with pytest.raises(ProjectFormatError, match="unknown type"):
        decode({"$dataclass": "os.system", "fields": {}})
    with pytest.raises(ProjectFormatError, match="unexpected fields"):
        decode({"$dataclass": "AnalysisGrid", "fields": {"evil": 1}})
    with pytest.raises(ProjectFormatError, match="unknown enum"):
        decode({"$enum": "Anything", "value": 1})


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------


def test_empty_project_round_trip(tmp_path):
    state = ApplicationState()
    path = save_state(state, tmp_path / "empty")
    assert path.suffix == ".rivelero" and path.is_file()
    assert state.project.name == "empty" and not state.project.dirty

    result = open_state(path)
    restored = result.state
    assert restored.survey.viewpoint_configuration is None
    assert restored.analysis.environment is None
    assert restored.project.name == "empty"
    assert not restored.project.dirty
    manifest = _json_member(path, "project.json")
    assert manifest["format"] == "rivelero-project"
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert "software" in manifest


def test_full_round_trip(tmp_path):
    state, _dem = _built(tmp_path)
    snapshot = _with_design(state)
    path = save_state(state, tmp_path / "survey.rivelero")

    result = open_state(path)
    restored = result.state
    assert result.warnings == [] and result.field_status == "restored"

    # Survey
    assert restored.survey.viewpoint_configuration == state.survey.viewpoint_configuration
    assert restored.survey.sensors == state.survey.sensors
    events = restored.survey.viewpoint_configuration.observation_events
    assert events == state.survey.viewpoint_configuration.observation_events
    assert all(event.timestamp.tzinfo is not None for event in events)

    # World
    a, b = restored.analysis, state.analysis
    assert a.environment == b.environment
    assert a.analysis_grid == b.analysis_grid
    assert a.analysis_domain.geometry.equals(b.analysis_domain.geometry)
    assert np.array_equal(a.analysis_domain.analysis_mask, b.analysis_domain.analysis_mask)
    assert np.array_equal(a.analysis_domain.valid_mask, b.analysis_domain.valid_mask)
    assert a.analysis_domain.domain_id == b.analysis_domain.domain_id

    # Assumptions and cache settings
    assert a.visibility_configuration == b.visibility_configuration
    assert a.visibility_store.cache_directory == b.visibility_store.cache_directory

    # Derived field
    sof, original = a.survey_observability_field, b.survey_observability_field
    assert restored.observability_ready
    assert sof.sof_id == original.sof_id
    assert sof.exposure_count.dtype == original.exposure_count.dtype
    for name in ("exposure_count", "analysis_mask", "valid_mask"):
        assert np.array_equal(getattr(sof, name), getattr(original, name))
    assert sof.active_keys == original.active_keys
    assert sof.transform == original.transform and sof.crs == original.crs
    assert a.build_report == b.build_report

    # Design work
    workspace = a.scenario_workspace
    assert workspace.snapshots == (snapshot,)
    assert workspace.snapshots[0].description == "Ωmega ✓"
    assert workspace.snapshots[0].compatible_with(sof)
    live = a.design_scenario
    assert live is not None and live.baseline is sof
    assert live.summary() == b.design_scenario.summary()
    assert not restored.project.dirty


def test_missing_metadata_repeated_coordinates_and_provenance(tmp_path):
    dem = _local_dem(tmp_path, DEM_FLAT)
    state = make_ready_state(tmp_path / "cache", dem=dem, viewpoint_ids=("vp_center_360",))
    base = state.survey.viewpoint_configuration.viewpoints[0]
    twin = replace(
        base, viewpoint_id="vp_twin", heading_deg=None, observer_height_m=None,
        provenance={"source_file": Path("raw/gsv.csv"), "imported": datetime(2026, 2, 1)},
        extra_metadata={"pano": "αβγ", "angles": (1.5, math.nan), "tags": ["a", "b"]},
    )
    event = ObservationEvent(event_id="e1", viewpoint_id="vp_twin", timestamp=None,
                             acquisition_conditions={"weather": "sol ☀"})
    state.set_viewpoint_configuration(ViewpointConfiguration(
        configuration_id="survey", name="Survey", viewpoints=[base, twin],
        observation_events=[event],
    ))
    path = save_state(state, tmp_path / "meta")
    restored = open_state(path).state.survey.viewpoint_configuration
    got = restored.get_viewpoint("vp_twin")
    assert (got.x, got.y) == (base.x, base.y)  # repeated coordinates kept
    assert got.heading_deg is None and got.observer_height_m is None
    assert got.provenance == twin.provenance
    assert got.extra_metadata["pano"] == "αβγ"
    assert got.extra_metadata["angles"][0] == 1.5 and math.isnan(got.extra_metadata["angles"][1])
    assert restored.get_event("e1").timestamp is None
    assert restored.get_event("e1").acquisition_conditions == {"weather": "sol ☀"}


def test_project_without_field(tmp_path):
    state, _dem = _built(tmp_path)
    data = ProjectData(
        viewpoint_configuration=state.survey.viewpoint_configuration,
        sensors=state.survey.sensors,
        environment=state.analysis.environment,
        grid=state.analysis.analysis_grid,
        domain=state.analysis.analysis_domain,
        visibility_configuration=state.analysis.visibility_configuration,
    )
    path = save_project(data, tmp_path / "nofield.rivelero")
    with zipfile.ZipFile(path) as archive:
        assert "field.npz" not in archive.namelist()
    result = open_state(path)
    assert result.field_status == "not saved"
    assert not result.state.observability_ready
    assert result.state.observability_build_blockers() == []  # user can rebuild


def test_masks_are_not_serialized_in_snapshots(tmp_path):
    state, _dem = _built(tmp_path)
    _with_design(state)
    path = save_state(state, tmp_path / "p")
    with zipfile.ZipFile(path) as archive:
        with np.load(io.BytesIO(archive.read("analysis_arrays.npz"))) as arrays:
            largest = max((a.size for a in arrays.values()), default=0)
    grid = state.analysis.analysis_grid
    assert largest < grid.width * grid.height  # only small summary counts


# ---------------------------------------------------------------------------
# Derived-state validation and resources
# ---------------------------------------------------------------------------


def test_edited_inputs_reject_saved_field(tmp_path):
    state, _dem = _built(tmp_path)
    path = save_state(state, tmp_path / "p")
    survey = _json_member(path, "survey.json")
    viewpoint = survey["viewpoint_configuration"]["fields"]["viewpoints"][0]["fields"]
    viewpoint["x"] += 10.0
    _rewrite(path, **{"survey.json": json.dumps(survey).encode()})

    result = open_state(path)
    assert result.field_status == "rejected"
    assert not result.state.observability_ready
    assert result.state.survey_ready  # inputs are still restored
    assert any("does not match the saved inputs" in w for w in result.warnings)


def test_changed_dem_rejects_field_and_refreshes_validity(tmp_path):
    state, dem = _built(tmp_path)
    path = save_state(state, tmp_path / "p")

    with rasterio.open(dem, "r+") as dataset:  # same grid, new content
        band = dataset.read(1)
        band[50:60, 50:60] = dataset.nodata
        dataset.write(band, 1)

    result = open_state(path)
    restored = result.state
    assert result.field_status == "rejected"
    assert not restored.observability_ready
    assert restored.world_ready  # terrain and domain geometry kept
    newly_invalid = ~restored.analysis.analysis_domain.valid_mask
    assert newly_invalid[50:60, 50:60].all()
    assert any("differs from the file used" in w for w in result.warnings)
    assert any("recomputed" in w for w in result.warnings)


def test_missing_dem_keeps_survey_and_warns(tmp_path):
    state, dem = _built(tmp_path)
    _with_design(state)
    path = save_state(state, tmp_path / "p")
    dem.unlink()

    result = open_state(path)
    restored = result.state
    assert restored.survey_ready
    assert restored.analysis.environment is None and restored.analysis.analysis_domain is None
    assert not restored.observability_ready
    assert restored.analysis.visibility_configuration == state.analysis.visibility_configuration
    assert any("not found" in w for w in result.warnings)
    # Snapshots are kept but no longer compatible.
    snapshot = restored.analysis.scenario_workspace.snapshots[0]
    assert not snapshot.compatible_with(restored.analysis.survey_observability_field)


def test_project_moved_with_terrain_resolves_relative_path(tmp_path):
    state, _dem = _built(tmp_path)
    path = save_state(state, tmp_path / "p")
    reference = _json_member(path, "project.json")["resources"]["elevation"]
    assert reference["relative_path"] == "data/terrain.tif"
    assert Path(reference["absolute_path"]).is_absolute()

    moved = tmp_path / "elsewhere"
    moved.mkdir()
    shutil.copy2(path, moved / path.name)
    shutil.copytree(tmp_path / "data", moved / "data")
    (tmp_path / "data" / "terrain.tif").unlink()

    data = load_project(moved / path.name)
    assert data.elevation.status == ResourceStatus.MOVED
    result = open_state(moved / path.name)
    assert result.field_status == "restored"
    assert result.state.observability_ready
    assert Path(result.state.analysis.environment.elevation_model.source).parent == (moved / "data")


def test_deleted_visibility_cache_does_not_block_opening(tmp_path):
    state, _dem = _built(tmp_path)
    _with_design(state)
    cache = state.analysis.visibility_store.cache_directory
    path = save_state(state, tmp_path / "p")
    shutil.rmtree(cache)

    result = open_state(path)
    assert result.state.observability_ready  # the field lives in the project
    assert result.state.analysis.design_scenario is None  # masks needed to restore
    assert any("unsaved scenario was not restored" in w for w in result.warnings)
    assert result.state.analysis.scenario_workspace.snapshots  # saved record kept


def test_dem_rewritten_in_place_changes_cache_identity(tmp_path):
    state, dem = _built(tmp_path)
    arguments = dict(
        environment=state.analysis.environment,
        domain=state.analysis.analysis_domain,
        visibility_configuration=state.analysis.visibility_configuration,
    )
    before = visibility_context_fingerprint(**arguments)
    with rasterio.open(dem, "r+") as dataset:
        band = dataset.read(1)
        band[0, 0] += 5
        dataset.write(band, 1)
    assert visibility_context_fingerprint(**arguments) != before


# ---------------------------------------------------------------------------
# Malformed input and transactions
# ---------------------------------------------------------------------------


def test_unsupported_future_schema_is_rejected(tmp_path):
    path = save_state(ApplicationState(), tmp_path / "p")
    manifest = _json_member(path, "project.json")
    manifest["schema_version"] = SCHEMA_VERSION + 1
    _rewrite(path, **{"project.json": json.dumps(manifest).encode()})
    with pytest.raises(UnsupportedSchemaError, match="Update Rivelero"):
        load_project(path)


def test_not_a_project_and_missing_members(tmp_path):
    bogus = tmp_path / "bogus.rivelero"
    bogus.write_text("not a zip")
    with pytest.raises(ProjectFormatError):
        load_project(bogus)

    path = save_state(ApplicationState(), tmp_path / "p")
    _rewrite(path, **{"world.json": None})
    with pytest.raises(ProjectFormatError, match="world.json"):
        load_project(path)

    other = save_state(ApplicationState(), tmp_path / "q")
    _rewrite(other, **{"project.json": json.dumps({"format": "something-else"}).encode()})
    with pytest.raises(ProjectFormatError, match="not a Rivelero project"):
        load_project(other)


def test_corrupted_or_pickled_field_is_rejected_but_project_opens(tmp_path):
    state, _dem = _built(tmp_path)
    path = save_state(state, tmp_path / "p")
    _rewrite(path, **{"field.npz": b"garbage"})
    result = open_state(path)
    assert result.field_status == "rejected"
    assert not result.state.observability_ready
    assert result.state.world_ready  # inputs still restored
    assert any("field.npz" in w for w in result.warnings)

    path2 = save_state(state, tmp_path / "p2")
    buffer = io.BytesIO()
    np.savez(buffer, field_0=np.array([{"x": 1}], dtype=object))
    _rewrite(path2, **{"field.npz": buffer.getvalue()})
    result = open_state(path2)
    assert result.field_status == "rejected"
    assert any("unsafe" in w or "unsupported" in w for w in result.warnings)

    # Required array files are strict: a pickled domain mask fails opening.
    path3 = save_state(state, tmp_path / "p3")
    _rewrite(path3, **{"world_arrays.npz": buffer.getvalue()})
    with pytest.raises(ProjectFormatError, match="world_arrays.npz"):
        load_project(path3)


def test_invalid_references_are_rejected(tmp_path):
    state, _dem = _built(tmp_path)
    path = save_state(state, tmp_path / "p")
    survey = _json_member(path, "survey.json")
    survey["sensors"] = {}  # Viewpoints still reference their Sensors
    _rewrite(path, **{"survey.json": json.dumps(survey).encode()})
    with pytest.raises(ProjectFormatError, match="undefined Sensors"):
        open_state(path)


def test_failed_save_keeps_previous_project(tmp_path):
    state, _dem = _built(tmp_path)
    path = save_state(state, tmp_path / "p")
    before = path.read_bytes()

    state.project.metadata["bad"] = object()
    state.project.mark_dirty()
    with pytest.raises(ProjectSaveError, match="bad"):
        save_state(state, path)
    assert path.read_bytes() == before
    assert state.project.dirty
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".p.rivelero.tmp")] == []

    del state.project.metadata["bad"]
    save_state(state, path)
    assert not state.project.dirty


def test_dirty_state_transitions(tmp_path):
    state, _dem = _built(tmp_path)
    save_state(state, tmp_path / "p")
    assert not state.project.dirty

    state.select_viewpoint("vp_center_360")
    state.set_active_page("world")
    state.set_map_extent((0.0, 1.0, 0.0, 1.0))
    state.clear_selection()
    assert not state.project.dirty  # UI state is not project state

    state.notify_design_changed()
    assert state.project.dirty

    save_state(state, tmp_path / "p")
    state.set_visibility_configuration(visibility_configuration(max_distance_m=80.0))
    assert state.project.dirty

    result = open_state(tmp_path / "p.rivelero")
    assert not result.state.project.dirty
