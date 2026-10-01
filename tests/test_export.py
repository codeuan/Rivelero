"""P2 tests: scientific data export (Qt-free).

Rasters are checked by reopening them with rasterio (values, CRS, transform,
shape, dtype, NoData, codes, tags, sidecar); tables by reading them back and
by re-importing the Survey tables with the survey importer.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone

import numpy as np
import pytest
import rasterio

from rivelero.analysis.comparison import (
    BASELINE_ID,
    LIVE_SCENARIO_ID,
    baseline_state,
    compare_states,
    live_state,
)
from rivelero.analysis.contribution import analyse_contributions
from rivelero.analysis.scenario import ScenarioChangeClass, SurveyDesignScenario
from rivelero.core.configuration import ViewpointConfiguration
from rivelero.core.observation import ObservationEvent
from rivelero.core.sensor import Sensor
from rivelero.core.viewpoint import Viewpoint
from rivelero.export import files as export_files
from rivelero.export import rasters, tables
from rivelero.export.catalog import (
    ExportConflictError,
    ExportContext,
    ExportUnavailableError,
    export_catalog,
    run_export,
)
from rivelero.export.files import PathTooLongError, atomic_output, safe_filename_part
from rivelero.export.metadata import ExportProvenance, sidecar_path
from rivelero.gui.survey_import import SurveyImportOptions, import_survey_csv
from rivelero.observability.storage import StoredVisibility, VisibilityStore
from test_analysis_scenario import CRS_UTM, TRANSFORM, _baseline, _cells, _key

PROVENANCE = ExportProvenance(
    project_name="Test project", sof_id="baseline", analysis_domain_id="domain",
    visibility_configuration_id="config", visibility_configuration_name="Test visibility",
)
CANDIDATE_MASK = _cells((1, 4), (2, 4))  # both blind in the baseline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _store(tmp_path, masks):
    store = VisibilityStore(tmp_path / "store")
    for name, mask in {**masks, "candidate_001": CANDIDATE_MASK}.items():
        store.put(StoredVisibility(key=_key(name), visibility_mask=mask,
                                   transform=TRANSFORM, crs=CRS_UTM, metadata={}))
    return store


def _scenario(sof, masks):
    """Deactivate A (loses (1,1)) and add a candidate (gains (1,4),(2,4))."""
    scenario = SurveyDesignScenario(sof)
    scenario.deactivate({_key("A"): masks["A"]})
    scenario.add_candidate(Viewpoint(viewpoint_id="candidate_001", x=500045.0,
                                     y=4099975.0, crs="EPSG:32633"))
    scenario.set_candidate_visibility("candidate_001", key=_key("candidate_001"),
                                      mask=CANDIDATE_MASK)
    return scenario


def _survey():
    viewpoints = [
        Viewpoint(viewpoint_id="vp_α", x=500000.123456789, y=4099000.5, crs="EPSG:32633",
                  heading_deg=45.0, sensor_id="cam", source="field notes, day 1",
                  extra_metadata={"observer": "Zoë", "crs": "legacy"}),
        # Repeated coordinates are kept as separate rows.
        Viewpoint(viewpoint_id="vp_b", x=500000.123456789, y=4099000.5, crs="EPSG:32633",
                  observer_height_m=1.6),
    ]
    events = [
        ObservationEvent(event_id="ev_1", viewpoint_id="vp_α",
                         timestamp=datetime(2026, 5, 1, 9, 30, tzinfo=timezone.utc),
                         sequence_id="s", sequence_index=0, image_id="IMG 1.jpg"),
        ObservationEvent(event_id="ev_2", viewpoint_id="vp_b"),
    ]
    configuration = ViewpointConfiguration(
        configuration_id="survey", name="Survey", viewpoints=viewpoints,
        observation_events=events,
    )
    sensors = (
        Sensor(sensor_id="cam", horizontal_fov_deg=84.0, spectral_bands=("red", "green"),
               wavelength_range_nm=(400.0, 700.0), extra_metadata={"name": "Main camera"}),
    )
    return configuration, sensors


def _context(tmp_path, *, comparison_sides=(BASELINE_ID, LIVE_SCENARIO_ID), **overrides):
    sof, masks = _baseline()
    store = _store(tmp_path, masks)
    scenario = _scenario(sof, masks)
    states = {BASELINE_ID: baseline_state(sof), LIVE_SCENARIO_ID: live_state(scenario)}
    comparison = compare_states(states[comparison_sides[0]], states[comparison_sides[1]], sof)
    configuration, sensors = _survey()
    values = dict(
        provenance=PROVENANCE, prefix="test", configuration=configuration, sensors=sensors,
        survey_reason=None, sof=sof, sof_reason=None,
        contribution=analyse_contributions(sof, store), contribution_reason=None,
        scenario_exposure=scenario.exposure, scenario_change=scenario.change_classes(),
        scenario_reason=None,
        scenario_rows=(tables.scenario_row(
            state_id="baseline", kind="baseline", label="Baseline", compatible=True,
            baseline_sof_id="baseline", coverage=states[BASELINE_ID].summary, baseline=None,
            active_existing_units=3, deactivated_units=0, included_candidates=0),),
        scenario_rows_reason=None, comparison=comparison, comparison_reason=None,
    )
    values.update(overrides)
    return ExportContext(**values), sof, scenario


def _read(path):
    with rasterio.open(path) as dataset:
        return dataset.read(1), dataset.profile, dataset.tags()


def _sidecar(path):
    return json.loads(sidecar_path(path).read_text(encoding="utf-8"))


def _rows(path):
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


# ---------------------------------------------------------------------------
# Rasters
# ---------------------------------------------------------------------------


def _assert_grid(profile, sof):
    assert profile["crs"] == sof.crs
    assert profile["transform"] == sof.transform
    assert (profile["height"], profile["width"]) == sof.exposure_count.shape


def test_exposure_count_georeferencing_and_nodata(tmp_path):
    sof, _ = _baseline()
    path, sidecar = rasters.write_sof_product("exposure_count", sof, tmp_path / "e.tif", PROVENANCE)
    data, profile, tags = _read(path)
    _assert_grid(profile, sof)
    assert profile["dtype"] == "uint16"
    nodata = np.iinfo(np.uint16).max
    assert profile["nodata"] == nodata
    analysable = sof.analysable_mask
    np.testing.assert_array_equal(data[analysable], sof.exposure_count[analysable])
    assert np.all(data[~analysable] == nodata)
    # Blind spots are real zeros, never NoData.
    assert np.all(data[sof.blindspot_mask] == 0) and sof.blindspot_mask.sum() == 4
    assert tags["RIVELERO_PRODUCT"] == "exposure_count"
    assert tags["RIVELERO_SOF_ID"] == "baseline"
    assert "Blind spots are 0" in tags["RIVELERO_NODATA"]
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    assert record["provenance"]["analysis_domain_id"] == "domain"
    assert record["provenance"]["visibility_configuration_id"] == "config"
    assert record["units"] == "count of sampling units"
    assert record["grid"]["crs"] == "EPSG:32633"
    assert record["software"]["name"] == "rivelero"


def test_observability_state_keeps_all_four_codes(tmp_path):
    sof, _ = _baseline()
    path, _ = rasters.write_sof_product("observability_state", sof, tmp_path / "s.tif", PROVENANCE)
    data, profile, tags = _read(path)
    _assert_grid(profile, sof)
    assert profile["nodata"] is None and profile["dtype"] == "uint8"
    np.testing.assert_array_equal(data, sof.observability_state)
    assert set(np.unique(data)) == {0, 1, 2, 3}
    # Outside (0), invalid (1) and blind (2) stay distinct.
    assert data[0, 0] == 0 and data[1, 0] == 1 and data[1, 4] == 2 and data[1, 1] == 3
    assert tags["RIVELERO_CODE_1"] == "invalid_unanalysable"
    assert tags["RIVELERO_CODE_2"] == "blind_spot"
    assert _sidecar(path)["codes"]["0"] == "outside_analysis_domain"


def test_normalized_exposure_and_masks(tmp_path):
    sof, _ = _baseline()
    analysable = sof.analysable_mask
    path, _ = rasters.write_sof_product("normalized_exposure", sof, tmp_path / "n.tif", PROVENANCE)
    data, profile, _ = _read(path)
    assert profile["dtype"] == "float32"
    assert np.all(np.isnan(data[~analysable]))
    np.testing.assert_allclose(data[analysable], sof.normalized_exposure[analysable])
    assert np.all(data[sof.blindspot_mask] == 0.0)

    path, _ = rasters.write_sof_product("blindspot_mask", sof, tmp_path / "b.tif", PROVENANCE)
    data, profile, _ = _read(path)
    assert profile["nodata"] == 255
    assert np.all(data[~analysable] == 255)
    np.testing.assert_array_equal(data[analysable], sof.blindspot_mask[analysable])

    for product, expected in (("analysis_mask", sof.analysis_mask), ("valid_mask", sof.valid_mask)):
        path, _ = rasters.write_sof_product(product, sof, tmp_path / f"{product}.tif", PROVENANCE)
        data, profile, _ = _read(path)
        assert profile["nodata"] is None
        np.testing.assert_array_equal(data.astype(bool), expected)


def test_coverage_class_raster(tmp_path):
    sof, _ = _baseline()
    path, _ = rasters.write_coverage_class(sof, tmp_path / "c.tif", PROVENANCE)
    data, profile, tags = _read(path)
    _assert_grid(profile, sof)
    assert profile["nodata"] is None
    assert data[1, 1] == 3 and data[1, 2] == 4 and data[1, 4] == 2  # unique, repeated, blind
    assert data[0, 0] == 0 and data[1, 0] == 1
    assert tags["RIVELERO_CODE_4"] == "observable_repeated_exposure_2_or_more"


def test_scenario_exposure_and_change(tmp_path):
    sof, masks = _baseline()
    scenario = _scenario(sof, masks)
    path, _ = rasters.write_scenario_exposure(
        scenario.exposure, sof, tmp_path / "x.tif", PROVENANCE,
        scenario_record={"deactivated_sampling_units": ["A"]},
    )
    data, profile, tags = _read(path)
    _assert_grid(profile, sof)
    analysable = sof.analysable_mask
    assert profile["dtype"] == str(sof.exposure_count.dtype)
    np.testing.assert_array_equal(data[analysable], scenario.exposure[analysable])
    assert np.all(data[~analysable] == np.iinfo(sof.exposure_count.dtype).max)
    assert data[1, 1] == 0  # lost: now a blind spot, still not NoData
    assert json.loads(tags["RIVELERO_SCENARIO"])["deactivated_sampling_units"] == ["A"]

    path, _ = rasters.write_scenario_change(scenario.change_classes(), sof, tmp_path / "c.tif", PROVENANCE)
    data, profile, tags = _read(path)
    assert profile["nodata"] is None
    assert data[1, 1] == ScenarioChangeClass.LOST_COVERAGE
    assert data[1, 4] == data[2, 4] == ScenarioChangeClass.GAINED_COVERAGE
    assert data[0, 0] == ScenarioChangeClass.OUTSIDE_DOMAIN
    assert data[1, 0] == ScenarioChangeClass.INVALID
    assert tags["RIVELERO_CODE_4"] == "observable_in_baseline_blind_in_scenario"
    assert tags["RIVELERO_DIRECTION"] == "baseline_to_scenario"


def test_comparison_difference_is_right_minus_left(tmp_path):
    sof, masks = _baseline()
    scenario = _scenario(sof, masks)
    left, right = baseline_state(sof), live_state(scenario)
    forward = compare_states(left, right, sof)
    backward = compare_states(right, left, sof)

    path, _ = rasters.write_comparison_difference(forward, sof, tmp_path / "f.tif", PROVENANCE)
    data, profile, tags = _read(path)
    _assert_grid(profile, sof)
    nodata = np.iinfo(np.int32).min
    assert profile["dtype"] == "int32" and profile["nodata"] == nodata
    analysable = sof.analysable_mask
    expected = scenario.exposure.astype(np.int64) - sof.exposure_count.astype(np.int64)
    np.testing.assert_array_equal(data[analysable], expected[analysable])
    assert np.all(data[~analysable] == nodata)
    assert data[1, 1] == -1 and data[1, 4] == 1
    # Unchanged blind or observable cells are 0, a real value.
    assert data[2, 0] == 0
    assert tags["RIVELERO_DIRECTION"] == "right_minus_left"
    record = _sidecar(path)
    assert record["left"]["state_id"] == BASELINE_ID and record["right"]["state_id"] == LIVE_SCENARIO_ID

    path, _ = rasters.write_comparison_difference(backward, sof, tmp_path / "b.tif", PROVENANCE)
    swapped, _, _ = _read(path)
    np.testing.assert_array_equal(swapped[analysable], -data[analysable])

    path, _ = rasters.write_comparison_change(forward, sof, tmp_path / "cf.tif", PROVENANCE)
    change, _, tags = _read(path)
    assert change[1, 1] == ScenarioChangeClass.LOST_COVERAGE
    assert tags["RIVELERO_CODE_5"] == "blind_in_left_observable_in_right"
    path, _ = rasters.write_comparison_change(backward, sof, tmp_path / "cb.tif", PROVENANCE)
    change_back, _, _ = _read(path)
    assert change_back[1, 1] == ScenarioChangeClass.GAINED_COVERAGE


# ---------------------------------------------------------------------------
# Survey tables
# ---------------------------------------------------------------------------


def test_survey_tables_content(tmp_path):
    configuration, sensors = _survey()
    path, sidecar = tables.write_viewpoints(configuration, tmp_path / "vp.csv", PROVENANCE)
    rows = _rows(path)
    assert [row["viewpoint_id"] for row in rows] == ["vp_α", "vp_b"]
    # Repeated coordinates are preserved with full precision.
    assert rows[0]["x"] == rows[1]["x"] == "500000.123456789"
    # Missing values are empty cells, not None / nan / 0.
    assert rows[0]["observer_height_m"] == "" and rows[1]["heading_deg"] == ""
    assert rows[0]["source"] == "field notes, day 1"
    assert rows[0]["observer"] == "Zoë"
    assert rows[0]["extra_crs"] == "legacy" and rows[0]["crs"] == "EPSG:32633"
    assert json.loads(sidecar.read_text(encoding="utf-8"))["rows"] == 2

    path, _ = tables.write_observation_events(configuration, tmp_path / "ev.csv", PROVENANCE)
    rows = _rows(path)
    assert rows[0]["viewpoint_id"] == "vp_α"
    assert rows[0]["timestamp"] == "2026-05-01T09:30:00+00:00"
    assert rows[1]["timestamp"] == "" and rows[1]["sequence_index"] == ""

    path, _ = tables.write_sensors(sensors, tmp_path / "s.csv", PROVENANCE)
    row = _rows(path)[0]
    assert row["spectral_bands"] == "red;green"
    assert row["wavelength_range_nm"] == "400.0,700.0"
    assert row["name"] == "Main camera" and row["model"] == ""


def test_survey_tables_reimport(tmp_path):
    configuration, sensors = _survey()
    vp, _ = tables.write_viewpoints(configuration, tmp_path / "vp.csv", PROVENANCE)
    ev, _ = tables.write_observation_events(configuration, tmp_path / "ev.csv", PROVENANCE)
    se, _ = tables.write_sensors(sensors, tmp_path / "s.csv", PROVENANCE)
    result = import_survey_csv(
        viewpoints_path=vp, sensors_path=se, observation_events_path=ev,
        configuration_id="again", configuration_name="Again",
        options=SurveyImportOptions(source_crs="EPSG:32633", target_crs="EPSG:32633",
                                    strict=True, allow_duplicate_coordinates=True),
    )
    again = result.viewpoint_configuration
    for original, imported in zip(configuration.viewpoints, again.viewpoints):
        for name in ("viewpoint_id", "x", "y", "heading_deg", "observer_height_m",
                     "sensor_id", "source"):
            assert getattr(imported, name) == getattr(original, name), name
    assert again.viewpoints[0].extra_metadata["observer"] == "Zoë"
    event = again.observation_events[0]
    assert event.viewpoint_id == "vp_α"
    assert event.timestamp == configuration.observation_events[0].timestamp
    assert event.image_id == "IMG 1.jpg"
    sensor = result.sensors["cam"]
    assert sensor.spectral_bands == ("red", "green")
    assert tuple(sensor.wavelength_range_nm) == (400.0, 700.0)
    assert sensor.horizontal_fov_deg == 84.0


# ---------------------------------------------------------------------------
# Analysis tables
# ---------------------------------------------------------------------------


def test_contribution_table(tmp_path):
    context, sof, _ = _context(tmp_path)
    path, sidecar = tables.write_contributions(context.contribution, tmp_path / "c.csv", PROVENANCE)
    rows = {row["sampling_unit_id"]: row for row in _rows(path)}
    assert set(rows) == {"A", "B", "C"}
    a = rows["A"]
    # A: (1,1) unique; (1,2) and (2,1) repeated; 9 analysable cells.
    assert (a["visible_cells"], a["unique_cells"], a["repeated_cells"]) == ("3", "1", "2")
    assert a["cells_lost_if_removed"] == "1" and a["analysable_cells"] == "9"
    assert float(a["coverage_lost_if_removed_share_of_analysable"]) == pytest.approx(1 / 9)
    assert float(a["unique_share_of_unit_visibility"]) == pytest.approx(1 / 3)
    assert a["observation_event_id"] == "" and a["status"] == "available"
    text = path.read_text(encoding="utf-8").lower() + sidecar.read_text(encoding="utf-8").lower()
    assert "useless" not in text
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    assert "analysable_cells" in record["columns"]["coverage_lost_if_removed_share_of_analysable"]


def test_comparison_table_direction(tmp_path):
    context, _, _ = _context(tmp_path)
    path, sidecar = tables.write_comparison(context.comparison, tmp_path / "f.csv", PROVENANCE)
    rows = {row["metric"]: row for row in _rows(path)}
    assert list(_rows(path)[0]) == ["metric", "left", "right", "difference_right_minus_left", "unit"]
    observable = rows["observable_cells"]
    # Baseline 5 observable; scenario loses (1,1) and gains (1,4),(2,4).
    assert (observable["left"], observable["right"]) == ("5", "6")
    assert observable["difference_right_minus_left"] == "1"
    assert rows["cells_gained_coverage"]["difference_right_minus_left"] == "2"
    assert rows["cells_lost_coverage"]["difference_right_minus_left"] == "1"
    assert rows["cells_gained_coverage"]["left"] == ""
    pp = rows["observable_percent_of_analysable"]
    assert float(pp["difference_right_minus_left"]) == pytest.approx(100 / 9)
    assert "percentage points" in pp["unit"]
    assert json.loads(sidecar.read_text(encoding="utf-8"))["direction"] == "right_minus_left"
    assert "score" not in path.read_text(encoding="utf-8")

    swapped, _, _ = _context(tmp_path / "b", comparison_sides=(LIVE_SCENARIO_ID, BASELINE_ID))
    path, _ = tables.write_comparison(swapped.comparison, tmp_path / "b.csv", PROVENANCE)
    back = {row["metric"]: row for row in _rows(path)}
    assert back["observable_cells"]["difference_right_minus_left"] == "-1"
    assert back["cells_gained_coverage"]["difference_right_minus_left"] == "1"


# ---------------------------------------------------------------------------
# Catalog and runner
# ---------------------------------------------------------------------------


def test_run_export_writes_everything_available(tmp_path):
    context, _, _ = _context(tmp_path)
    catalog = export_catalog(context)
    assert all(product.available for product in catalog)
    keys = [product.key for product in catalog]
    result = run_export(context, keys, tmp_path / "out")
    assert result.successful and set(result.products_written) == set(keys)
    names = {path.name for path in (tmp_path / "out").iterdir()}
    assert len(names) == 2 * len(keys)  # every product has a JSON sidecar
    assert "test_comparison_Baseline_vs_Current_scenario_unsaved_exposure_difference_right_minus_left.tif" in names
    assert not [name for name in names if ".tmp" in name]


def test_unavailable_products_have_reasons(tmp_path):
    context = ExportContext(provenance=PROVENANCE)
    catalog = export_catalog(context)
    assert not any(product.available for product in catalog)
    assert all(product.reason for product in catalog)
    with pytest.raises(ExportUnavailableError):
        run_export(context, ["exposure_count"], tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_existing_file_refuses_whole_export_before_writing(tmp_path):
    context, _, _ = _context(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    existing = out / "test_observability_state.tif.json"
    existing.write_text("keep me", encoding="utf-8")
    unrelated = out / "notes.txt"
    unrelated.write_text("unrelated", encoding="utf-8")
    with pytest.raises(ExportConflictError) as error:
        run_export(context, ["exposure_count", "observability_state"], out)
    assert error.value.conflicts == [existing.resolve()]
    assert sorted(path.name for path in out.iterdir()) == ["notes.txt", existing.name]
    assert existing.read_text(encoding="utf-8") == "keep me"

    result = run_export(context, ["exposure_count", "observability_state"], out, overwrite=True)
    assert result.successful
    assert json.loads(existing.read_text(encoding="utf-8"))["product"] == "observability_state"
    assert unrelated.read_text(encoding="utf-8") == "unrelated"


def test_failed_product_is_reported_and_others_still_written(tmp_path, monkeypatch):
    context, _, _ = _context(tmp_path)
    original = rasters.sof_io.write_raster

    def failing(**kwargs):
        if kwargs["tags"]["rivelero_product"] == "coverage_class":
            raise OSError("disk full")
        return original(**kwargs)

    monkeypatch.setattr(rasters.sof_io, "write_raster", failing)
    out = tmp_path / "out"
    result = run_export(context, ["exposure_count", "coverage_class", "viewpoints"], out)
    assert [failure.key for failure in result.failures] == ["coverage_class"]
    assert "disk full" in result.failures[0].message
    assert set(result.products_written) == {"exposure_count", "viewpoints"}
    names = sorted(path.name for path in out.iterdir())
    assert not [name for name in names if "coverage_class" in name or ".tmp" in name]


def test_failed_sidecar_removes_new_raster(tmp_path, monkeypatch):
    sof, _ = _baseline()

    def broken(path, record, *, overwrite):
        raise OSError("cannot write sidecar")

    monkeypatch.setattr(rasters, "write_sidecar", broken)
    with pytest.raises(OSError):
        rasters.write_sof_product("exposure_count", sof, tmp_path / "e.tif", PROVENANCE)
    assert list(tmp_path.iterdir()) == []


def test_atomic_output_keeps_existing_file_on_failure(tmp_path):
    target = tmp_path / "table.csv"
    target.write_text("original", encoding="utf-8")
    with pytest.raises(RuntimeError):
        with atomic_output(target, overwrite=True) as temporary:
            temporary.write_text("half", encoding="utf-8")
            raise RuntimeError("interrupted")
    assert target.read_text(encoding="utf-8") == "original"
    assert [path.name for path in tmp_path.iterdir()] == ["table.csv"]
    with pytest.raises(FileExistsError):
        with atomic_output(target):
            pass


def test_unwritable_destination(tmp_path):
    context, _, _ = _context(tmp_path)
    blocker = tmp_path / "a_file"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        run_export(context, ["exposure_count"], blocker)
    assert blocker.read_text(encoding="utf-8") == "x"


def test_too_long_path_refused_before_writing(tmp_path, monkeypatch):
    context, _, _ = _context(tmp_path)
    monkeypatch.setattr(export_files, "MAX_PATH_LENGTH", len(str(tmp_path)) + 30)
    with pytest.raises(PathTooLongError):
        run_export(context, ["exposure_count", "viewpoints"], tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_safe_filename_part():
    assert safe_filename_part("Sicily Survey: día 1/2") == "Sicily_Survey_dia_1_2"
    assert safe_filename_part("///", fallback="rivelero") == "rivelero"
    assert len(safe_filename_part("x" * 200)) == 80
