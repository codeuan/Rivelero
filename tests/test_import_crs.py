"""Regression: projected survey coordinates imported with a geographic CRS.

The synthetic ``viewpoints.csv`` stores UTM 33N metres and declares
``crs = EPSG:32633`` per row. Imported with the dialog's former default
Source CRS (EPSG:4326) its Viewpoints landed at "latitude 4 099 500 degrees":
nowhere near the DEM, and the build failed with a raw PROJ error.
"""

from __future__ import annotations

import pytest

from observability_fixtures import (
    DATA_ROOT, EVENTS_PATH, SENSORS_PATH, VIEWPOINTS_PATH, visibility_configuration,
)
from rivelero.core.viewpoint import Viewpoint, geographic_coordinate_problem
from rivelero.gui.survey_import import (
    SurveyImportError,
    SurveyImportOptions,
    SurveyImportReport,
    SurveyImportRowError,
    detect_crs_column,
    import_survey_csv,
    import_viewpoints_csv,
)

pytestmark = pytest.mark.filterwarnings("ignore")

DEM_RIDGE = DATA_ROOT / "environments" / "dem_ridge.tif"


def _csv(tmp_path, text, name="viewpoints.csv"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Coordinate validity
# ---------------------------------------------------------------------------


def test_geographic_coordinate_problem():
    assert geographic_coordinate_problem(500500, 4099500, "EPSG:4326").startswith(
        "longitude 500500 and latitude 4.0995e+06 out of range for EPSG:4326")
    assert geographic_coordinate_problem(14.9, 37.5, "EPSG:4326") is None
    assert geographic_coordinate_problem(-180, 90, "EPSG:4326") is None
    assert "latitude 91" in geographic_coordinate_problem(10, 91, "EPSG:4326")
    # Projected CRS: any finite value is acceptable here.
    assert geographic_coordinate_problem(500500, 4099500, "EPSG:32633") is None


def test_projected_coordinates_with_geographic_source_crs_are_rejected():
    with pytest.raises(SurveyImportRowError) as error:
        import_viewpoints_csv(VIEWPOINTS_PATH, options=SurveyImportOptions(
            source_crs="EPSG:4326", strict=True, allow_duplicate_coordinates=True))
    message = str(error.value)
    assert "row 2" in message and "not valid in the Source CRS" in message
    assert "The file's 'crs' column says EPSG:32633" in message

    # Permissive mode: every row is reported, none imported...
    report = SurveyImportReport()
    viewpoints = import_viewpoints_csv(
        VIEWPOINTS_PATH, report=report,
        options=SurveyImportOptions(source_crs="EPSG:4326", strict=False,
                                    allow_duplicate_coordinates=True),
    )
    assert viewpoints == [] and len(report.errors) == 10
    assert all("look like projected coordinates" in e for e in report.errors)
    # ...and a complete Survey import refuses to produce an empty Survey.
    with pytest.raises(SurveyImportError, match="No valid Viewpoints"):
        import_survey_csv(
            viewpoints_path=VIEWPOINTS_PATH, sensors_path=SENSORS_PATH,
            configuration_id="s", configuration_name="s",
            options=SurveyImportOptions(source_crs="EPSG:4326", strict=False,
                                        allow_duplicate_coordinates=True),
        )


def test_correct_source_crs_imports_inside_the_dem():
    import rasterio

    viewpoints = import_viewpoints_csv(VIEWPOINTS_PATH, options=SurveyImportOptions(
        source_crs="EPSG:32633", strict=True, allow_duplicate_coordinates=True))
    with rasterio.open(DEM_RIDGE) as dem:
        bounds = dem.bounds
    assert viewpoints and all(
        bounds.left <= vp.x <= bounds.right and bounds.bottom <= vp.y <= bounds.top
        for vp in viewpoints
    )


def test_real_longitude_latitude_are_still_accepted(tmp_path):
    path = _csv(tmp_path, "viewpoint_id,x,y\nvp_1,14.93,37.52\nvp_2,-3.70,40.42\n")
    viewpoints = import_viewpoints_csv(path, options=SurveyImportOptions(
        source_crs="EPSG:4326", target_crs="EPSG:32633", strict=True))
    assert [vp.viewpoint_id for vp in viewpoints] == ["vp_1", "vp_2"]
    assert viewpoints[0].crs.to_string() == "EPSG:32633"


def test_hint_without_crs_column(tmp_path):
    path = _csv(tmp_path, "viewpoint_id,x,y\nvp_1,500500,4099500\n")
    with pytest.raises(SurveyImportRowError, match="Set the Source CRS to the CRS"):
        import_viewpoints_csv(path, options=SurveyImportOptions(source_crs="EPSG:4326",
                                                                strict=True))


# ---------------------------------------------------------------------------
# CRS column detection
# ---------------------------------------------------------------------------


def test_detect_crs_column_from_synthetic_file():
    detection = detect_crs_column(VIEWPOINTS_PATH)
    assert detection.present and detection.crs == "EPSG:32633" and detection.message is None


@pytest.mark.parametrize("rows, crs, message", [
    ("vp_1,1,2,EPSG:32633\nvp_2,1,2,epsg:32633\n", "EPSG:32633", None),
    ("vp_1,1,2,EPSG:32633\nvp_2,1,2,\n", "EPSG:32633", None),
    ("vp_1,1,2,EPSG:32633\nvp_2,1,2,EPSG:4326\n", None, "mixes several CRSs"),
    ("vp_1,1,2,not-a-crs\n", None, "not a valid CRS"),
    ("vp_1,1,2,\n", None, "is empty"),
])
def test_detect_crs_column_cases(tmp_path, rows, crs, message):
    detection = detect_crs_column(_csv(tmp_path, "viewpoint_id,x,y,crs\n" + rows))
    assert detection.present and detection.crs == crs
    if message is None:
        assert detection.message is None
    else:
        assert message in detection.message


def test_detect_without_crs_column(tmp_path):
    detection = detect_crs_column(_csv(tmp_path, "viewpoint_id,x,y\nvp_1,1,2\n"))
    assert not detection.present and detection.crs is None


# ---------------------------------------------------------------------------
# Visibility engine: a clear per-Viewpoint error, not a PROJ traceback
# ---------------------------------------------------------------------------


def test_build_explains_viewpoints_that_cannot_be_placed(tmp_path):
    from rivelero.core.configuration import ViewpointConfiguration
    from rivelero.gui.application_state import ApplicationState
    from rivelero.gui.domain_service import build_domain_from_grid_extent, elevation_valid_mask
    from rivelero.gui.environment_import import import_elevation_environment
    from rivelero.gui.observability_service import create_visibility_store, prepare_build
    from rivelero.gui.task_controller import describe_error
    from rivelero.observability.builder import build_survey_observability_field

    # Bypass the importer, as an older project or a script could.
    survey = ViewpointConfiguration(
        configuration_id="s", name="s",
        viewpoints=[Viewpoint(viewpoint_id="vp_center_360", x=500500.0, y=4099500.0,
                              crs="EPSG:4326", observer_height_m=1.75)],
    )
    state = ApplicationState()
    state.set_viewpoint_configuration(survey)
    environment = import_elevation_environment(DEM_RIDGE, environment_id="e", name="ridge")
    state.set_environment(environment.environment, grid=environment.grid)
    state.set_analysis_domain(build_domain_from_grid_extent(
        grid=environment.grid, domain_id="d", name="all",
        valid_mask=elevation_valid_mask(DEM_RIDGE, environment.grid)).domain)
    state.set_visibility_configuration(visibility_configuration())
    state.set_visibility_store(create_visibility_store(tmp_path / "cache"))

    with pytest.raises(ValueError) as error:
        build_survey_observability_field(**prepare_build(state).build_kwargs())
    message = describe_error(error.value)
    assert message.startswith("Viewpoint 'vp_center_360' cannot be placed on the terrain")
    assert "latitude" in message and "wrong source CRS" in message
    assert "CPLE" not in message and "PROJ" not in message


# ---------------------------------------------------------------------------
# Import dialog: Source CRS from the file
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


_DIALOGS = []


def _dialog():
    from rivelero.gui.survey_import_dialog import SurveyImportDialog

    dialog = SurveyImportDialog()
    _DIALOGS.append(dialog)
    return dialog


def test_dialog_takes_source_crs_from_the_file(app):
    dialog = _dialog()
    assert dialog.source_crs_edit.text() == "EPSG:4326"
    dialog.viewpoints_selector.set_path(VIEWPOINTS_PATH)
    assert dialog.source_crs_edit.text() == "EPSG:32633"
    assert "set from the file's 'crs' column (EPSG:32633)" in dialog.crs_status.text()
    assert dialog._validate_crs_step()
    assert "Coordinates will remain in EPSG:32633" in dialog.crs_status.text()


def test_dialog_never_overrides_a_typed_crs(app, tmp_path):
    dialog = _dialog()
    dialog.source_crs_edit.setText("EPSG:25833")
    dialog.viewpoints_selector.set_path(VIEWPOINTS_PATH)
    assert dialog.source_crs_edit.text() == "EPSG:25833"
    assert "says EPSG:32633, but the Source CRS is EPSG:25833" in dialog.crs_status.text()
    assert dialog.crs_status.property("statusWarning")


def test_dialog_warns_about_mixed_crs_column(app, tmp_path):
    dialog = _dialog()
    path = _csv(tmp_path, "viewpoint_id,x,y,crs\nvp_1,1,2,EPSG:32633\nvp_2,1,2,EPSG:4326\n")
    dialog.viewpoints_selector.set_path(path)
    assert dialog.source_crs_edit.text() == "EPSG:4326"  # unchanged
    assert "mixes several CRSs" in dialog.crs_status.text()


def test_dialog_follows_a_newly_selected_file(app, tmp_path):
    dialog = _dialog()
    dialog.viewpoints_selector.set_path(VIEWPOINTS_PATH)
    other = _csv(tmp_path, "viewpoint_id,x,y,crs\nvp_1,1,2,EPSG:25830\n")
    dialog.viewpoints_selector.set_path(other)
    # The earlier automatic value is replaced; a user value would not be.
    assert dialog.source_crs_edit.text() == "EPSG:25830"
