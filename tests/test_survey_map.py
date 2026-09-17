from __future__ import annotations

from PySide6.QtWidgets import QApplication
from rasterio.crs import CRS

from rivelero.core.configuration import ViewpointConfiguration
from rivelero.core.viewpoint import Viewpoint
from rivelero.gui.survey_map import SurveyMapWidget
from rivelero.gui.survey_qc import SurveySpatialQC


def _make_viewpoints():
    return [
        Viewpoint(
            viewpoint_id="vp-1",
            x=0.0,
            y=0.0,
            crs="EPSG:32633",
            heading_deg=0.0,
            horizontal_fov_deg=90.0,
            observer_height_m=1.7,
            sensor_id="sensor-a",
            source="source-1",
            platform="platform-a",
        ),
        Viewpoint(
            viewpoint_id="vp-2",
            x=10.0,
            y=0.0,
            crs="EPSG:32633",
            heading_deg=None,
            horizontal_fov_deg=None,
            observer_height_m=None,
            sensor_id="sensor-b",
            source="source-2",
            platform="platform-a",
        ),
        Viewpoint(
            viewpoint_id="vp-3",
            x=10.0,
            y=0.0,
            crs="EPSG:32633",
            heading_deg=180.0,
            horizontal_fov_deg=80.0,
            observer_height_m=1.5,
            sensor_id=None,
            source=None,
            platform="platform-b",
        ),
        Viewpoint(
            viewpoint_id="vp-4",
            x=50.0,
            y=50.0,
            crs="EPSG:32633",
            heading_deg=270.0,
            horizontal_fov_deg=None,
            observer_height_m=None,
            sensor_id="sensor-a",
            source="source-1",
            platform=None,
        ),
    ]


app = QApplication.instance() or QApplication([])


def test_survey_qc_counts_repeated_locations_and_missing_metadata():
    configuration = ViewpointConfiguration(
        configuration_id="qc-test",
        name="QC Test",
        viewpoints=_make_viewpoints(),
    )

    qc = SurveySpatialQC.from_configuration(configuration)

    assert qc.total_viewpoints == 4
    assert qc.unique_coordinate_locations == 3
    assert qc.repeated_location_count == 1
    assert qc.missing_heading_count == 1
    assert qc.missing_fov_count == 2
    assert qc.missing_observer_height_count == 2
    assert qc.crs == CRS.from_epsg(32633)
    assert qc.repeated_coordinate_locations == {(10.0, 0.0)}


def test_survey_qc_empty_and_inconsistent_crs():
    empty = ViewpointConfiguration(
        configuration_id="empty",
        name="Empty",
        viewpoints=[],
    )
    empty_qc = SurveySpatialQC.from_configuration(empty)
    assert empty_qc.total_viewpoints == 0
    assert empty_qc.crs is None

    mismatched = ViewpointConfiguration(
        configuration_id="mixed-crs",
        name="Mixed CRS",
        viewpoints=[
            Viewpoint(viewpoint_id="a", x=0.0, y=0.0, crs="EPSG:4326"),
            Viewpoint(viewpoint_id="b", x=1.0, y=1.0, crs="EPSG:32633"),
        ],
    )

    assert SurveySpatialQC.from_configuration(mismatched).has_consistent_crs is False


def test_selection_lookup_handles_duplicate_coordinates_and_empty_cases():
    configuration = ViewpointConfiguration(
        configuration_id="lookup-test",
        name="Lookup test",
        viewpoints=_make_viewpoints(),
    )

    qc = SurveySpatialQC.from_configuration(configuration)

    assert qc.lookup_viewpoint_id("vp-2") == "vp-2"
    assert qc.lookup_viewpoint_by_point(10.0, 0.0) == "vp-2"
    assert qc.lookup_viewpoints_at_point(10.0, 0.0) == ["vp-2", "vp-3"]
    assert qc.lookup_viewpoint_id("missing") is None


def test_survey_map_widget_constructs_and_syncs_selection():
    widget = SurveyMapWidget()
    configuration = ViewpointConfiguration(
        configuration_id="gui-test",
        name="GUI Map Test",
        viewpoints=_make_viewpoints(),
    )

    widget.set_configuration(configuration)

    assert widget.configuration is configuration
    assert widget.current_display_mode == "all"
    assert widget.selected_viewpoint_id is None

    widget.set_selected_viewpoint_id("vp-3")
    assert widget.selected_viewpoint_id == "vp-3"

    widget.set_display_mode("sensor")
    assert widget.current_display_mode == "sensor"
