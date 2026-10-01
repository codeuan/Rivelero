from datetime import datetime, timezone

from PySide6.QtWidgets import QApplication

from rivelero.core.configuration import ViewpointConfiguration
from rivelero.core.observation import ObservationEvent
from rivelero.core.sensor import Sensor, SensorModality
from rivelero.core.viewpoint import Viewpoint
from rivelero.gui.observation_event_dialog import ObservationEventDialog
from rivelero.gui.sensor_dialog import SensorDialog
from rivelero.gui.viewpoint_dialog import ViewpointDialog


app = QApplication.instance() or QApplication([])


def test_viewpoint_dialog_preserves_blank_optional_metadata_as_none():
    dialog = ViewpointDialog(sensors={})
    dialog._fields["viewpoint_id"].setText("manual-1")
    dialog._fields["x"].setText("1")
    dialog._fields["y"].setText("2")
    dialog._fields["crs"].setText("EPSG:4326")
    dialog._accept()
    assert dialog.result_viewpoint is not None
    assert dialog.result_viewpoint.heading_deg is None
    assert dialog.result_viewpoint.horizontal_fov_deg is None
    assert dialog.result_viewpoint.observer_height_m is None


def test_sensor_dialog_preserves_canonical_modality():
    dialog = SensorDialog()
    dialog._fields["sensor_id"].setText("thermal-1")
    dialog.modality_combo.setCurrentIndex(dialog.modality_combo.findData(SensorModality.THERMAL.value))
    dialog._accept()
    assert dialog.result_sensor is not None
    assert dialog.result_sensor.modality is SensorModality.THERMAL


def test_observation_event_dialog_round_trips_timezone_and_viewpoint_reference():
    configuration = ViewpointConfiguration.from_viewpoints(
        "survey-1",
        "Survey",
        [],
    )
    configuration.viewpoints.append(Viewpoint("vp-1", 0, 0, "EPSG:4326"))
    dialog = ObservationEventDialog(configuration=configuration)
    dialog._fields["event_id"].setText("event-1")
    dialog._fields["viewpoint_id"].setText("vp-1")
    dialog._fields["timestamp"].setText("2024-01-01T12:00:00+05:30")
    dialog._accept()
    assert dialog.result_event is not None
    assert dialog.result_event.timestamp == datetime.fromisoformat("2024-01-01T12:00:00+05:30")
    assert dialog.result_event.timestamp.tzinfo is not None
