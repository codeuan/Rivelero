from datetime import datetime, timezone

import pytest

from rivelero.core.configuration import ViewpointConfiguration
from rivelero.core.observation import ObservationEvent
from rivelero.core.sensor import Sensor, SensorModality
from rivelero.core.viewpoint import Viewpoint
from rivelero.gui.application_state import ApplicationState


def _viewpoint(viewpoint_id="vp-1", sensor_id=None):
    return Viewpoint(
        viewpoint_id=viewpoint_id,
        x=1.0,
        y=2.0,
        crs="EPSG:32633",
        sensor_id=sensor_id,
    )


def _state(with_event=False):
    events = []
    if with_event:
        events.append(ObservationEvent("event-1", "vp-1"))
    state = ApplicationState()
    state.set_viewpoint_configuration(
        ViewpointConfiguration(
            configuration_id="survey-1",
            name="Survey",
            viewpoints=[_viewpoint()],
            observation_events=events,
            configuration_type="observed",
            description="keep me",
            provenance={"source": "test"},
            extra_metadata={"order": 1},
        )
    )
    return state


def test_viewpoint_operations_preserve_configuration_and_allow_repeated_coordinates():
    state = _state()
    original = state.survey.viewpoint_configuration
    state.add_viewpoint(_viewpoint("vp-2"))
    assert state.survey.viewpoint_configuration.viewpoint_ids == ("vp-1", "vp-2")
    assert state.survey.viewpoint_configuration.description == "keep me"
    assert state.survey.viewpoint_configuration.extra_metadata == {"order": 1}

    state.replace_viewpoint(_viewpoint("vp-2"), previous_id="vp-2")
    assert state.survey.viewpoint_configuration.viewpoints[1].x == 1.0
    assert state.survey.viewpoint_configuration is not original
    state.remove_viewpoint("vp-2")
    assert state.survey.viewpoint_configuration.viewpoint_ids == ("vp-1",)


def test_viewpoint_duplicate_and_referenced_delete_are_rejected():
    state = _state(with_event=True)
    with pytest.raises(ValueError, match="Duplicate viewpoint_id"):
        state.add_viewpoint(_viewpoint("vp-1"))
    with pytest.raises(ValueError, match="referenced"):
        state.remove_viewpoint("vp-1")


def test_sensor_operations_preserve_modality_and_block_referenced_delete():
    state = _state()
    sensor = Sensor("sensor-1", modality=SensorModality.THERMAL)
    state.add_sensor(sensor)
    assert state.survey.sensors["sensor-1"].modality is SensorModality.THERMAL
    state.replace_viewpoint(_viewpoint("vp-1", sensor_id="sensor-1"), previous_id="vp-1")
    with pytest.raises(ValueError, match="referenced"):
        state.remove_sensor("sensor-1")
    state.replace_viewpoint(_viewpoint("vp-1"), previous_id="vp-1")
    state.remove_sensor("sensor-1")
    assert "sensor-1" not in state.survey.sensors


def test_event_operations_preserve_order_repeated_references_and_invalidate_sof():
    state = _state()
    state.add_observation_event(
        ObservationEvent(
            "event-1",
            "vp-1",
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
    )
    state.add_observation_event(ObservationEvent("event-2", "vp-1"))
    assert state.survey.viewpoint_configuration.event_ids == ("event-1", "event-2")
    with pytest.raises(ValueError, match="Duplicate event_id"):
        state.add_observation_event(ObservationEvent("event-1", "vp-1"))
    with pytest.raises(ValueError, match="not present"):
        state.add_observation_event(ObservationEvent("event-3", "missing"))
    state.replace_observation_event(ObservationEvent("event-1", "vp-1", sequence_index=4))
    assert state.survey.viewpoint_configuration.event_ids == ("event-1", "event-2")
    state.remove_observation_event("event-1")
    assert state.survey.viewpoint_configuration.event_ids == ("event-2",)
    with pytest.raises(KeyError):
        state.remove_observation_event("missing")
