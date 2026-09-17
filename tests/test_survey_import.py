"""Integration tests for Rivelero survey CSV import."""

from pathlib import Path

from rasterio.crs import CRS

from rivelero.gui.survey_import import (
    SurveyImportOptions,
    import_survey_csv,
)


DATA_ROOT = Path(
    r"C:\Users\zool2620\VISTA\rivelero_synthetic_observability"
)

VIEWPOINTS_PATH = (
    DATA_ROOT
    / "viewpoints"
    / "viewpoints.csv"
)

SENSORS_PATH = (
    DATA_ROOT
    / "viewpoints"
    / "sensors.csv"
)

EVENTS_PATH = (
    DATA_ROOT
    / "viewpoints"
    / "observation_events.csv"
)


def test_synthetic_survey_import():

    options = SurveyImportOptions(
        source_crs="EPSG:32633",
        target_crs="EPSG:32633",
        strict=True,
        allow_duplicate_coordinates=True,
    )

    result = import_survey_csv(
        viewpoints_path=VIEWPOINTS_PATH,
        sensors_path=SENSORS_PATH,
        observation_events_path=EVENTS_PATH,
        configuration_id="synthetic_import_test",
        configuration_name="Synthetic import test",
        options=options,
    )

    configuration = (
        result.viewpoint_configuration
    )

    report = result.report

    sensors = result.sensors

    viewpoints = list(
        configuration.viewpoints
    )

    events = list(
        configuration.observation_events
    )

    # ---------------------------------------------------------
    # Basic import
    # ---------------------------------------------------------

    assert len(viewpoints) > 0

    assert report.viewpoints_imported == len(
        viewpoints
    )

    assert report.viewpoints_skipped == 0

    # ---------------------------------------------------------
    # Viewpoint identity
    # ---------------------------------------------------------

    viewpoint_ids = [
        viewpoint.viewpoint_id
        for viewpoint in viewpoints
    ]

    assert len(
        viewpoint_ids
    ) == len(
        set(viewpoint_ids)
    )

    # ---------------------------------------------------------
    # CRS
    # ---------------------------------------------------------

    expected_crs = CRS.from_epsg(
        32633
    )

    for viewpoint in viewpoints:
        assert viewpoint.crs == expected_crs

    # ---------------------------------------------------------
    # Sensors
    # ---------------------------------------------------------

    assert len(sensors) > 0

    assert report.sensors_imported == len(
        sensors
    )

    assert report.sensors_skipped == 0

    # ---------------------------------------------------------
    # ObservationEvents
    # ---------------------------------------------------------

    assert len(events) > 0

    assert report.events_imported == len(
        events
    )

    assert report.events_skipped == 0

    event_ids = [
        event.event_id
        for event in events
    ]

    assert len(
        event_ids
    ) == len(
        set(event_ids)
    )

    # Every event must reference a real Viewpoint.
    viewpoint_id_set = set(
        viewpoint_ids
    )

    for event in events:
        assert (
            event.viewpoint_id
            in viewpoint_id_set
        )

    # ---------------------------------------------------------
    # Duplicate coordinates are legitimate
    # ---------------------------------------------------------

    coordinates = [
        (
            viewpoint.x,
            viewpoint.y,
        )
        for viewpoint in viewpoints
    ]

    # Do NOT assert that coordinates are unique.
    # Rivelero explicitly permits multiple Viewpoints at the
    # same physical location.

    assert len(
        coordinates
    ) == len(
        viewpoints
    )

    # ---------------------------------------------------------
    # Missing metadata must remain missing
    # ---------------------------------------------------------

    viewpoints_with_missing_heading = [
        viewpoint
        for viewpoint in viewpoints
        if viewpoint.heading_deg is None
    ]

    # The synthetic dataset was designed to include missing
    # metadata. Import must not silently replace it with defaults.
    assert len(
        viewpoints_with_missing_heading
    ) > 0

    for viewpoint in viewpoints_with_missing_heading:
        assert viewpoint.heading_deg is None

    # ---------------------------------------------------------
    # Event ordering
    # ---------------------------------------------------------

    # Configuration must preserve the source event order.
    assert list(
        configuration.observation_events
    ) == events

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------

    assert report.successful

    print(
        "\n--- RIVELERO SURVEY IMPORT ---"
    )

    print(
        "Viewpoints:",
        len(viewpoints),
    )

    print(
        "Sensors:",
        len(sensors),
    )

    print(
        "ObservationEvents:",
        len(events),
    )

    print(
        "Viewpoint errors:",
        report.viewpoints_skipped,
    )

    print(
        "Sensor errors:",
        report.sensors_skipped,
    )

    print(
        "Event errors:",
        report.events_skipped,
    )

    print(
        "Coordinate transformation:",
        report.transformed_coordinates,
    )

    print(
        "Warnings:",
        len(report.warnings),
    )

    print(
        "Errors:",
        len(report.errors),
    )

    print(
        "\nSUCCESS: synthetic survey import passed."
    )