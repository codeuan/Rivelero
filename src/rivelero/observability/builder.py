"""Survey Observability Field builder for Rivelero.

This module orchestrates construction of a SurveyObservabilityField (SOF)
from a ViewpointConfiguration.

It connects:

    ViewpointConfiguration
    Environment
    AnalysisDomain
    VisibilityConfiguration
            |
            v
    VisibilityStore
            |
            +-- cache hit -> stored visibility
            |
            +-- cache miss -> visibility.engine -> GDAL
            |
            v
    SurveyObservabilityField

The builder deliberately contains no viewshed mathematics, exposure
mathematics, or storage implementation. Those responsibilities belong to
their respective Rivelero modules.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable

from rivelero.core.configuration import ViewpointConfiguration
from rivelero.core.domain import AnalysisDomain
from rivelero.core.environment import Environment
from rivelero.core.observation import ObservationEvent
from rivelero.core.sensor import Sensor
from rivelero.core.viewpoint import Viewpoint
from rivelero.observability.storage import (
    StoredVisibility,
    VisibilityKey,
    VisibilityStore,
)
from rivelero.observability.survey_field import (
    SurveyObservabilityField,
)
from rivelero.visibility.configuration import (
    SamplingUnit,
    VisibilityConfiguration,
)
from rivelero.visibility.engine import (
    SingleViewpointVisibility,
    ViewpointExcludedError,
    compute_viewpoint_visibility,
)


@dataclass(slots=True)
class SOFBuildReport:
    """Summary of one SurveyObservabilityField build operation.

    The report records what happened during construction without becoming
    part of the scientific SOF state itself.
    """

    requested_units: int = 0
    added_units: int = 0
    excluded_units: int = 0
    failed_units: int = 0

    cache_hits: int = 0
    computed_units: int = 0

    excluded_ids: list[str] = field(default_factory=list)
    failed_ids: list[str] = field(default_factory=list)

    errors: dict[str, str] = field(default_factory=dict)

    @property
    def successful_units(self) -> int:
        """Return the number of units successfully added to the SOF."""

        return self.added_units

    @property
    def complete(self) -> bool:
        """Return whether every requested unit was successfully added."""

        return (
            self.requested_units
            == self.added_units
            and self.excluded_units == 0
            and self.failed_units == 0
        )


@dataclass(slots=True)
class SOFBuildResult:
    """Result returned by the SOF builder."""

    sof: SurveyObservabilityField
    report: SOFBuildReport


def build_survey_observability_field(
    *,
    sof_id: str,
    viewpoint_configuration: ViewpointConfiguration,
    environment: Environment,
    domain: AnalysisDomain,
    visibility_configuration: VisibilityConfiguration,
    store: VisibilityStore,
    sensors: Mapping[str, Sensor] | None = None,
    skip_excluded: bool = True,
    continue_on_error: bool = False,
    metadata: dict[str, Any] | None = None,
    progress_callback: Callable[
        [int, int, str],
        None,
    ]
    | None = None,
) -> SOFBuildResult:
    """Build a SurveyObservabilityField from a viewpoint configuration.

    Parameters
    ----------
    sof_id
        Identifier assigned to the resulting SOF.

    viewpoint_configuration
        Survey configuration whose observability should be reconstructed.

    environment
        Physical spatial Environment.

    domain
        AnalysisDomain defining the raster grid and valid analysis space.

    visibility_configuration
        Scientific assumptions controlling visibility.

    store
        VisibilityStore used for lazy visibility calculation, disk
        persistence, and LRU memory caching.

    sensors
        Optional mapping from ``sensor_id`` to Sensor objects.

        When a Viewpoint references a Sensor, the builder resolves it through
        this mapping and passes it to the visibility engine.

    skip_excluded
        If True, sampling units rejected by a missing-metadata EXCLUDE policy
        are recorded and skipped.

        If False, ViewpointExcludedError is propagated.

    continue_on_error
        If False, unexpected errors stop the build immediately.

        If True, failed units are recorded in the build report and processing
        continues with the remaining units.

        This option should be used carefully because the resulting SOF then
        represents only the successfully processed subset.

    metadata
        Optional metadata attached to the resulting SOF.

    progress_callback
        Optional callback invoked after each processed sampling unit as:

            callback(processed, total, sampling_unit_id)

        This provides a future connection point for the GUI without making
        the builder depend on GUI code.

    Returns
    -------
    SOFBuildResult
        The constructed SurveyObservabilityField and a build report.

    Notes
    -----
    The sampling unit is controlled by VisibilityConfiguration:

    ``VIEWPOINT``
        Each unique Viewpoint contributes once.

    ``OBSERVATION_EVENT``
        Each ObservationEvent contributes separately. Repeated events may
        therefore reference the same underlying Viewpoint.

    Visibility is calculated lazily. If a requested VisibilityKey already
    exists in the VisibilityStore, the cached result is reused and GDAL is
    not called.
    """

    _validate_builder_inputs(
        viewpoint_configuration=viewpoint_configuration,
        environment=environment,
        domain=domain,
        visibility_configuration=visibility_configuration,
        store=store,
        sensors=sensors,
        progress_callback=progress_callback,
    )

    sensor_lookup: Mapping[str, Sensor] = (
        {}
        if sensors is None
        else sensors
    )

    sof = SurveyObservabilityField.empty(
        sof_id=sof_id,
        viewpoint_configuration=viewpoint_configuration,
        environment=environment,
        domain=domain,
        visibility_configuration=visibility_configuration,
        metadata=metadata,
    )

    units = _sampling_units(
        viewpoint_configuration=viewpoint_configuration,
        visibility_configuration=visibility_configuration,
    )

    report = SOFBuildReport(
        requested_units=len(units)
    )

    total = len(units)

    for index, unit in enumerate(
        units,
        start=1,
    ):
        unit_id = _sampling_unit_id(unit)

        try:
            viewpoint, event = _resolve_unit(
                unit=unit,
                viewpoint_configuration=viewpoint_configuration,
            )

            sensor = _resolve_sensor(
                viewpoint=viewpoint,
                sensors=sensor_lookup,
            )

            key = make_visibility_key(
                viewpoint=viewpoint,
                event=event,
                environment=environment,
                domain=domain,
                visibility_configuration=visibility_configuration,
            )

            was_cached = store.contains(key)

            stored_visibility = store.get_or_compute(
                key,
                compute=lambda vp=viewpoint, ev=event, sn=sensor: (
                    compute_viewpoint_visibility(
                        viewpoint=vp,
                        event=ev,
                        sensor=sn,
                        environment=environment,
                        domain=domain,
                        configuration=visibility_configuration,
                    )
                ),
            )

            sof.add_visibility(
                stored_visibility
            )

            report.added_units += 1

            if was_cached:
                report.cache_hits += 1
            else:
                report.computed_units += 1

        except ViewpointExcludedError as exc:
            report.excluded_units += 1
            report.excluded_ids.append(
                unit_id
            )
            report.errors[unit_id] = str(exc)

            if not skip_excluded:
                raise

        except Exception as exc:
            report.failed_units += 1
            report.failed_ids.append(
                unit_id
            )
            report.errors[unit_id] = (
                f"{type(exc).__name__}: {exc}"
            )

            if not continue_on_error:
                raise

        finally:
            if progress_callback is not None:
                progress_callback(
                    index,
                    total,
                    unit_id,
                )

    _attach_build_metadata(
        sof=sof,
        report=report,
    )

    return SOFBuildResult(
        sof=sof,
        report=report,
    )


# ---------------------------------------------------------------------------
# Visibility-key construction
# ---------------------------------------------------------------------------


def make_visibility_key(
    *,
    viewpoint: Viewpoint,
    environment: Environment,
    domain: AnalysisDomain,
    visibility_configuration: VisibilityConfiguration,
    event: ObservationEvent | None = None,
) -> VisibilityKey:
    """Construct the cache identity for one visibility calculation."""

    if not isinstance(
        viewpoint,
        Viewpoint,
    ):
        raise TypeError(
            "viewpoint must be a Viewpoint."
        )

    sampling_unit = (
        visibility_configuration.sampling_unit
    )

    if sampling_unit == SamplingUnit.VIEWPOINT:
        if event is not None:
            raise ValueError(
                "An ObservationEvent must not be supplied when "
                "sampling_unit='viewpoint'."
            )

        sampling_unit_id = (
            viewpoint.viewpoint_id
        )
        sampling_unit_type = (
            SamplingUnit.VIEWPOINT.value
        )

    elif (
        sampling_unit
        == SamplingUnit.OBSERVATION_EVENT
    ):
        if event is None:
            raise ValueError(
                "An ObservationEvent is required when "
                "sampling_unit='observation_event'."
            )

        if (
            event.viewpoint_id
            != viewpoint.viewpoint_id
        ):
            raise ValueError(
                "ObservationEvent references a different Viewpoint."
            )

        sampling_unit_id = event.event_id
        sampling_unit_type = (
            SamplingUnit.OBSERVATION_EVENT.value
        )

    else:
        raise ValueError(
            f"Unsupported sampling unit: {sampling_unit!r}."
        )

    return VisibilityKey(
        sampling_unit_id=sampling_unit_id,
        sampling_unit_type=sampling_unit_type,
        viewpoint_id=viewpoint.viewpoint_id,
        environment_id=environment.environment_id,
        analysis_domain_id=domain.domain_id,
        visibility_configuration_id=(
            visibility_configuration.configuration_id
        ),
    )


# ---------------------------------------------------------------------------
# Sampling-unit selection
# ---------------------------------------------------------------------------


def _sampling_units(
    *,
    viewpoint_configuration: ViewpointConfiguration,
    visibility_configuration: VisibilityConfiguration,
) -> list[Viewpoint | ObservationEvent]:
    """Return sampling units in deterministic configuration order."""

    if (
        visibility_configuration.sampling_unit
        == SamplingUnit.VIEWPOINT
    ):
        return list(
            viewpoint_configuration.viewpoints
        )

    if (
        visibility_configuration.sampling_unit
        == SamplingUnit.OBSERVATION_EVENT
    ):
        if not viewpoint_configuration.observation_events:
            raise ValueError(
                "VisibilityConfiguration requests observation-event "
                "sampling, but the ViewpointConfiguration contains no "
                "ObservationEvents."
            )

        return list(
            viewpoint_configuration.observation_events
        )

    raise ValueError(
        "Unsupported VisibilityConfiguration sampling unit."
    )


def _sampling_unit_id(
    unit: Viewpoint | ObservationEvent,
) -> str:
    """Return the identifier of a sampling unit."""

    if isinstance(
        unit,
        Viewpoint,
    ):
        return unit.viewpoint_id

    if isinstance(
        unit,
        ObservationEvent,
    ):
        return unit.event_id

    raise TypeError(
        "Sampling unit must be a Viewpoint or ObservationEvent."
    )


def _resolve_unit(
    *,
    unit: Viewpoint | ObservationEvent,
    viewpoint_configuration: ViewpointConfiguration,
) -> tuple[
    Viewpoint,
    ObservationEvent | None,
]:
    """Resolve one sampling unit to its Viewpoint and optional event."""

    if isinstance(
        unit,
        Viewpoint,
    ):
        return unit, None

    if isinstance(
        unit,
        ObservationEvent,
    ):
        viewpoint = (
            viewpoint_configuration.get_viewpoint(
                unit.viewpoint_id
            )
        )

        return viewpoint, unit

    raise TypeError(
        "Sampling unit must be a Viewpoint or ObservationEvent."
    )


# ---------------------------------------------------------------------------
# Sensor resolution
# ---------------------------------------------------------------------------


def _resolve_sensor(
    *,
    viewpoint: Viewpoint,
    sensors: Mapping[str, Sensor],
) -> Sensor | None:
    """Resolve a Viewpoint's optional Sensor reference."""

    if viewpoint.sensor_id is None:
        return None

    if viewpoint.sensor_id not in sensors:
        raise KeyError(
            f"Viewpoint {viewpoint.viewpoint_id!r} references "
            f"Sensor {viewpoint.sensor_id!r}, but that Sensor "
            "was not supplied to the SOF builder."
        )

    sensor = sensors[
        viewpoint.sensor_id
    ]

    if not isinstance(
        sensor,
        Sensor,
    ):
        raise TypeError(
            f"Sensor mapping entry {viewpoint.sensor_id!r} "
            "is not a Sensor."
        )

    return sensor


# ---------------------------------------------------------------------------
# Build metadata
# ---------------------------------------------------------------------------


def _attach_build_metadata(
    *,
    sof: SurveyObservabilityField,
    report: SOFBuildReport,
) -> None:
    """Attach non-scientific build information to SOF metadata."""

    sof.metadata["build_report"] = {
        "requested_units": (
            report.requested_units
        ),
        "added_units": (
            report.added_units
        ),
        "excluded_units": (
            report.excluded_units
        ),
        "failed_units": (
            report.failed_units
        ),
        "cache_hits": (
            report.cache_hits
        ),
        "computed_units": (
            report.computed_units
        ),
        "excluded_ids": list(
            report.excluded_ids
        ),
        "failed_ids": list(
            report.failed_ids
        ),
    }


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _validate_builder_inputs(
    *,
    viewpoint_configuration: ViewpointConfiguration,
    environment: Environment,
    domain: AnalysisDomain,
    visibility_configuration: VisibilityConfiguration,
    store: VisibilityStore,
    sensors: Mapping[str, Sensor] | None,
    progress_callback: Callable[
        [int, int, str],
        None,
    ]
    | None,
) -> None:
    """Validate top-level SOF builder inputs."""

    if not isinstance(
        viewpoint_configuration,
        ViewpointConfiguration,
    ):
        raise TypeError(
            "viewpoint_configuration must be a "
            "ViewpointConfiguration."
        )

    if not isinstance(
        environment,
        Environment,
    ):
        raise TypeError(
            "environment must be an Environment."
        )

    if not isinstance(
        domain,
        AnalysisDomain,
    ):
        raise TypeError(
            "domain must be an AnalysisDomain."
        )

    if not isinstance(
        visibility_configuration,
        VisibilityConfiguration,
    ):
        raise TypeError(
            "visibility_configuration must be a "
            "VisibilityConfiguration."
        )

    if not isinstance(
        store,
        VisibilityStore,
    ):
        raise TypeError(
            "store must be a VisibilityStore."
        )

    if sensors is not None:
        if not isinstance(
            sensors,
            Mapping,
        ):
            raise TypeError(
                "sensors must be a mapping from sensor_id to Sensor."
            )

        for sensor_id, sensor in sensors.items():
            if not isinstance(
                sensor_id,
                str,
            ):
                raise TypeError(
                    "Sensor mapping keys must be strings."
                )

            if not isinstance(
                sensor,
                Sensor,
            ):
                raise TypeError(
                    "Sensor mapping values must be Sensor objects."
                )

            if sensor_id != sensor.sensor_id:
                raise ValueError(
                    f"Sensor mapping key {sensor_id!r} does not match "
                    f"Sensor.sensor_id {sensor.sensor_id!r}."
                )

    if (
        progress_callback is not None
        and not callable(progress_callback)
    ):
        raise TypeError(
            "progress_callback must be callable or None."
        )