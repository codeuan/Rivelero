"""C1 regression: schema-1 projects written before the legacy cleanup load.

``tests/data/schema1_golden.rivelero`` was written by commit 24c31e9 (before
the C1 cleanup). It must keep loading with the same scientific content; its
DEM is linked relative to the project file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rivelero.gui.project_service import open_state, save_state

DATA = Path(__file__).resolve().parent / "data"
GOLDEN = DATA / "schema1_golden.rivelero"
EXPECTED = json.loads((DATA / "schema1_golden.expected.json").read_text(encoding="utf-8"))

pytestmark = pytest.mark.filterwarnings("ignore")


def _check(state):
    sof = state.analysis.survey_observability_field
    assert state.project.name == EXPECTED["project_name"]
    assert state.survey.n_viewpoints == EXPECTED["viewpoints"]
    assert state.survey.n_observation_events == EXPECTED["events"]
    assert state.survey.n_sensors == EXPECTED["sensors"]
    assert state.analysis.environment.environment_id == EXPECTED["environment_id"]
    assert state.analysis.analysis_domain.domain_id == EXPECTED["domain_id"]
    configuration = state.analysis.visibility_configuration
    assert configuration.configuration_id == EXPECTED["visibility_configuration_id"]
    assert configuration.max_distance_m == EXPECTED["max_distance_m"]
    assert sof is not None and sof.n_active_units == EXPECTED["active_units"]
    assert int(sof.n_observable_cells) == EXPECTED["observable_cells"]
    assert {str(int(k)): int(v) for k, v in sof.state_counts().items()} == EXPECTED["state_counts"]
    snapshots = state.analysis.scenario_workspace.snapshots
    assert [s.name for s in snapshots] == EXPECTED["snapshots"]
    assert snapshots[0].summary.scenario.observable_cells == EXPECTED["snapshot_observable"]
    live = state.analysis.design_scenario
    assert [k.sampling_unit_id for k in live.deactivated_keys] == EXPECTED["live_deactivated"]
    assert [c.candidate_id for c in live.candidates] == EXPECTED["live_candidates"]


def test_pre_cleanup_schema1_project_loads(tmp_path):
    result = open_state(GOLDEN)
    _check(result.state)
    assert not result.state.project.dirty


def test_schema1_round_trip_after_cleanup(tmp_path):
    # Opened in place (its DEM link is relative to tests/data); the re-saved
    # copy records links valid on the machine running the test.
    state = open_state(GOLDEN).state
    saved = save_state(state, tmp_path / "again")
    _check(open_state(saved).state)
