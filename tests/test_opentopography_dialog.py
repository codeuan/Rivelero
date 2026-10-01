"""OpenTopography tab of the terrain dialog: API key, resolution, area."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

import rivelero.io.dem as dem
from rivelero.core.environment import ElevationModelType
from rivelero.gui.environment_import_dialog import API_KEY_SETTING, EnvironmentImportDialog

from test_opentopography import KEY, FakeSession, tiff_response

app = QApplication.instance() or QApplication([])


@dataclass
class VP:
    x: float
    y: float
    crs: str = "EPSG:32633"


# Two Viewpoints ~ 1 km apart in Sicily (UTM 33N).
SURVEY = [VP(420000.0, 4096000.0), VP(421000.0, 4097000.0)]


@pytest.fixture
def settings(tmp_path):
    return QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)


@pytest.fixture(autouse=True)
def no_environment_key(monkeypatch):
    monkeypatch.delenv("OPENTOPO_API_KEY", raising=False)


def _dialog(settings, viewpoints=SURVEY):
    return EnvironmentImportDialog(viewpoints=viewpoints, settings=settings)


def test_key_entry_validates_format_and_is_masked(settings):
    dialog = _dialog(settings)

    assert dialog.api_key_edit.echoMode() == dialog.api_key_edit.EchoMode.Password
    assert not dialog.download_button.isEnabled()
    assert "Enter your OpenTopography API key" in dialog.api_status.text()

    dialog.api_key_edit.setText("abc def")
    assert dialog.api_status.property("statusError")
    assert not dialog.download_button.isEnabled()

    dialog.api_key_edit.setText(f"OPENTOPO_API_KEY={KEY}")
    assert dialog.api_status.text().startswith("✓ Key format is valid")
    assert dialog.download_button.isEnabled()

    dialog.show_key_button.setChecked(True)
    assert dialog.api_key_edit.echoMode() == dialog.api_key_edit.EchoMode.Normal


def test_key_is_read_from_environment(settings, monkeypatch):
    monkeypatch.setenv("OPENTOPO_API_KEY", KEY)
    dialog = _dialog(settings)
    assert dialog.api_key_edit.text() == KEY
    assert "OPENTOPO_API_KEY environment variable" in dialog.api_status.text()


def test_resolution_is_explicit(settings):
    dialog = _dialog(settings)
    combo_texts = [dialog.ot_dataset.itemText(i) for i in range(dialog.ot_dataset.count())]
    assert "Copernicus GLO-30 — 1 arc-second (≈ 30 m)" in combo_texts
    assert "Copernicus GLO-90 — 3 arc-seconds (≈ 90 m)" in combo_texts

    assert "30.9 m north–south" in dialog.ot_dataset_info.text()
    summary = dialog.ot_resolution_summary.text()
    assert "30 m × 30 m cells in EPSG:32633" in summary
    assert "native 1 arc-second (≈ 30 m)" in summary

    # Changing dataset resets the cell size to its native resolution.
    dialog.ot_dataset.setCurrentIndex(dialog.ot_dataset.findData("COP90"))
    assert dialog.ot_resolution.value() == 90.0
    assert "90 m × 90 m cells" in dialog.ot_resolution_summary.text()

    dialog.ot_resolution.setValue(10.0)
    assert "finer than the dataset's native resolution" in dialog.ot_resolution_summary.text()


def test_area_of_interest_and_crs_validation(settings):
    dialog = _dialog(settings)
    dialog.api_key_edit.setText(KEY)
    assert dialog.ot_area_survey.isChecked()
    assert "km²" in dialog.ot_area_summary.text()

    dialog.ot_area_custom.setChecked(True)
    dialog.ot_bounds["north"].setValue(dialog.ot_bounds["south"].value() - 1)
    assert "South must be less than North" in dialog.ot_area_summary.text()
    assert not dialog.download_button.isEnabled()

    dialog.ot_bounds["north"].setValue(dialog.ot_bounds["south"].value() + 0.05)
    assert dialog.download_button.isEnabled()

    dialog.ot_crs.setText("EPSG:4326")
    dialog.ot_crs.textEdited.emit("EPSG:4326")
    assert "projected in metres" in dialog.ot_resolution_summary.text()
    assert not dialog.download_button.isEnabled()


def test_without_survey_a_custom_box_is_used(settings):
    dialog = _dialog(settings, viewpoints=())
    assert not dialog.ot_area_survey.isEnabled()
    assert dialog.ot_area_custom.isChecked()


def test_download_task_builds_projected_environment(tmp_path, monkeypatch):
    session = FakeSession(tiff_response(tmp_path))
    monkeypatch.setattr(dem, "requests", type("R", (), {
        "get": staticmethod(session.get), "RequestException": Exception,
    }))

    result = EnvironmentImportDialog._download_and_import(
        bbox=(37.00, 37.05, 14.00, 14.05), dataset="EU_DTM", api_key=KEY,
        output_path=str(tmp_path / "terrain.tif"), crs="EPSG:32633",
        resolution_m=30.0, resampling="bilinear",
        environment_name="Downloaded", survey_buffer_m=500.0,
    )

    elevation = result.environment.elevation_model
    assert elevation.crs.to_epsg() == 32633
    assert elevation.resolution_m == 30.0
    assert elevation.model_type == ElevationModelType.DTM
    assert elevation.provenance["native_resolution"] == "1 arc-second (≈ 30 m)"
    assert elevation.provenance["survey_buffer_m"] == 500.0
    assert KEY not in repr(elevation.provenance)


def test_key_is_remembered_only_when_asked(settings, monkeypatch):
    dialog = _dialog(settings)
    dialog.api_key_edit.setText(KEY)
    dialog.ot_output.setText("unused.tif")

    class Controller:
        def start(self, **kwargs):
            return "task"

    dialog._task_controller = Controller()
    dialog._download_opentopo()
    assert not settings.value(API_KEY_SETTING)

    dialog._active_task_id = None
    dialog.remember_key_checkbox.setChecked(True)
    dialog._download_opentopo()
    assert settings.value(API_KEY_SETTING) == KEY

    again = _dialog(settings)
    assert again.api_key_edit.text() == KEY
    assert again.remember_key_checkbox.isChecked()
