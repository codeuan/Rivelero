import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from rivelero.gui.main_window import MainWindow


def test_experiment_menu_and_go_buttons(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    action = next(a for a in window.menuBar().actions() if "Experiments" in a.text())
    action.trigger()
    dialog = window.experiment_dialog
    dem = tmp_path / "terrain.tif"
    dem.touch()
    dialog.dem.setText(str(dem))
    dialog.folder.setText(str(tmp_path))
    submitted = []

    def record(**kwargs):
        submitted.append(kwargs["args"][0])
        return "task"

    monkeypatch.setattr(dialog.controller, "start", record)
    assert dialog.tabs.count() == 3
    for index in range(3):
        dialog.tabs.setCurrentIndex(index)
        dialog.forms[index]["camera_height_m"].setValue(2.25 + index)
        dialog.buttons[index].click()
        assert submitted[-1].experiment == index + 1
        assert submitted[-1].camera_height_m == 2.25 + index
        assert dialog.cancel.isEnabled()
        dialog.controller.task_finished.emit("task")
        assert dialog.buttons[index].isEnabled()
    dialog.close()
    window.close()
    app.processEvents()
