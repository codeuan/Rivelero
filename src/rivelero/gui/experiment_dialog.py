"""Controls for the server-backed Rivelero experiments."""
from pathlib import Path
from urllib.parse import urlparse

try:
    from PySide6 import QtWidgets as W
except ImportError:
    from PyQt6 import QtWidgets as W

from rivelero.experiments.studies import Settings, run_experiment


def number(value, low, high, decimals=None):
    box = W.QSpinBox() if decimals is None else W.QDoubleSpinBox()
    if decimals is not None:
        box.setDecimals(decimals)
        box.setSingleStep(10**-decimals)
    box.setRange(low, high)
    box.setValue(value)
    return box


class ExperimentDialog(W.QDialog):
    def __init__(self, window):
        super().__init__(window)
        self.setWindowTitle("Rivelero experiments")
        self.controller = window.task_controller
        self.task = None
        self.resize(600, 550)
        outer = W.QVBoxLayout(self)
        shared = W.QFormLayout()
        outer.addLayout(shared)
        self.dem = self.path_field(shared, "Terrain DEM", directory=False)
        environment = window.state.analysis.environment
        if environment:
            self.dem.setText(str(environment.elevation_model.source))
        self.folder = self.path_field(shared, "Shared results folder", directory=True)
        self.server = W.QLineEdit("http://127.0.0.1:8765")
        shared.addRow("Unreal experiment server", self.server)
        self.tabs = W.QTabWidget()
        outer.addWidget(self.tabs)
        self.buttons = []
        self.forms = []
        for exp, title in enumerate(("Viewpoint selection", "Observation gaps", "Spatial redundancy"), 1):
            tab = W.QWidget()
            form = W.QFormLayout(tab)
            controls = dict(
                cameras=number(10 if exp == 1 else 20, 1, 100),
                replicates=number(50 if exp == 2 else 100, 1, 10000),
                distance=number(250, 1, 100000), seed=number(42, 0, 2147483647),
                threshold=number(0.95, 0, 1, 3))
            for key, label in (("cameras", "Maximum cameras" if exp == 1 else "Cameras per layout"),
                               ("replicates", ("Random layouts", "Plant redistributions", "Layouts per mode")[exp-1]),
                               ("distance", "Observation distance (m)"), ("seed", "Seed"),
                               ("threshold", "Minimum IoU (must exceed)")):
                form.addRow(label, controls[key])
            if exp == 2:
                controls["layout"] = W.QComboBox()
                controls["layout"].addItems(["random", "dispersed", "clustered"])
                form.addRow("Fixed camera layout", controls["layout"])
                controls["plants"] = number(100, 1, 1000000)
                controls["scale_m"] = number(100, 1, 100000)
                controls["contrast"] = number(5, 0, 20, 1)
                form.addRow("Plants per replicate", controls["plants"])
                form.addRow("Perlin patch scale (m)", controls["scale_m"])
                form.addRow("Density contrast", controls["contrast"])
            self.forms.append(controls)
            button = W.QPushButton("Go")
            button.clicked.connect(lambda checked=False, i=exp: self.launch(i))
            form.addRow(button)
            self.buttons.append(button)
            self.tabs.addTab(tab, f"{exp}. {title}")
        self.status = W.QLabel()
        self.status.setWordWrap(True)
        outer.addWidget(self.status)
        self.cancel = W.QPushButton("Cancel run")
        self.cancel.setEnabled(False)
        self.cancel.clicked.connect(lambda: self.controller.cancel())
        outer.addWidget(self.cancel)
        self.controller.task_progress.connect(self.progress)
        self.controller.task_result.connect(self.show_status)
        self.controller.task_error.connect(lambda task, message, details: self.show_status(task, message))
        self.controller.task_cancelled.connect(self.show_status)
        self.controller.task_finished.connect(self.task_done)

    def path_field(self, form, label, directory):
        field = W.QLineEdit()
        row = W.QHBoxLayout()
        row.addWidget(field)
        browse = W.QPushButton("Browse…")
        row.addWidget(browse)

        def choose():
            if directory:
                value = W.QFileDialog.getExistingDirectory(self, label, field.text())
            else:
                value, _ = W.QFileDialog.getOpenFileName(self, label, field.text(), "Terrain (*.tif *.tiff);;All files (*)")
            if value:
                field.setText(value)

        browse.clicked.connect(choose)
        form.addRow(label, row)
        return field

    def launch(self, exp):
        values = {k: w.currentText() if isinstance(w, W.QComboBox) else w.value()
                  for k, w in self.forms[exp-1].items()}
        dem = Path(self.dem.text()).expanduser()
        if not dem.is_file() or not self.folder.text().strip():
            self.status.setText("Choose a DEM file and the results folder shared with Unreal.")
            return
        endpoint = self.server.text().strip()
        url = urlparse(endpoint)
        if url.scheme not in ("http", "https") or not url.hostname:
            self.status.setText("Enter the Unreal server URL, including http:// or https://.")
            return
        try:
            self.task = self.controller.start(
                task_name=f"Experiment {exp}", function=run_experiment, inject_context=True,
                args=(Settings(exp, dem, **values), Path(self.folder.text()).expanduser(), endpoint))
        except RuntimeError as exc:
            self.status.setText(str(exc))
            return
        for button in self.buttons:
            button.setEnabled(False)
        self.cancel.setEnabled(True)
        self.status.setText("Preparing experiment…")

    def progress(self, task, processed, total, unit, message):
        self.show_status(task, message)

    def show_status(self, task, message):
        if task == self.task:
            self.status.setText(str(message))

    def task_done(self, task):
        if task == self.task:
            self.task = None
            self.cancel.setEnabled(False)
            for button in self.buttons:
                button.setEnabled(True)
