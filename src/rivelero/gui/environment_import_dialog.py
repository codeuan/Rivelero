"""Terrain acquisition dialog for the Rivelero World workflow.

The dialog supports:

- local DEM/DTM GeoTIFF import;
- OpenTopography DEM acquisition around the current survey;
- inspection before installation into ApplicationState.

DSM is represented in the canonical Environment model but remains disabled
in the GUI until the current visibility workflow has been validated for it.
"""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

try:
    from PySide6.QtCore import Signal
    from PySide6.QtWidgets import (
        QComboBox,
        QDialog,
        QFileDialog,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPushButton,
        QSpinBox,
        QStackedWidget,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    from PyQt6.QtCore import pyqtSignal as Signal
    from PyQt6.QtWidgets import (
        QComboBox,
        QDialog,
        QFileDialog,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPushButton,
        QSpinBox,
        QStackedWidget,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )

from rivelero.core.environment import ElevationModelType
from rivelero.gui.components import (
    ContentCard,
    LabeledValue,
    PageHeader,
    StatusBadge,
    BadgeType,
    make_primary_button,
    make_secondary_button,
)
from rivelero.gui.environment_import import (
    EnvironmentImportResult,
    import_elevation_environment,
    inspect_elevation_raster,
)
from rivelero.gui.theme import SPACING
from rivelero.gui.task_controller import TaskController
from rivelero.io.dem import (
    download_dem_for_viewpoints,
)


class EnvironmentImportDialog(QDialog):
    """Acquire and validate one canonical elevation Environment."""

    environment_ready = Signal(object)

    def __init__(
        self,
        *,
        viewpoints=(),
        task_controller: TaskController | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)

        self._viewpoints = list(viewpoints)
        self._result: EnvironmentImportResult | None = None
        self._downloaded_path: Path | None = None
        self._task_controller = task_controller
        self._active_task_id: str | None = None

        if self._task_controller is not None:
            self._task_controller.task_result.connect(self._on_task_result)
            self._task_controller.task_error.connect(self._on_task_error)
            self._task_controller.task_cancelled.connect(self._on_task_cancelled)

        self.setWindowTitle("Add terrain — Rivelero")
        self.setMinimumSize(760, 650)
        self.resize(850, 720)
        self.setModal(True)

        self._build_interface()

    @property
    def environment_result(self) -> EnvironmentImportResult | None:
        return self._result

    def _build_interface(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(
            SPACING.xl,
            SPACING.xl,
            SPACING.xl,
            SPACING.xl,
        )
        root.setSpacing(SPACING.lg)

        root.addWidget(
            PageHeader(
                "Add terrain",
                "Load an existing elevation raster or acquire terrain "
                "from OpenTopography.",
            )
        )

        self.tabs = QTabWidget()
        self.tabs.addTab(
            self._build_local_tab(),
            "Local DEM",
        )
        self.tabs.addTab(
            self._build_opentopo_tab(),
            "OpenTopography",
        )
        self.tabs.addTab(
            self._build_dsm_tab(),
            "DSM",
        )

        root.addWidget(self.tabs, 1)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setProperty("secondaryText", True)
        root.addWidget(self.status_label)

        actions = QHBoxLayout()
        actions.addStretch(1)

        cancel = QPushButton("Cancel")
        self.accept_button = make_primary_button("Use terrain")
        self.accept_button.setEnabled(False)

        actions.addWidget(cancel)
        actions.addWidget(self.accept_button)

        root.addLayout(actions)

        cancel.clicked.connect(self.reject)
        self.accept_button.clicked.connect(self._accept_result)

    # ------------------------------------------------------------------
    # Local raster
    # ------------------------------------------------------------------

    def _build_local_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(SPACING.lg)

        card = ContentCard(
            title="Elevation raster",
            description=(
                "Select a georeferenced DEM or DTM. Rivelero reads "
                "metadata first and does not load the complete raster "
                "into memory."
            ),
        )

        path_row = QHBoxLayout()

        self.local_path = QLineEdit()
        self.local_path.setPlaceholderText("Select GeoTIFF…")

        browse = QPushButton("Browse")

        path_row.addWidget(self.local_path, 1)
        path_row.addWidget(browse)

        card.add_layout(path_row)

        form = QFormLayout()

        self.local_type = QComboBox()
        self.local_type.addItem("DEM", ElevationModelType.DEM)
        self.local_type.addItem("DTM", ElevationModelType.DTM)

        self.local_name = QLineEdit("Imported terrain")

        form.addRow("Model type", self.local_type)
        form.addRow("Environment name", self.local_name)

        card.add_layout(form)

        self.local_metadata = QLabel(
            "Select a raster to inspect its spatial metadata."
        )
        self.local_metadata.setWordWrap(True)
        self.local_metadata.setProperty("secondaryText", True)

        card.add_widget(self.local_metadata)

        layout.addWidget(card)
        layout.addStretch(1)

        browse.clicked.connect(self._browse_local)
        self.local_path.textChanged.connect(self._inspect_local)

        return page

    def _browse_local(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select elevation raster",
            "",
            "GeoTIFF (*.tif *.tiff);;Raster files (*)",
        )

        if filename:
            self.local_path.setText(filename)

    def _inspect_local(self) -> None:
        self._result = None
        self.accept_button.setEnabled(False)

        text = self.local_path.text().strip()

        if not text:
            self.local_metadata.setText(
                "Select a raster to inspect its spatial metadata."
            )
            return

        try:
            metadata = inspect_elevation_raster(text)

        except Exception as exc:
            self.local_metadata.setText(
                f"Unable to inspect raster: {exc}"
            )
            return

        resolution = (
            f"{metadata.resolution_x:g} × "
            f"{metadata.resolution_y:g}"
        )

        if metadata.linear_units:
            resolution += f" {metadata.linear_units}"

        self.local_metadata.setText(
            f"CRS: {metadata.crs.to_string()}\n"
            f"Size: {metadata.width:,} × {metadata.height:,}\n"
            f"Resolution: {resolution}\n"
            f"NoData: {metadata.nodata_value}\n"
            f"Driver: {metadata.driver}"
        )

        self.accept_button.setEnabled(True)

    # ------------------------------------------------------------------
    # OpenTopography
    # ------------------------------------------------------------------

    def _build_opentopo_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(SPACING.lg)

        card = ContentCard(
            title="Download elevation",
            description=(
                "Download a global DEM covering the current survey. "
                "Survey coordinates are explicitly transformed to WGS84 "
                "for the API request."
            ),
        )

        form = QFormLayout()

        self.ot_dataset = QComboBox()
        self.ot_dataset.addItem(
            "Copernicus 30 m",
            "COP30",
        )
        self.ot_dataset.addItem(
            "SRTM GL1",
            "SRTMGL1",
        )

        self.ot_buffer = QSpinBox()
        self.ot_buffer.setRange(0, 100000)
        self.ot_buffer.setValue(500)
        self.ot_buffer.setSuffix(" m")

        self.ot_name = QLineEdit(
            "OpenTopography terrain"
        )

        self.ot_output = QLineEdit()
        self.ot_output.setPlaceholderText(
            "Choose output GeoTIFF…"
        )

        output_button = QPushButton("Browse")

        output_row = QWidget()
        output_layout = QHBoxLayout(output_row)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.addWidget(self.ot_output, 1)
        output_layout.addWidget(output_button)

        form.addRow("Dataset", self.ot_dataset)
        form.addRow("Survey buffer", self.ot_buffer)
        form.addRow("Environment name", self.ot_name)
        form.addRow("Save as", output_row)

        card.add_layout(form)

        key_available = bool(
            os.getenv("OPENTOPO_API_KEY")
        )

        self.api_status = QLabel(
            (
                "✓ OPENTOPO_API_KEY detected"
                if key_available
                else "OPENTOPO_API_KEY is not configured."
            )
        )

        self.api_status.setProperty(
            "statusSuccess" if key_available else "statusWarning",
            True,
        )

        card.add_widget(self.api_status)

        if not self._viewpoints:
            no_survey = QLabel(
                "A survey is required to derive the download extent."
            )
            no_survey.setProperty("statusWarning", True)
            card.add_widget(no_survey)

        self.download_button = make_primary_button(
            "Download DEM"
        )

        self.download_button.setEnabled(
            key_available
            and bool(self._viewpoints)
        )

        card.add_widget(self.download_button)

        layout.addWidget(card)
        layout.addStretch(1)

        output_button.clicked.connect(
            self._choose_download_output
        )
        self.download_button.clicked.connect(
            self._download_opentopo
        )

        return page

    def _choose_download_output(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save downloaded DEM",
            "rivelero_dem.tif",
            "GeoTIFF (*.tif)",
        )

        if filename:
            self.ot_output.setText(filename)

    def _download_opentopo(self) -> None:
        output = self.ot_output.text().strip()

        if not output:
            self._choose_download_output()
            output = self.ot_output.text().strip()

        if not output:
            return

        if self._task_controller is None:
            QMessageBox.warning(
                self,
                "Download unavailable",
                "OpenTopography downloads require the shared task controller.",
            )
            return

        self.download_button.setEnabled(False)
        self.accept_button.setEnabled(False)
        self.status_label.setText(
            "Downloading terrain from OpenTopography…"
        )
        self._active_task_id = self._task_controller.start(
            task_name="Download terrain",
            function=self._download_and_import,
            kwargs={
                "viewpoints": tuple(self._viewpoints),
                "buffer_m": float(self.ot_buffer.value()),
                "demtype": str(self.ot_dataset.currentData()),
                "output_path": output,
                "environment_name": self.ot_name.text().strip()
                or "OpenTopography terrain",
                "dataset": str(self.ot_dataset.currentData()),
            },
            message="Downloading terrain from OpenTopography…",
        )

    @staticmethod
    def _download_and_import(
        *,
        viewpoints,
        buffer_m: float,
        demtype: str,
        output_path: str,
        environment_name: str,
        dataset: str,
    ) -> EnvironmentImportResult:
        path = download_dem_for_viewpoints(
            viewpoints,
            buffer_m=buffer_m,
            demtype=demtype,
            output_path=output_path,
        )
        return import_elevation_environment(
            path,
            environment_id=uuid4().hex,
            name=environment_name,
            model_type=ElevationModelType.DEM,
            source_name="OpenTopography",
            provenance={
                "acquisition": "OpenTopography",
                "dataset": dataset,
                "survey_buffer_m": buffer_m,
            },
        )

    def _on_task_result(self, task_id: str, result: object) -> None:
        if task_id != self._active_task_id:
            return
        if not isinstance(result, EnvironmentImportResult):
            self._finish_download_error("The terrain task returned an invalid result.")
            return
        self._result = result
        self._downloaded_path = Path(result.environment.elevation_model.source)
        self.status_label.setText("Terrain downloaded and validated.")
        self.accept_button.setEnabled(True)
        self.download_button.setEnabled(True)

    def _on_task_error(self, task_id: str, message: str, _traceback: str) -> None:
        if task_id != self._active_task_id:
            return
        self._finish_download_error(message)

    def _on_task_cancelled(self, task_id: str, message: object) -> None:
        if task_id != self._active_task_id:
            return
        self._finish_download_error(str(message or "Terrain download cancelled."))

    def _finish_download_error(self, message: str) -> None:
        self._active_task_id = None
        self._result = None
        self.status_label.setText(f"Terrain download failed: {message}")
        self.download_button.setEnabled(True)
        self.accept_button.setEnabled(False)

    # ------------------------------------------------------------------
    # DSM future placeholder
    # ------------------------------------------------------------------

    def _build_dsm_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        card = ContentCard(
            title="Digital Surface Model",
            description=(
                "The canonical Rivelero Environment already supports "
                "DSM sources. Integration with the current visibility "
                "workflow will be validated in a future implementation."
            ),
            badge=BadgeType.COMING_SOON,
            subtle=True,
        )

        layout.addWidget(card)
        layout.addStretch(1)

        return page

    # ------------------------------------------------------------------
    # Accept
    # ------------------------------------------------------------------

    def _accept_result(self) -> None:
        if self.tabs.currentIndex() == 0:
            path = self.local_path.text().strip()

            if not path:
                return

            try:
                self._result = import_elevation_environment(
                    path,
                    environment_id=uuid4().hex,
                    name=(
                        self.local_name.text().strip()
                        or "Imported terrain"
                    ),
                    model_type=self.local_type.currentData(),
                )

            except Exception as exc:
                QMessageBox.critical(
                    self,
                    "Unable to import terrain",
                    str(exc),
                )
                return

        elif self.tabs.currentIndex() == 1:
            if self._result is None:
                QMessageBox.information(
                    self,
                    "Download terrain first",
                    "Download and validate the DEM before using it.",
                )
                return

        else:
            return

        self.environment_ready.emit(self._result)
        self.accept()