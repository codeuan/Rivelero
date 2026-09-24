"""Terrain acquisition dialog for the Rivelero World workflow.

The dialog supports:

- local DEM/DTM GeoTIFF import;
- OpenTopography DEM acquisition for an area of interest, with an API key
  entered in the dialog (or OPENTOPO_API_KEY), reprojected to a projected
  CRS with an explicit cell size so the visibility engine can use it;
- inspection before installation into ApplicationState.

DSM is represented in the canonical Environment model but remains disabled
in the GUI until the current visibility workflow has been validated for it.
"""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

try:
    from PySide6.QtCore import QSettings, Qt, Signal
    from PySide6.QtWidgets import (
        QButtonGroup,
        QCheckBox,
        QComboBox,
        QDialog,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPushButton,
        QRadioButton,
        QScrollArea,
        QSpinBox,
        QStackedWidget,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    from PyQt6.QtCore import QSettings, Qt, pyqtSignal as Signal
    from PyQt6.QtWidgets import (
        QButtonGroup,
        QCheckBox,
        QComboBox,
        QDialog,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPushButton,
        QRadioButton,
        QScrollArea,
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
from rivelero.io.dem import wgs84_bbox_from_viewpoints
from rivelero.io.opentopography import (
    LARGE_GRID_CELLS,
    OPENTOPO_API_KEY_ENVIRONMENT_VARIABLE,
    OPENTOPO_API_KEY_URL,
    OPENTOPO_DATASETS,
    bbox_area_km2,
    bbox_size_m,
    check_opentopo_api_key,
    default_output_crs,
    download_projected_dem,
    estimated_grid_shape,
    native_cell_size_m,
    raw_download_path,
    validate_projected_output_crs,
)

# QSettings key of a remembered OpenTopography API key (opt-in).
API_KEY_SETTING = "opentopography/api_key"


def _default_settings() -> QSettings:
    return QSettings("Rivelero", "Rivelero")


def _set_status(label: QLabel, text: str, level: str | None) -> None:
    """Show ``text`` styled as success / warning / error (or plain)."""
    for name in ("statusSuccess", "statusWarning", "statusError"):
        label.setProperty(name, name == level)
    label.setText(text)
    label.style().unpolish(label)
    label.style().polish(label)


class EnvironmentImportDialog(QDialog):
    """Acquire and validate one canonical elevation Environment."""

    environment_ready = Signal(object)

    def __init__(
        self,
        *,
        viewpoints=(),
        task_controller: TaskController | None = None,
        settings: QSettings | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)

        # Per-user settings; only an opted-in API key is stored there.
        self._settings = settings if settings is not None else _default_settings()

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
        self.setMinimumSize(780, 650)
        self.resize(880, 820)
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
                "Load an existing elevation raster or download terrain "
                "from OpenTopography for your area of interest.",
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

        layout.addWidget(self._build_api_key_card())
        layout.addWidget(self._build_dataset_card())
        layout.addWidget(self._build_area_card())
        layout.addWidget(self._build_download_card())
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(page)

        self._on_dataset_changed()
        self._update_api_key_status()
        self._update_opentopo_summary()
        return scroll

    # API key -----------------------------------------------------------

    def _build_api_key_card(self) -> ContentCard:
        card = ContentCard(
            title="OpenTopography API key",
            description=(
                "OpenTopography requires a free API key. Request one at "
                f"{OPENTOPO_API_KEY_URL} (My OpenTopo › Request API key), "
                "then paste it here."
            ),
        )

        row = QHBoxLayout()
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.setPlaceholderText("e.g. 0123456789abcdef0123456789abcdef")
        self.api_key_edit.setToolTip(
            "Format: 32 hexadecimal characters (0–9, a–f), without spaces.\n"
            "Pasting OPENTOPO_API_KEY=<key> or API_Key=<key> also works."
        )
        self.show_key_button = QPushButton("Show")
        self.show_key_button.setCheckable(True)
        row.addWidget(self.api_key_edit, 1)
        row.addWidget(self.show_key_button)
        card.add_layout(row)

        format_note = QLabel(
            "Format: 32 hexadecimal characters (0–9, a–f), no spaces. The key "
            "is sent only to OpenTopography and is never written to projects, "
            "exports or reports. If you tick “Remember”, it is kept unencrypted "
            "in your Rivelero user settings."
        )
        format_note.setWordWrap(True)
        format_note.setProperty("secondaryText", True)
        card.add_widget(format_note)

        self.api_status = QLabel()
        self.api_status.setWordWrap(True)
        card.add_widget(self.api_status)

        self.remember_key_checkbox = QCheckBox("Remember this key on this computer")
        self.remember_key_checkbox.setToolTip(
            "Stored unencrypted in your Rivelero user settings. Leave unticked "
            "to use the key for this session only, or set OPENTOPO_API_KEY."
        )
        card.add_widget(self.remember_key_checkbox)

        # Prefill: a remembered key, else the OPENTOPO_API_KEY variable.
        self._key_source = None
        remembered = str(self._settings.value(API_KEY_SETTING, "") or "")
        environment = os.getenv(OPENTOPO_API_KEY_ENVIRONMENT_VARIABLE, "")
        if remembered:
            self.api_key_edit.setText(remembered)
            self.remember_key_checkbox.setChecked(True)
            self._key_source = "remembered"
        elif environment:
            self.api_key_edit.setText(environment)
            self._key_source = "environment"

        self.api_key_edit.textChanged.connect(self._on_api_key_edited)
        self.show_key_button.toggled.connect(self._toggle_key_visibility)
        return card

    def _toggle_key_visibility(self, visible: bool) -> None:
        self.api_key_edit.setEchoMode(
            QLineEdit.EchoMode.Normal if visible else QLineEdit.EchoMode.Password
        )
        self.show_key_button.setText("Hide" if visible else "Show")

    def _on_api_key_edited(self, _text: str) -> None:
        self._key_source = None
        self._update_api_key_status()
        self._update_download_enabled()

    def _api_key_check(self):
        return check_opentopo_api_key(self.api_key_edit.text())

    def _update_api_key_status(self) -> None:
        check = self._api_key_check()
        text = ("✓ " if check.status == "valid" else "") + check.message
        if check.usable and self._key_source == "environment":
            text += f" Read from the {OPENTOPO_API_KEY_ENVIRONMENT_VARIABLE} environment variable."
        elif check.usable and self._key_source == "remembered":
            text += " Remembered on this computer."
        _set_status(self.api_status, text, {
            "valid": "statusSuccess", "unusual": "statusWarning",
            "invalid": "statusError", "empty": "statusWarning",
        }[check.status])

    # Dataset and resolution ---------------------------------------------

    def _build_dataset_card(self) -> ContentCard:
        card = ContentCard(
            title="Dataset and resolution",
            description=(
                "OpenTopography delivers these datasets on a geographic "
                "(longitude/latitude) grid. Rivelero reprojects the download "
                "to a projected CRS in metres with square cells of the size "
                "below, which is the resolution the visibility analysis uses."
            ),
        )

        form = QFormLayout()

        self.ot_dataset = QComboBox()
        for dataset in OPENTOPO_DATASETS.values():
            self.ot_dataset.addItem(
                f"{dataset.label} — {dataset.native_resolution_text}", dataset.code
            )

        self.ot_dataset_info = QLabel()
        self.ot_dataset_info.setWordWrap(True)
        self.ot_dataset_info.setProperty("secondaryText", True)

        self.ot_crs = QLineEdit()
        self.ot_crs.setToolTip(
            "Projected CRS in metres for the terrain, e.g. EPSG:32633. "
            "Defaults to the survey's CRS, or the UTM zone of the area."
        )
        self._crs_is_default = True

        self.ot_resolution = QDoubleSpinBox()
        self.ot_resolution.setRange(1.0, 1000.0)
        self.ot_resolution.setDecimals(1)
        self.ot_resolution.setSingleStep(5.0)
        self.ot_resolution.setSuffix(" m")
        self.ot_resolution.setToolTip(
            "Side of each square terrain cell after reprojection. The native "
            "resolution of the dataset is the finest meaningful value."
        )

        self.ot_resampling = QComboBox()
        self.ot_resampling.addItem("Bilinear (recommended for elevation)", "bilinear")
        self.ot_resampling.addItem("Cubic", "cubic")
        self.ot_resampling.addItem("Nearest neighbour", "nearest")

        form.addRow("Dataset", self.ot_dataset)
        form.addRow("", self.ot_dataset_info)
        form.addRow("Terrain CRS", self.ot_crs)
        form.addRow("Terrain cell size", self.ot_resolution)
        form.addRow("Resampling", self.ot_resampling)
        card.add_layout(form)

        self.ot_resolution_summary = QLabel()
        self.ot_resolution_summary.setWordWrap(True)
        card.add_widget(self.ot_resolution_summary)

        self.ot_dataset.currentIndexChanged.connect(self._on_dataset_changed)
        self.ot_crs.textEdited.connect(self._on_crs_edited)
        self.ot_resolution.valueChanged.connect(lambda _value: self._update_opentopo_summary())
        return card

    def _dataset(self):
        return OPENTOPO_DATASETS[str(self.ot_dataset.currentData())]

    def _on_dataset_changed(self, _index: int = 0) -> None:
        # The native resolution is the default cell size of each dataset.
        self.ot_resolution.blockSignals(True)
        self.ot_resolution.setValue(self._dataset().nominal_resolution_m)
        self.ot_resolution.blockSignals(False)
        self._update_opentopo_summary()

    def _on_crs_edited(self, _text: str) -> None:
        self._crs_is_default = not self.ot_crs.text().strip()
        self._update_opentopo_summary()

    # Area of interest ---------------------------------------------------

    def _build_area_card(self) -> ContentCard:
        card = ContentCard(
            title="Area of interest",
            description=(
                "The area to download, as a longitude/latitude box. Survey "
                "coordinates are explicitly transformed to WGS84."
            ),
        )

        self.ot_area_survey = QRadioButton("Survey extent + buffer")
        self.ot_area_custom = QRadioButton("Custom bounding box (WGS84 degrees)")
        group = QButtonGroup(card)
        group.addButton(self.ot_area_survey)
        group.addButton(self.ot_area_custom)

        self.ot_buffer = QSpinBox()
        self.ot_buffer.setRange(0, 100000)
        self.ot_buffer.setValue(500)
        self.ot_buffer.setSingleStep(100)
        self.ot_buffer.setSuffix(" m")
        self.ot_buffer.setToolTip(
            "Distance added around the survey. Use at least the maximum "
            "visibility distance so every sight line stays on the terrain."
        )

        buffer_row = QHBoxLayout()
        buffer_row.addWidget(self.ot_area_survey)
        buffer_row.addWidget(self.ot_buffer)
        buffer_row.addStretch(1)
        card.add_layout(buffer_row)
        card.add_widget(self.ot_area_custom)

        self.ot_bounds = {}
        bounds_form = QFormLayout()
        for name, low, high in (
            ("south", -90.0, 90.0), ("north", -90.0, 90.0),
            ("west", -180.0, 180.0), ("east", -180.0, 180.0),
        ):
            spin = QDoubleSpinBox()
            spin.setRange(low, high)
            spin.setDecimals(6)
            spin.setSingleStep(0.01)
            spin.setSuffix(" °")
            spin.valueChanged.connect(lambda _value: self._update_opentopo_summary())
            self.ot_bounds[name] = spin
            bounds_form.addRow(name.capitalize(), spin)
        self._bounds_widget = QWidget()
        self._bounds_widget.setLayout(bounds_form)
        card.add_widget(self._bounds_widget)

        self.ot_area_summary = QLabel()
        self.ot_area_summary.setWordWrap(True)
        card.add_widget(self.ot_area_summary)

        survey_bbox = self._survey_bbox()
        if survey_bbox is not None:
            self._set_custom_bounds(survey_bbox)
            self.ot_area_survey.setChecked(True)
        else:
            self.ot_area_survey.setEnabled(False)
            self.ot_area_survey.setToolTip("Import a survey to derive the area from it.")
            self.ot_area_custom.setChecked(True)

        self.ot_area_survey.toggled.connect(self._on_area_mode_changed)
        self.ot_buffer.valueChanged.connect(lambda _value: self._update_opentopo_summary())
        self._on_area_mode_changed(self.ot_area_survey.isChecked())
        return card

    def _on_area_mode_changed(self, survey_mode: bool) -> None:
        if not survey_mode:
            # Start the custom box from the current survey extent + buffer.
            bbox = self._survey_bbox(float(self.ot_buffer.value()))
            if bbox is not None:
                self._set_custom_bounds(bbox)
        self.ot_buffer.setEnabled(survey_mode)
        self._bounds_widget.setEnabled(not survey_mode)
        self._update_opentopo_summary()

    def _survey_bbox(self, buffer_m: float = 0.0):
        if not self._viewpoints:
            return None
        try:
            return wgs84_bbox_from_viewpoints(self._viewpoints, buffer_m=buffer_m)
        except Exception:  # noqa: BLE001 - reported as unusable survey extent
            return None

    def _set_custom_bounds(self, bbox) -> None:
        for name, value in zip(("south", "north", "west", "east"), bbox):
            self.ot_bounds[name].blockSignals(True)
            self.ot_bounds[name].setValue(value)
            self.ot_bounds[name].blockSignals(False)

    def _area_bbox(self):
        """(south, north, west, east) of the area of interest, or an error text."""
        if self.ot_area_survey.isChecked():
            bbox = self._survey_bbox(float(self.ot_buffer.value()))
            if bbox is None:
                return None, "The survey extent cannot be transformed to WGS84."
            return bbox, None
        bbox = tuple(self.ot_bounds[name].value() for name in ("south", "north", "west", "east"))
        south, north, west, east = bbox
        if not (south < north and west < east):
            return None, "South must be less than North and West less than East."
        return bbox, None

    # Download -----------------------------------------------------------

    def _build_download_card(self) -> ContentCard:
        card = ContentCard(title="Download")

        form = QFormLayout()
        self.ot_name = QLineEdit("OpenTopography terrain")
        self.ot_output = QLineEdit()
        self.ot_output.setPlaceholderText("Choose output GeoTIFF…")
        output_button = QPushButton("Browse")
        output_row = QWidget()
        output_layout = QHBoxLayout(output_row)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.addWidget(self.ot_output, 1)
        output_layout.addWidget(output_button)
        form.addRow("Environment name", self.ot_name)
        form.addRow("Save terrain as", output_row)
        card.add_layout(form)

        self.ot_output_note = QLabel()
        self.ot_output_note.setWordWrap(True)
        self.ot_output_note.setProperty("secondaryText", True)
        card.add_widget(self.ot_output_note)

        self.download_button = make_primary_button("Download and reproject DEM")
        card.add_widget(self.download_button)

        output_button.clicked.connect(self._choose_download_output)
        self.ot_output.textChanged.connect(lambda _text: self._update_opentopo_summary())
        self.download_button.clicked.connect(self._download_opentopo)
        return card

    def _choose_download_output(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save terrain",
            f"rivelero_{self._dataset().code.lower()}.tif",
            "GeoTIFF (*.tif)",
        )
        if filename:
            if not filename.lower().endswith((".tif", ".tiff")):
                filename += ".tif"
            self.ot_output.setText(filename)

    # Summary ------------------------------------------------------------

    def _output_crs(self):
        """Projected output CRS, or an error text."""
        bbox, _error = self._area_bbox()
        if self._crs_is_default:
            if bbox is None:
                return None, None
            crs = default_output_crs(self._viewpoints, bbox)
            self.ot_crs.blockSignals(True)
            self.ot_crs.setText(crs.to_string())
            self.ot_crs.blockSignals(False)
            return crs, None
        try:
            return validate_projected_output_crs(self.ot_crs.text().strip()), None
        except ValueError as exc:
            return None, str(exc)

    def _update_opentopo_summary(self) -> None:
        if not hasattr(self, "ot_output_note"):
            return  # still building the tab
        dataset = self._dataset()
        bbox, area_error = self._area_bbox()
        crs, crs_error = self._output_crs()
        resolution = float(self.ot_resolution.value())
        self._opentopo_problem = area_error or crs_error

        latitude = 0.0 if bbox is None else (bbox[0] + bbox[1]) / 2.0
        east_west, north_south = native_cell_size_m(dataset, latitude)
        self.ot_dataset_info.setText(
            f"{dataset.surface.capitalize()}. Coverage: {dataset.coverage}.\n"
            f"Native grid: {dataset.native_arcsec:g}″ × {dataset.native_arcsec:g}″ "
            f"= {north_south:.1f} m north–south × {east_west:.1f} m east–west"
            f" at {latitude:.2f}° latitude."
        )

        if bbox is None:
            _set_status(self.ot_area_summary, area_error or "", "statusError")
        else:
            south, north, west, east = bbox
            width_m, height_m = bbox_size_m(bbox)
            area = bbox_area_km2(bbox)
            text = (
                f"S {south:.5f}° · N {north:.5f}° · W {west:.5f}° · E {east:.5f}°\n"
                f"≈ {width_m / 1000:.2f} km × {height_m / 1000:.2f} km = {area:,.2f} km²"
            )
            if area > dataset.max_area_km2:
                self._opentopo_problem = (
                    f"The area exceeds OpenTopography's limit of "
                    f"{dataset.max_area_km2:,.0f} km² per {dataset.label} request."
                )
                _set_status(self.ot_area_summary, f"{text}\n{self._opentopo_problem}", "statusError")
            else:
                _set_status(self.ot_area_summary, text, None)

        if crs is None or bbox is None:
            _set_status(self.ot_resolution_summary, crs_error or "", "statusError" if crs_error else None)
        else:
            rows, columns = estimated_grid_shape(bbox, resolution)
            cells = rows * columns
            lines = [
                f"Terrain for visibility: {resolution:g} m × {resolution:g} m cells in "
                f"{crs.to_string()} ({crs.name}), ≈ {columns:,} × {rows:,} cells "
                f"({cells / 1e6:.2f} million).",
                f"Source: {dataset.label}, native {dataset.native_resolution_text}.",
            ]
            level = None
            if resolution < min(east_west, north_south) * 0.95:
                lines.append(
                    f"The cell size is finer than the dataset's native resolution "
                    f"(≈ {min(east_west, north_south):.0f} m here): it adds cells, not detail."
                )
                level = "statusWarning"
            if cells > LARGE_GRID_CELLS:
                lines.append(
                    "This is a very large grid: visibility computation will be slow. "
                    "Consider a larger cell size or a smaller area."
                )
                level = "statusWarning"
            _set_status(self.ot_resolution_summary, "\n".join(lines), level)

        output = self.ot_output.text().strip()
        if output:
            self.ot_output_note.setText(
                f"The projected terrain is saved as {Path(output).name}; the original "
                f"download is kept as {raw_download_path(output, dataset.code).name}."
            )
        else:
            self.ot_output_note.setText(
                "The projected terrain and the original download "
                "(<name>_<dataset>_wgs84.tif) are saved in the same folder."
            )
        self._update_download_enabled()

    def _update_download_enabled(self) -> None:
        if not hasattr(self, "download_button"):
            return
        ready = (
            self._api_key_check().usable
            and not getattr(self, "_opentopo_problem", None)
            and self._active_task_id is None
        )
        self.download_button.setEnabled(ready)

    # Download task --------------------------------------------------------

    def _download_opentopo(self) -> None:
        check = self._api_key_check()
        if not check.usable:
            QMessageBox.warning(self, "API key required", check.message)
            return
        bbox, error = self._area_bbox()
        crs, crs_error = self._output_crs()
        if bbox is None or crs is None:
            QMessageBox.warning(self, "Check the settings", error or crs_error or "")
            return

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

        # Remember (or forget) the key only when it is used.
        if self.remember_key_checkbox.isChecked():
            self._settings.setValue(API_KEY_SETTING, check.key)
        else:
            self._settings.remove(API_KEY_SETTING)

        dataset = self._dataset()
        resolution = float(self.ot_resolution.value())
        self.accept_button.setEnabled(False)
        message = (
            f"Downloading {dataset.label} from OpenTopography and reprojecting "
            f"to {resolution:g} m cells in {crs.to_string()}…"
        )
        self.status_label.setText(message)
        self._active_task_id = self._task_controller.start(
            task_name="Download terrain",
            function=self._download_and_import,
            kwargs={
                "bbox": tuple(bbox),
                "dataset": dataset.code,
                "api_key": check.key,
                "output_path": output,
                "crs": crs.to_string(),
                "resolution_m": resolution,
                "resampling": str(self.ot_resampling.currentData()),
                "environment_name": self.ot_name.text().strip() or "OpenTopography terrain",
                "survey_buffer_m": (
                    float(self.ot_buffer.value()) if self.ot_area_survey.isChecked() else None
                ),
            },
            message=message,
        )
        self._update_download_enabled()

    @staticmethod
    def _download_and_import(
        *,
        bbox,
        dataset: str,
        api_key: str,
        output_path: str,
        crs: str,
        resolution_m: float,
        resampling: str,
        environment_name: str,
        survey_buffer_m: float | None,
    ) -> EnvironmentImportResult:
        download = download_projected_dem(
            bbox,
            dataset=dataset,
            api_key=api_key,
            output_path=output_path,
            crs=crs,
            resolution_m=resolution_m,
            resampling=resampling,
        )
        provenance = download.provenance()
        provenance["area_of_interest"] = (
            "survey_extent_plus_buffer" if survey_buffer_m is not None else "custom_bbox"
        )
        if survey_buffer_m is not None:
            provenance["survey_buffer_m"] = survey_buffer_m
        return import_elevation_environment(
            download.output_path,
            environment_id=uuid4().hex,
            name=environment_name,
            model_type=ElevationModelType(download.dataset.model_type),
            source_name=f"OpenTopography {download.dataset.label}",
            description=(
                f"{download.dataset.label} ({download.dataset.native_resolution_text} "
                f"native) reprojected to {download.resolution_m:g} m cells in "
                f"{download.output_crs}."
            ),
            provenance=provenance,
        )

    def _on_task_result(self, task_id: str, result: object) -> None:
        if task_id != self._active_task_id:
            return
        self._active_task_id = None
        if not isinstance(result, EnvironmentImportResult):
            self._finish_download_error("The terrain task returned an invalid result.")
            return
        self._result = result
        self._downloaded_path = Path(result.environment.elevation_model.source)
        provenance = result.environment.elevation_model.provenance or {}
        grid = result.grid
        self.status_label.setText(
            "Terrain downloaded, reprojected and validated: "
            f"{provenance.get('dataset_name', 'OpenTopography')} "
            f"(native {provenance.get('native_resolution', '?')}) → "
            f"{provenance.get('output_resolution_m', '?'):g} m cells in "
            f"{provenance.get('output_crs', '?')}, "
            f"{grid.width:,} × {grid.height:,} cells."
        )
        self.accept_button.setEnabled(True)
        self._update_download_enabled()

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
        self.accept_button.setEnabled(False)
        self._update_download_enabled()

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