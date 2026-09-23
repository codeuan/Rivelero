"""Output page: data export (P2), figures and report (P3).

Three tabs share one page:

* **Data** lists every machine-readable product grouped by origin;
* **Figures** exports standalone publication figures;
* **Report** generates the HTML report with its provenance manifest.

Products that cannot be produced now are disabled with the reason. Every
export runs on the shared TaskController, on an ExportContext frozen when it
starts, so the GUI stays responsive and later edits cannot change files that
are already being written.

Exporting never modifies ApplicationState, so it never marks the project
as modified.
"""

from __future__ import annotations

from pathlib import Path

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QCheckBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
        QListWidget, QProgressBar, QScrollArea, QTabWidget, QVBoxLayout, QWidget,
    )
except ImportError:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QCheckBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
        QListWidget, QProgressBar, QScrollArea, QTabWidget, QVBoxLayout, QWidget,
    )

from rivelero.export.catalog import (
    GROUPS,
    ExportProduct,
    export_catalog,
    run_export,
)
from rivelero.gui.application_state import ApplicationState
from rivelero.gui.components import (
    ContentCard, PageHeader, make_primary_button, make_secondary_button,
)
from rivelero.gui.export_service import export_context_from_state
from rivelero.gui.output_panels import FiguresPanel, ReportPanel
from rivelero.gui.task_controller import TaskController, make_progress_task
from rivelero.gui.theme import SPACING, refresh_style


EXPORT_TASK_NAME = "Exporting data"

# Checked by default the first time they become available.
DEFAULT_PRODUCTS = frozenset({"exposure_count", "observability_state", "blindspot_mask"})

GROUP_DESCRIPTIONS = {
    "Survey": "Canonical Survey tables (CSV) using the importer's column names.",
    "Observability": (
        "GeoTIFFs on the exact grid of the current Survey Observability Field. "
        "Blind spots are 0, never NoData; categorical rasters keep every code."
    ),
    "Analysis": "Descriptive coverage classes and per-unit contributions.",
    "Scenarios and comparison": (
        "The current what-if scenario and the comparison selected in "
        "Analysis › Compare. Comparison differences are right − left."
    ),
}


class OutputPage(QWidget):
    """Choose products and a folder; export on the TaskController."""

    def __init__(
        self,
        state: ApplicationState,
        *,
        task_controller: TaskController,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.state = state
        self.task_controller = task_controller
        self._task_id: str | None = None
        self._products: dict[str, ExportProduct] = {}
        self._seen: set[str] = set()
        self.last_result = None

        self._build_interface()
        controller = self.task_controller
        controller.task_progress.connect(self._on_task_progress)
        controller.task_result.connect(self._on_task_result)
        controller.task_error.connect(self._on_task_error)
        controller.task_cancelled.connect(self._on_task_cancelled)
        controller.task_finished.connect(self._on_task_finished)
        controller.task_started.connect(lambda _id, _name: self._refresh_controls())
        self.refresh_from_state()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._task_id is not None

    def selected_keys(self) -> list[str]:
        return [
            key for key, box in self.checkboxes.items()
            if box.isEnabled() and box.isChecked()
        ]

    def set_selection(self, keys) -> None:
        keys = set(keys)
        for key, box in self.checkboxes.items():
            box.setChecked(key in keys and box.isEnabled())

    def choose_directory(self) -> str | None:
        """Dialog hook (replaced in tests)."""
        path = QFileDialog.getExistingDirectory(
            self, "Export folder", self.directory_edit.text() or str(Path.home())
        )
        return path or None

    def refresh_from_state(self) -> None:
        context = export_context_from_state(self.state)
        self.figures_panel.refresh(context)
        self.report_panel.refresh(context)
        if self.running:
            self._refresh_controls()
            return
        catalog = export_catalog(context)
        self._products = {product.key: product for product in catalog}
        for product in catalog:
            box = self.checkboxes[product.key]
            reason = self.reason_labels[product.key]
            box.setEnabled(product.available)
            box.setToolTip(product.filename if product.available else product.reason or "")
            if not product.available:
                box.setChecked(False)
                self._seen.discard(product.key)
            elif product.key not in self._seen:
                self._seen.add(product.key)
                box.setChecked(product.key in DEFAULT_PRODUCTS)
            reason.setText("" if product.available else product.reason or "")
            reason.setVisible(not product.available)
        self._refresh_controls()

    def start_export(self) -> str | None:
        """Start exporting the selected products; returns the task id."""
        if self.running or self.state.busy:
            return None
        directory = self.directory_edit.text().strip()
        keys = self.selected_keys()
        if not directory:
            self._set_status("Choose an export folder first.", kind="warning")
            return None
        if not keys:
            self._set_status("Select at least one product.", kind="warning")
            return None

        # Freeze the current state: the export writes exactly these results.
        context = export_context_from_state(self.state)
        task = make_progress_task(
            function=run_export,
            kwargs={
                "context": context,
                "keys": keys,
                "directory": directory,
                "overwrite": self.overwrite_box.isChecked(),
            },
        )
        self.last_result = None
        self.results_list.clear()
        self._task_id = self.task_controller.start(
            task_name=EXPORT_TASK_NAME,
            function=task,
            total=len(keys),
            message=f"Exporting {len(keys)} product(s)",
            inject_context=True,
        )
        self.progress_bar.setRange(0, len(keys))
        self.progress_bar.setValue(0)
        self._set_status(None)
        self._refresh_controls()
        return self._task_id

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def _build_interface(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(SPACING.page, SPACING.xxxl, SPACING.page, SPACING.xxxl)
        layout.setSpacing(SPACING.xl)

        layout.addWidget(PageHeader(
            "Output",
            "Export machine-readable data (GeoTIFF, CSV), standalone figures and a "
            "scientific report with its provenance record. Save the project "
            "(File › Save) to keep the full analysis.",
        ))

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        layout.addWidget(self.tabs)
        page_layout = layout
        data_tab = QWidget()
        layout = QVBoxLayout(data_tab)
        layout.setContentsMargins(0, SPACING.lg, 0, 0)
        layout.setSpacing(SPACING.xl)
        note = QLabel(
            "GeoTIFF rasters and CSV tables, each with a JSON metadata sidecar, for GIS, "
            "Python/R and archiving."
        )
        note.setWordWrap(True)
        note.setProperty("secondaryText", True)
        layout.addWidget(note)

        self.checkboxes: dict[str, QCheckBox] = {}
        self.reason_labels: dict[str, QLabel] = {}
        catalog = export_catalog(export_context_from_state(self.state))
        grid = QGridLayout()
        grid.setHorizontalSpacing(SPACING.xl)
        grid.setVerticalSpacing(SPACING.xl)
        for index, group in enumerate(GROUPS):
            card = ContentCard(title=group, description=GROUP_DESCRIPTIONS.get(group))
            for product in (p for p in catalog if p.group == group):
                box = QCheckBox(product.label)
                box.setObjectName(f"export_{product.key}")
                reason = QLabel()
                reason.setWordWrap(True)
                reason.setProperty("secondaryText", True)
                reason.setContentsMargins(24, 0, 0, 0)
                reason.setVisible(False)
                self.checkboxes[product.key] = box
                self.reason_labels[product.key] = reason
                card.add_widget(box)
                card.add_widget(reason)
            card.add_stretch()
            grid.addWidget(card, index // 2, index % 2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)

        selection = QHBoxLayout()
        self.select_all_button = make_secondary_button("Select all available")
        self.select_none_button = make_secondary_button("Clear selection")
        selection.addWidget(self.select_all_button)
        selection.addWidget(self.select_none_button)
        selection.addStretch(1)
        layout.addLayout(selection)

        destination = ContentCard(
            title="Export",
            description=(
                "Files are named <project>_<product> and each is written "
                "atomically. Existing files are never replaced unless allowed; "
                "files not being exported are never touched."
            ),
        )
        folder_row = QHBoxLayout()
        self.directory_edit = QLineEdit()
        self.directory_edit.setPlaceholderText("Export folder")
        self.browse_button = make_secondary_button("Choose folder…")
        folder_row.addWidget(self.directory_edit, 1)
        folder_row.addWidget(self.browse_button)
        destination.add_layout(folder_row)
        self.overwrite_box = QCheckBox("Replace existing files with the same names")
        destination.add_widget(self.overwrite_box)

        buttons = QHBoxLayout()
        self.export_button = make_primary_button("Export selected")
        self.cancel_button = make_secondary_button("Cancel")
        self.cancel_button.setVisible(False)
        buttons.addWidget(self.export_button)
        buttons.addWidget(self.cancel_button)
        buttons.addStretch(1)
        destination.add_layout(buttons)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        destination.add_widget(self.progress_bar)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.status_label.setVisible(False)
        destination.add_widget(self.status_label)
        self.results_list = QListWidget()
        self.results_list.setMinimumHeight(140)
        self.results_list.setVisible(False)
        destination.add_widget(self.results_list)
        layout.addWidget(destination)
        layout.addStretch(1)
        self.tabs.addTab(data_tab, "Data")
        self.figures_panel = FiguresPanel(self.state, self.task_controller)
        self.tabs.addTab(self.figures_panel, "Figures")
        self.report_panel = ReportPanel(self.state, self.task_controller)
        self.tabs.addTab(self.report_panel, "Report")
        page_layout.addStretch(1)

        scroll.setWidget(content)
        root.addWidget(scroll)

        self.select_all_button.clicked.connect(
            lambda: self.set_selection(self.checkboxes)
        )
        self.select_none_button.clicked.connect(lambda: self.set_selection(()))
        self.browse_button.clicked.connect(self._browse)
        self.export_button.clicked.connect(self.start_export)
        self.cancel_button.clicked.connect(self._cancel)
        self.directory_edit.textChanged.connect(lambda _text: self._refresh_controls())
        for box in self.checkboxes.values():
            box.toggled.connect(lambda _checked: self._refresh_controls())

    def _browse(self) -> None:
        path = self.choose_directory()
        if path:
            self.directory_edit.setText(path)

    def _cancel(self) -> None:
        if self._task_id is not None and self.task_controller.cancel():
            self.cancel_button.setEnabled(False)
            self._set_status(
                "Cancellation requested; stopping after the current file. "
                "Files already written are kept.",
                kind="warning",
            )

    # ------------------------------------------------------------------
    # Task signals
    # ------------------------------------------------------------------

    def _on_task_progress(self, task_id, processed, total, unit_id, _message) -> None:
        if task_id != self._task_id:
            return
        if total:
            self.progress_bar.setRange(0, int(total))
        self.progress_bar.setValue(int(processed))

    def _on_task_result(self, task_id, result) -> None:
        if task_id != self._task_id:
            return
        self.last_result = result
        self.results_list.clear()
        for path in result.written:
            self.results_list.addItem(Path(path).name)
        for failure in result.failures:
            self.results_list.addItem(f"FAILED  {failure.label}: {failure.message}")
        self.results_list.setVisible(True)
        written = len(result.products_written)
        if result.failures:
            self._set_status(
                f"Exported {written} product(s) to {result.directory}; "
                f"{len(result.failures)} failed (listed below). Written files are "
                "complete and were kept.",
                kind="warning",
            )
        else:
            self._set_status(
                f"Exported {written} product(s) ({len(result.written)} files) to "
                f"{result.directory}.",
                kind="success",
            )

    def _on_task_error(self, task_id, message, _details) -> None:
        if task_id == self._task_id:
            self._set_status(f"Export failed: {message}", kind="error")

    def _on_task_cancelled(self, task_id, _message) -> None:
        if task_id == self._task_id:
            self._set_status(
                "Export cancelled. Files written before cancellation are complete "
                "and were kept; no partial file was left.",
                kind="warning",
            )

    def _on_task_finished(self, task_id) -> None:
        if task_id == self._task_id:
            self._task_id = None
        self.refresh_from_state()

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def _refresh_controls(self) -> None:
        running = self.running
        idle = not running and not self.state.busy
        for box in self.checkboxes.values():
            box.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, running)
        self.export_button.setEnabled(
            idle and bool(self.selected_keys()) and bool(self.directory_edit.text().strip())
        )
        self.browse_button.setEnabled(not running)
        self.directory_edit.setEnabled(not running)
        self.overwrite_box.setEnabled(not running)
        self.cancel_button.setVisible(running)
        self.cancel_button.setEnabled(running and not self.task_controller.cancellation_requested)
        self.progress_bar.setVisible(running)

    def _set_status(self, text: str | None, *, kind: str = "success") -> None:
        if not text:
            self.status_label.setVisible(False)
            return
        for name in ("statusSuccess", "statusWarning", "statusError"):
            self.status_label.setProperty(name, False)
        self.status_label.setProperty(
            {"success": "statusSuccess", "warning": "statusWarning"}.get(kind, "statusError"),
            True,
        )
        refresh_style(self.status_label)
        self.status_label.setText(text)
        self.status_label.setVisible(True)
