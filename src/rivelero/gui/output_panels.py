"""Figures and Report tabs of the Output page (P3).

Both panels only select outputs and invoke the Qt-free services in
rivelero.export.figures / rivelero.export.report on the shared
TaskController. The ExportContext is frozen when the task starts, and
nothing is written to ApplicationState, so generating figures or reports
never marks the project as modified.
"""

from __future__ import annotations

from pathlib import Path

try:
    from PySide6.QtCore import Qt, QUrl
    from PySide6.QtGui import QDesktopServices
    from PySide6.QtWidgets import (
        QCheckBox, QComboBox, QFileDialog, QFormLayout, QGridLayout, QHBoxLayout, QLabel,
        QLineEdit, QListWidget, QProgressBar, QSpinBox, QVBoxLayout, QWidget,
    )
except ImportError:
    from PyQt6.QtCore import Qt, QUrl
    from PyQt6.QtGui import QDesktopServices
    from PyQt6.QtWidgets import (
        QCheckBox, QComboBox, QFileDialog, QFormLayout, QGridLayout, QHBoxLayout, QLabel,
        QLineEdit, QListWidget, QProgressBar, QSpinBox, QVBoxLayout, QWidget,
    )

from rivelero.export.catalog import ExportContext
from rivelero.export.figures import (
    DEFAULT_DPI,
    FIGURE_GROUPS,
    FigureOptions,
    figure_catalog,
    run_figure_export,
)
from rivelero.export.report import DEFAULT_REPORT_FIGURES, ReportOptions, generate_report
from rivelero.gui.application_state import ApplicationState
from rivelero.gui.components import ContentCard, make_primary_button, make_secondary_button
from rivelero.gui.export_service import export_context_from_state
from rivelero.gui.task_controller import TaskController, make_progress_task
from rivelero.gui.theme import SPACING, refresh_style

FIGURE_TASK_NAME = "Exporting figures"
REPORT_TASK_NAME = "Generating report"
DEFAULT_FIGURES = frozenset({"analysis_state", "exposure"})


class _TaskPanel(QWidget):
    """Shared task plumbing: one task id, progress bar, status label."""

    def __init__(self, state: ApplicationState, task_controller: TaskController, parent=None):
        super().__init__(parent)
        self.state = state
        self.task_controller = task_controller
        self._task_id: str | None = None
        self.last_result = None
        self.checkboxes: dict[str, QCheckBox] = {}
        self.reason_labels: dict[str, QLabel] = {}
        self._seen: set[str] = set()

    @property
    def running(self) -> bool:
        return self._task_id is not None

    def selected_keys(self) -> list[str]:
        return [k for k, box in self.checkboxes.items() if box.isEnabled() and box.isChecked()]

    def set_selection(self, keys) -> None:
        keys = set(keys)
        for key, box in self.checkboxes.items():
            box.setChecked(key in keys and box.isEnabled())

    def choose_directory(self) -> str | None:
        """Dialog hook (replaced in tests)."""
        path = QFileDialog.getExistingDirectory(
            self, "Choose folder", self.directory_edit.text() or str(Path.home())
        )
        return path or None

    # -- building blocks -------------------------------------------------

    def _add_product_cards(self, layout, catalog, groups, defaults) -> None:
        grid = QGridLayout()
        grid.setHorizontalSpacing(SPACING.xl)
        grid.setVerticalSpacing(SPACING.xl)
        for index, group in enumerate(groups):
            card = ContentCard(title=group)
            for product in (p for p in catalog if p.group == group):
                box = QCheckBox(product.label)
                box.setToolTip(product.meaning)
                reason = QLabel()
                reason.setWordWrap(True)
                reason.setProperty("secondaryText", True)
                reason.setContentsMargins(24, 0, 0, 0)
                reason.setVisible(False)
                self.checkboxes[product.key] = box
                self.reason_labels[product.key] = reason
                card.add_widget(box)
                card.add_widget(reason)
                box.toggled.connect(lambda _c: self._refresh_controls())
            card.add_stretch()
            grid.addWidget(card, 0, index)
        layout.addLayout(grid)
        self._defaults = defaults

    def _add_folder_row(self, card, placeholder: str) -> None:
        row = QHBoxLayout()
        self.directory_edit = QLineEdit()
        self.directory_edit.setPlaceholderText(placeholder)
        self.browse_button = make_secondary_button("Choose folder…")
        row.addWidget(self.directory_edit, 1)
        row.addWidget(self.browse_button)
        card.add_layout(row)
        self.browse_button.clicked.connect(self._browse)
        self.directory_edit.textChanged.connect(lambda _t: self._refresh_controls())

    def _add_run_controls(self, card, text: str) -> None:
        buttons = QHBoxLayout()
        self.run_button = make_primary_button(text)
        self.cancel_button = make_secondary_button("Cancel")
        self.cancel_button.setVisible(False)
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.cancel_button)
        buttons.addStretch(1)
        card.add_layout(buttons)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        card.add_widget(self.progress_bar)
        self.progress_label = QLabel()
        self.progress_label.setProperty("secondaryText", True)
        self.progress_label.setVisible(False)
        card.add_widget(self.progress_label)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.status_label.setVisible(False)
        card.add_widget(self.status_label)
        self.results_list = QListWidget()
        self.results_list.setMinimumHeight(110)
        self.results_list.setVisible(False)
        card.add_widget(self.results_list)
        self.run_button.clicked.connect(self.start)
        self.cancel_button.clicked.connect(self._cancel)
        controller = self.task_controller
        controller.task_progress.connect(self._on_task_progress)
        controller.task_result.connect(self._on_task_result)
        controller.task_error.connect(self._on_task_error)
        controller.task_cancelled.connect(self._on_task_cancelled)
        controller.task_finished.connect(self._on_task_finished)
        # Any task (on any page) disables starting another one.
        controller.task_started.connect(lambda _id, _name: self._refresh_controls())

    def _browse(self) -> None:
        path = self.choose_directory()
        if path:
            self.directory_edit.setText(path)

    # -- state -----------------------------------------------------------

    def refresh(self, context: ExportContext | None = None) -> None:
        if self.running:
            self._refresh_controls()
            return
        context = context or export_context_from_state(self.state)
        for product in figure_catalog(context):
            box = self.checkboxes.get(product.key)
            if box is None:
                continue
            box.setEnabled(product.available)
            if not product.available:
                box.setChecked(False)
                self._seen.discard(product.key)
            elif product.key not in self._seen:
                self._seen.add(product.key)
                box.setChecked(product.key in self._defaults)
            self.reason_labels[product.key].setText(product.reason or "")
            self.reason_labels[product.key].setVisible(not product.available)
        self._refresh_controls()

    def _ready(self) -> bool:
        return bool(self.selected_keys()) and bool(self.directory_edit.text().strip())

    def _refresh_controls(self) -> None:
        running = self.running
        self.run_button.setEnabled(not running and not self.state.busy and self._ready())
        self.cancel_button.setVisible(running)
        self.cancel_button.setEnabled(running and not self.task_controller.cancellation_requested)
        self.progress_bar.setVisible(running)
        self.progress_label.setVisible(running)
        for widget in self._inputs():
            widget.setEnabled(not running)
        for box in self.checkboxes.values():
            box.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, running)

    def _inputs(self):
        return (self.directory_edit, self.browse_button)

    def _start_task(self, name: str, function, kwargs: dict, total: int) -> str:
        task = make_progress_task(function=function, kwargs=kwargs)
        self.last_result = None
        self.results_list.clear()
        self.results_list.setVisible(False)
        self._task_id = self.task_controller.start(
            task_name=name, function=task, total=total, message=name, inject_context=True,
        )
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(0)
        self.progress_label.setText("Starting…")
        self._set_status(None)
        self._refresh_controls()
        return self._task_id

    def _cancel(self) -> None:
        if self._task_id is not None and self.task_controller.cancel():
            self.cancel_button.setEnabled(False)
            self._set_status("Cancellation requested; stopping after the current step.",
                             kind="warning")

    def _on_task_progress(self, task_id, processed, total, unit_id, _message) -> None:
        if task_id != self._task_id:
            return
        if total:
            self.progress_bar.setRange(0, int(total))
        self.progress_bar.setValue(int(processed))
        if unit_id:
            self.progress_label.setText(str(unit_id))

    def _on_task_error(self, task_id, message, _details) -> None:
        if task_id == self._task_id:
            self._set_status(f"Failed: {message}", kind="error")

    def _on_task_finished(self, task_id) -> None:
        if task_id == self._task_id:
            self._task_id = None
            self.refresh()
        else:
            self._refresh_controls()

    def _set_status(self, text: str | None, *, kind: str = "success") -> None:
        if not text:
            self.status_label.setVisible(False)
            return
        for name in ("statusSuccess", "statusWarning", "statusError"):
            self.status_label.setProperty(name, False)
        self.status_label.setProperty(
            {"success": "statusSuccess", "warning": "statusWarning"}.get(kind, "statusError"), True
        )
        refresh_style(self.status_label)
        self.status_label.setText(text)
        self.status_label.setVisible(True)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


class FiguresPanel(_TaskPanel):
    """Export individual publication figures."""

    def __init__(self, state, task_controller, parent=None):
        super().__init__(state, task_controller, parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, SPACING.lg, 0, 0)
        layout.setSpacing(SPACING.xl)
        note = QLabel(
            "Standalone figures use the same colours, scales, masks and legends as the "
            "application maps and charts. Maps always show the full grid; exposure "
            "differences are right − left on a scale centred on zero."
        )
        note.setWordWrap(True)
        note.setProperty("secondaryText", True)
        layout.addWidget(note)
        catalog = figure_catalog(export_context_from_state(state))
        self._add_product_cards(layout, catalog, FIGURE_GROUPS, DEFAULT_FIGURES)

        card = ContentCard(title="Export figures")
        form = QFormLayout()
        self.format_combo = QComboBox()
        for fmt, label in (("png", "PNG (raster)"), ("svg", "SVG (vector)"), ("pdf", "PDF (vector)")):
            self.format_combo.addItem(label, fmt)
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(50, 1200)
        self.dpi_spin.setValue(DEFAULT_DPI)
        self.dpi_spin.setSuffix(" dpi")
        self.viewpoints_box = QCheckBox("Show Viewpoints on maps")
        self.viewpoints_box.setChecked(True)
        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("Default title (used for every selected figure)")
        self.caption_edit = QLineEdit()
        self.caption_edit.setPlaceholderText("Optional caption")
        form.addRow("Format", self.format_combo)
        form.addRow("Resolution", self.dpi_spin)
        form.addRow("", self.viewpoints_box)
        form.addRow("Title", self.title_edit)
        form.addRow("Caption", self.caption_edit)
        card.add_layout(form)
        self._add_folder_row(card, "Figure folder")
        self.overwrite_box = QCheckBox("Replace existing files with the same names")
        card.add_widget(self.overwrite_box)
        self._add_run_controls(card, "Export figures")
        layout.addWidget(card)
        layout.addStretch(1)
        self.format_combo.currentIndexChanged.connect(lambda _i: self._refresh_controls())
        self.refresh()

    def _inputs(self):
        return (self.directory_edit, self.browse_button, self.format_combo, self.dpi_spin,
                self.viewpoints_box, self.title_edit, self.caption_edit, self.overwrite_box)

    def _refresh_controls(self) -> None:
        super()._refresh_controls()
        self.dpi_spin.setEnabled(not self.running and self.format_combo.currentData() == "png")

    def options(self) -> FigureOptions:
        return FigureOptions(
            format=self.format_combo.currentData(), dpi=self.dpi_spin.value(),
            include_viewpoints=self.viewpoints_box.isChecked(),
            title=self.title_edit.text().strip() or None,
            caption=self.caption_edit.text().strip() or None,
        )

    def start(self) -> str | None:
        if self.running or self.state.busy or not self._ready():
            return None
        keys = self.selected_keys()
        return self._start_task(FIGURE_TASK_NAME, run_figure_export, {
            "context": export_context_from_state(self.state), "keys": keys,
            "directory": self.directory_edit.text().strip(), "options": self.options(),
            "overwrite": self.overwrite_box.isChecked(),
        }, len(keys))

    def _on_task_result(self, task_id, result) -> None:
        if task_id != self._task_id:
            return
        self.last_result = result
        for path in result.written:
            self.results_list.addItem(Path(path).name)
        for key, message in result.failures:
            self.results_list.addItem(f"FAILED  {key}: {message}")
        self.results_list.setVisible(True)
        figures = sum(1 for p in result.written if not p.name.endswith(".json"))
        if result.failures:
            self._set_status(f"Exported {figures} figure(s) to {result.directory}; "
                             f"{len(result.failures)} failed.", kind="warning")
        else:
            self._set_status(f"Exported {figures} figure(s) with metadata sidecars to "
                             f"{result.directory}.")

    def _on_task_cancelled(self, task_id, _message) -> None:
        if task_id == self._task_id:
            self._set_status("Figure export cancelled. Figures written before cancellation "
                             "are complete and were kept.", kind="warning")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


class ReportPanel(_TaskPanel):
    """Generate the compact HTML report with provenance."""

    def __init__(self, state, task_controller, parent=None):
        super().__init__(state, task_controller, parent)
        self.report_directory: Path | None = None
        self._auto_name = ""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, SPACING.lg, 0, 0)
        layout.setSpacing(SPACING.xl)
        note = QLabel(
            "A self-contained report folder: report.html (opens offline in any browser), "
            "provenance.json, provenance.md and the selected figures. Sections for results "
            "that do not exist yet are omitted. Data files are exported separately (Data tab)."
        )
        note.setWordWrap(True)
        note.setProperty("secondaryText", True)
        layout.addWidget(note)
        catalog = figure_catalog(export_context_from_state(state))
        self._add_product_cards(layout, catalog, FIGURE_GROUPS, frozenset(DEFAULT_REPORT_FIGURES))

        card = ContentCard(title="Generate report")
        form = QFormLayout()
        self.title_edit = QLineEdit(ReportOptions().title)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Report folder name")
        self.format_combo = QComboBox()
        self.format_combo.addItem("PNG", "png")
        self.format_combo.addItem("SVG (vector)", "svg")
        self.viewpoints_box = QCheckBox("Show Viewpoints on maps")
        self.viewpoints_box.setChecked(True)
        form.addRow("Report title", self.title_edit)
        form.addRow("Folder name", self.name_edit)
        form.addRow("Figure format", self.format_combo)
        form.addRow("", self.viewpoints_box)
        card.add_layout(form)
        self._add_folder_row(card, "Parent folder")
        self.overwrite_box = QCheckBox("Replace an existing report folder with the same name")
        card.add_widget(self.overwrite_box)
        self._add_run_controls(card, "Generate report")
        self.open_button = make_secondary_button("Open report")
        self.open_button.setVisible(False)
        card.add_widget(self.open_button)
        self.open_button.clicked.connect(self._open_report)
        layout.addWidget(card)
        layout.addStretch(1)
        self.name_edit.textChanged.connect(lambda _t: self._refresh_controls())
        self.refresh()

    def _inputs(self):
        return (self.directory_edit, self.browse_button, self.title_edit, self.name_edit,
                self.format_combo, self.viewpoints_box, self.overwrite_box)

    def refresh(self, context: ExportContext | None = None) -> None:
        context = context or export_context_from_state(self.state)
        # Follow the project name (e.g. after Save As) until the user types
        # a name of their own.
        current = self.name_edit.text().strip()
        if not self.running and current in ("", self._auto_name):
            self._auto_name = f"{context.prefix}_report"
            self.name_edit.setText(self._auto_name)
        super().refresh(context)

    def _ready(self) -> bool:
        # A report needs no figure: a Survey-only report is valid.
        return bool(self.directory_edit.text().strip()) and bool(self.name_edit.text().strip())

    def target(self) -> Path:
        return Path(self.directory_edit.text().strip()) / self.name_edit.text().strip()

    def start(self) -> str | None:
        if self.running or self.state.busy or not self._ready():
            return None
        options = ReportOptions(
            title=self.title_edit.text().strip() or ReportOptions().title,
            figures=tuple(self.selected_keys()),
            figure_format=self.format_combo.currentData(),
            include_viewpoints=self.viewpoints_box.isChecked(),
        )
        self.open_button.setVisible(False)
        return self._start_task(REPORT_TASK_NAME, generate_report, {
            "context": export_context_from_state(self.state), "directory": self.target(),
            "options": options, "overwrite": self.overwrite_box.isChecked(),
        }, len(options.figures) + 3)

    def _on_task_result(self, task_id, result) -> None:
        if task_id != self._task_id:
            return
        self.last_result = result
        self.report_directory = result.directory
        for path in result.files:
            self.results_list.addItem(str(path.relative_to(result.directory)))
        self.results_list.setVisible(True)
        self.open_button.setVisible(True)
        self._set_status(f"Report written to {result.directory} "
                         f"({len(result.figures)} figure(s)).")

    def _on_task_cancelled(self, task_id, _message) -> None:
        if task_id == self._task_id:
            self._set_status("Report generation cancelled. No report was written and any "
                             "existing report was left unchanged.", kind="warning")

    def _open_report(self) -> None:
        if self.report_directory is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.report_directory / "report.html")))
