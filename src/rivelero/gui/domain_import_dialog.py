"""Analysis-area polygon import dialog for Rivelero."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

try:
    from PySide6.QtWidgets import (
        QDialog,
        QFileDialog,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPushButton,
        QVBoxLayout,
    )
except ImportError:
    from PyQt6.QtWidgets import (
        QDialog,
        QFileDialog,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPushButton,
        QVBoxLayout,
    )

from rivelero.core.domain import AnalysisGrid
from rivelero.gui.components import (
    ContentCard,
    PageHeader,
    make_primary_button,
)
from rivelero.gui.domain_service import (
    DomainConstructionResult,
    build_domain_from_geojson,
    read_geojson_geometry,
)
from rivelero.gui.theme import SPACING


class DomainImportDialog(QDialog):
    """Import a polygonal AnalysisDomain from GeoJSON."""

    def __init__(
        self,
        *,
        grid: AnalysisGrid,
        valid_mask=None,
        parent=None,
    ) -> None:
        super().__init__(parent)

        # Elevation-validity mask applied to the imported domain.
        self.valid_mask = valid_mask

        if not isinstance(grid, AnalysisGrid):
            raise TypeError(
                "grid must be an AnalysisGrid."
            )

        self.grid = grid
        self._result: DomainConstructionResult | None = None

        self.setWindowTitle(
            "Import analysis area — Rivelero"
        )
        self.setMinimumSize(680, 480)
        self.setModal(True)

        self._build_interface()

    @property
    def domain_result(self) -> DomainConstructionResult | None:
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
                "Import analysis area",
                (
                    "Import polygon geometry and explicitly define "
                    "its source coordinate reference system."
                ),
            )
        )

        card = ContentCard(
            title="Polygon source",
        )

        path_row = QHBoxLayout()

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText(
            "Select GeoJSON…"
        )

        browse = QPushButton("Browse")

        path_row.addWidget(
            self.path_edit,
            1,
        )
        path_row.addWidget(browse)

        card.add_layout(path_row)

        form = QFormLayout()

        self.crs_edit = QLineEdit(
            self.grid.crs.to_string()
        )

        self.role_edit = QLineEdit()
        self.role_edit.setPlaceholderText(
            "Optional, e.g. analysis_domain"
        )

        self.name_edit = QLineEdit(
            "Imported analysis area"
        )

        form.addRow(
            "Source CRS",
            self.crs_edit,
        )
        form.addRow(
            "Feature role",
            self.role_edit,
        )
        form.addRow(
            "Name",
            self.name_edit,
        )

        card.add_layout(form)

        self.preview = QLabel(
            "Select a polygon file."
        )
        self.preview.setWordWrap(True)
        self.preview.setProperty(
            "secondaryText",
            True,
        )

        card.add_widget(self.preview)

        root.addWidget(card)
        root.addStretch(1)

        actions = QHBoxLayout()
        actions.addStretch(1)

        cancel = QPushButton("Cancel")
        self.import_button = make_primary_button(
            "Import area"
        )
        self.import_button.setEnabled(False)

        actions.addWidget(cancel)
        actions.addWidget(self.import_button)

        root.addLayout(actions)

        browse.clicked.connect(self._browse)
        self.path_edit.textChanged.connect(
            self._inspect
        )
        cancel.clicked.connect(self.reject)
        self.import_button.clicked.connect(
            self._accept_import
        )

    def _browse(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select analysis-area polygon",
            "",
            "GeoJSON (*.geojson *.json);;All files (*)",
        )

        if filename:
            self.path_edit.setText(filename)

    def _inspect(self) -> None:
        self.import_button.setEnabled(False)

        path = self.path_edit.text().strip()

        if not path:
            self.preview.setText(
                "Select a polygon file."
            )
            return

        try:
            geometry = read_geojson_geometry(
                path,
                role=(
                    self.role_edit.text().strip()
                    or None
                ),
            )

        except Exception as exc:
            self.preview.setText(
                f"Unable to inspect geometry: {exc}"
            )
            return

        self.preview.setText(
            f"Geometry: {geometry.geom_type}\n"
            f"Area in source coordinates: {geometry.area:,.3f}\n"
            f"Target grid CRS: {self.grid.crs.to_string()}"
        )

        self.import_button.setEnabled(True)

    def _accept_import(self) -> None:
        path = self.path_edit.text().strip()

        try:
            self._result = build_domain_from_geojson(
                path=path,
                source_crs=self.crs_edit.text().strip(),
                grid=self.grid,
                domain_id=uuid4().hex,
                name=(
                    self.name_edit.text().strip()
                    or "Imported analysis area"
                ),
                role=(
                    self.role_edit.text().strip()
                    or None
                ),
                valid_mask=self.valid_mask,
                clip_to_grid=True,
            )

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Unable to import analysis area",
                str(exc),
            )
            return

        self.accept()