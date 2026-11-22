"""Coordinate input for DEM lattice sampling."""

from rasterio.errors import RasterioError

from rivelero.analysis.dem_distribution import (
    DISTRIBUTION_MODES,
    distribute_dem_coordinates,
)
from rivelero.core.environment import Environment
from rivelero.core.viewpoint import Viewpoint

try:
    from PySide6.QtWidgets import (
        QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel,
        QMessageBox, QPlainTextEdit, QSpinBox,
    )
except ImportError:
    from PyQt6.QtWidgets import (
        QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel,
        QMessageBox, QPlainTextEdit, QSpinBox,
    )


class DemDistributionDialog(QDialog):
    """Generate a batch of Viewpoints from pasted DEM-CRS coordinate pairs."""

    def __init__(
        self, environment: Environment, existing_ids: tuple[str, ...], parent=None
    ) -> None:
        super().__init__(parent)
        self.environment = environment
        self.existing_ids = set(existing_ids)
        self.viewpoints: list[Viewpoint] = []
        self.setWindowTitle("Distribute coordinates on DEM")
        self.resize(470, 340)

        form = QFormLayout(self)
        crs = environment.elevation_model.crs or "the DEM coordinate reference system"
        instructions = QLabel(
            f"Enter one X, Y pair per line in {crs}. Each pair becomes a new "
            "Viewpoint at a distinct valid site on the 10 × 10 DEM lattice."
        )
        instructions.setWordWrap(True)
        form.addRow(instructions)
        self.coordinates = QPlainTextEdit()
        self.coordinates.setPlaceholderText("X, Y")
        form.addRow("Coordinates", self.coordinates)
        self.mode = QComboBox()
        for mode in DISTRIBUTION_MODES:
            self.mode.addItem(mode.capitalize(), mode)
        self.mode.setToolTip(
            "Random samples sites uniformly. Clustered uses sites nearest a seed site. "
            "Dispersed spreads sites apart."
        )
        form.addRow("Distribution", self.mode)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2**31 - 1)
        self.seed.setToolTip("Use the same seed to reproduce the selected sites.")
        form.addRow("Random seed", self.seed)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Add Viewpoints")
        buttons.accepted.connect(self._generate)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _generate(self) -> None:
        coordinates = []
        mode = self.mode.currentData()
        seed = self.seed.value()
        try:
            for number, line in enumerate(self.coordinates.toPlainText().splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    x, y = (float(value) for value in line.split(","))
                except ValueError as exc:
                    raise ValueError(f"Line {number} must contain one X, Y pair.") from exc
                coordinates.append((x, y))
            crs, positions = distribute_dem_coordinates(
                self.environment.elevation_model.source, coordinates, mode=mode, seed=seed
            )
        except (ValueError, RasterioError) as exc:
            QMessageBox.warning(self, "Cannot distribute coordinates", str(exc))
            return

        number = 0
        for original, (x, y) in zip(coordinates, positions, strict=True):
            number += 1
            while f"dem-{mode}-{number}" in self.existing_ids:
                number += 1
            self.viewpoints.append(Viewpoint(
                viewpoint_id=f"dem-{mode}-{number}",
                x=x,
                y=y,
                crs=crs,
                source="DEM lattice simulation",
                provenance={
                    "environment_id": self.environment.environment_id,
                    "grid_shape": [10, 10],
                    "distribution": mode,
                    "seed": seed,
                    "input_coordinates": list(original),
                    "input_crs": crs.to_string(),
                },
            ))
        self.accept()
