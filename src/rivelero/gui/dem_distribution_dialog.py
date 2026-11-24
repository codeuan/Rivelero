"""Controls for placing simulated viewpoints on a DEM."""

from rasterio.errors import RasterioError

from rivelero.analysis.dem_distribution import (
    DISTRIBUTION_MODES,
    generate_dem_viewpoints,
)
from rivelero.core.environment import Environment
from rivelero.core.viewpoint import Viewpoint

try:
    from PySide6.QtWidgets import (
        QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel,
        QMessageBox, QSpinBox,
    )
except ImportError:
    from PyQt6.QtWidgets import (
        QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel,
        QMessageBox, QSpinBox,
    )


class DemDistributionDialog(QDialog):
    """Generate a batch of Viewpoints at selected DEM grid centres."""

    def __init__(
        self, environment: Environment, existing_ids: tuple[str, ...], parent=None
    ) -> None:
        super().__init__(parent)
        self.environment = environment
        self.existing_ids = existing_ids
        self.viewpoints: list[Viewpoint] = []
        self.setWindowTitle("Distribute viewpoints on DEM")
        self.resize(470, 340)

        form = QFormLayout(self)
        instructions = QLabel(
            "Place viewpoints at distinct sites on a 10 × 10 grid over the DEM. "
            "Each camera has a 120° horizontal field of view aimed at the terrain centre. "
            "Sites without valid elevation are skipped."
        )
        instructions.setWordWrap(True)
        form.addRow(instructions)
        self.count = QSpinBox()
        self.count.setRange(1, 100)
        form.addRow("Number of viewpoints", self.count)
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
        mode = self.mode.currentData()
        seed = self.seed.value()
        try:
            self.viewpoints = generate_dem_viewpoints(
                self.environment.elevation_model.source,
                self.count.value(),
                environment_id=self.environment.environment_id,
                mode=mode,
                seed=seed,
                existing_ids=self.existing_ids,
            )
        except (ValueError, RasterioError) as exc:
            QMessageBox.warning(self, "Cannot distribute viewpoints", str(exc))
            return

        self.accept()
