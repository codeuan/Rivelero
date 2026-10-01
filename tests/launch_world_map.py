"""Development launcher for the Rivelero World map."""

from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from PySide6.QtWidgets import QApplication

from rivelero.gui.domain_service import (
    build_domain_from_drawn_polygon,
)
from rivelero.gui.environment_import import (
    inspect_elevation_raster,
)
from rivelero.gui.survey_import import (
    SurveyImportOptions,
    import_survey_csv,
)
from rivelero.gui.theme import (
    apply_theme,
)
from rivelero.gui.world_map import (
    WorldMapWidget,
)


DATA_ROOT = Path(
    r"C:\Users\zool2620\VISTA"
    r"\rivelero_synthetic_observability"
)

DEM = (
    DATA_ROOT
    / "environments"
    / "dem_flat.tif"
)

VIEWPOINTS = (
    DATA_ROOT
    / "viewpoints"
    / "viewpoints.csv"
)

SENSORS = (
    DATA_ROOT
    / "viewpoints"
    / "sensors.csv"
)

EVENTS = (
    DATA_ROOT
    / "viewpoints"
    / "observation_events.csv"
)


def main() -> int:

    app = QApplication(
        sys.argv
    )

    app.setApplicationName(
        "Rivelero World Map Test"
    )

    apply_theme(
        app
    )

    # --------------------------------------------------------------
    # Import canonical synthetic survey
    # --------------------------------------------------------------

    survey_result = import_survey_csv(
        viewpoints_path=VIEWPOINTS,
        sensors_path=SENSORS,
        observation_events_path=EVENTS,
        configuration_id="world_map_test",
        configuration_name="World map test",
        options=SurveyImportOptions(
            source_crs="EPSG:32633",
            target_crs="EPSG:32633",
            strict=True,
            allow_duplicate_coordinates=True,
        ),
    )

    viewpoints = list(
        survey_result
        .viewpoint_configuration
        .viewpoints
    )

    # --------------------------------------------------------------
    # Terrain
    # --------------------------------------------------------------

    metadata = inspect_elevation_raster(
        DEM
    )

    grid = metadata.analysis_grid

    # --------------------------------------------------------------
    # Map
    # --------------------------------------------------------------

    widget = WorldMapWidget()

    widget.resize(
        1200,
        850,
    )

    widget.set_raster(
        DEM
    )

    widget.set_viewpoints(
        viewpoints
    )

    # --------------------------------------------------------------
    # Selection diagnostic
    # --------------------------------------------------------------

    def on_viewpoint_selected(
        viewpoint_id: str,
    ) -> None:

        print(
            "Selected Viewpoint:",
            viewpoint_id,
        )

    widget.viewpoint_selected.connect(
        on_viewpoint_selected
    )

    # --------------------------------------------------------------
    # Draw-domain diagnostic
    # --------------------------------------------------------------

    def on_polygon(
        vertices,
    ) -> None:

        result = (
            build_domain_from_drawn_polygon(
                vertices=vertices,
                map_crs=grid.crs,
                grid=grid,
                domain_id="drawn_test",
                name="Drawn test",
            )
        )

        widget.set_domain(
            result.domain
        )

        print(
            "Analysis domain created:",
            result.domain.domain_id,
        )

        print(
            "Domain cells:",
            int(
                result
                .domain
                .analysis_mask
                .sum()
            ),
        )

    widget.domain_polygon_drawn.connect(
        on_polygon
    )

    widget.show()

    print(
        "World map loaded with",
        len(viewpoints),
        "Viewpoints.",
    )

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(
        main()
    )