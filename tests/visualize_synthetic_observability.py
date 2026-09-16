from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"

for path in (PROJECT_ROOT, TESTS_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from test_synthetic_observability import (  # noqa: E402
    DATA_ROOT,
    _build,
    make_context as context_fixture,
    sensors as sensors_fixture,
    viewpoints as viewpoints_fixture,
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from rivelero.visualization.observability import (  # noqa: E402
    plot_blindspots,
    plot_exposure,
    plot_normalized_exposure,
    plot_observability_state,
)


OUTPUT_ROOT = DATA_ROOT / "visual_validation"


def _save_figure(plotter, sof, output_path: Path, **kwargs) -> None:
    fig, _ = plotter(sof, **kwargs)
    try:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    viewpoints = viewpoints_fixture.__wrapped__()
    sensors = sensors_fixture.__wrapped__()

    with tempfile.TemporaryDirectory(prefix="rivelero_visual_validation_") as cache_root:
        cache_path = Path(cache_root)
        make_context = context_fixture.__wrapped__(cache_path)

        flat_context = make_context("dem_flat.tif", "visual_flat")
        flat_sof, _, _, _ = _build(
            context=flat_context,
            viewpoints=viewpoints,
            sensors=sensors,
            tmp_path=cache_path,
            ids=["vp_center_360"],
            max_distance_m=200,
            configuration_id="visual_flat_single",
        )
        _save_figure(
            plot_exposure,
            flat_sof,
            OUTPUT_ROOT / "flat_single_exposure.png",
            title="Synthetic flat single: exposure",
        )
        _save_figure(
            plot_blindspots,
            flat_sof,
            OUTPUT_ROOT / "flat_single_blindspots.png",
            title="Synthetic flat single: blind spots",
        )

        overlap_context = make_context("dem_flat.tif", "visual_overlap")
        overlap_sof, _, _, _ = _build(
            context=overlap_context,
            viewpoints=viewpoints,
            sensors=sensors,
            tmp_path=cache_path,
            ids=["vp_overlap_west", "vp_overlap_east"],
            max_distance_m=300,
            configuration_id="visual_overlap",
        )
        _save_figure(
            plot_exposure,
            overlap_sof,
            OUTPUT_ROOT / "overlap_exposure.png",
            title="Synthetic overlap: exposure",
        )
        _save_figure(
            plot_normalized_exposure,
            overlap_sof,
            OUTPUT_ROOT / "overlap_normalized_exposure.png",
            title="Synthetic overlap: normalized exposure",
        )

        nodata_context = make_context("dem_nodata.tif", "visual_nodata")
        nodata_sof, _, _, _ = _build(
            context=nodata_context,
            viewpoints=viewpoints,
            sensors=sensors,
            tmp_path=cache_path,
            ids=["vp_center_360"],
            max_distance_m=300,
            configuration_id="visual_nodata_states",
        )
        _save_figure(
            plot_observability_state,
            nodata_sof,
            OUTPUT_ROOT / "nodata_observability_state.png",
            title="Synthetic nodata: observability state",
        )

    for path in sorted(OUTPUT_ROOT.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()