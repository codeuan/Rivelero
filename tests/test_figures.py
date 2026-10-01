"""P3 tests: standalone scientific figures (Qt-free).

No pixel comparisons: tests inspect the Matplotlib objects (source arrays,
extent, norms, legends, colourbars, titles) and the saved files.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from matplotlib.figure import Figure
from PIL import Image

from rivelero.analysis.comparison import BASELINE_ID, LIVE_SCENARIO_ID
from rivelero.analysis.coverage import summarize_coverage
from rivelero.export.catalog import ExportConflictError, ExportContext, ExportUnavailableError
from rivelero.export.figures import (
    FigureOptions,
    figure_catalog,
    run_figure_export,
)
from rivelero.visualization.analysis import plot_exposure_distribution
from rivelero.visualization.maps import raster_extent
from test_export import PROVENANCE, _context

pytestmark = pytest.mark.filterwarnings("ignore")


def _build(context, key, **options):
    product = {p.key: p for p in figure_catalog(context)}[key]
    assert product.available, product.reason
    return product.build(FigureOptions(**options))


def _map_axes(fig):
    return fig.axes[0]


def _legend_labels(fig):
    ax = _map_axes(fig)
    legend = ax.get_legend() or (fig.legends[0] if fig.legends else None)
    return [text.get_text() for text in legend.get_texts()] if legend else []


def _colorbar_label(fig):
    return fig.axes[1].get_ylabel() if len(fig.axes) > 1 else None


def test_catalog_availability(tmp_path):
    context, _, _ = _context(tmp_path)
    assert all(p.available for p in figure_catalog(context))
    empty = figure_catalog(ExportContext(provenance=PROVENANCE))
    assert not any(p.available for p in empty)
    assert all(p.reason for p in empty)


def test_analysis_state_map_semantics(tmp_path):
    context, sof, _ = _context(tmp_path)
    fig = _build(context, "analysis_state")
    assert isinstance(fig, Figure)
    ax = _map_axes(fig)
    image = ax.images[0]
    np.testing.assert_array_equal(np.asarray(image.get_array()), sof.observability_state)
    extent = raster_extent(shape=sof.exposure_count.shape, transform=sof.transform)
    assert tuple(image.get_extent()) == pytest.approx((extent[0], extent[1], extent[2], extent[3]))
    assert ax.get_xlim() == pytest.approx((extent[0], extent[1]))
    assert ax.get_ylim() == pytest.approx((extent[2], extent[3]))
    labels = _legend_labels(fig)
    for label in ("Outside domain", "Invalid / unanalysable", "Blind spot", "Observable"):
        assert label in labels
    assert _colorbar_label(fig) is None  # categorical: legend, no colourbar
    assert "EPSG:32633" in ax.get_xlabel() and "(m" in ax.get_xlabel()
    assert fig.get_suptitle() == "Analysis state"


def test_exposure_map_whole_field_scale_and_masking(tmp_path):
    context, sof, _ = _context(tmp_path)
    fig = _build(context, "exposure")
    image = _map_axes(fig).images[0]
    data = image.get_array()
    assert np.array_equal(np.ma.getmaskarray(data), ~sof.analysable_mask)
    np.testing.assert_array_equal(data.compressed(), sof.exposure_count[sof.analysable_mask])
    # Blind spots are drawn as 0, never masked.
    assert not np.ma.getmaskarray(data)[sof.blindspot_mask].any()
    assert image.norm.vmin == 0 and image.norm.vmax == sof.maximum_exposure
    assert "Exposure" in _colorbar_label(fig)

    fig = _build(context, "normalized_exposure")
    image = _map_axes(fig).images[0]
    assert (image.norm.vmin, image.norm.vmax) == (0.0, 1.0)


def test_viewpoints_optional(tmp_path):
    context, _, _ = _context(tmp_path)
    with_points = _build(context, "exposure")
    assert any(c.get_gid() == "rivelero_viewpoints" for c in _map_axes(with_points).collections)
    without = _build(context, "exposure", include_viewpoints=False)
    assert not any(c.get_gid() == "rivelero_viewpoints" for c in _map_axes(without).collections)


def test_scenario_change_map(tmp_path):
    context, sof, scenario = _context(tmp_path)
    fig = _build(context, "scenario_change")
    np.testing.assert_array_equal(np.asarray(_map_axes(fig).images[0].get_array()),
                                  scenario.change_classes())
    labels = _legend_labels(fig)
    for label in ("Remains blind", "Remains observable", "Lost coverage (becomes blind)",
                  "Gained coverage (becomes observable)", "Outside domain",
                  "Invalid / unanalysable"):
        assert label in labels


def test_exposure_difference_direction_and_symmetric_scale(tmp_path):
    context, sof, scenario = _context(tmp_path)
    fig = _build(context, "exposure_difference")
    ax = _map_axes(fig)
    image = ax.images[0]
    data = image.get_array()
    expected = scenario.exposure.astype(np.int64) - sof.exposure_count.astype(np.int64)
    analysable = sof.analysable_mask
    np.testing.assert_array_equal(data[analysable], expected[analysable])
    assert np.ma.getmaskarray(data)[~analysable].all()
    assert image.norm.vmin == -image.norm.vmax > -np.inf and image.norm.vmax >= 1
    assert "right − left" in _colorbar_label(fig)
    # Title names RIGHT first: "right − left".
    assert fig.get_suptitle() == "Exposure difference: Current scenario (unsaved) − Baseline"
    assert ax.patch.get_hatch() == "////"

    swapped, _, _ = _context(tmp_path / "b", comparison_sides=(LIVE_SCENARIO_ID, BASELINE_ID))
    back = _map_axes(_build(swapped, "exposure_difference")).images[0].get_array()
    np.testing.assert_array_equal(back[analysable], -expected[analysable])

    change = _build(context, "comparison_change")
    assert "Lost coverage (observable left, blind right)" in _legend_labels(change)


def test_a1_charts_use_the_analysis_page_semantics(tmp_path):
    context, sof, _ = _context(tmp_path)
    fig = _build(context, "exposure_distribution")
    reference = Figure()
    plot_exposure_distribution(summarize_coverage(sof), ax=reference.add_subplot(), title="")
    heights = [bar.get_height() for bar in fig.axes[0].patches]
    expected = [bar.get_height() for bar in reference.axes[0].patches]
    assert heights == pytest.approx(expected)

    fig = _build(context, "coverage_composition")
    widths = [bar.get_width() for bar in fig.axes[0].patches]
    summary = summarize_coverage(sof)
    assert widths == pytest.approx([100 * summary.blind_fraction, 100 * summary.unique_fraction,
                                    100 * summary.repeated_fraction])
    assert len(_legend_labels(fig)) == 3


def test_contribution_distribution_is_scalable_and_descriptive(tmp_path):
    context, _, _ = _context(tmp_path)
    fig = _build(context, "contribution_distribution")
    left, right = fig.axes[:2]
    units = context.contribution.available_units
    assert sum(p.get_height() for p in left.patches) == len(units)
    with_unique = [u for u in units if u.unique_cells > 0]
    assert sum(p.get_height() for p in right.patches) == len(with_unique)
    without = len(units) - len(with_unique)
    assert f"{without} with none" in right.get_title()
    texts = " ".join(t.get_text() for t in fig.findobj(match=lambda a: hasattr(a, "get_text")))
    assert "useless" not in texts.lower() and "bad" not in texts.lower().split()


def test_user_text_is_not_mathtext(tmp_path):
    context, _, _ = _context(tmp_path)
    fig = _build(context, "exposure", title="Cost $5 survey")
    assert fig.get_suptitle() == r"Cost \$5 survey"


def test_options_validation():
    with pytest.raises(ValueError):
        FigureOptions(format="jpg")
    with pytest.raises(ValueError):
        FigureOptions(dpi=10)


def test_save_formats_and_metadata(tmp_path):
    context, sof, _ = _context(tmp_path)
    keys = ["analysis_state", "exposure_difference", "coverage_composition"]
    for fmt in ("png", "svg", "pdf"):
        result = run_figure_export(context, keys, tmp_path / fmt, FigureOptions(format=fmt, dpi=72))
        assert result.successful
        files = sorted(p.name for p in (tmp_path / fmt).iterdir())
        assert files == sorted([f"test_{k}.{fmt}" for k in keys] + [f"test_{k}.{fmt}.json" for k in keys])
        for path in (tmp_path / fmt).glob(f"*.{fmt}"):
            assert path.stat().st_size > 1000
    with Image.open(tmp_path / "png" / "test_analysis_state.png") as image:
        description = json.loads(image.info["Description"])
        assert description["sof_id"] == sof.sof_id and description["project"] == "Test project"
        assert image.info["Title"] == "Analysis state"
    svg = (tmp_path / "svg" / "test_exposure_difference.svg").read_text(encoding="utf-8")
    assert "<text" in svg and "right − left" in svg  # vector text, not paths
    assert b"/Title" in (tmp_path / "pdf" / "test_analysis_state.pdf").read_bytes()
    record = json.loads((tmp_path / "png" / "test_exposure_difference.png.json").read_text(encoding="utf-8"))
    assert record["sources"]["direction"] == "right_minus_left"
    assert record["figure"]["dpi"] == 72


def test_figure_export_refuses_conflicts_and_unavailable(tmp_path):
    context, _, _ = _context(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "test_exposure.png").write_text("keep", encoding="utf-8")
    with pytest.raises(ExportConflictError):
        run_figure_export(context, ["analysis_state", "exposure"], out)
    assert sorted(p.name for p in out.iterdir()) == ["test_exposure.png"]
    with pytest.raises(ExportUnavailableError):
        run_figure_export(ExportContext(provenance=PROVENANCE), ["exposure"], out)


def test_scenario_maps_mark_deactivated_units_and_candidates(tmp_path):
    record = {
        "deactivated_sampling_units": ["vp_b"],
        "candidates": [
            {"viewpoint_id": "candidate_001", "x": 500045.0, "y": 4099975.0,
             "crs": "EPSG:32633", "included": True},
            {"viewpoint_id": "candidate_002", "x": 500015.0, "y": 4099975.0,
             "crs": "EPSG:32633", "included": False},
        ],
    }
    context, _, _ = _context(tmp_path, scenario_record=record)
    ax = _map_axes(_build(context, "scenario_change"))
    gids = {c.get_gid(): c for c in ax.collections}
    assert len(gids["rivelero_candidates"].get_offsets()) == 1  # only the included one
    assert len(gids["rivelero_deactivated"].get_offsets()) == 1
    assert "Included candidate Viewpoint" in [t.get_text() for t in ax.get_legend().get_texts()]
    plain = _map_axes(_build(context, "exposure"))
    assert "rivelero_candidates" not in {c.get_gid() for c in plain.collections}
