"""Regression tests for the plant placement model used by Experiment 2."""
import numpy as np

from rivelero.experiments.plants import sample_plants


def draw(seed, contrast=5):
    return sample_plants((31, 47), 100, (5, 5), 40, contrast, np.random.default_rng(seed))


def test_reproducible_fresh_placements_without_replacement():
    first = draw(47)
    np.testing.assert_array_equal(first, draw(47))
    assert not np.array_equal(first, draw(48))
    assert len(np.unique(first, axis=0)) == 100
    assert np.all(first >= 0) and np.all(first < (31, 47))


def test_contrast_changes_sampling_and_full_grid_is_supported():
    assert not np.array_equal(draw(47, 0), draw(47, 8))
    cells = sample_plants((3, 2), 6, (5, 5), 2, 5, np.random.default_rng(0))
    assert set(map(tuple, cells)) == set(np.ndindex((3, 2)))


def test_unreal_othonna_readback_and_scoring(tmp_path, monkeypatch):
    import json
    import sys
    from pathlib import Path
    from types import SimpleNamespace as NS

    import pytest
    from rivelero.experiments.studies import Settings, score_plants, write_planting_list

    cfg = dict(batch="test", shape=[30, 40], east_step=5, south_step=5,
               west_edge=600000, north_edge=4100000)
    (tmp_path / "batch.json").write_text(json.dumps(cfg))
    np.full(cfg["shape"], 11.5, dtype="<f4").tofile(tmp_path / "terrain.f32")
    settings = Settings(2, tmp_path, 2, 3, 42, 150, .95, plants=5)
    write_planting_list(tmp_path, settings, cfg, lambda: None)
    scene = []
    offset = 0

    class Mesh:
        pass

    mesh = Mesh()

    class Actor:
        def __init__(self, pos):
            self.pos = pos

        def set_actor_label(self, label):
            self.label = label

        def get_actor_label(self):
            return self.label

        def get_component_by_class(self, cls):
            return NS(get_world_location=lambda: self.pos,
                      get_editor_property=lambda name: mesh)

    def spawn(asset, pos, transient):
        pos.x += offset
        actor = Actor(pos)
        scene.append(actor)
        return actor

    subsystem = NS(spawn_actor_from_object=spawn, get_all_level_actors=lambda: list(scene),
                   destroy_actor=scene.remove)
    unreal = NS(StaticMesh=Mesh, load_asset=lambda path: mesh,
                get_editor_subsystem=lambda cls: subsystem, EditorActorSubsystem=object,
                StaticMeshComponent=object, SystemLibrary=NS(line_trace_single=lambda *args: object()),
                TraceTypeQuery=NS(TRACE_TYPE_QUERY1=1), DrawDebugTrace=NS(NONE=None),
                Vector=lambda x,y,z: NS(x=x,y=y,z=z))
    monkeypatch.setitem(sys.modules, "unreal", unreal)
    import runpy
    validator = runpy.run_path(str(Path(__file__).parents[1] / "src/Camera/validate_othonna.py"))["validate_plants"]

    def verify():
        return list(validator(tmp_path, (500, -200), 50, "mesh", vars_settings))

    vars_settings = {"replicates": settings.replicates, "plants": settings.plants}
    assert len(verify()) == 3
    assert not scene
    visible = np.ones(cfg["shape"], bool)
    assert [r["detected"] for r in score_plants(tmp_path, settings, visible, lambda: None)] == [5]*3
    offset = 6
    with pytest.raises(RuntimeError, match="placement"):
        verify()
    assert not scene
    with pytest.raises(RuntimeError, match="verify every expected"):
        score_plants(tmp_path, settings, visible, lambda: None)
