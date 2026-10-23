"""Storage-retention tests for debug frames and local optimizer checkpoints."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import dispatchevolve.optimizer.genetic.controller as controller
from dispatchevolve.workflows.dispatchevolve_v2.orchestrator import _load_debug_frame


def test_prune_complete_checkpoints_keeps_latest_and_incomplete(
    tmp_path: Path, monkeypatch,
) -> None:
    root = tmp_path / "checkpoints"
    complete = []
    for iteration in (1, 2, 10):
        path = root / f"checkpoint_{iteration}"
        path.mkdir(parents=True)
        (path / "payload").write_text(str(iteration), encoding="utf-8")
        complete.append(path)
    incomplete = root / "checkpoint_11"
    incomplete.mkdir()
    monkeypatch.setattr(controller, "is_complete_checkpoint", lambda path: path != incomplete)

    retained = controller.prune_complete_checkpoints(root)

    assert retained == (complete[-1],)
    assert not complete[0].exists()
    assert not complete[1].exists()
    assert complete[-1].is_dir()
    assert incomplete.is_dir()


def test_debug_frame_is_not_persisted_by_default(tmp_path: Path) -> None:
    data_path = tmp_path / "debug.csv"
    pd.DataFrame({"batch_id": [1, 2], "value": [3, 4]}).to_csv(data_path, index=False)
    config = SimpleNamespace(
        data_path=data_path,
        cache_root=tmp_path / "cache",
        debug_max_batches=None,
        csv_chunk_size=1,
        persist_debug_frames=False,
    )

    frame = _load_debug_frame(config)

    assert frame["value"].tolist() == [3, 4]
    assert not (config.cache_root / "debug_frames").exists()


def test_debug_frame_persistence_can_be_enabled(tmp_path: Path) -> None:
    data_path = tmp_path / "debug.csv"
    pd.DataFrame({"batch_id": [1], "value": [3]}).to_csv(data_path, index=False)
    config = SimpleNamespace(
        data_path=data_path,
        cache_root=tmp_path / "cache",
        debug_max_batches=None,
        csv_chunk_size=1,
        persist_debug_frames=True,
    )

    _load_debug_frame(config)

    assert len(list((config.cache_root / "debug_frames").glob("*.pkl"))) == 1
