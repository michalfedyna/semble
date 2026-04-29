from pathlib import Path
from unittest.mock import MagicMock, patch

from semble.cache import DiskIndexCache
from semble.types import IndexStats


def _fake_index(marker: str) -> MagicMock:
    index = MagicMock()
    index.stats = IndexStats(indexed_files=1, total_chunks=1, languages={"python": 1})
    index.to_artifact.return_value = {"marker": marker}
    return index


def test_cache_builds_then_loads(tmp_path: Path) -> None:
    """DiskIndexCache builds once, then loads the persisted artifact."""
    source = tmp_path / "project"
    source.mkdir()
    (source / "a.py").write_text("def a(): pass\n")
    built = _fake_index("built")
    loaded = _fake_index("loaded")

    cache = DiskIndexCache(tmp_path / "cache")
    with (
        patch("semble.cache.load_model", return_value=MagicMock()),
        patch("semble.cache.SembleIndex.from_path", return_value=built) as mock_build,
        patch("semble.cache.SembleIndex.from_artifact", return_value=loaded) as mock_load,
    ):
        assert cache.get_or_build(source) is built
        assert cache.get_or_build(source) is loaded

    mock_build.assert_called_once()
    mock_load.assert_called_once()


def test_cache_rebuilds_when_fingerprint_changes(tmp_path: Path) -> None:
    """A changed local file fingerprint creates a fresh cache entry."""
    source = tmp_path / "project"
    source.mkdir()
    file_path = source / "a.py"
    file_path.write_text("def a(): pass\n")

    cache = DiskIndexCache(tmp_path / "cache")
    with (
        patch("semble.cache.load_model", return_value=MagicMock()),
        patch(
            "semble.cache.SembleIndex.from_path",
            side_effect=[_fake_index("first"), _fake_index("second")],
        ) as mock_build,
    ):
        cache.get_or_build(source)
        file_path.write_text("def a():\n    return 1\n")
        cache.get_or_build(source)

    assert mock_build.call_count == 2


def test_cache_clear_path(tmp_path: Path) -> None:
    """clear(path) removes matching cache entries."""
    source = tmp_path / "project"
    source.mkdir()
    (source / "a.py").write_text("def a(): pass\n")
    cache = DiskIndexCache(tmp_path / "cache")

    with patch("semble.cache.load_model", return_value=MagicMock()), patch(
        "semble.cache.SembleIndex.from_path", return_value=_fake_index("built")
    ):
        cache.get_or_build(source)

    assert cache.clear(source) == 1
    assert cache.stats()["entries"] == 0


def test_cache_lock_waits_for_existing_build(tmp_path: Path) -> None:
    """A second cache instance can load a completed entry instead of rebuilding."""
    source = tmp_path / "project"
    source.mkdir()
    (source / "a.py").write_text("def a(): pass\n")
    cache_dir = tmp_path / "cache"

    with patch("semble.cache.load_model", return_value=MagicMock()), patch(
        "semble.cache.SembleIndex.from_path", return_value=_fake_index("built")
    ) as mock_build:
        DiskIndexCache(cache_dir).get_or_build(source)

    with patch("semble.cache.load_model", return_value=MagicMock()), patch(
        "semble.cache.SembleIndex.from_artifact", return_value=_fake_index("loaded")
    ):
        DiskIndexCache(cache_dir).get_or_build(source)

    mock_build.assert_called_once()
