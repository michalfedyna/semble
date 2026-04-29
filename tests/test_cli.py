from unittest.mock import MagicMock, patch

from semble import cli
from semble.types import SearchMode, SearchResult
from tests.conftest import make_chunk


def test_cli_search(capsys) -> None:
    """Search prints markdown snippets."""
    index = MagicMock()
    index.search.return_value = [SearchResult(make_chunk("def foo(): pass", "foo.py"), 0.9, SearchMode.HYBRID)]
    cache = MagicMock()
    cache.get_or_build.return_value = index

    with patch("semble.cli.DiskIndexCache", return_value=cache):
        assert cli.main(["search", "foo", "--path", ".", "--top-k", "1"]) == 0

    out = capsys.readouterr().out
    assert "Search results for" in out
    assert "foo.py:1-1" in out


def test_cli_related(capsys) -> None:
    """Related resolves FILE:LINE and prints markdown snippets."""
    chunk = make_chunk("def foo(): pass", "foo.py")
    index = MagicMock()
    index.chunks = [chunk]
    index.find_related.return_value = [SearchResult(make_chunk("def bar(): pass", "bar.py"), 0.8, SearchMode.SEMANTIC)]
    cache = MagicMock()
    cache.get_or_build.return_value = index

    with patch("semble.cli.DiskIndexCache", return_value=cache):
        assert cli.main(["related", "foo.py:1", "--path", "."]) == 0

    out = capsys.readouterr().out
    assert "Chunks related to foo.py:1" in out
    assert "bar.py:1-1" in out


def test_cli_related_missing_chunk(capsys) -> None:
    """Related exits non-zero when the target chunk is missing."""
    index = MagicMock()
    index.chunks = []
    cache = MagicMock()
    cache.get_or_build.return_value = index

    with patch("semble.cli.DiskIndexCache", return_value=cache):
        assert cli.main(["related", "foo.py:1", "--path", "."]) == 1

    assert "No chunk found" in capsys.readouterr().err


def test_cli_cache_stats(capsys) -> None:
    """Cache stats prints JSON."""
    cache = MagicMock()
    cache.stats.return_value = {"entries": 1}
    with patch("semble.cli.DiskIndexCache", return_value=cache):
        assert cli.main(["cache", "stats"]) == 0
    assert '"entries": 1' in capsys.readouterr().out
