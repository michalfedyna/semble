from pathlib import Path

import pytest

from semble.source import is_git_url, normalize_local_path


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("https://github.com/org/repo", True),
        ("http://github.com/org/repo", True),
        ("git://github.com/org/repo", True),
        ("ssh://git@github.com/org/repo", True),
        ("git+ssh://git@github.com/org/repo", True),
        ("file:///tmp/repo", True),
        ("git@github.com:org/repo", True),
        ("/local/path/to/repo", False),
        ("./relative/path", False),
        ("repo_name", False),
    ],
)
def test_is_git_url(path: str, expected: bool) -> None:
    """Git-like URLs are detected while local paths are not."""
    assert is_git_url(path) is expected


def test_normalize_local_path_non_git(tmp_path: Path) -> None:
    """Non-git paths normalize to their resolved path."""
    assert normalize_local_path(tmp_path) == tmp_path.resolve()
