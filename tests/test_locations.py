from semble.locations import resolve_chunk
from tests.conftest import make_chunk


def test_resolve_chunk() -> None:
    """resolve_chunk returns the containing chunk and handles misses."""
    interior = make_chunk("line1\nline2\nline3", "src/a.py")
    boundary = make_chunk("last line", "src/a.py")

    assert resolve_chunk([interior], "src/a.py", 2) is interior
    assert resolve_chunk([boundary], "src/a.py", 1) is boundary
    assert resolve_chunk([interior], "src/other.py", 1) is None
    assert resolve_chunk([interior], "src/a.py", 99) is None
