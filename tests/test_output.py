from semble.output import format_results
from semble.types import SearchMode, SearchResult
from tests.conftest import make_chunk


def test_format_results() -> None:
    """format_results renders numbered fenced markdown blocks with scores."""
    chunks = [make_chunk(f"def fn_{i}(): pass", f"f{i}.py") for i in range(2)]
    results = [SearchResult(chunk=c, score=0.1 * (i + 1), source=SearchMode.HYBRID) for i, c in enumerate(chunks)]

    out = format_results("Results", results)

    assert "Results" in out
    assert "## 1. f0.py:1-1  [score=0.100]" in out
    assert "## 2. f1.py:1-1  [score=0.200]" in out
    assert out.count("```") == 4
    assert "def fn_0(): pass" in out


def test_format_results_empty() -> None:
    """Empty results render only the header."""
    out = format_results("Nothing", [])
    assert out == "Nothing\n"
