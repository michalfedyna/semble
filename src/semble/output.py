from semble.types import SearchResult


def format_results(header: str, results: list[SearchResult]) -> str:
    """Render search results as numbered markdown code blocks."""
    lines: list[str] = [header, ""]
    for i, result in enumerate(results, 1):
        lines.append(f"## {i}. {result.chunk.location}  [score={result.score:.3f}]")
        lines.append("```")
        lines.append(result.chunk.content.strip())
        lines.append("```")
        lines.append("")
    return "\n".join(lines)
