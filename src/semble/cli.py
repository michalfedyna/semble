from __future__ import annotations

import argparse
import json
import sys

from semble.cache import DiskIndexCache
from semble.locations import resolve_chunk
from semble.output import format_results


def main(argv: list[str] | None = None) -> int:
    """Run the Semble command-line interface."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "search":
            return _search(args)
        if args.command == "related":
            return _related(args)
        if args.command == "cache":
            return _cache(args)
    except Exception as exc:
        sys.stderr.write(f"Semble failed: {exc}\n")
        return 1
    parser.print_help(sys.stderr)
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="semble", description="Fast local code search for agents.")
    subparsers = parser.add_subparsers(dest="command")

    search = subparsers.add_parser("search", help="Search code and return markdown snippets.")
    search.add_argument("query", help="Natural-language or code query.")
    search.add_argument("--path", default=".", help="Local path or git URL to search. Defaults to cwd.")
    search.add_argument("--mode", choices=["hybrid", "semantic", "bm25"], default="hybrid")
    search.add_argument("--top-k", type=int, default=5, help="Number of snippets to return.")

    related = subparsers.add_parser("related", help="Find snippets related to a file and line.")
    related.add_argument("location", nargs="?", help="Location in FILE:LINE form.")
    related.add_argument("--file", dest="file_path", help="File path from a search result.")
    related.add_argument("--line", type=int, help="Line number in the indexed file.")
    related.add_argument("--path", default=".", help="Local path or git URL to search. Defaults to cwd.")
    related.add_argument("--top-k", type=int, default=5, help="Number of snippets to return.")

    cache = subparsers.add_parser("cache", help="Inspect or clear the on-disk cache.")
    cache_subparsers = cache.add_subparsers(dest="cache_command", required=True)
    stats = cache_subparsers.add_parser("stats", help="Show cache stats.")
    stats.add_argument("--path", default=None, help="Optional local path or git URL to inspect.")
    clear = cache_subparsers.add_parser("clear", help="Clear cache entries.")
    clear.add_argument("--path", default=None, help="Optional local path or git URL to clear.")

    return parser


def _search(args: argparse.Namespace) -> int:
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")
    index = DiskIndexCache().get_or_build(args.path)
    results = index.search(args.query, top_k=args.top_k, mode=args.mode)
    if not results:
        sys.stdout.write("No results found.\n")
        return 0
    sys.stdout.write(format_results(f"Search results for: {args.query!r} (mode={args.mode})", results) + "\n")
    return 0


def _related(args: argparse.Namespace) -> int:
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")
    file_path, line = _parse_location(args)
    index = DiskIndexCache().get_or_build(args.path)
    chunk = resolve_chunk(index.chunks, file_path, line)
    if chunk is None:
        sys.stderr.write(
            f"No chunk found at {file_path}:{line}. "
            "Make sure the file is indexed and the line number is within a known chunk.\n"
        )
        return 1
    results = index.find_related(chunk, top_k=args.top_k)
    if not results:
        sys.stdout.write(f"No related chunks found for {file_path}:{line}.\n")
        return 0
    sys.stdout.write(format_results(f"Chunks related to {file_path}:{line}", results) + "\n")
    return 0


def _cache(args: argparse.Namespace) -> int:
    cache = DiskIndexCache()
    if args.cache_command == "stats":
        stats = cache.stats(args.path) if args.path else cache.stats()
        sys.stdout.write(json.dumps(stats, indent=2, sort_keys=True) + "\n")
        return 0
    if args.cache_command == "clear":
        count = cache.clear(args.path) if args.path else cache.clear()
        sys.stdout.write(f"Cleared {count} cache entr{'y' if count == 1 else 'ies'}.\n")
        return 0
    return 2


def _parse_location(args: argparse.Namespace) -> tuple[str, int]:
    if args.file_path and args.line:
        return args.file_path, args.line
    if args.location:
        file_path, sep, line_s = args.location.rpartition(":")
        if sep and file_path and line_s.isdigit():
            return file_path, int(line_s)
    raise ValueError("related requires FILE:LINE or --file FILE --line LINE")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
