from __future__ import annotations

import re
import subprocess
from pathlib import Path

_GIT_URL_SCHEMES = ("https://", "http://", "ssh://", "git://", "git+ssh://", "file://")
_SCP_GIT_URL_RE = re.compile(r"^[\w.-]+@[\w.-]+:(?!/)")


def is_git_url(path: str) -> bool:
    """Return True if path looks like a git URL rather than a local path."""
    return path.startswith(_GIT_URL_SCHEMES) or _SCP_GIT_URL_RE.match(path) is not None


def git_root(path: Path) -> Path | None:
    """Return the git worktree root containing path, or None if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    return Path(result.stdout.strip()).resolve()


def normalize_local_path(path: str | Path) -> Path:
    """Resolve a local source path, using the worktree root when inside git."""
    resolved = Path(path).expanduser().resolve()
    return git_root(resolved) or resolved
