from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from semble.index import SembleIndex
from semble.index.dense import DEFAULT_MODEL_NAME, load_model
from semble.index.file_walker import filter_extensions, walk_files
from semble.source import is_git_url, normalize_local_path
from semble.types import Encoder
from semble.version import __version__

_CACHE_FORMAT_VERSION = 1
_CHUNKER_FORMAT_VERSION = 1
_LOCK_STALE_SECONDS = 300


class DiskIndexCache:
    """Persistent Semble index cache safe for concurrent CLI processes."""

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        """Create a disk cache rooted at cache_dir or SEMBLE_CACHE_DIR."""
        root = cache_dir or os.environ.get("SEMBLE_CACHE_DIR") or Path.home() / ".cache" / "semble"
        self.root = Path(root).expanduser().resolve()
        self.indexes_dir = self.root / "indexes"
        self.locks_dir = self.root / "locks"
        self.indexes_dir.mkdir(parents=True, exist_ok=True)
        self.locks_dir.mkdir(parents=True, exist_ok=True)

    def get_or_build(self, source: str | Path, *, ref: str | None = None, model: Encoder | None = None) -> SembleIndex:
        """Return a fresh index for source, rebuilding when the cache is missing or stale."""
        model = model or load_model()
        descriptor = self._describe_source(str(source), ref=ref)
        entry_dir = self.indexes_dir / descriptor["key"]
        loaded = self._try_load(entry_dir, descriptor, model)
        if loaded is not None:
            return loaded

        with self._lock(descriptor["key"]):
            loaded = self._try_load(entry_dir, descriptor, model)
            if loaded is not None:
                return loaded
            index = self._build(descriptor, model)
            self._write(entry_dir, descriptor, index)
            return index

    def stats(self, source: str | Path | None = None, *, ref: str | None = None) -> dict[str, Any]:
        """Return cache statistics, optionally scoped to a source path or git URL."""
        entries = [p for p in self.indexes_dir.iterdir() if p.is_dir()]
        if source is None:
            return {"entries": len(entries), "cache_dir": str(self.root)}

        descriptor = self._describe_source(str(source), ref=ref)
        entry_dir = self.indexes_dir / descriptor["key"]
        metadata = self._read_metadata(entry_dir)
        if metadata is None:
            metadata = next(self._matching_source_metadata(descriptor), None)
        return {
            "cache_dir": str(self.root),
            "key": descriptor["key"],
            "path": descriptor.get("source"),
            "exists": metadata is not None,
            "fresh": metadata is not None and self._metadata_matches(metadata, descriptor),
            "metadata": metadata,
        }

    def clear(self, source: str | Path | None = None, *, ref: str | None = None) -> int:
        """Remove cache entries, returning the number removed."""
        if source is None:
            count = 0
            for entry in list(self.indexes_dir.iterdir()):
                if entry.is_dir():
                    shutil.rmtree(entry, ignore_errors=True)
                    count += 1
            return count

        descriptor = self._describe_source(str(source), ref=ref)
        count = 0
        for entry in list(self.indexes_dir.iterdir()):
            metadata = self._read_metadata(entry)
            if metadata and self._same_source(metadata, descriptor):
                shutil.rmtree(entry, ignore_errors=True)
                count += 1
        return count

    def _describe_source(self, source: str, *, ref: str | None) -> dict[str, Any]:
        config_hash = self._index_config_hash()
        if is_git_url(source):
            commit = self._resolve_remote_commit(source, ref)
            raw_key = f"git:{source}:{commit}:{config_hash}"
            return {
                "key": _sha256(raw_key),
                "source_type": "git",
                "source": source,
                "git_url": source,
                "git_ref": ref,
                "git_commit": commit,
                "fingerprint": commit,
                "config_hash": config_hash,
            }

        path = normalize_local_path(source)
        fingerprint = self._local_fingerprint(path)
        raw_key = f"local:{path}:{fingerprint}:{config_hash}"
        return {
            "key": _sha256(raw_key),
            "source_type": "local",
            "source": str(path),
            "git_url": None,
            "git_ref": None,
            "git_commit": self._local_head(path),
            "fingerprint": fingerprint,
            "config_hash": config_hash,
        }

    def _build(self, descriptor: dict[str, Any], model: Encoder) -> SembleIndex:
        if descriptor["source_type"] == "git":
            return SembleIndex.from_git(descriptor["git_url"], ref=descriptor["git_ref"], model=model)
        return SembleIndex.from_path(descriptor["source"], model=model)

    def _write(self, entry_dir: Path, descriptor: dict[str, Any], index: SembleIndex) -> None:
        tmp_dir = Path(tempfile.mkdtemp(prefix=f"{entry_dir.name}.", dir=self.indexes_dir))
        try:
            metadata = {
                "version": _CACHE_FORMAT_VERSION,
                "semble_version": __version__,
                "source_type": descriptor["source_type"],
                "source": descriptor["source"],
                "git_url": descriptor["git_url"],
                "git_ref": descriptor["git_ref"],
                "git_commit": descriptor["git_commit"],
                "fingerprint": descriptor["fingerprint"],
                "config_hash": descriptor["config_hash"],
                "model": DEFAULT_MODEL_NAME,
                "index_format": _CACHE_FORMAT_VERSION,
                "chunker_format": _CHUNKER_FORMAT_VERSION,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "stats": asdict(index.stats),
            }
            with (tmp_dir / "index.pkl").open("wb") as f:
                pickle.dump(index.to_artifact(), f, protocol=pickle.HIGHEST_PROTOCOL)
            (tmp_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
            if entry_dir.exists():
                shutil.rmtree(entry_dir, ignore_errors=True)
            tmp_dir.replace(entry_dir)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

    def _try_load(self, entry_dir: Path, descriptor: dict[str, Any], model: Encoder) -> SembleIndex | None:
        metadata = self._read_metadata(entry_dir)
        if metadata is None or not self._metadata_matches(metadata, descriptor):
            return None
        try:
            with (entry_dir / "index.pkl").open("rb") as f:
                artifact = pickle.load(f)
            return SembleIndex.from_artifact(artifact, model)
        except Exception:
            shutil.rmtree(entry_dir, ignore_errors=True)
            return None

    def _read_metadata(self, entry_dir: Path) -> dict[str, Any] | None:
        try:
            with (entry_dir / "metadata.json").open(encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def _metadata_matches(self, metadata: dict[str, Any], descriptor: dict[str, Any]) -> bool:
        return (
            metadata.get("version") == _CACHE_FORMAT_VERSION
            and metadata.get("config_hash") == descriptor["config_hash"]
            and metadata.get("source_type") == descriptor["source_type"]
            and metadata.get("source") == descriptor["source"]
            and metadata.get("fingerprint") == descriptor["fingerprint"]
        )

    def _same_source(self, metadata: dict[str, Any], descriptor: dict[str, Any]) -> bool:
        return (
            metadata.get("source_type") == descriptor["source_type"]
            and metadata.get("source") == descriptor["source"]
        )

    def _matching_source_metadata(self, descriptor: dict[str, Any]) -> Any:
        for entry in self.indexes_dir.iterdir():
            metadata = self._read_metadata(entry)
            if metadata is not None and self._same_source(metadata, descriptor):
                yield metadata

    def _lock(self, key: str) -> _CacheLock:
        return _CacheLock(self.locks_dir / f"{key}.lock")

    def _index_config_hash(self) -> str:
        return _sha256(f"{_CACHE_FORMAT_VERSION}:{_CHUNKER_FORMAT_VERSION}:{__version__}:{DEFAULT_MODEL_NAME}")

    def _resolve_remote_commit(self, url: str, ref: str | None) -> str:
        target = ref or "HEAD"
        try:
            result = subprocess.run(
                ["git", "ls-remote", url, target],
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            return target
        if result.returncode != 0:
            return target
        first = result.stdout.splitlines()[0].split() if result.stdout.splitlines() else []
        return first[0] if first else target

    def _local_head(self, path: Path) -> str | None:
        result = _git(path, "rev-parse", "HEAD")
        return result if result else None

    def _local_fingerprint(self, path: Path) -> str:
        head = self._local_head(path)
        if head is not None:
            status = _git(path, "status", "--porcelain") or ""
            changed = self._changed_path_stats(path, status)
            return _sha256(f"git:{head}\n{status}\n{changed}")
        return self._directory_fingerprint(path)

    def _changed_path_stats(self, root: Path, status: str) -> str:
        lines = []
        for raw in status.splitlines():
            rel = raw[3:]
            if " -> " in rel:
                rel = rel.split(" -> ", 1)[1]
            target = root / rel
            try:
                stat = target.stat()
            except OSError:
                lines.append(f"{rel}:missing")
            else:
                lines.append(f"{rel}:{stat.st_mtime_ns}:{stat.st_size}")
        return "\n".join(sorted(lines))

    def _directory_fingerprint(self, path: Path) -> str:
        extensions = filter_extensions(None, include_text_files=False)
        rows = []
        for file_path in walk_files(path, extensions, None):
            try:
                stat = file_path.stat()
            except OSError:
                continue
            rows.append(f"{file_path.relative_to(path)}:{stat.st_mtime_ns}:{stat.st_size}")
        return _sha256("\n".join(sorted(rows)))


class _CacheLock:
    def __init__(self, path: Path) -> None:
        self.path = path

    def __enter__(self) -> _CacheLock:
        while True:
            try:
                self.path.mkdir(parents=True)
                (self.path / "owner.json").write_text(
                    json.dumps({"pid": os.getpid(), "created_at": time.time()}),
                    encoding="utf-8",
                )
                return self
            except FileExistsError:
                if self._is_stale():
                    shutil.rmtree(self.path, ignore_errors=True)
                    continue
                time.sleep(0.05)

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        shutil.rmtree(self.path, ignore_errors=True)

    def _is_stale(self) -> bool:
        try:
            created_at = json.loads((self.path / "owner.json").read_text(encoding="utf-8")).get("created_at", 0)
        except (OSError, json.JSONDecodeError):
            created_at = 0
        return time.time() - float(created_at) > _LOCK_STALE_SECONDS


def _git(path: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), *args],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
