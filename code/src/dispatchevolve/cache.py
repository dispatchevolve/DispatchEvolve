"""Unified content-addressed cache key and cache-root helpers."""
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

from dispatchevolve.repo_paths import find_shared_root


def _file_sha256(path: Union[str, Path]) -> str:
    """SHA256 hex digest of file contents."""
    path = Path(path)
    if not path.exists():
        return "nonexistent"
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class CacheKey:
    """Deterministic cache key derived from content fingerprints.

    Replaces manual CACHE_VERSION constants with automatic content-based
    invalidation. When any fingerprint input changes, the key changes,
    and old cached data is naturally invalidated.
    """

    def __init__(self, domain: str, **fingerprints: Any):
        self.domain = domain
        self.fingerprints = fingerprints

    def digest(self) -> str:
        payload = json.dumps(
            {"domain": self.domain, "fingerprints": self.fingerprints},
            sort_keys=True, default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def __str__(self) -> str:
        return f"{self.domain}/{self.digest()}"

    def __repr__(self) -> str:
        return f"CacheKey(domain={self.domain!r}, digest={self.digest()!r})"


def experiment_cache_root(start_path: Union[str, Path, None] = None) -> Path:
    """Root for all persistent experiment caches."""
    start = Path(start_path).resolve() if start_path is not None else Path(__file__).resolve()
    try:
        shared_root = find_shared_root(start)
    except RuntimeError:
        shared_root = find_shared_root(Path(__file__).resolve())
    return shared_root / ".cache" / "experiments"


def cache_path_component(raw_value: object, *, default: str = "cache") -> str:
    component = re.sub(r"[^A-Za-z0-9]+", "_", str(raw_value)).strip("_").lower()
    return component or default


def experiment_cache_dir(
    *components: object,
    start_path: Union[str, Path, None] = None,
) -> Path:
    root = experiment_cache_root(start_path)
    safe_components = [cache_path_component(component) for component in components]
    return root.joinpath(*safe_components)


def require_experiment_cache_path(
    path: Union[str, Path],
    *,
    purpose: str = "cache",
    start_path: Union[str, Path, None] = None,
) -> Path:
    resolved = Path(path).expanduser().resolve()
    cache_root = experiment_cache_root(start_path).resolve()
    try:
        resolved.relative_to(cache_root)
    except ValueError as exc:
        raise ValueError(
            f"{purpose} must be stored under {cache_root}; got {resolved}"
        ) from exc
    return resolved
