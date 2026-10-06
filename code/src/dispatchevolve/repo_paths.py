"""Repository path helpers for code roots and shared data roots.

Code paths should resolve against the current checkout. Shared paths such as
data, outputs, logs, and outputs_sync should resolve against the parent
repository when code is running from an in-repo git worktree.
"""

from __future__ import annotations

from pathlib import Path


DEFAULT_SHARED_DIR_NAMES = ("data", "outputs", "logs", "outputs_sync")


def find_repo_root(start: Path | None = None) -> Path:
    """Return the current checkout root directory."""
    start_path = Path(__file__).resolve() if start is None else Path(start).resolve()
    for candidate in (start_path, *start_path.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError(f"Could not find repository root from {start}")


def worktree_parent_repo_root(repo_root: Path) -> Path | None:
    """Return the parent repository root when repo_root is under .worktree/."""
    parts = repo_root.resolve().parts
    if ".worktree" not in parts:
        return None
    worktree_index = parts.index(".worktree")
    if worktree_index <= 0:
        return None
    parent_root = Path(*parts[:worktree_index])
    if (
        (parent_root / "pyproject.toml").is_file()
        and (parent_root / "src").is_dir()
        and (parent_root / ".worktree").is_dir()
    ):
        return parent_root
    return None


def find_shared_root(start: Path | None = None) -> Path:
    """Return the root for shared directories.

    In a worktree under ``<repo>/.worktree/...``, this returns ``<repo>``.
    Otherwise it returns the current checkout root.
    """
    repo_root = find_repo_root(start)
    return worktree_parent_repo_root(repo_root) or repo_root


def remap_shared_path(
    path: str | Path,
    *,
    code_root: Path,
    shared_root: Path,
    shared_dir_names: tuple[str, ...] = DEFAULT_SHARED_DIR_NAMES,
) -> Path:
    """Map code-root shared-directory paths to shared-root paths."""
    path_value = Path(path).expanduser().resolve()
    code_root_value = Path(code_root).resolve()
    shared_root_value = Path(shared_root).resolve()
    if code_root_value == shared_root_value:
        return path_value

    for dirname in shared_dir_names:
        source_root = (code_root_value / dirname).resolve()
        try:
            relative_path = path_value.relative_to(source_root)
        except ValueError:
            continue
        return shared_root_value / dirname / relative_path
    return path_value


def shared_repo_path(
    code_root: Path,
    *parts: str,
    shared_root: Path | None = None,
) -> Path:
    """Return a path under the shared root for a known shared directory."""
    root = shared_root or find_shared_root(code_root)
    return root.joinpath(*parts)
