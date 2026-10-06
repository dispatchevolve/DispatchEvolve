"""Canonical repository genomes and immutable content-addressed snapshots."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Mapping

from .contracts import CandidateProposal, EngineCandidate
from .mutations import (
    CandidateValidationError,
    MAX_EDITABLE_FILE_BYTES,
    MAX_EDITABLE_FILE_COUNT,
    MAX_EDITABLE_TOTAL_BYTES,
    REPOSITORY_PATCH_FINGERPRINT,
    escape_repository_source,
    extract_repository_patch,
    normalize_python_source,
    parse_repository_patch,
    validate_editable_path,
    validate_python_source,
)


_CANDIDATE_ID = re.compile(r"^[0-9a-f]{64}$")
_HASH_DOMAIN = b"dispatchevolve-repository-genome-v1\0"
REPOSITORY_GENOME_PROTOCOL_VERSION = "repository-genome-v3"
CANDIDATE_TRANSPORT_FINGERPRINT = hashlib.sha256(
    (
        f"{REPOSITORY_GENOME_PROTOCOL_VERSION}:"
        f"patch={REPOSITORY_PATCH_FINGERPRINT}:"
        f"file={MAX_EDITABLE_FILE_BYTES}:total={MAX_EDITABLE_TOTAL_BYTES}:"
        f"count={MAX_EDITABLE_FILE_COUNT}"
    ).encode("ascii")
).hexdigest()


def _checked_source_size(path: str, source: str, running_total: int) -> int:
    source_bytes = len(source.encode("utf-8"))
    if source_bytes > MAX_EDITABLE_FILE_BYTES:
        raise CandidateValidationError(f"{path} exceeds the per-file size limit")
    total = running_total + source_bytes
    if total > MAX_EDITABLE_TOTAL_BYTES:
        raise CandidateValidationError(
            "candidate repository exceeds the editable total size limit"
        )
    return total


def _reject_symlinks(root: Path) -> None:
    if root.is_symlink():
        raise CandidateValidationError(f"candidate root cannot be a symlink: {root}")
    for directory, dir_names, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in (*dir_names, *file_names):
            path = directory_path / name
            if path.is_symlink():
                raise CandidateValidationError(
                    f"candidate source cannot contain a symlink: {path}"
                )


def _read_utf8_source(path: Path, relative_path: str) -> str:
    if path.is_symlink():
        raise CandidateValidationError(f"candidate source cannot be a symlink: {path}")
    data = path.read_bytes()
    if b"\x00" in data:
        raise CandidateValidationError(f"{relative_path} contains a binary NUL byte")
    try:
        source = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise CandidateValidationError(
            f"{relative_path} must be valid UTF-8"
        ) from exc
    return normalize_python_source(source)


def _collect_python_files(engine_dir: Path) -> dict[str, str]:
    root = Path(engine_dir)
    if not root.is_dir():
        raise CandidateValidationError(f"candidate root is not a directory: {root}")
    _reject_symlinks(root)

    engine_path = root / "engine.py"
    if not engine_path.is_file():
        raise CandidateValidationError("candidate must contain engine.py")

    selected = [engine_path]
    policies_dir = root / "policies"
    if policies_dir.exists():
        if not policies_dir.is_dir():
            raise CandidateValidationError("policies must be a directory")
        selected.extend(
            path
            for path in policies_dir.rglob("*.py")
            if path.is_file()
        )

    files: dict[str, str] = {}
    total_bytes = 0
    for path in selected:
        relative_path = path.relative_to(root).as_posix()
        canonical_path = validate_editable_path(relative_path)
        source = _read_utf8_source(path, canonical_path)
        total_bytes = _checked_source_size(canonical_path, source, total_bytes)
        validate_python_source(canonical_path, source)
        files[canonical_path] = source
        if len(files) > MAX_EDITABLE_FILE_COUNT:
            raise CandidateValidationError(
                "candidate repository exceeds the editable file count limit"
            )
    return dict(sorted(files.items()))


def _validate_files(files: Mapping[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    total_bytes = 0
    for raw_path, raw_source in files.items():
        path = validate_editable_path(raw_path)
        source = normalize_python_source(raw_source)
        total_bytes = _checked_source_size(path, source, total_bytes)
        validate_python_source(path, source)
        normalized[path] = source
        if len(normalized) > MAX_EDITABLE_FILE_COUNT:
            raise CandidateValidationError(
                "candidate repository exceeds the editable file count limit"
            )
    if "engine.py" not in normalized:
        raise CandidateValidationError("candidate must contain engine.py")
    return dict(sorted(normalized.items()))


def _candidate_id_from_files(files: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    digest.update(_HASH_DOMAIN)
    for path, source in sorted(files.items()):
        path_bytes = path.encode("utf-8")
        source_bytes = source.encode("utf-8")
        digest.update(len(path_bytes).to_bytes(8, byteorder="big"))
        digest.update(path_bytes)
        digest.update(len(source_bytes).to_bytes(8, byteorder="big"))
        digest.update(source_bytes)
    return digest.hexdigest()


def _write_code_files(destination: Path, files: Mapping[str, str]) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for relative_path, source in sorted(files.items()):
        output_path = destination / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(source, encoding="utf-8", newline="\n")


def _make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_file():
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        elif path.is_dir():
            path.chmod(
                stat.S_IRUSR
                | stat.S_IXUSR
                | stat.S_IRGRP
                | stat.S_IXGRP
                | stat.S_IROTH
                | stat.S_IXOTH
            )
    root.chmod(
        stat.S_IRUSR
        | stat.S_IXUSR
        | stat.S_IRGRP
        | stat.S_IXGRP
        | stat.S_IROTH
        | stat.S_IXOTH
    )


def _remove_read_only_tree(root: Path) -> None:
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_dir():
            path.chmod(stat.S_IRWXU)
        else:
            path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    root.chmod(stat.S_IRWXU)
    shutil.rmtree(root)


class RepositoryGenomeCodec:
    """Encode and hash the editable subset of one dispatch engine repository."""

    def encode(self, engine_dir: Path) -> str:
        blocks: list[str] = []
        for path, source in _collect_python_files(engine_dir).items():
            blocks.extend(
                (
                    f"<<<FILE {path}>>>\n",
                    escape_repository_source(source),
                    "<<<END FILE>>>\n",
                )
            )
        return "".join(blocks)

    def decode(self, genome: str | bytes, destination: Path) -> EngineCandidate:
        patch = parse_repository_patch(genome)
        if any(mutation.content is None for mutation in patch.mutations):
            raise CandidateValidationError("a complete repository genome cannot delete files")
        files = _validate_files(
            {
                mutation.path: mutation.content
                for mutation in patch.mutations
                if mutation.content is not None
            }
        )
        destination = Path(destination)
        _write_code_files(destination, files)
        candidate_id = _candidate_id_from_files(files)
        return EngineCandidate(candidate_id, destination, (), 0, None)

    def candidate_id(self, engine_dir: Path) -> str:
        return _candidate_id_from_files(_collect_python_files(engine_dir))


class CandidateStore:
    """Materialize immutable repository snapshots under a caller-owned workspace.

    Source identity is content-only. Native parents and the transport base belong
    to the generation event returned by :meth:`materialize`, not to the shared
    snapshot, so two events can reuse one tree without inventing lineage.
    """

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)
        self.candidates_dir = self.workspace / "candidates"
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        self._codec = RepositoryGenomeCodec()

    def materialize_seed(self, engine_dir: Path) -> EngineCandidate:
        source_root = Path(engine_dir)
        files = _collect_python_files(source_root)
        readme = self._read_readme(source_root)
        candidate_id = _candidate_id_from_files(files)
        destination = self._ensure_snapshot(candidate_id, files, readme)
        return EngineCandidate(candidate_id, destination, (), 0, None)

    def materialize(
        self,
        proposal: CandidateProposal,
        *,
        base_candidate: EngineCandidate,
    ) -> EngineCandidate:
        if proposal.base_candidate_id != base_candidate.candidate_id:
            raise CandidateValidationError(
                "proposal base_candidate_id does not match the explicit base candidate"
            )

        actual_base_id = self._codec.candidate_id(base_candidate.engine_dir)
        if actual_base_id != base_candidate.candidate_id:
            raise CandidateValidationError(
                "explicit base candidate content does not match its candidate_id"
            )

        files = _collect_python_files(base_candidate.engine_dir)
        patch = extract_repository_patch(proposal.raw_text)
        updated = _validate_files(patch.apply(files))
        candidate_id = _candidate_id_from_files(updated)
        destination = self._ensure_snapshot(
            candidate_id,
            updated,
            self._read_readme(base_candidate.engine_dir),
        )
        return EngineCandidate(
            candidate_id=candidate_id,
            engine_dir=destination,
            parent_ids=proposal.parent_ids,
            native_step=proposal.native_step,
            base_candidate_id=proposal.base_candidate_id,
        )

    def get(self, candidate_id: str) -> EngineCandidate:
        if not _CANDIDATE_ID.fullmatch(candidate_id):
            raise KeyError(f"invalid candidate id: {candidate_id!r}")
        destination = self.candidates_dir / candidate_id
        if not destination.is_dir():
            raise KeyError(candidate_id)
        actual_id = self._codec.candidate_id(destination)
        if actual_id != candidate_id:
            raise CandidateValidationError(
                f"candidate snapshot {candidate_id} failed content verification"
            )
        return EngineCandidate(candidate_id, destination, (), 0, None)

    @staticmethod
    def _read_readme(root: Path) -> bytes | None:
        path = root / "README.md"
        if not path.exists():
            return None
        if path.is_symlink() or not path.is_file():
            raise CandidateValidationError("README.md must be a regular file")
        return path.read_bytes()

    def _ensure_snapshot(
        self,
        candidate_id: str,
        files: Mapping[str, str],
        readme: bytes | None,
    ) -> Path:
        destination = self.candidates_dir / candidate_id
        if destination.exists():
            actual_id = self._codec.candidate_id(destination)
            if actual_id != candidate_id:
                raise CandidateValidationError(
                    f"candidate snapshot collision for {candidate_id}"
                )
            return destination

        temporary = Path(
            tempfile.mkdtemp(prefix=f".{candidate_id[:12]}-", dir=self.candidates_dir)
        )
        # mkdtemp creates the root; _write_code_files deliberately owns root
        # creation, so remove the empty placeholder before filling it.
        temporary.rmdir()
        try:
            _write_code_files(temporary, files)
            if readme is not None:
                (temporary / "README.md").write_bytes(readme)
            _make_read_only(temporary)
            try:
                temporary.rename(destination)
            except FileExistsError:
                _remove_read_only_tree(temporary)
            except OSError:
                if destination.exists():
                    _remove_read_only_tree(temporary)
                else:
                    raise
        except Exception:
            _remove_read_only_tree(temporary)
            raise

        actual_id = self._codec.candidate_id(destination)
        if actual_id != candidate_id:
            raise CandidateValidationError(
                f"materialized candidate {candidate_id} failed content verification"
            )
        return destination


__all__ = [
    "CANDIDATE_TRANSPORT_FINGERPRINT",
    "CandidateStore",
    "REPOSITORY_GENOME_PROTOCOL_VERSION",
    "RepositoryGenomeCodec",
]
