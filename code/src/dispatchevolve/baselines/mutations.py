"""Shared, non-JSON repository mutation transport for every baseline method."""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass
from pathlib import PurePosixPath


class MutationFormatError(ValueError):
    """Raised when a repository patch does not follow the wire format."""


class CandidateValidationError(ValueError):
    """Raised when candidate source violates a repository safety gate."""


@dataclass(frozen=True)
class FileMutation:
    """One full-file replacement/addition, or a deletion when content is None."""

    path: str
    content: str | None


@dataclass(frozen=True)
class RepositoryPatch:
    """An ordered collection of validated, non-overlapping file mutations."""

    mutations: tuple[FileMutation, ...]

    def apply(self, files: dict[str, str]) -> dict[str, str]:
        updated = dict(files)
        for mutation in self.mutations:
            if mutation.content is None:
                updated.pop(mutation.path, None)
            else:
                updated[mutation.path] = mutation.content
        if "engine.py" not in updated:
            raise CandidateValidationError("engine.py cannot be deleted")
        return updated


_FILE_START = re.compile(r"^<<<FILE ([^>\r\n]+)>>>$")
_DELETE = re.compile(r"^<<<DELETE ([^>\r\n]+)>>>$")
_END_FILE = "<<<END FILE>>>"

REPOSITORY_PATCH_PROTOCOL_VERSION = "full-file-markers-v3"
MAX_EDITABLE_FILE_BYTES = 2 * 1024 * 1024
MAX_EDITABLE_TOTAL_BYTES = 8 * 1024 * 1024
MAX_PATCH_RESPONSE_BYTES = (2 * MAX_EDITABLE_TOTAL_BYTES) + (1 * 1024 * 1024)
MAX_EDITABLE_FILE_COUNT = 1024
REPOSITORY_PATCH_FINGERPRINT = hashlib.sha256(
    (
        f"{REPOSITORY_PATCH_PROTOCOL_VERSION}:"
        f"file={MAX_EDITABLE_FILE_BYTES}:total={MAX_EDITABLE_TOTAL_BYTES}:"
        f"response={MAX_PATCH_RESPONSE_BYTES}:count={MAX_EDITABLE_FILE_COUNT}"
    ).encode("ascii")
).hexdigest()

FORBIDDEN_IMPORT_PREFIXES = (
    "data.reference",
    "dispatchevolve.baselines.methods",
    "dispatchevolve.baselines.registry",
    "dispatchevolve.optimizer.genetic",
)
FORBIDDEN_REFERENCE_LITERAL_FRAGMENTS = (
    "data/reference",
    "data.reference",
    "baseline_source_code",
)
_MAX_ANALYZED_LITERAL_CHARS = 4096


def normalize_python_source(source: str) -> str:
    """Return the canonical LF representation used by hashing and snapshots."""

    normalized = source.replace("\r\n", "\n").replace("\r", "\n")
    if normalized and not normalized.endswith("\n"):
        normalized += "\n"
    return normalized


def escape_repository_source(source: str) -> str:
    """Escape source lines that would collide with v3 transport markers.

    One leading backslash is added to every canonical source line beginning
    with either a backslash or ``<<<``. The parser removes exactly that one
    transport backslash, preserving arbitrary legal Python source losslessly.
    """

    normalized = normalize_python_source(source)
    return "".join(
        f"\\{line}" if line.startswith(("\\", "<<<")) else line
        for line in normalized.splitlines(keepends=True)
    )


def _unescape_repository_source_line(line: str) -> str:
    return line[1:] if line.startswith("\\") else line


def validate_editable_path(raw_path: str) -> str:
    """Validate and return one canonical repository-relative editable path."""

    if raw_path != raw_path.strip() or not raw_path:
        raise CandidateValidationError(f"invalid path: {raw_path!r}")
    if "\\" in raw_path or raw_path.startswith("/"):
        raise CandidateValidationError(f"invalid path: {raw_path!r}")
    if "//" in raw_path:
        raise CandidateValidationError(f"invalid path: {raw_path!r}")

    path = PurePosixPath(raw_path)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise CandidateValidationError(f"invalid path: {raw_path!r}")
    canonical = path.as_posix()
    editable = canonical == "engine.py" or (
        len(path.parts) >= 2
        and path.parts[0] == "policies"
        and path.suffix == ".py"
    )
    if not editable:
        raise CandidateValidationError(f"path is not editable: {raw_path!r}")
    return canonical


def _decode_patch(raw_text: str | bytes) -> str:
    if isinstance(raw_text, bytes):
        if len(raw_text) > MAX_PATCH_RESPONSE_BYTES:
            raise CandidateValidationError("repository response exceeds the size limit")
        try:
            raw_text = raw_text.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise MutationFormatError("repository patch must be valid UTF-8") from exc
    if not isinstance(raw_text, str):
        raise MutationFormatError("repository patch must be text or UTF-8 bytes")
    if len(raw_text.encode("utf-8")) > MAX_PATCH_RESPONSE_BYTES:
        raise CandidateValidationError("repository response exceeds the size limit")
    if "\x00" in raw_text:
        raise CandidateValidationError("repository patch contains a binary NUL byte")
    return raw_text.replace("\r\n", "\n").replace("\r", "\n")


def parse_repository_patch(raw_text: str | bytes) -> RepositoryPatch:
    """Parse the shared full-file replacement/delete transport.

    The protocol deliberately carries complete files rather than JSON or a
    repository patch format::

        <<<FILE engine.py>>>
        ...complete UTF-8 Python source...
        <<<END FILE>>>
        <<<DELETE policies/obsolete.py>>>
    """

    text = _decode_patch(raw_text)
    lines = text.splitlines(keepends=True)
    mutations: list[FileMutation] = []
    seen: set[str] = set()
    total_bytes = 0
    index = 0

    while index < len(lines):
        line = lines[index]
        marker = line.removesuffix("\n")
        if not marker.strip():
            index += 1
            continue

        file_match = _FILE_START.fullmatch(marker)
        delete_match = _DELETE.fullmatch(marker)
        if file_match:
            path = validate_editable_path(file_match.group(1))
            if path in seen:
                raise MutationFormatError(f"duplicate directive for {path}")
            seen.add(path)
            index += 1
            content_lines: list[str] = []
            while index < len(lines):
                candidate_line = lines[index]
                if candidate_line.removesuffix("\n") == _END_FILE:
                    break
                content_lines.append(_unescape_repository_source_line(candidate_line))
                index += 1
            if index >= len(lines):
                raise MutationFormatError(f"unterminated FILE directive for {path}")
            content = normalize_python_source("".join(content_lines))
            if "\x00" in content:
                raise CandidateValidationError(f"{path} contains a binary NUL byte")
            content_bytes = len(content.encode("utf-8"))
            if content_bytes > MAX_EDITABLE_FILE_BYTES:
                raise CandidateValidationError(
                    f"{path} exceeds the per-file size limit"
                )
            total_bytes += content_bytes
            if total_bytes > MAX_EDITABLE_TOTAL_BYTES:
                raise CandidateValidationError(
                    "repository patch exceeds the editable total size limit"
                )
            mutations.append(FileMutation(path=path, content=content))
            if len(mutations) > MAX_EDITABLE_FILE_COUNT:
                raise CandidateValidationError(
                    "repository patch exceeds the editable file count limit"
                )
            index += 1
            continue

        if delete_match:
            path = validate_editable_path(delete_match.group(1))
            if path == "engine.py":
                raise CandidateValidationError("engine.py cannot be deleted")
            if path in seen:
                raise MutationFormatError(f"duplicate directive for {path}")
            seen.add(path)
            mutations.append(FileMutation(path=path, content=None))
            if len(mutations) > MAX_EDITABLE_FILE_COUNT:
                raise CandidateValidationError(
                    "repository patch exceeds the editable file count limit"
                )
            index += 1
            continue

        if marker.startswith("<<<"):
            raise MutationFormatError(f"malformed repository patch marker: {marker!r}")
        raise MutationFormatError("non-whitespace content outside a FILE directive")

    if not mutations:
        raise MutationFormatError("repository patch contains no directives")
    return RepositoryPatch(tuple(mutations))


def _scan_patch_sequence(lines: list[str], start: int) -> int:
    """Return the exclusive end of one contiguous top-level directive sequence."""

    index = start
    end = start
    while index < len(lines):
        marker = lines[index].removesuffix("\n")
        file_match = _FILE_START.fullmatch(marker)
        delete_match = _DELETE.fullmatch(marker)
        if file_match:
            index += 1
            while index < len(lines):
                if lines[index].removesuffix("\n") == _END_FILE:
                    break
                index += 1
            if index >= len(lines):
                raise MutationFormatError("unterminated FILE directive in response")
            index += 1
            end = index
        elif delete_match:
            index += 1
            end = index
        else:
            break

        lookahead = index
        while lookahead < len(lines) and not lines[lookahead].strip():
            lookahead += 1
        if lookahead < len(lines) and (
            _FILE_START.fullmatch(lines[lookahead].removesuffix("\n"))
            or _DELETE.fullmatch(lines[lookahead].removesuffix("\n"))
        ):
            index = lookahead
            continue
        break
    return end


def _validate_outer_fence(lines: list[str], start: int, end: int) -> None:
    fence_lines = [
        index for index, line in enumerate(lines) if line.strip().startswith("```")
    ]
    if not fence_lines:
        return
    if len(fence_lines) != 2:
        raise MutationFormatError("ambiguous or unbalanced Markdown code fence")
    opening, closing = fence_lines
    if opening >= start or closing < end:
        raise MutationFormatError("Markdown code fence must wrap the repository patch")
    if not re.fullmatch(r"```[A-Za-z0-9_.+-]*", lines[opening].strip()):
        raise MutationFormatError("malformed opening Markdown code fence")
    if lines[closing].strip() != "```":
        raise MutationFormatError("malformed closing Markdown code fence")
    if any(line.strip() for line in lines[opening + 1 : start]):
        raise MutationFormatError("content exists between code fence and repository patch")
    if any(line.strip() for line in lines[end:closing]):
        raise MutationFormatError("content exists between repository patch and code fence")


def extract_repository_patch(raw_response: str | bytes) -> RepositoryPatch:
    """Extract one unambiguous marker sequence from a model response.

    Method prompts demand marker-only output. This compatibility layer also
    accepts ordinary prose around that sequence and one optional outer Markdown
    code fence, while rejecting multiple alternatives or stray marker text.
    """

    text = _decode_patch(raw_response)
    try:
        return parse_repository_patch(text)
    except MutationFormatError:
        pass

    lines = text.splitlines(keepends=True)
    starts = [
        index
        for index, line in enumerate(lines)
        if _FILE_START.fullmatch(line.removesuffix("\n"))
        or _DELETE.fullmatch(line.removesuffix("\n"))
    ]
    if not starts:
        raise MutationFormatError("model response contains no repository patch markers")

    start = starts[0]
    end = _scan_patch_sequence(lines, start)
    if end <= start:
        raise MutationFormatError("model response contains no well-formed repository patch")

    outside = "".join(lines[:start] + lines[end:])
    if "<<<" in outside or ">>>" in outside:
        raise MutationFormatError("ambiguous repository patch marker text")
    _validate_outer_fence(lines, start, end)
    return parse_repository_patch("".join(lines[start:end]))


def _is_forbidden_import(name: str) -> bool:
    return any(
        name == prefix or name.startswith(f"{prefix}.")
        for prefix in FORBIDDEN_IMPORT_PREFIXES
    )


def _static_import_names(node: ast.Import | ast.ImportFrom) -> tuple[str, ...]:
    if isinstance(node, ast.Import):
        return tuple(alias.name for alias in node.names)
    if node.level:
        return ()
    module = node.module or ""
    names = [module] if module else []
    names.extend(
        f"{module}.{alias.name}" if module else alias.name for alias in node.names
    )
    return tuple(names)


def _assignment_target_names(target: ast.expr) -> tuple[str, ...]:
    if isinstance(target, ast.Name):
        return (target.id,)
    if isinstance(target, (ast.Tuple, ast.List)):
        return tuple(
            name for element in target.elts for name in _assignment_target_names(element)
        )
    return ()


def _assignment_values(tree: ast.AST) -> tuple[tuple[tuple[str, ...], ast.expr], ...]:
    assignments: list[tuple[tuple[str, ...], ast.expr]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            names = tuple(
                name
                for target in node.targets
                for name in _assignment_target_names(target)
            )
            assignments.append((names, node.value))
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            assignments.append((_assignment_target_names(node.target), node.value))
        elif isinstance(node, ast.NamedExpr):
            assignments.append((_assignment_target_names(node.target), node.value))
    return tuple(assignments)


def _is_import_callable(expression: ast.expr, aliases: set[str]) -> bool:
    if isinstance(expression, ast.Name):
        return expression.id in aliases
    return isinstance(expression, ast.Attribute) and expression.attr in {
        "__import__",
        "import_module",
    }


def _literal_dynamic_import_aliases(tree: ast.AST) -> set[str]:
    aliases = {"__import__", "import_module"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in {"builtins", "importlib"}:
            for alias in node.names:
                if alias.name in {"__import__", "import_module"}:
                    aliases.add(alias.asname or alias.name)
    assignments = _assignment_values(tree)
    changed = True
    while changed:
        changed = False
        for names, value in assignments:
            if _is_import_callable(value, aliases):
                for name in names:
                    if name not in aliases:
                        aliases.add(name)
                        changed = True
    return aliases


def _bounded_literal_join(left: str, separator: str, right: str) -> str:
    combined = f"{left}{separator}{right}"
    if len(combined) <= _MAX_ANALYZED_LITERAL_CHARS:
        return combined
    quarter = _MAX_ANALYZED_LITERAL_CHARS // 4
    # Keep both expression ends and the newly-created join boundary. Every
    # subexpression is inspected separately, so this bounded summary cannot
    # hide a forbidden token that is formed only by the current concatenation.
    return (
        f"{left[:quarter]}{left[-quarter:]}"
        f"{separator}"
        f"{right[:quarter]}{right[-quarter:]}"
    )


def _constant_text(expression: ast.AST, environment: dict[str, str]) -> str | None:
    if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
        return expression.value
    if isinstance(expression, ast.Constant) and isinstance(expression.value, bytes):
        return expression.value.decode("utf-8", errors="ignore")
    if isinstance(expression, ast.Name):
        return environment.get(expression.id)
    if isinstance(expression, ast.BinOp) and isinstance(expression.op, (ast.Add, ast.Div)):
        left = _constant_text(expression.left, environment)
        right = _constant_text(expression.right, environment)
        if left is None or right is None:
            return None
        separator = "" if isinstance(expression.op, ast.Add) else "/"
        return _bounded_literal_join(left, separator, right)
    if isinstance(expression, ast.JoinedStr):
        result = ""
        for value in expression.values:
            if isinstance(value, ast.FormattedValue):
                piece = _constant_text(value.value, environment)
            else:
                piece = _constant_text(value, environment)
            if piece is None:
                return None
            result = _bounded_literal_join(result, "", piece)
        return result
    if isinstance(expression, ast.Call):
        function = expression.func
        is_path_constructor = (
            isinstance(function, ast.Name)
            and function.id in {"Path", "PurePath", "PurePosixPath"}
        ) or (
            isinstance(function, ast.Attribute)
            and function.attr in {"Path", "PurePath", "PurePosixPath"}
        )
        if is_path_constructor and expression.args:
            parts = [_constant_text(argument, environment) for argument in expression.args]
            if any(part is None for part in parts):
                return None
            result = ""
            for index, part in enumerate(parts):
                assert part is not None
                result = _bounded_literal_join(result, "/" if index else "", part)
            return result
        if (
            isinstance(function, ast.Attribute)
            and function.attr == "joinpath"
            and expression.args
        ):
            root = _constant_text(function.value, environment)
            parts = [_constant_text(argument, environment) for argument in expression.args]
            if root is None or any(part is None for part in parts):
                return None
            result = root
            for part in parts:
                assert part is not None
                result = _bounded_literal_join(result, "/", part)
            return result
        if (
            isinstance(function, ast.Attribute)
            and function.attr == "join"
            and len(expression.args) > 1
        ):
            parts = [_constant_text(argument, environment) for argument in expression.args]
            if any(part is None for part in parts):
                return None
            result = ""
            for index, part in enumerate(parts):
                assert part is not None
                result = _bounded_literal_join(result, "/" if index else "", part)
            return result
        if (
            isinstance(function, ast.Attribute)
            and function.attr == "join"
            and len(expression.args) == 1
        ):
            separator = _constant_text(function.value, environment)
            collection = expression.args[0]
            if separator is None or not isinstance(collection, (ast.List, ast.Tuple)):
                return None
            parts = [_constant_text(item, environment) for item in collection.elts]
            if any(part is None for part in parts):
                return None
            result = ""
            for index, part in enumerate(parts):
                assert part is not None
                result = _bounded_literal_join(
                    result,
                    separator if index else "",
                    part,
                )
            return result
    return None


def _literal_environment(tree: ast.AST) -> dict[str, str]:
    environment: dict[str, str] = {}
    assignments = _assignment_values(tree)
    changed = True
    while changed:
        changed = False
        for names, expression in assignments:
            value = _constant_text(expression, environment)
            if value is None:
                continue
            for name in names:
                if name not in environment:
                    environment[name] = value
                    changed = True
    return environment


def _contains_forbidden_reference_literal(value: str) -> bool:
    normalized = value.replace("\\", "/").lower()
    return any(
        fragment in normalized for fragment in FORBIDDEN_REFERENCE_LITERAL_FRAGMENTS
    )


def _references_dynamic_import_callable(node: ast.AST) -> bool:
    """Conservatively identify access to Python's dynamic import callables."""

    if isinstance(node, ast.Name):
        return node.id == "__import__"
    if isinstance(node, ast.Attribute):
        return node.attr in {"__import__", "import_module"}
    if isinstance(node, ast.ImportFrom):
        return (
            node.module in {"builtins", "importlib"}
            and any(
                alias.name in {"__import__", "import_module"}
                for alias in node.names
            )
        )
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return (
            node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in {"__import__", "import_module"}
        )
    if isinstance(node, ast.Subscript):
        return (
            isinstance(node.slice, ast.Constant)
            and node.slice.value in {"__import__", "import_module"}
        )
    return False


def validate_python_source(path: str, source: str) -> None:
    """Reject invalid Python and imports across baseline/reference boundaries."""

    if "\x00" in source:
        raise CandidateValidationError(f"{path} contains a binary NUL byte")
    try:
        tree = ast.parse(source, filename=path)
    except (SyntaxError, ValueError) as exc:
        raise CandidateValidationError(f"syntax error in {path}: {exc}") from exc

    aliases = _literal_dynamic_import_aliases(tree)
    literal_environment = _literal_environment(tree)
    for node in ast.walk(tree):
        if _references_dynamic_import_callable(node):
            raise CandidateValidationError(
                f"forbidden import callable use in {path}"
            )
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            forbidden = next(
                (name for name in _static_import_names(node) if _is_forbidden_import(name)),
                None,
            )
            if forbidden is not None:
                raise CandidateValidationError(
                    f"forbidden import in {path}: {forbidden}"
                )
        elif isinstance(node, ast.Call):
            if _is_import_callable(node.func, aliases):
                raise CandidateValidationError(
                    f"forbidden import callable use in {path}"
                )

    for node in ast.walk(tree):
        if not isinstance(node, ast.expr):
            continue
        value = _constant_text(node, literal_environment)
        if value is not None and _contains_forbidden_reference_literal(value):
            raise CandidateValidationError(
                f"forbidden reference path literal in {path}"
            )
