from __future__ import annotations

import argparse
import contextlib
import fnmatch
import importlib.util
import json
import os
import re
import resource
import signal
import subprocess
import sys
import time
import types
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from dispatchevolve.cache import experiment_cache_dir, require_experiment_cache_path



ArtifactValidator = Callable[[Path], "ArtifactValidationResult"]

DEFAULT_SANDBOX_THREAD_ENV = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _relative_to(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _extract_markdown_headings(markdown_text: str) -> set[str]:
    headings: set[str] = set()
    for line in markdown_text.splitlines():
        match = re.match(r"^\s*#+\s+(.*?)\s*$", line)
        if match:
            headings.add(match.group(1).strip().lower())
    return headings


@contextlib.contextmanager
def _temporary_env(overrides: dict[str, str]) -> Any:
    previous = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _load_module_from_path(module_path: Path, module_name_prefix: str) -> types.ModuleType:
    module_name = f"{module_name_prefix}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _child_ensure_program_loadable(program_path: Path) -> dict[str, Any]:
    module = _load_module_from_path(program_path, "sandbox_candidate_program")
    if not hasattr(module, "compute_scores"):
        raise AttributeError("Candidate program must define compute_scores(df)")
    return {
        "program_path": str(program_path.resolve()),
        "loadable": True,
    }


def _child_evaluate_train(
    *,
    program_path: Path,
    evaluator_path: Path,
    backend_env: dict[str, str],
) -> dict[str, Any]:
    with _temporary_env(backend_env):
        evaluator_module = _load_module_from_path(evaluator_path, "sandbox_train_evaluator")
        result = evaluator_module.evaluate(str(program_path))
    return {
        "metrics": dict(result.metrics),
        "artifacts": dict(result.artifacts),
    }


@dataclass(frozen=True)
class SandboxProfile:
    timeout_secs: int = 900
    memory_mb: int | None = None
    cpu_time_secs: int | None = None
    network_policy: str = "disabled"
    env_allowlist: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "SandboxProfile":
        data = dict(payload or {})
        env_allowlist = tuple(str(item) for item in data.get("env_allowlist", ()) if item)
        return cls(
            timeout_secs=int(data.get("timeout_secs", 900)),
            memory_mb=int(data["memory_mb"]) if data.get("memory_mb") is not None else None,
            cpu_time_secs=int(data["cpu_time_secs"]) if data.get("cpu_time_secs") is not None else None,
            network_policy=str(data.get("network_policy", "disabled")),
            env_allowlist=env_allowlist,
        )

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "timeout_secs": self.timeout_secs,
            "memory_mb": self.memory_mb,
            "cpu_time_secs": self.cpu_time_secs,
            "network_policy": self.network_policy,
            "env_allowlist": list(self.env_allowlist),
        }


@dataclass(frozen=True)
class SandboxConfig:
    enabled: bool
    workspace_root: Path
    cache_root: Path
    log_root: Path
    keep_failed_workspace: bool = True
    profiles: dict[str, SandboxProfile] = field(default_factory=dict)

    @classmethod
    def default(cls, root_dir: Path) -> "SandboxConfig":
        default_cache_root = experiment_cache_dir("sandbox", start_path=root_dir)
        return cls.from_dict(
            {
                "enabled": True,
                "workspace_root": "outputs/_sandbox/workspaces",
                "cache_root": str(default_cache_root),
                "log_root": "logs/dispatchevolve/sandbox",
                "keep_failed_workspace": True,
                "profiles": {
                    "scene_analysis_debug": {
                        "timeout_secs": 900,
                        "memory_mb": 102400,
                        "network_policy": "disabled",
                    },
                    "candidate_validation": {
                        "timeout_secs": 900,
                        "memory_mb": 102400,
                        "network_policy": "disabled",
                    },
                    "merge_validation": {
                        "timeout_secs": 900,
                        "memory_mb": 102400,
                        "network_policy": "disabled",
                    },
                },
            },
            root_dir=root_dir,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None, *, root_dir: Path) -> "SandboxConfig":
        data = dict(payload or {})
        profiles = {
            name: SandboxProfile.from_dict(profile_payload)
            for name, profile_payload in dict(data.get("profiles", {})).items()
        }
        if not profiles:
            profiles = cls.default(root_dir).profiles
        raw_cache_root = data.get("cache_root")
        cache_root = (
            Path(str(raw_cache_root)).expanduser()
            if raw_cache_root is not None
            else experiment_cache_dir("sandbox", start_path=root_dir)
        )
        if not cache_root.is_absolute():
            cache_root = root_dir / cache_root
        cache_root = require_experiment_cache_path(
            cache_root.resolve(),
            purpose="sandbox cache",
            start_path=root_dir,
        )
        return cls(
            enabled=bool(data.get("enabled", True)),
            workspace_root=(root_dir / str(data.get("workspace_root", "outputs/_sandbox/workspaces"))).resolve(),
            cache_root=cache_root,
            log_root=(root_dir / str(data.get("log_root", "logs/dispatchevolve/sandbox"))).resolve(),
            keep_failed_workspace=bool(data.get("keep_failed_workspace", True)),
            profiles=profiles,
        )

    def get_profile(self, profile_name: str) -> SandboxProfile:
        if profile_name not in self.profiles:
            raise KeyError(f"Sandbox profile '{profile_name}' is not configured")
        return self.profiles[profile_name]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "workspace_root": str(self.workspace_root),
            "cache_root": str(self.cache_root),
            "log_root": str(self.log_root),
            "keep_failed_workspace": self.keep_failed_workspace,
            "profiles": {name: profile.to_jsonable() for name, profile in self.profiles.items()},
        }


@dataclass(frozen=True)
class ArtifactValidationResult:
    success: bool
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "message": self.message,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ExecutionArtifactContract:
    required_files: tuple[str, ...] = ()
    optional_globs: tuple[str, ...] = ()
    forbidden_suffixes: tuple[str, ...] = ()
    required_markdown_sections: tuple[str, ...] = ()
    markdown_path: str | None = None
    custom_validator: ArtifactValidator | None = None

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "required_files": list(self.required_files),
            "optional_globs": list(self.optional_globs),
            "forbidden_suffixes": list(self.forbidden_suffixes),
            "required_markdown_sections": list(self.required_markdown_sections),
            "markdown_path": self.markdown_path,
            "custom_validator": getattr(self.custom_validator, "__name__", None),
        }

    def validate(self, attempt_dir: Path) -> ArtifactValidationResult:
        metadata: dict[str, Any] = {}
        missing = [path for path in self.required_files if not (attempt_dir / path).exists()]
        if missing:
            return ArtifactValidationResult(
                success=False,
                message=f"Missing required artifact files: {', '.join(missing)}",
                metadata={"missing_files": missing},
            )

        forbidden_matches: list[str] = []
        if self.forbidden_suffixes:
            for path in attempt_dir.rglob("*"):
                if path.is_file() and path.suffix.lower() in self.forbidden_suffixes:
                    forbidden_matches.append(_relative_to(path, attempt_dir))
        if forbidden_matches:
            return ArtifactValidationResult(
                success=False,
                message="Forbidden artifact suffixes found: " + ", ".join(forbidden_matches),
                metadata={"forbidden_files": forbidden_matches},
            )

        if self.required_markdown_sections:
            markdown_path = attempt_dir / (self.markdown_path or "outputs/result.md")
            if not markdown_path.exists():
                return ArtifactValidationResult(
                    success=False,
                    message=f"Markdown artifact missing for section validation: {_relative_to(markdown_path, attempt_dir)}",
                    metadata={"markdown_path": _relative_to(markdown_path, attempt_dir)},
                )
            headings = _extract_markdown_headings(markdown_path.read_text(encoding="utf-8"))
            missing_headings = [
                section for section in self.required_markdown_sections if section.strip().lower() not in headings
            ]
            if missing_headings:
                return ArtifactValidationResult(
                    success=False,
                    message="Missing required markdown sections: " + ", ".join(missing_headings),
                    metadata={"missing_markdown_sections": missing_headings},
                )
            metadata["markdown_path"] = _relative_to(markdown_path, attempt_dir)

        matched_optional: list[str] = []
        if self.optional_globs:
            for path in attempt_dir.rglob("*"):
                if not path.is_file():
                    continue
                rel_path = _relative_to(path, attempt_dir)
                if any(fnmatch.fnmatch(rel_path, pattern) for pattern in self.optional_globs):
                    matched_optional.append(rel_path)
        metadata["matched_optional_files"] = sorted(matched_optional)

        if self.custom_validator is not None:
            custom_result = self.custom_validator(attempt_dir)
            if not custom_result.success:
                return custom_result
            metadata.update(custom_result.metadata)

        return ArtifactValidationResult(success=True, message="Artifact contract satisfied.", metadata=metadata)


@dataclass(frozen=True)
class SandboxCommandSpec:
    name: str
    argv: tuple[str, ...]
    cwd: Path
    env: dict[str, str] = field(default_factory=dict)
    timeout_secs: int | None = None

    @classmethod
    def python_file(
        cls,
        *,
        name: str,
        repo_root: Path,
        script_path: Path,
        cwd: Path,
        env: dict[str, str] | None = None,
        timeout_secs: int | None = None,
        args: list[str] | None = None,
    ) -> "SandboxCommandSpec":
        return cls(
            name=name,
            argv=tuple(
                [
                    "uv",
                    "run",
                    "--project",
                    str(repo_root),
                    "python",
                    str(script_path),
                    *(args or []),
                ]
            ),
            cwd=cwd,
            env=dict(env or {}),
            timeout_secs=timeout_secs,
        )

    @classmethod
    def python_module(
        cls,
        *,
        name: str,
        repo_root: Path,
        module_name: str,
        cwd: Path,
        env: dict[str, str] | None = None,
        timeout_secs: int | None = None,
        args: list[str] | None = None,
    ) -> "SandboxCommandSpec":
        return cls(
            name=name,
            argv=tuple(
                [
                    "uv",
                    "run",
                    "--project",
                    str(repo_root),
                    "python",
                    "-m",
                    module_name,
                    *(args or []),
                ]
            ),
            cwd=cwd,
            env=dict(env or {}),
            timeout_secs=timeout_secs,
        )

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "argv": list(self.argv),
            "cwd": str(self.cwd),
            "env": dict(self.env),
            "timeout_secs": self.timeout_secs,
        }


@dataclass(frozen=True)
class SandboxExecutionSpec:
    execution_id: str
    workflow_name: str
    stage_name: str
    profile_name: str
    stage_root: Path
    attempt_index: int
    repo_root: Path
    commands: tuple[SandboxCommandSpec, ...]
    input_json_files: dict[str, Any] = field(default_factory=dict)
    input_text_files: dict[str, str] = field(default_factory=dict)
    code_files: dict[str, str] = field(default_factory=dict)
    artifact_contract: ExecutionArtifactContract | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "workflow_name": self.workflow_name,
            "stage_name": self.stage_name,
            "profile_name": self.profile_name,
            "stage_root": str(self.stage_root),
            "attempt_index": self.attempt_index,
            "repo_root": str(self.repo_root),
            "commands": [command.to_jsonable() for command in self.commands],
            "input_json_files": self.input_json_files,
            "input_text_files": self.input_text_files,
            "code_files": list(self.code_files.keys()),
            "artifact_contract": self.artifact_contract.to_jsonable() if self.artifact_contract is not None else None,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class SandboxCommandResult:
    name: str
    return_code: int | None
    timed_out: bool
    duration_secs: float
    stdout_offset: int
    stderr_offset: int

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "return_code": self.return_code,
            "timed_out": self.timed_out,
            "duration_secs": self.duration_secs,
            "stdout_offset": self.stdout_offset,
            "stderr_offset": self.stderr_offset,
        }


@dataclass(frozen=True)
class SandboxExecutionResult:
    success: bool
    timed_out: bool
    return_code: int | None
    duration_secs: float
    attempt_dir: Path
    input_dir: Path
    code_dir: Path
    outputs_dir: Path
    sandbox_dir: Path
    stdout_path: Path
    stderr_path: Path
    combined_log_path: Path
    result_json_path: Path
    artifact_manifest_path: Path
    mirrored_log_path: Path | None
    policy_violation: str | None
    error_type: str | None
    summary: str
    metadata: dict[str, Any]
    command_results: tuple[SandboxCommandResult, ...]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "timed_out": self.timed_out,
            "return_code": self.return_code,
            "duration_secs": self.duration_secs,
            "attempt_dir": str(self.attempt_dir),
            "input_dir": str(self.input_dir),
            "code_dir": str(self.code_dir),
            "outputs_dir": str(self.outputs_dir),
            "sandbox_dir": str(self.sandbox_dir),
            "stdout_path": str(self.stdout_path),
            "stderr_path": str(self.stderr_path),
            "combined_log_path": str(self.combined_log_path),
            "result_json_path": str(self.result_json_path),
            "artifact_manifest_path": str(self.artifact_manifest_path),
            "mirrored_log_path": str(self.mirrored_log_path) if self.mirrored_log_path is not None else None,
            "policy_violation": self.policy_violation,
            "error_type": self.error_type,
            "summary": self.summary,
            "metadata": self.metadata,
            "command_results": [item.to_jsonable() for item in self.command_results],
        }


@dataclass(frozen=True)
class SandboxAttemptPaths:
    attempt_dir: Path
    input_dir: Path
    code_dir: Path
    outputs_dir: Path
    sandbox_dir: Path


class SandboxService:
    def __init__(self, *, config: SandboxConfig):
        self.config = config

    @staticmethod
    def attempt_paths(stage_root: Path, attempt_index: int) -> SandboxAttemptPaths:
        attempt_dir = stage_root / f"attempt_{attempt_index:02d}"
        input_dir = _ensure_dir(attempt_dir / "input")
        code_dir = _ensure_dir(attempt_dir / "code")
        outputs_dir = _ensure_dir(attempt_dir / "outputs")
        sandbox_dir = _ensure_dir(attempt_dir / "sandbox")
        return SandboxAttemptPaths(
            attempt_dir=attempt_dir,
            input_dir=input_dir,
            code_dir=code_dir,
            outputs_dir=outputs_dir,
            sandbox_dir=sandbox_dir,
        )

    def _check_policy(self, *, profile: SandboxProfile, spec: SandboxExecutionSpec) -> str | None:
        if not profile.env_allowlist:
            return None
        provided_keys = {key for command in spec.commands for key in command.env}
        disallowed = sorted(key for key in provided_keys if key not in profile.env_allowlist)
        if not disallowed:
            return None
        return "Disallowed environment keys for profile: " + ", ".join(disallowed)

    def _write_attempt_inputs(self, *, paths: SandboxAttemptPaths, spec: SandboxExecutionSpec, profile: SandboxProfile) -> None:
        _write_json(paths.input_dir / "sandbox_spec.json", spec.to_jsonable())
        _write_json(
            paths.input_dir / "sandbox_policy.json",
            {
                "profile_name": spec.profile_name,
                "profile": profile.to_jsonable(),
                "config": self.config.to_jsonable(),
            },
        )
        for relative_path, payload in spec.input_json_files.items():
            _write_json(paths.input_dir / relative_path, payload)
        for relative_path, text in spec.input_text_files.items():
            target = paths.input_dir / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        for relative_path, text in spec.code_files.items():
            target = paths.code_dir / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        _write_json(
            paths.sandbox_dir / "command.json",
            {
                "commands": [command.to_jsonable() for command in spec.commands],
                "execution_id": spec.execution_id,
            },
        )

    def _build_preexec(self, profile: SandboxProfile) -> Callable[[], None] | None:
        if profile.memory_mb is None and profile.cpu_time_secs is None:
            return None

        def _apply_limits() -> None:
            if profile.memory_mb is not None and hasattr(resource, "RLIMIT_AS"):
                memory_bytes = int(profile.memory_mb) * 1024 * 1024
                try:
                    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
                except (ValueError, OSError):
                    pass
            if profile.cpu_time_secs is not None and hasattr(resource, "RLIMIT_CPU"):
                try:
                    resource.setrlimit(resource.RLIMIT_CPU, (int(profile.cpu_time_secs), int(profile.cpu_time_secs)))
                except (ValueError, OSError):
                    pass

        return _apply_limits

    def _append_log_block(self, path: Path, *, header: str, content: str) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"=== {header} ===\n")
            handle.write(content)
            if content and not content.endswith("\n"):
                handle.write("\n")
            handle.write("\n")

    def _run_command(
        self,
        *,
        command: SandboxCommandSpec,
        profile: SandboxProfile,
        stdout_path: Path,
        stderr_path: Path,
        combined_path: Path,
        heartbeat_path: Path,
    ) -> SandboxCommandResult:
        start_time = time.monotonic()
        merged_env = os.environ.copy()
        merged_env.update(command.env)
        for key, value in DEFAULT_SANDBOX_THREAD_ENV.items():
            merged_env.setdefault(key, value)
        process = subprocess.Popen(
            list(command.argv),
            cwd=command.cwd,
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            preexec_fn=self._build_preexec(profile),
        )

        stdout_text = ""
        stderr_text = ""
        timed_out = False
        try:
            stdout_text, stderr_text = process.communicate(timeout=command.timeout_secs or profile.timeout_secs)
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            remaining_stdout, remaining_stderr = process.communicate()
            stdout_text = (exc.stdout or "") + (remaining_stdout or "")
            stderr_text = (exc.stderr or "") + (remaining_stderr or "")

        duration_secs = time.monotonic() - start_time
        _write_json(
            heartbeat_path,
            {
                "status": "running" if process.returncode is None else "completed",
                "command_name": command.name,
                "updated_at": _timestamp(),
                "timed_out": timed_out,
                "return_code": process.returncode,
            },
        )
        stdout_offset = stdout_path.stat().st_size if stdout_path.exists() else 0
        stderr_offset = stderr_path.stat().st_size if stderr_path.exists() else 0
        self._append_log_block(stdout_path, header=f"{command.name} stdout", content=stdout_text or "")
        self._append_log_block(stderr_path, header=f"{command.name} stderr", content=stderr_text or "")
        combined_body = (
            f"$ {' '.join(command.argv)}\n"
            f"cwd={command.cwd}\n"
            f"timeout_secs={command.timeout_secs or profile.timeout_secs}\n\n"
            f"## STDOUT\n{stdout_text or ''}\n"
            f"## STDERR\n{stderr_text or ''}\n"
            f"## RETURN CODE\n{process.returncode}\n"
            f"## TIMED OUT\n{timed_out}\n"
        )
        self._append_log_block(combined_path, header=command.name, content=combined_body)
        return SandboxCommandResult(
            name=command.name,
            return_code=process.returncode,
            timed_out=timed_out,
            duration_secs=duration_secs,
            stdout_offset=stdout_offset,
            stderr_offset=stderr_offset,
        )

    def _artifact_manifest(self, outputs_dir: Path, attempt_dir: Path) -> dict[str, Any]:
        files: list[dict[str, Any]] = []
        for path in sorted(outputs_dir.rglob("*")):
            if not path.is_file():
                continue
            files.append(
                {
                    "path": _relative_to(path, attempt_dir),
                    "size_bytes": path.stat().st_size,
                }
            )
        return {
            "output_root": str(outputs_dir),
            "files": files,
        }

    def _mirrored_log_path(self, spec: SandboxExecutionSpec) -> Path:
        workflow_dir = self.config.log_root / spec.workflow_name
        workflow_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"{spec.stage_name}__{_timestamp()}__attempt_{spec.attempt_index:02d}.log"
        return workflow_dir / file_name

    def run(self, spec: SandboxExecutionSpec) -> SandboxExecutionResult:
        profile = self.config.get_profile(spec.profile_name)
        paths = self.attempt_paths(spec.stage_root, spec.attempt_index)
        stdout_path = paths.sandbox_dir / "stdout.log"
        stderr_path = paths.sandbox_dir / "stderr.log"
        combined_path = paths.sandbox_dir / "combined.log"
        result_json_path = paths.sandbox_dir / "result.json"
        artifact_manifest_path = paths.sandbox_dir / "artifact_manifest.json"
        heartbeat_path = paths.sandbox_dir / "heartbeat.json"
        policy_violation = self._check_policy(profile=profile, spec=spec)
        self._write_attempt_inputs(paths=paths, spec=spec, profile=profile)
        _write_json(
            heartbeat_path,
            {
                "status": "preparing_workspace",
                "updated_at": _timestamp(),
                "execution_id": spec.execution_id,
            },
        )

        command_results: list[SandboxCommandResult] = []
        error_type: str | None = None
        summary = "Sandbox execution succeeded."
        start_time = time.monotonic()

        if policy_violation is not None:
            error_type = "policy_blocked"
            summary = policy_violation
        else:
            for command in spec.commands:
                command_result = self._run_command(
                    command=command,
                    profile=profile,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                    combined_path=combined_path,
                    heartbeat_path=heartbeat_path,
                )
                command_results.append(command_result)
                if command_result.timed_out:
                    error_type = "timed_out"
                    summary = f"Command '{command.name}' timed out after {command.timeout_secs or profile.timeout_secs} seconds."
                    break
                if command_result.return_code != 0:
                    error_type = "execution_failed"
                    summary = f"Command '{command.name}' exited with code {command_result.return_code}."
                    break

        artifact_manifest = self._artifact_manifest(paths.outputs_dir, paths.attempt_dir)
        artifact_validation: ArtifactValidationResult | None = None
        if error_type is None and spec.artifact_contract is not None:
            artifact_validation = spec.artifact_contract.validate(paths.attempt_dir)
            if not artifact_validation.success:
                error_type = "artifact_contract_failed"
                summary = artifact_validation.message
        _write_json(artifact_manifest_path, artifact_manifest)

        duration_secs = time.monotonic() - start_time
        return_code = None
        timed_out = False
        if command_results:
            return_code = command_results[-1].return_code
            timed_out = any(item.timed_out for item in command_results)
        if policy_violation is not None:
            return_code = None
            timed_out = False

        mirrored_log_path = self._mirrored_log_path(spec)
        mirrored_log_path.write_text(_read_text(combined_path), encoding="utf-8")

        metadata = {
            "execution_id": spec.execution_id,
            "workflow_name": spec.workflow_name,
            "stage_name": spec.stage_name,
            "profile_name": spec.profile_name,
            "artifact_validation": artifact_validation.to_jsonable() if artifact_validation is not None else None,
            "artifact_manifest": artifact_manifest,
            "spec_metadata": spec.metadata,
        }
        success = error_type is None
        _write_json(
            heartbeat_path,
            {
                "status": "succeeded" if success else error_type,
                "updated_at": _timestamp(),
                "execution_id": spec.execution_id,
                "return_code": return_code,
            },
        )
        result = SandboxExecutionResult(
            success=success,
            timed_out=timed_out,
            return_code=return_code,
            duration_secs=duration_secs,
            attempt_dir=paths.attempt_dir,
            input_dir=paths.input_dir,
            code_dir=paths.code_dir,
            outputs_dir=paths.outputs_dir,
            sandbox_dir=paths.sandbox_dir,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            combined_log_path=combined_path,
            result_json_path=result_json_path,
            artifact_manifest_path=artifact_manifest_path,
            mirrored_log_path=mirrored_log_path,
            policy_violation=policy_violation,
            error_type=error_type,
            summary=summary,
            metadata=metadata,
            command_results=tuple(command_results),
        )
        _write_json(result_json_path, result.to_jsonable())
        return result

    def materialize_cached_execution(
        self,
        *,
        workflow_name: str,
        stage_name: str,
        profile_name: str,
        stage_root: Path,
        repo_root: Path,
        attempt_index: int,
        outputs: dict[str, str],
        input_json_files: dict[str, Any] | None = None,
        input_text_files: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
        artifact_contract: ExecutionArtifactContract | None = None,
    ) -> SandboxExecutionResult:
        paths = self.attempt_paths(stage_root, attempt_index)
        profile = self.config.get_profile(profile_name)
        spec = SandboxExecutionSpec(
            execution_id=f"{stage_name}_cache",
            workflow_name=workflow_name,
            stage_name=stage_name,
            profile_name=profile_name,
            stage_root=stage_root,
            attempt_index=attempt_index,
            repo_root=repo_root,
            commands=tuple(),
            input_json_files=dict(input_json_files or {}),
            input_text_files=dict(input_text_files or {}),
            code_files={},
            artifact_contract=artifact_contract,
            metadata={"source": "cache", **(metadata or {})},
        )
        self._write_attempt_inputs(paths=paths, spec=spec, profile=profile)
        for relative_path, text in outputs.items():
            target = paths.attempt_dir / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        artifact_manifest = self._artifact_manifest(paths.outputs_dir, paths.attempt_dir)
        _write_json(paths.sandbox_dir / "artifact_manifest.json", artifact_manifest)
        artifact_validation = artifact_contract.validate(paths.attempt_dir) if artifact_contract is not None else None
        success = artifact_validation is None or artifact_validation.success
        error_type = None if success else "artifact_contract_failed"
        summary = "Sandbox execution materialized from cache." if success else artifact_validation.message
        mirrored_log_path = self._mirrored_log_path(spec)
        mirrored_log_path.write_text("cache hit\n", encoding="utf-8")
        result = SandboxExecutionResult(
            success=success,
            timed_out=False,
            return_code=0 if success else None,
            duration_secs=0.0,
            attempt_dir=paths.attempt_dir,
            input_dir=paths.input_dir,
            code_dir=paths.code_dir,
            outputs_dir=paths.outputs_dir,
            sandbox_dir=paths.sandbox_dir,
            stdout_path=paths.sandbox_dir / "stdout.log",
            stderr_path=paths.sandbox_dir / "stderr.log",
            combined_log_path=paths.sandbox_dir / "combined.log",
            result_json_path=paths.sandbox_dir / "result.json",
            artifact_manifest_path=paths.sandbox_dir / "artifact_manifest.json",
            mirrored_log_path=mirrored_log_path,
            policy_violation=None,
            error_type=error_type,
            summary=summary,
            metadata={
                "execution_id": spec.execution_id,
                "workflow_name": workflow_name,
                "stage_name": stage_name,
                "profile_name": profile_name,
                "artifact_validation": artifact_validation.to_jsonable() if artifact_validation is not None else None,
                "artifact_manifest": artifact_manifest,
                "spec_metadata": spec.metadata,
            },
            command_results=tuple(),
        )
        (paths.sandbox_dir / "stdout.log").write_text("", encoding="utf-8")
        (paths.sandbox_dir / "stderr.log").write_text("", encoding="utf-8")
        (paths.sandbox_dir / "combined.log").write_text("cache hit\n", encoding="utf-8")
        _write_json(
            paths.sandbox_dir / "heartbeat.json",
            {
                "status": "succeeded" if success else error_type,
                "updated_at": _timestamp(),
                "execution_id": spec.execution_id,
                "return_code": result.return_code,
            },
        )
        _write_json(result.result_json_path, result.to_jsonable())
        return result


@dataclass(frozen=True)
class SandboxDebugAttemptFailure:
    error_type: str
    failure_code: str
    summary: str


class SandboxDebugLoop:
    def __init__(self, *, sandbox_service: SandboxService):
        self.sandbox_service = sandbox_service

    def run(
        self,
        *,
        workflow_name: str,
        stage_name: str,
        profile_name: str,
        stage_root: Path,
        repo_root: Path,
        max_attempts: int,
        prompt_budget_tokens: int | None,
        llm_call: Callable[[str, str], str],
        prompt_builder: Callable[[int, SandboxAttemptPaths, str | None], tuple[str, str]],
        response_validator: Callable[[str, SandboxAttemptPaths], str],
        execution_builder: Callable[[int, SandboxAttemptPaths, str], SandboxExecutionSpec],
        on_success: Callable[[SandboxExecutionResult], tuple[Any, dict[str, Any]]],
        retry_context_builder: Callable[[str, SandboxExecutionResult], str],
    ) -> tuple[Any, dict[str, Any]]:
        retry_context: str | None = None
        latest_failure: SandboxDebugAttemptFailure | None = None

        for attempt in range(1, max_attempts + 1):
            paths = self.sandbox_service.attempt_paths(stage_root, attempt)
            system_prompt, user_prompt = prompt_builder(attempt, paths, retry_context)
            prompt_tokens = max(1, (len(system_prompt) + len(user_prompt)) // 4)
            if prompt_budget_tokens is not None and prompt_budget_tokens > 0 and prompt_tokens > prompt_budget_tokens:
                message = (
                    f"Prompt exceeds budget for {stage_name}: estimated_tokens={prompt_tokens} "
                    f"budget_tokens={prompt_budget_tokens}"
                )
                (paths.sandbox_dir / "validation_error.txt").write_text(message, encoding="utf-8")
                _write_json(
                    paths.input_dir / "prompt_audit.json",
                    {
                        "attempt": attempt,
                        "prompt_estimated_tokens": prompt_tokens,
                        "prompt_budget_tokens": prompt_budget_tokens,
                        "prompt_within_budget": False,
                    },
                )
                raise ValueError(message)
            (paths.input_dir / "system_prompt.txt").write_text(system_prompt, encoding="utf-8")
            (paths.input_dir / "user_prompt.txt").write_text(user_prompt, encoding="utf-8")

            raw_response = llm_call(system_prompt, user_prompt)
            (paths.input_dir / "response.md").write_text(raw_response, encoding="utf-8")

            try:
                code = response_validator(raw_response, paths)
            except Exception as exc:
                latest_failure = SandboxDebugAttemptFailure(
                    error_type="invalid_response",
                    failure_code="analysis_codegen_failed",
                    summary=str(exc),
                )
                (paths.sandbox_dir / "validation_error.txt").write_text(latest_failure.summary, encoding="utf-8")
                _write_json(
                    paths.input_dir / "prompt_audit.json",
                    {
                        "attempt": attempt,
                        "prompt_estimated_tokens": prompt_tokens,
                        "prompt_budget_tokens": prompt_budget_tokens,
                        "prompt_within_budget": True,
                        "validated": False,
                        "failure_code": latest_failure.failure_code,
                    },
                )
                retry_context = (
                    f"Previous response failed validation: {latest_failure.summary}\n"
                    "Rewrite from scratch and follow the exact wrapper contract."
                )
                continue

            (paths.code_dir / "analysis.py").write_text(code + "\n", encoding="utf-8")
            _write_json(
                paths.input_dir / "prompt_audit.json",
                {
                    "attempt": attempt,
                    "prompt_estimated_tokens": prompt_tokens,
                    "prompt_budget_tokens": prompt_budget_tokens,
                    "prompt_within_budget": True,
                    "validated": True,
                },
            )
            spec = execution_builder(attempt, paths, code)
            result = self.sandbox_service.run(spec)
            if result.success:
                return on_success(result)

            latest_failure = SandboxDebugAttemptFailure(
                error_type=result.error_type or "execution_failed",
                failure_code="analysis_execution_failed",
                summary=result.summary,
            )
            (paths.sandbox_dir / "validation_error.txt").write_text(latest_failure.summary, encoding="utf-8")
            retry_context = retry_context_builder(code, result)

        if latest_failure is None:
            latest_failure = SandboxDebugAttemptFailure(
                error_type="unknown_failure",
                failure_code="analysis_execution_failed",
                summary=f"{stage_name} failed without a captured attempt result.",
            )
        raise RuntimeError(json.dumps({"failure_code": latest_failure.failure_code, "message": latest_failure.summary}))


def run_sandbox_child_task(
    *,
    sandbox_service: SandboxService,
    workflow_name: str,
    stage_name: str,
    profile_name: str,
    stage_root: Path,
    repo_root: Path,
    attempt_index: int,
    request_payload: dict[str, Any],
) -> dict[str, Any]:
    paths = sandbox_service.attempt_paths(stage_root, attempt_index)
    request_path = paths.input_dir / "request.json"
    response_path = paths.outputs_dir / "response.json"
    _write_json(request_path, request_payload)
    command = SandboxCommandSpec.python_module(
        name=request_payload["task_name"],
        repo_root=repo_root,
        module_name="dispatchevolve.sandbox",
        cwd=repo_root,
        args=[
            "--child-task",
            request_payload["task_name"],
            "--request-path",
            str(request_path),
            "--response-path",
            str(response_path),
        ],
    )
    spec = SandboxExecutionSpec(
        execution_id=f"{stage_name}_{attempt_index:02d}",
        workflow_name=workflow_name,
        stage_name=stage_name,
        profile_name=profile_name,
        stage_root=stage_root,
        attempt_index=attempt_index,
        repo_root=repo_root,
        commands=(command,),
        input_json_files={"request.json": request_payload},
        metadata={"child_task": request_payload["task_name"]},
        artifact_contract=ExecutionArtifactContract(required_files=("outputs/response.json",)),
    )
    result = sandbox_service.run(spec)
    if not result.success:
        raise RuntimeError(result.summary)
    return json.loads(response_path.read_text(encoding="utf-8"))


def _run_child_task(task_name: str, request_path: Path, response_path: Path) -> int:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if task_name == "ensure_program_loadable":
        payload = _child_ensure_program_loadable(Path(request["program_path"]))
    elif task_name == "evaluate_program_train":
        payload = _child_evaluate_train(
            program_path=Path(request["program_path"]),
            evaluator_path=Path(request["evaluator_path"]),
            backend_env={str(key): str(value) for key, value in dict(request["backend_env"]).items()},
        )
    else:
        raise ValueError(f"Unsupported child task '{task_name}'")
    _write_json(response_path, payload)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DispatchEvolve sandbox child tasks")
    parser.add_argument("--child-task", default=None)
    parser.add_argument("--request-path", default=None)
    parser.add_argument("--response-path", default=None)
    args = parser.parse_args(argv)
    if not args.child_task:
        parser.print_help()
        return 0
    if not args.request_path or not args.response_path:
        raise ValueError("Sandbox child task requires --request-path and --response-path")
    return _run_child_task(
        task_name=str(args.child_task),
        request_path=Path(str(args.request_path)).resolve(),
        response_path=Path(str(args.response_path)).resolve(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
