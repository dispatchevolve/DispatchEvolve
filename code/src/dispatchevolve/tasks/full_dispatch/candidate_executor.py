"""Candidate execution backend for the Full Dispatch benchmark."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import os
import pickle
import select
import signal
import socket
import struct
import subprocess
import sys
import sysconfig
import threading
import time
import uuid
from pathlib import Path
from typing import Any, BinaryIO, Callable

import numpy as np
import pandas as pd
import pyarrow as pa

from .column_policy import OPAQUE_ROW_ID_COLUMN
from .utils import validate_engine_codebase


ProgressCallback = Callable[[str, int, int], Any]

CANDIDATE_RUNNER_PROTOCOL = "full-dispatch-batch-stream-v3"
MAX_CANDIDATE_MESSAGE_BYTES = 64 * 1024 * 1024
MAX_CANDIDATE_TRACE_BYTES = 64 * 1024
MAX_CANDIDATE_LOG_BYTES = 16 * 1024 * 1024
MAX_CANDIDATE_OUTPUT_EXTRA_COLUMNS = 32
DEFAULT_BATCH_TIMEOUT_SECONDS = 120.0
MIN_PARALLEL_BATCHES_PER_WORKER = 16
_LENGTH = struct.Struct("!Q")
_TWO_LENGTHS = struct.Struct("!QQ")


class CandidateProcessError(RuntimeError):
    """Structured failure from an isolated candidate process."""

    def __init__(
        self,
        message: str,
        *,
        returncode: int,
        run_dir: Path,
        failure_kind: str = "candidate_process_failure",
    ):
        super().__init__(message)
        self.returncode = int(returncode)
        self.run_dir = Path(run_dir)
        self.failure_kind = str(failure_kind)


def _candidate_worker_fingerprint() -> str:
    """Bind runner identity to the executable worker implementation."""
    return hashlib.sha256(
        Path(__file__).with_name("candidate_worker.py").read_bytes()
    ).hexdigest()


class _OuterTimeout(RuntimeError):
    pass


class _BatchTimeout(RuntimeError):
    pass


class _PrematureCandidateExit(RuntimeError):
    pass


class _CandidateRejected(RuntimeError):
    pass


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        os.fchmod(handle.fileno(), 0o600)
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _candidate_environment(run_dir: Path) -> dict[str, str]:
    """Build a minimal environment without provider or user credentials."""

    environment = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": str(run_dir),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
        "ARROW_NUM_THREADS": "1",
        "POLARS_MAX_THREADS": "1",
        "RAYON_NUM_THREADS": "1",
        "UV_THREADPOOL_SIZE": "1",
        "MALLOC_ARENA_MAX": "1",
        "MALLOC_TRIM_THRESHOLD_": "131072",
        "PYTHONPATH": os.pathsep.join(
            dict.fromkeys(
                str(Path(path).resolve())
                for path in (
                    sysconfig.get_paths().get("purelib"),
                    sysconfig.get_paths().get("platlib"),
                )
                if path
            )
        ),
    }
    for key in ("LANG", "LC_ALL", "LC_CTYPE"):
        value = os.environ.get(key)
        if value:
            environment[key] = value
    return environment


def _notify_safely(
    callback: ProgressCallback | None,
    stage: str,
    current: int,
    total: int,
) -> bool:
    if callback is None:
        return True
    try:
        callback(stage, int(current), int(total))
        return True
    except Exception:
        return False


def _trusted_frame_to_pickle(frame: pd.DataFrame) -> bytes:
    """Serialize evaluator-owned candidate input without repeated Arrow schemas.

    This direction is trusted parent to untrusted child only. Candidate output
    continues to use validated Arrow IPC, so candidate-controlled bytes are
    never deserialized with pickle in the evaluator process.
    """

    return pickle.dumps(pd.DataFrame(frame), protocol=pickle.HIGHEST_PROTOCOL)


def _frame_from_arrow(
    payload: bytes,
    *,
    max_rows: int,
    max_columns: int,
) -> pd.DataFrame:
    try:
        reader = pa.ipc.open_file(pa.BufferReader(payload))
        if len(reader.schema) > max_columns:
            raise ValueError("candidate batch output has too many columns")
        if len(set(reader.schema.names)) != len(reader.schema.names):
            raise ValueError("candidate batch output has duplicate columns")
        if any(pa.types.is_nested(field.type) for field in reader.schema):
            raise ValueError("candidate batch output contains nested Arrow values")
        rows = sum(
            reader.get_batch(index).num_rows
            for index in range(reader.num_record_batches)
        )
        if rows > max_rows:
            raise ValueError("candidate batch output expanded its visible rows")
        table = reader.read_all()
    except pa.ArrowException as exc:
        raise ValueError(f"candidate batch output is invalid Arrow IPC: {exc}") from exc
    return table.to_pandas()


def _receive_exact(channel: socket.socket, size: int, *, deadline: float) -> bytes:
    chunks: list[bytes] = []
    remaining = int(size)
    while remaining:
        timeout = deadline - time.monotonic()
        if timeout <= 0:
            raise _BatchTimeout("candidate batch exceeded its wall-time boundary")
        channel.settimeout(timeout)
        try:
            chunk = channel.recv(remaining)
        except TimeoutError as exc:
            raise _BatchTimeout(
                "candidate batch exceeded its wall-time boundary"
            ) from exc
        if not chunk:
            raise _PrematureCandidateExit(
                "candidate batch exited before acknowledgement"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class _CappedPipeDrainer(threading.Thread):
    def __init__(self, source: BinaryIO, destination: Path, limit: int):
        super().__init__(daemon=True)
        self.source = source
        self.destination = destination
        self.limit = int(limit)
        self.total = 0
        self.overflowed = False

    def run(self) -> None:
        with self.destination.open("wb") as handle:
            os.fchmod(handle.fileno(), 0o600)
            while True:
                chunk = self.source.read(64 * 1024)
                if not chunk:
                    break
                previous = self.total
                self.total += len(chunk)
                remaining = max(self.limit - previous, 0)
                if remaining:
                    handle.write(chunk[:remaining])
                if self.total > self.limit:
                    self.overflowed = True
            handle.flush()
            os.fsync(handle.fileno())


def _reap_process_group(
    process: subprocess.Popen[bytes], *, terminate_leader: bool
) -> None:
    if terminate_leader:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    if process.poll() is None:
        process.wait()


def _append_protocol_error(path: Path, detail: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"candidate protocol error: {detail}\n")
    path.chmod(0o600)


class LocalProcessCandidateRunner:
    """Run candidate code in one isolated process with batch-stream Arrow IPC."""

    execution_isolation = "uv_local_subprocess"
    paper_eligible = True

    def __init__(
        self,
        *,
        workspace: Path,
        wall_timeout_seconds: float = 900.0,
        batch_timeout_seconds: float = DEFAULT_BATCH_TIMEOUT_SECONDS,
        poll_interval_seconds: float = 0.1,
        heartbeat_interval_seconds: float = 60.0,
        batch_child_process_limit: int = 0,
        candidate_process_workers: int = 1,
        parallel_bundle_batches: bool = True,
        max_candidate_output_bytes: int | None = None,
        max_message_bytes: int | None = None,
        max_trace_bytes: int = MAX_CANDIDATE_TRACE_BYTES,
        max_log_bytes: int = MAX_CANDIDATE_LOG_BYTES,
    ):
        if max_message_bytes is None:
            max_message_bytes = (
                MAX_CANDIDATE_MESSAGE_BYTES
                if max_candidate_output_bytes is None
                else int(max_candidate_output_bytes)
            )
        elif (
            max_candidate_output_bytes is not None
            and int(max_candidate_output_bytes) != int(max_message_bytes)
        ):
            raise ValueError(
                "candidate output and message byte limits must match when both are set"
            )
        for name, value in {
            "wall_timeout_seconds": wall_timeout_seconds,
            "batch_timeout_seconds": batch_timeout_seconds,
            "poll_interval_seconds": poll_interval_seconds,
            "heartbeat_interval_seconds": heartbeat_interval_seconds,
            "max_message_bytes": max_message_bytes,
            "max_trace_bytes": max_trace_bytes,
            "max_log_bytes": max_log_bytes,
        }.items():
            if float(value) <= 0:
                raise ValueError(f"candidate {name} must be positive")
        if batch_child_process_limit < 0:
            raise ValueError("candidate batch_child_process_limit must be non-negative")
        if isinstance(candidate_process_workers, bool) or int(candidate_process_workers) <= 0:
            raise ValueError("candidate_process_workers must be positive")
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.workspace.chmod(0o700)
        self.wall_timeout_seconds = float(wall_timeout_seconds)
        self.batch_timeout_seconds = float(batch_timeout_seconds)
        self.poll_interval_seconds = float(poll_interval_seconds)
        self.heartbeat_interval_seconds = float(heartbeat_interval_seconds)
        self.batch_child_process_limit = int(batch_child_process_limit)
        self.candidate_process_workers = int(candidate_process_workers)
        self.parallel_bundle_batches = bool(parallel_bundle_batches)
        self.max_message_bytes = int(max_message_bytes)
        self.max_candidate_output_bytes = self.max_message_bytes
        self.max_trace_bytes = int(max_trace_bytes)
        self.max_log_bytes = int(max_log_bytes)
        self._thread_state = threading.local()
        fingerprint_payload = self._execution_metadata()
        self.execution_fingerprint = hashlib.sha256(
            json.dumps(
                fingerprint_payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()

    @property
    def last_run_metadata(self) -> dict[str, Any] | None:
        value = getattr(self._thread_state, "last_run_metadata", None)
        return dict(value) if isinstance(value, dict) else None

    def _execution_metadata(self) -> dict[str, Any]:
        return {
            "candidate_runner_protocol": CANDIDATE_RUNNER_PROTOCOL,
            "candidate_worker_sha256": _candidate_worker_fingerprint(),
            "wall_timeout_seconds": self.wall_timeout_seconds,
            "batch_timeout_seconds": self.batch_timeout_seconds,
            "max_message_bytes": self.max_message_bytes,
            "max_trace_bytes": self.max_trace_bytes,
            "max_log_bytes": self.max_log_bytes,
            "max_output_extra_columns": MAX_CANDIDATE_OUTPUT_EXTRA_COLUMNS,
            "rlimit_nofile": 64,
            "rlimit_nproc": self.batch_child_process_limit,
            "python_process_creation": "disabled",
            "module_state": "fresh_per_batch",
            "candidate_process_workers": self.candidate_process_workers,
            "parallel_bundle_batches": self.parallel_bundle_batches,
            "parallel_trace_false_transport": (
                "trusted_pickle_bundle_v2"
                if self.parallel_bundle_batches
                else "strict_per_batch"
            ),
            "candidate_input_transport": "trusted_pickle_protocol_5",
            "candidate_output_transport": "validated_arrow_ipc_file",
            "min_parallel_batches_per_worker": MIN_PARALLEL_BATCHES_PER_WORKER,
        }

    def _wait_until_readable(
        self,
        channel: socket.socket,
        process: subprocess.Popen[bytes],
        *,
        started: float,
        outer_deadline: float,
        batch_deadline: float,
        next_heartbeat: float,
        progress_callback: ProgressCallback | None,
        callback_errors: int,
    ) -> tuple[float, int]:
        while True:
            now = time.monotonic()
            if now >= outer_deadline:
                raise _OuterTimeout("candidate evaluation timed out")
            if now >= batch_deadline:
                raise _BatchTimeout("candidate batch exceeded its wall-time boundary")
            if now >= next_heartbeat:
                if not _notify_safely(
                    progress_callback,
                    "candidate_process_heartbeat",
                    int(now - started),
                    int(self.wall_timeout_seconds),
                ):
                    callback_errors += 1
                next_heartbeat = now + self.heartbeat_interval_seconds
            timeout = min(
                self.poll_interval_seconds,
                outer_deadline - now,
                batch_deadline - now,
                max(next_heartbeat - now, 0.0),
            )
            ready, _, _ = select.select([channel], [], [], max(timeout, 0.0))
            if ready:
                return next_heartbeat, callback_errors
            if process.poll() is not None:
                raise _PrematureCandidateExit(
                    "candidate batch exited before acknowledgement"
                )

    def _exchange_batches(
        self,
        channel: socket.socket,
        process: subprocess.Popen[bytes],
        visible: pd.DataFrame,
        *,
        started: float,
        progress_callback: ProgressCallback | None,
        callback_errors: int,
    ) -> tuple[pd.DataFrame, list[dict[str, Any]], str, str, int]:
        outer_deadline = started + self.wall_timeout_seconds
        next_heartbeat = started + self.heartbeat_interval_seconds
        grouped = (
            visible.groupby("batch_id", sort=False, dropna=False)
            if "batch_id" in visible.columns
            else [(None, visible)]
        )
        frames: list[pd.DataFrame] = []
        traces: list[dict[str, Any]] = []
        input_digest = hashlib.sha256()
        output_digest = hashlib.sha256()
        for batch_id, batch in grouped:
            if time.monotonic() >= outer_deadline:
                raise _OuterTimeout("candidate evaluation timed out")
            input_payload = _trusted_frame_to_pickle(batch)
            if len(input_payload) > self.max_message_bytes:
                raise ValueError("visible batch exceeds candidate IPC message limit")
            input_digest.update(_LENGTH.pack(len(input_payload)))
            input_digest.update(input_payload)
            channel.sendall(
                b"B" + _LENGTH.pack(len(input_payload)) + input_payload
            )
            batch_deadline = min(
                outer_deadline, time.monotonic() + self.batch_timeout_seconds
            )
            next_heartbeat, callback_errors = self._wait_until_readable(
                channel,
                process,
                started=started,
                outer_deadline=outer_deadline,
                batch_deadline=batch_deadline,
                next_heartbeat=next_heartbeat,
                progress_callback=progress_callback,
                callback_errors=callback_errors,
            )
            status = _receive_exact(channel, 1, deadline=batch_deadline)
            if status == b"E":
                error_size = _LENGTH.unpack(
                    _receive_exact(channel, _LENGTH.size, deadline=batch_deadline)
                )[0]
                if error_size > self.max_trace_bytes:
                    raise ValueError("candidate error exceeds IPC error limit")
                error = _receive_exact(
                    channel, error_size, deadline=batch_deadline
                ).decode("utf-8", errors="replace")
                raise _CandidateRejected(error)
            if status != b"O":
                raise ValueError("candidate returned an invalid IPC status")
            output_size, trace_size = _TWO_LENGTHS.unpack(
                _receive_exact(channel, _TWO_LENGTHS.size, deadline=batch_deadline)
            )
            if output_size > self.max_message_bytes:
                raise ValueError("candidate batch output exceeds IPC message limit")
            if trace_size > self.max_trace_bytes:
                raise ValueError("candidate batch trace exceeds IPC trace limit")
            output_payload = _receive_exact(
                channel, output_size, deadline=batch_deadline
            )
            trace_payload = _receive_exact(
                channel, trace_size, deadline=batch_deadline
            )
            output = _frame_from_arrow(
                output_payload,
                max_rows=len(batch),
                max_columns=len(batch.columns) + MAX_CANDIDATE_OUTPUT_EXTRA_COLUMNS,
            )
            trace_item = json.loads(trace_payload.decode("utf-8"))
            output_digest.update(_LENGTH.pack(len(output_payload)))
            output_digest.update(output_payload)
            frames.append(output)
            if isinstance(trace_item, dict):
                item = dict(trace_item)
                item.setdefault("batch_id", batch_id)
                traces.append(item)
        channel.sendall(b"D")
        output = (
            pd.concat(frames, axis=0, ignore_index=True)
            if frames
            else visible.iloc[0:0].copy()
        )
        return (
            output,
            traces,
            input_digest.hexdigest(),
            output_digest.hexdigest(),
            callback_errors,
        )

    def _exchange_bundle(
        self,
        channel: socket.socket,
        process: subprocess.Popen[bytes],
        visible: pd.DataFrame,
        *,
        started: float,
        progress_callback: ProgressCallback | None,
        callback_errors: int,
    ) -> tuple[pd.DataFrame, list[dict[str, Any]], str, str, int]:
        outer_deadline = started + self.wall_timeout_seconds
        next_heartbeat = started + self.heartbeat_interval_seconds
        input_payload = _trusted_frame_to_pickle(visible)
        if len(input_payload) > self.max_message_bytes:
            raise ValueError("visible batch bundle exceeds candidate IPC message limit")
        input_digest = hashlib.sha256()
        output_digest = hashlib.sha256()
        input_digest.update(_LENGTH.pack(len(input_payload)))
        input_digest.update(input_payload)
        channel.sendall(b"B" + _LENGTH.pack(len(input_payload)) + input_payload)
        batch_deadline = min(outer_deadline, time.monotonic() + self.batch_timeout_seconds)
        next_heartbeat, callback_errors = self._wait_until_readable(
            channel,
            process,
            started=started,
            outer_deadline=outer_deadline,
            batch_deadline=batch_deadline,
            next_heartbeat=next_heartbeat,
            progress_callback=progress_callback,
            callback_errors=callback_errors,
        )
        status = _receive_exact(channel, 1, deadline=batch_deadline)
        if status == b"E":
            error_size = _LENGTH.unpack(
                _receive_exact(channel, _LENGTH.size, deadline=batch_deadline)
            )[0]
            if error_size > self.max_trace_bytes:
                raise ValueError("candidate error exceeds IPC error limit")
            error = _receive_exact(
                channel, error_size, deadline=batch_deadline
            ).decode("utf-8", errors="replace")
            raise _CandidateRejected(error)
        if status != b"O":
            raise ValueError("candidate returned an invalid IPC status")
        output_size, trace_size = _TWO_LENGTHS.unpack(
            _receive_exact(channel, _TWO_LENGTHS.size, deadline=batch_deadline)
        )
        if output_size > self.max_message_bytes:
            raise ValueError("candidate batch output exceeds IPC message limit")
        if trace_size > self.max_trace_bytes:
            raise ValueError("candidate batch trace exceeds IPC trace limit")
        output_payload = _receive_exact(channel, output_size, deadline=batch_deadline)
        trace_payload = _receive_exact(channel, trace_size, deadline=batch_deadline)
        output = _frame_from_arrow(
            output_payload,
            max_rows=len(visible),
            max_columns=len(visible.columns) + MAX_CANDIDATE_OUTPUT_EXTRA_COLUMNS,
        )
        trace_raw = json.loads(trace_payload.decode("utf-8"))
        traces = [dict(item) for item in trace_raw if isinstance(item, dict)] if isinstance(trace_raw, list) else []
        output_digest.update(_LENGTH.pack(len(output_payload)))
        output_digest.update(output_payload)
        channel.sendall(b"D")
        return (
            output,
            traces,
            input_digest.hexdigest(),
            output_digest.hexdigest(),
            callback_errors,
        )

    def _run_single_process(
        self,
        engine_path: Path,
        visible_df: pd.DataFrame,
        trace: bool,
        progress_callback: ProgressCallback | None,
        *,
        bundle_batches: bool = False,
    ) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        engine_path = validate_engine_codebase(engine_path)
        visible = pd.DataFrame(visible_df)
        run_dir = self.workspace / f"candidate-{uuid.uuid4().hex}"
        run_dir.mkdir(parents=False, exist_ok=False, mode=0o700)
        run_dir.chmod(0o700)
        stdout_path = run_dir / "stdout.log"
        stderr_path = run_dir / "stderr.log"
        manifest_path = run_dir / "worker_manifest.json"
        environment = _candidate_environment(run_dir)
        cpu_seconds = max(1, int(math.ceil(self.wall_timeout_seconds)) + 5)
        parent_channel, child_channel = socket.socketpair()
        command = [
            sys.executable,
            "-S",
            str(Path(__file__).with_name("candidate_worker.py").resolve()),
            "--engine-dir",
            str(engine_path),
            "--socket-fd",
            str(child_channel.fileno()),
            "--max-message-bytes",
            str(self.max_message_bytes),
            "--max-trace-bytes",
            str(self.max_trace_bytes),
            "--cpu-seconds",
            str(cpu_seconds),
            "--process-limit",
            str(self.batch_child_process_limit),
        ]
        if trace:
            command.append("--trace")
        if bundle_batches:
            command.append("--bundle-batches")
        manifest: dict[str, Any] = {
            "execution_isolation": self.execution_isolation,
            "engine_dir": str(engine_path),
            "input_rows": int(len(visible)),
            "input_columns": list(map(str, visible.columns)),
            "environment_keys": sorted(environment),
            "candidate_runner_fingerprint": self.execution_fingerprint,
            "status": "running",
            **self._execution_metadata(),
        }
        _atomic_json(manifest_path, manifest)
        callback_errors = 0 if _notify_safely(
            progress_callback, "candidate_process_start", 0, 1
        ) else 1
        started = time.monotonic()
        process: subprocess.Popen[bytes] | None = None
        stdout_drainer: _CappedPipeDrainer | None = None
        stderr_drainer: _CappedPipeDrainer | None = None
        failure_kind: str | None = None
        failure_detail: str | None = None
        output: pd.DataFrame | None = None
        traces: list[dict[str, Any]] = []
        input_sha256: str | None = None
        output_sha256: str | None = None
        try:
            process = subprocess.Popen(
                command,
                cwd=run_dir,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                close_fds=True,
                pass_fds=(child_channel.fileno(),),
                start_new_session=True,
            )
            child_channel.close()
            assert process.stdout is not None and process.stderr is not None
            stdout_drainer = _CappedPipeDrainer(
                process.stdout, stdout_path, self.max_log_bytes
            )
            stderr_drainer = _CappedPipeDrainer(
                process.stderr, stderr_path, self.max_log_bytes
            )
            stdout_drainer.start()
            stderr_drainer.start()
            if bundle_batches:
                (
                    output,
                    traces,
                    input_sha256,
                    output_sha256,
                    callback_errors,
                ) = self._exchange_bundle(
                    parent_channel,
                    process,
                    visible,
                    started=started,
                    progress_callback=progress_callback,
                    callback_errors=callback_errors,
                )
            else:
                (
                    output,
                    traces,
                    input_sha256,
                    output_sha256,
                    callback_errors,
                ) = self._exchange_batches(
                    parent_channel,
                    process,
                    visible,
                    started=started,
                    progress_callback=progress_callback,
                    callback_errors=callback_errors,
                )
            remaining = max(started + self.wall_timeout_seconds - time.monotonic(), 0.01)
            try:
                process.wait(timeout=min(10.0, remaining))
            except subprocess.TimeoutExpired as exc:
                raise _OuterTimeout("candidate evaluation timed out") from exc
            if process.returncode != 0:
                raise _PrematureCandidateExit(
                    f"candidate process exited with status {process.returncode}"
                )
        except _OuterTimeout as exc:
            failure_kind, failure_detail = "outer_timeout", str(exc)
        except _BatchTimeout as exc:
            failure_kind, failure_detail = "batch_timeout", str(exc)
        except _PrematureCandidateExit as exc:
            failure_kind, failure_detail = "premature_batch_exit", str(exc)
        except _CandidateRejected as exc:
            failure_kind, failure_detail = "candidate_error", str(exc)
        except Exception as exc:
            failure_kind = "candidate_process_failure"
            failure_detail = f"{type(exc).__name__}: {exc}"
        finally:
            parent_channel.close()
            child_channel.close()
            if process is not None:
                _reap_process_group(
                    process,
                    terminate_leader=failure_kind is not None or process.poll() is None,
                )
            for drainer in (stdout_drainer, stderr_drainer):
                if drainer is not None:
                    drainer.join(timeout=10)
        returncode = int(
            process.returncode
            if process is not None and process.returncode is not None
            else -9
        )
        log_overflow = bool(
            (stdout_drainer and stdout_drainer.overflowed)
            or (stderr_drainer and stderr_drainer.overflowed)
        )
        if failure_kind is None and log_overflow:
            failure_kind = "log_overflow"
            failure_detail = "exceeded the bounded log limit"
        elapsed = float(time.monotonic() - started)
        manifest.update(
            {
                "elapsed_seconds": elapsed,
                "returncode": returncode,
                "log_overflow": log_overflow,
                "callback_errors": callback_errors,
                "status": (
                    "timeout"
                    if failure_kind in {"outer_timeout", "batch_timeout"}
                    else ("failed" if failure_kind else "ok")
                ),
            }
        )
        if input_sha256 is not None:
            manifest["input_sha256"] = input_sha256
        if output_sha256 is not None:
            manifest["output_sha256"] = output_sha256
        if failure_kind is not None:
            manifest["failure_kind"] = failure_kind
            manifest["failure_detail"] = failure_detail
            _append_protocol_error(stderr_path, failure_detail or failure_kind)
            _atomic_json(manifest_path, manifest)
            self._remember_run(manifest, manifest_path)
            detail = (
                "timed out"
                if failure_kind == "outer_timeout"
                else failure_detail or failure_kind
            )
            raise CandidateProcessError(
                f"candidate worker failed: {detail}; logs: {stderr_path}",
                returncode=returncode,
                run_dir=run_dir,
                failure_kind=failure_kind,
            )
        assert output is not None
        manifest.update(
            {
                "output_rows": int(len(output)),
                "trace_count": len(traces),
            }
        )
        _atomic_json(manifest_path, manifest)
        self._remember_run(manifest, manifest_path)
        if not _notify_safely(progress_callback, "candidate_process_end", 1, 1):
            callback_errors += 1
            manifest["callback_errors"] = callback_errors
            _atomic_json(manifest_path, manifest)
            self._remember_run(manifest, manifest_path)
        return output, traces

    def _split_visible_batches(
        self,
        visible: pd.DataFrame,
    ) -> list[pd.DataFrame]:
        if self.candidate_process_workers <= 1 or "batch_id" not in visible.columns:
            return [visible]
        batch_positions = list(
            visible.groupby("batch_id", sort=False, dropna=False).indices.values()
        )
        if len(batch_positions) <= 1:
            return [visible]
        worker_count = min(self.candidate_process_workers, len(batch_positions))
        if OPAQUE_ROW_ID_COLUMN not in visible.columns:
            # Generic direct-runner callers have no trusted key for restoring
            # order after non-contiguous sharding. Keep the historical ordered
            # split for those callers.
            chunk_size = int(math.ceil(len(batch_positions) / worker_count))
            return [
                visible.iloc[np.concatenate(batch_positions[start : start + chunk_size])]
                .reset_index(drop=True)
                for start in range(0, len(batch_positions), chunk_size)
            ]

        # Candidate startup dominates tiny scenario partitions. Avoid spending
        # extra memory on workers that receive too few batches to amortize it;
        # larger local scenes and global partitions still use the configured
        # maximum. The opaque id permits canonical order restoration below.
        worker_count = min(
            worker_count,
            max(1, int(math.ceil(len(batch_positions) / MIN_PARALLEL_BATCHES_PER_WORKER))),
        )
        if worker_count <= 1:
            return [visible]

        # Use only engine-agnostic batch size as the cost estimate. This remains
        # valid for newly evolved engines regardless of which policies or input
        # features they use. The trusted opaque row id lets the parent restore
        # canonical order after non-contiguous LPT sharding.
        costs = {
            int(positions[0]): len(positions)
            for positions in batch_positions
        }
        loads = [0] * worker_count
        shards: list[list[np.ndarray]] = [[] for _ in range(worker_count)]
        ranked = sorted(
            enumerate(batch_positions),
            key=lambda item: (-costs[int(item[1][0])], item[0]),
        )
        for _batch_index, positions in ranked:
            shard = min(range(worker_count), key=lambda index: (loads[index], index))
            shards[shard].append(positions)
            loads[shard] += costs[int(positions[0])]
        return [
            visible.iloc[np.concatenate(sorted(parts, key=lambda item: int(item[0])))]
            .reset_index(drop=True)
            for parts in shards
        ]

    def __call__(
        self,
        engine_path: Path,
        visible_df: pd.DataFrame,
        trace: bool,
        progress_callback: ProgressCallback | None,
    ) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        visible = pd.DataFrame(visible_df)
        chunks = self._split_visible_batches(visible)
        if len(chunks) <= 1:
            return self._run_single_process(
                engine_path,
                visible,
                trace,
                progress_callback,
            )

        started = time.monotonic()

        def run_chunk(index: int, chunk: pd.DataFrame) -> tuple[int, pd.DataFrame, list[dict[str, Any]], dict[str, Any] | None]:
            output, traces = self._run_single_process(
                engine_path,
                chunk,
                trace,
                progress_callback,
                bundle_batches=not trace and self.parallel_bundle_batches,
            )
            metadata = getattr(self._thread_state, "last_run_metadata", None)
            return index, output, traces, dict(metadata) if isinstance(metadata, dict) else None

        results: list[tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any] | None] | None] = [
            None for _ in chunks
        ]
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(chunks),
            thread_name_prefix="full-dispatch-candidate",
        ) as pool:
            futures = [
                pool.submit(run_chunk, index, chunk)
                for index, chunk in enumerate(chunks)
            ]
            for future in concurrent.futures.as_completed(
                futures,
                timeout=self.wall_timeout_seconds + 10.0,
            ):
                index, output, traces, metadata = future.result()
                results[index] = (output, traces, metadata)

        if any(item is None for item in results):
            raise CandidateProcessError(
                "candidate parallel evaluation did not return all chunks",
                returncode=-9,
                run_dir=self.workspace,
                failure_kind="parallel_candidate_incomplete",
            )
        completed = [
            item for item in results if item is not None
        ]
        output = pd.concat([item[0] for item in completed], axis=0, ignore_index=True)
        if OPAQUE_ROW_ID_COLUMN in visible.columns and OPAQUE_ROW_ID_COLUMN in output.columns:
            canonical_order = pd.Series(
                np.arange(len(visible), dtype=np.int64),
                index=visible[OPAQUE_ROW_ID_COLUMN],
            )
            output_order = output[OPAQUE_ROW_ID_COLUMN].map(canonical_order)
            if bool(output_order.notna().all()):
                output = (
                    output.assign(__candidate_output_order__=output_order.to_numpy())
                    .sort_values("__candidate_output_order__", kind="stable")
                    .drop(columns=["__candidate_output_order__"])
                    .reset_index(drop=True)
                )
        traces = [
            trace_item
            for item in completed
            for trace_item in item[1]
        ]
        child_runs = [item[2] for item in completed if item[2] is not None]
        self._thread_state.last_run_metadata = {
            "execution_isolation": self.execution_isolation,
            "candidate_runner_fingerprint": self.execution_fingerprint,
            "status": "ok",
            "parallel_candidate_process_workers": len(chunks),
            "input_rows": int(len(visible)),
            "output_rows": int(len(output)),
            "trace_count": len(traces),
            "elapsed_seconds": float(time.monotonic() - started),
            "child_run_count": len(child_runs),
            "child_runs": child_runs,
            **self._execution_metadata(),
        }
        return output, traces

    def _remember_run(self, manifest: dict[str, Any], manifest_path: Path) -> None:
        metadata = dict(manifest)
        metadata["manifest_path"] = str(manifest_path)
        metadata["run_dir"] = str(manifest_path.parent)
        self._thread_state.last_run_metadata = metadata


__all__ = [
    "CANDIDATE_RUNNER_PROTOCOL",
    "CandidateProcessError",
    "LocalProcessCandidateRunner",
]
