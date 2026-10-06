"""Isolated Full Dispatch candidate process with bounded batch-stream IPC.

The trusted parent sends one visible batch at a time over a socket.  This
process loads only candidate code, invokes ``run_batch``, and returns one
bounded Arrow response plus an optional trace object.  It never receives an
aggregate split file or imports evaluator, objective, matching, or label code.
"""

from __future__ import annotations

import argparse
import importlib.abc
import importlib.util
import inspect
import json
import os
import pickle
import resource
import socket
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa


_LENGTH = struct.Struct("!Q")
_TWO_LENGTHS = struct.Struct("!QQ")
_MAX_ERROR_BYTES = 64 * 1024
_NUMERIC_RESULT_COLUMNS = (
    "weight",
    "stage",
    "driver_lock_time_s",
    "order_lock_time_s",
)


def _disable_python_process_creation() -> None:
    """Block ordinary Python process creation while allowing native threads."""

    def _raise_fork_disabled(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("candidate process creation is disabled")

    os.fork = _raise_fork_disabled  # type: ignore[assignment]
    if hasattr(os, "forkpty"):
        os.forkpty = _raise_fork_disabled  # type: ignore[assignment]
    subprocess.Popen = _raise_fork_disabled  # type: ignore[assignment]


def _validate_engine(engine_dir: Path) -> Path:
    root = engine_dir.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"candidate engine does not exist: {root}")
    if not (root / "engine.py").is_file() or not (root / "policies").is_dir():
        raise ValueError("candidate engine requires engine.py and policies/")
    return root


class _FreshCandidateLoader(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Reuse compiled sources while rebuilding candidate modules per batch."""

    ENGINE_NAME = "_candidate_engine"

    def __init__(self, engine_dir: Path):
        self.root = _validate_engine(engine_dir)
        self._sources: dict[str, tuple[Any, Path, bool]] = {}
        self._register(self.ENGINE_NAME, self.root / "engine.py", is_package=False)
        policies = self.root / "policies"
        init_path = policies / "__init__.py"
        if init_path.is_file():
            self._register("policies", init_path, is_package=True)
        else:
            self._sources["policies"] = (
                compile("", str(init_path), "exec"),
                init_path,
                True,
            )
        for path in sorted(policies.rglob("*.py")):
            if path == init_path:
                continue
            relative = path.relative_to(policies)
            parts = list(relative.with_suffix("").parts)
            if parts[-1] == "__init__":
                parts = parts[:-1]
                is_package = True
            else:
                is_package = False
            if not parts:
                continue
            for depth in range(1, len(parts)):
                package_name = "policies." + ".".join(parts[:depth])
                if package_name not in self._sources:
                    package_path = policies.joinpath(*parts[:depth], "__init__.py")
                    self._sources[package_name] = (
                        compile("", str(package_path), "exec"),
                        package_path,
                        True,
                    )
            self._register(
                "policies." + ".".join(parts),
                path,
                is_package=is_package,
            )

    def _register(self, name: str, path: Path, *, is_package: bool) -> None:
        source = path.read_text(encoding="utf-8")
        self._sources[name] = (compile(source, str(path), "exec"), path, is_package)

    def find_spec(
        self,
        fullname: str,
        path: Any = None,
        target: Any = None,
    ) -> Any:
        entry = self._sources.get(fullname)
        if entry is None:
            return None
        _code, source_path, is_package = entry
        return importlib.util.spec_from_loader(
            fullname,
            self,
            origin=str(source_path),
            is_package=is_package,
        )

    def create_module(self, spec: Any) -> None:
        return None

    def exec_module(self, module: Any) -> None:
        code, source_path, is_package = self._sources[module.__name__]
        module.__file__ = str(source_path)
        if is_package:
            module.__path__ = [str(source_path.parent)]
            module.__package__ = module.__name__
        exec(code, module.__dict__)

    def fresh_engine(self) -> Any:
        for name in tuple(sys.modules):
            if name == self.ENGINE_NAME or name == "policies" or name.startswith(
                "policies."
            ):
                sys.modules.pop(name, None)
        root_text = str(self.root)
        while root_text in sys.path:
            sys.path.remove(root_text)
        sys.meta_path.insert(0, self)
        try:
            spec = self.find_spec(self.ENGINE_NAME)
            if spec is None or spec.loader is None:
                raise ImportError("cannot load candidate engine.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[self.ENGINE_NAME] = module
            spec.loader.exec_module(module)
        finally:
            try:
                sys.meta_path.remove(self)
            except ValueError:
                pass
        if not hasattr(module, "run_batch"):
            raise AttributeError("candidate engine.py must define run_batch")
        return module


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NA:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _receive_exact(channel: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = int(size)
    while remaining:
        chunk = channel.recv(remaining)
        if not chunk:
            raise RuntimeError("candidate IPC closed before the declared payload")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _frame_to_arrow(frame: pd.DataFrame) -> bytes:
    table = pa.Table.from_pandas(
        pd.DataFrame(frame), preserve_index=False
    )
    sink = pa.BufferOutputStream()
    with pa.ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def _normalize_candidate_output(frame: pd.DataFrame) -> pd.DataFrame:
    """Give mutable numeric result fields a stable Arrow representation.

    Partition loading intentionally represents visible CSV values as strings.
    A product-scoped engine can therefore return numeric values for target rows
    and unchanged numeric strings for passthrough products. Arrow cannot encode
    that mixed object column. Strict numeric conversion preserves the values
    while still rejecting genuinely non-numeric candidate output.
    """
    # The worker owns the candidate result and never exposes this object again;
    # normalizing it in place avoids one full candidate-output copy per batch.
    normalized = pd.DataFrame(frame)
    for column in _NUMERIC_RESULT_COLUMNS:
        if column not in normalized.columns:
            continue
        try:
            normalized[column] = pd.to_numeric(normalized[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"candidate output column {column!r} must contain only numeric values"
            ) from exc
    return normalized


def _trusted_frame_from_pickle(payload: bytes) -> pd.DataFrame:
    """Decode input emitted only by the trusted evaluator parent.

    Pickle is deliberately one-way here. Candidate-controlled output is still
    encoded as Arrow and structurally validated by the parent before use.
    """

    frame = pickle.loads(payload)
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("candidate batch input must be a pandas DataFrame")
    if len(frame.columns) > 4_096:
        raise ValueError("candidate batch input has too many columns")
    if len(set(map(str, frame.columns))) != len(frame.columns):
        raise ValueError("candidate batch input has duplicate columns")
    return frame


def _send_error(channel: socket.socket, error: BaseException) -> None:
    message = f"{type(error).__name__}: {error}".encode(
        "utf-8", errors="replace"
    )[:_MAX_ERROR_BYTES]
    channel.sendall(b"E" + _LENGTH.pack(len(message)) + message)


def _run_candidate_batch(
    channel: socket.socket,
    loader: _FreshCandidateLoader,
    *,
    trace: bool,
    max_message_bytes: int,
    max_trace_bytes: int,
) -> None:
    input_size = _LENGTH.unpack(_receive_exact(channel, _LENGTH.size))[0]
    if input_size > max_message_bytes:
        raise ValueError("candidate batch input exceeds IPC message limit")
    batch = _trusted_frame_from_pickle(_receive_exact(channel, input_size))
    module = loader.fresh_engine()
    accepts_config = "config" in inspect.signature(module.run_batch).parameters
    if accepts_config:
        raw = module.run_batch(batch, config={}, trace=trace)
    else:
        raw = module.run_batch(batch, trace=trace)
    if isinstance(raw, tuple):
        output, batch_trace = raw
    else:
        output, batch_trace = raw, None
    output_payload = _frame_to_arrow(_normalize_candidate_output(output))
    trace_payload = json.dumps(
        _jsonable(batch_trace if isinstance(batch_trace, dict) else None),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    if len(output_payload) > max_message_bytes:
        raise ValueError("candidate batch output exceeds IPC message limit")
    if len(trace_payload) > max_trace_bytes:
        raise ValueError("candidate batch trace exceeds IPC trace limit")
    channel.sendall(
        b"O"
        + _TWO_LENGTHS.pack(len(output_payload), len(trace_payload))
        + output_payload
        + trace_payload
    )


def _call_candidate(
    loader: _FreshCandidateLoader,
    batch: pd.DataFrame,
    *,
    trace: bool,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    module = loader.fresh_engine()
    accepts_config = "config" in inspect.signature(module.run_batch).parameters
    if accepts_config:
        raw = module.run_batch(batch, config={}, trace=trace)
    else:
        raw = module.run_batch(batch, trace=trace)
    if isinstance(raw, tuple):
        output, batch_trace = raw
    else:
        output, batch_trace = raw, None
    return _normalize_candidate_output(output), batch_trace if isinstance(batch_trace, dict) else None


def _run_candidate_bundle(
    channel: socket.socket,
    loader: _FreshCandidateLoader,
    *,
    trace: bool,
    max_message_bytes: int,
    max_trace_bytes: int,
) -> None:
    input_size = _LENGTH.unpack(_receive_exact(channel, _LENGTH.size))[0]
    if input_size > max_message_bytes:
        raise ValueError("candidate batch input exceeds IPC message limit")
    frame = _trusted_frame_from_pickle(_receive_exact(channel, input_size))
    grouped = (
        frame.groupby("batch_id", sort=False, dropna=False)
        if "batch_id" in frame.columns
        else [(None, frame)]
    )
    outputs: list[pd.DataFrame] = []
    traces: list[dict[str, Any]] = []
    for batch_id, batch in grouped:
        output, batch_trace = _call_candidate(loader, pd.DataFrame(batch), trace=trace)
        outputs.append(output)
        if batch_trace is not None:
            item = dict(batch_trace)
            item.setdefault("batch_id", batch_id)
            traces.append(item)
    output_frame = (
        pd.concat(outputs, axis=0, ignore_index=True)
        if outputs
        else frame.iloc[0:0].copy()
    )
    output_payload = _frame_to_arrow(output_frame)
    trace_payload = json.dumps(
        _jsonable(traces),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    if len(output_payload) > max_message_bytes:
        raise ValueError("candidate batch output exceeds IPC message limit")
    if len(trace_payload) > max_trace_bytes:
        raise ValueError("candidate batch trace exceeds IPC trace limit")
    channel.sendall(
        b"O"
        + _TWO_LENGTHS.pack(len(output_payload), len(trace_payload))
        + output_payload
        + trace_payload
    )


def run_candidate_loop(
    channel: socket.socket,
    *,
    engine_dir: Path,
    trace: bool,
    max_message_bytes: int,
    max_trace_bytes: int,
    cpu_seconds: int,
    process_limit: int,
    bundle_batches: bool,
) -> None:
    resource.setrlimit(resource.RLIMIT_FSIZE, (max_message_bytes, max_message_bytes))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    if process_limit > 0:
        resource.setrlimit(resource.RLIMIT_NPROC, (process_limit, process_limit))
    _disable_python_process_creation()
    os.environ["TMPDIR"] = "/nonexistent"
    loader = _FreshCandidateLoader(engine_dir)
    while True:
        control = channel.recv(1)
        if control == b"D":
            return
        if control == b"":
            raise RuntimeError("candidate coordinator closed unexpectedly")
        if control != b"B":
            raise RuntimeError("candidate coordinator sent an invalid command")
        try:
            if bundle_batches:
                _run_candidate_bundle(
                    channel,
                    loader,
                    trace=trace,
                    max_message_bytes=max_message_bytes,
                    max_trace_bytes=max_trace_bytes,
                )
            else:
                _run_candidate_batch(
                    channel,
                    loader,
                    trace=trace,
                    max_message_bytes=max_message_bytes,
                    max_trace_bytes=max_trace_bytes,
                )
        except BaseException as exc:
            _send_error(channel, exc)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--socket-fd", type=int, required=True)
    parser.add_argument("--max-message-bytes", type=int, required=True)
    parser.add_argument("--max-trace-bytes", type=int, required=True)
    parser.add_argument("--cpu-seconds", type=int, required=True)
    parser.add_argument("--process-limit", type=int, default=0)
    parser.add_argument("--bundle-batches", action="store_true")
    parser.add_argument("--trace", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    channel = socket.socket(fileno=int(args.socket_fd))
    try:
        run_candidate_loop(
            channel,
            engine_dir=args.engine_dir,
            trace=bool(args.trace),
            max_message_bytes=int(args.max_message_bytes),
            max_trace_bytes=int(args.max_trace_bytes),
            cpu_seconds=int(args.cpu_seconds),
            process_limit=int(args.process_limit),
            bundle_batches=bool(args.bundle_batches),
        )
    finally:
        channel.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
