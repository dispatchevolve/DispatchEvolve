"""Backend-neutral evaluator adapter used by the DispatchEvolve workflow."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from dispatchevolve.artifacts import atomic_write_json, canonical_hash
from dispatchevolve.tasks.full_dispatch.candidate_executor import (
    LocalProcessCandidateRunner,
)
from dispatchevolve.tasks.full_dispatch.column_policy import COLUMN_POLICY_FINGERPRINT
from dispatchevolve.tasks.full_dispatch.evaluator import (
    evaluate_engine_candidate_only,
)
EVALUATION_PROTOCOL = "full-dispatch-evolution-evaluator-v3-behavior-signature"


def _fingerprint_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluator_implementation_document() -> dict[str, Any]:
    """Return the complete task-layer identity that can affect evaluation."""
    task_root = Path(__file__).resolve().parent
    paths = (
        Path(__file__),
        task_root / "evaluator.py",
        task_root / "candidate_executor.py",
        task_root / "candidate_worker.py",
        task_root / "column_policy.py",
        task_root / "utils.py",
    )
    return {
        "protocol": EVALUATION_PROTOCOL,
        "column_policy_fingerprint": COLUMN_POLICY_FINGERPRINT,
        "source_sha256": {
            path.name: _fingerprint_file(path) for path in paths if path.is_file()
        },
    }


def evaluator_implementation_fingerprint() -> str:
    return canonical_hash(evaluator_implementation_document())


def _handle_fingerprint(handle: Any) -> str:
    value = getattr(handle, "fingerprint", None)
    if isinstance(value, str) and value:
        return value
    if isinstance(handle, Mapping):
        document = dict(handle)
        sources = document.get("source_paths")
        if isinstance(sources, (list, tuple)):
            document["source_fingerprints"] = [
                {
                    "path": str(Path(item).expanduser().resolve()),
                    "sha256": _fingerprint_file(Path(item).expanduser().resolve()),
                }
                for item in sources
            ]
        return canonical_hash(document)
    if isinstance(handle, pd.DataFrame):
        row_hashes = pd.util.hash_pandas_object(handle, index=True).astype(str).tolist()
        return canonical_hash(
            {
                "rows": len(handle),
                "columns": list(map(str, handle.columns)),
                "content_hash": canonical_hash(row_hashes),
                "batch_ids": sorted(handle["batch_id"].astype(str).unique().tolist())
                if "batch_id" in handle
                else [],
            }
        )
    return canonical_hash(repr(handle))


def stable_sample_keys(frame: pd.DataFrame, columns: Iterable[str]) -> tuple[str, ...]:
    """Return deterministic row identities shared by scenario metrics and evaluator."""

    names = tuple(columns)
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise ValueError(f"stable sample id columns are missing: {missing}")
    keys = []
    for values in frame.loc[:, list(names)].itertuples(index=False, name=None):
        normalized = ["<NA>" if pd.isna(value) else str(value) for value in values]
        keys.append(json.dumps(normalized, ensure_ascii=True, separators=(",", ":")))
    if len(keys) != len(set(keys)):
        raise ValueError("stable sample id columns do not uniquely identify scenario rows")
    return tuple(sorted(keys))


def _iter_mapping_frames(
    handle: Mapping[str, Any], *, chunksize: int
) -> Iterable[pd.DataFrame]:
    sources = handle.get("source_paths")
    batch_ids = handle.get("batch_ids")
    if not isinstance(sources, (list, tuple)) or not isinstance(batch_ids, (list, tuple, set)):
        raise TypeError("mapping data handle requires source_paths and batch_ids")
    selected = {str(item) for item in batch_ids}
    for raw_path in sources:
        path = Path(raw_path).expanduser().resolve()
        for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
            keys = chunk["batch_id"].astype("string").fillna("<NA>").astype(str)
            filtered = chunk.loc[keys.isin(selected)]
            if not filtered.empty:
                yield filtered.copy()


def load_partition_frame(
    handle: Any,
    *,
    chunksize: int = 25_000,
    max_batches: int | None = None,
) -> pd.DataFrame:
    """Load one opaque split without consulting any other partition."""

    if isinstance(handle, pd.DataFrame):
        frame = handle.copy()
    elif callable(getattr(handle, "load_frame", None)):
        frame = pd.DataFrame(handle.load_frame(chunksize=chunksize)).copy()
    else:
        iterator = getattr(handle, "iter_frames", None)
        if callable(iterator):
            pieces = [pd.DataFrame(item).copy() for item in iterator(chunksize=chunksize)]
        elif callable(getattr(handle, "iter_rows", None)):
            rows = [dict(row) for _reference, row in handle.iter_rows(chunk_size=chunksize)]
            pieces = [pd.DataFrame.from_records(rows)] if rows else []
        elif isinstance(handle, Mapping):
            pieces = list(_iter_mapping_frames(handle, chunksize=chunksize))
        else:
            raise TypeError("unsupported opaque partition handle")
        if not pieces:
            raise ValueError("partition handle produced no rows")
        frame = pd.concat(pieces, ignore_index=True)

    if "batch_id" not in frame.columns:
        raise ValueError("partition frame requires batch_id")
    if max_batches is not None:
        if max_batches <= 0:
            raise ValueError("max_batches must be positive")
        keys = frame["batch_id"].astype("string").fillna("<NA>").astype(str)
        selected = sorted(keys.unique().tolist())[: int(max_batches)]
        frame = frame.loc[keys.isin(selected)].copy()
    if frame.empty:
        raise ValueError("partition frame is empty")
    return frame.reset_index(drop=True)


class FullDispatchEvolutionEvaluator:
    """Evaluate complete engines with offline local matching."""

    def __init__(
        self,
        *,
        cache_root: Path,
        runner_root: Path,
        backend: str,
        objective_protocol_hash: str,
        csv_chunk_size: int = 25_000,
        wall_timeout_seconds: float = 3600.0,
        candidate_process_workers: int = 1,
        parallel_final_evaluations: bool = False,
    ):
        if backend != "local":
            raise ValueError("backend must be local")
        self.cache_root = Path(cache_root).expanduser().resolve()
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.objective_protocol_hash = str(objective_protocol_hash)
        self.csv_chunk_size = int(csv_chunk_size)
        self.parallel_final_evaluations = bool(parallel_final_evaluations)
        self._partition_frame_cache: dict[tuple[str, int | None], pd.DataFrame] = {}
        self.runner = LocalProcessCandidateRunner(
            workspace=Path(runner_root).expanduser().resolve(),
            wall_timeout_seconds=wall_timeout_seconds,
            candidate_process_workers=candidate_process_workers,
        )

    def load_data_frame(
        self,
        data_handle: Any,
        *,
        max_batches: int | None = None,
    ) -> pd.DataFrame:
        """Load an opaque partition once per evaluator process.

        Full validation evaluates several candidates against the same immutable
        partition. Re-scanning the multi-gigabyte source CSV for every candidate
        makes a small batch-limited validation spend minutes on I/O even when the
        actual replay takes seconds. DataFrame handles are already materialized
        and remain uncached; opaque handles are cached by their frozen fingerprint
        and batch limit.
        """

        if isinstance(data_handle, pd.DataFrame):
            return load_partition_frame(
                data_handle,
                chunksize=self.csv_chunk_size,
                max_batches=max_batches,
            )
        key = (_handle_fingerprint(data_handle), max_batches)
        cached = self._partition_frame_cache.get(key)
        if cached is None:
            cached = load_partition_frame(
                data_handle,
                chunksize=self.csv_chunk_size,
                max_batches=max_batches,
            )
            # A run needs evolution and validation plus, at most, a small number
            # of alternate opaque handles. Keep at most four cached partitions.
            if len(self._partition_frame_cache) >= 4:
                self._partition_frame_cache.pop(next(iter(self._partition_frame_cache)))
            self._partition_frame_cache[key] = cached
        return cached.copy(deep=False)

    def _key(
        self,
        *,
        candidate_id: str,
        engine_dir: Path,
        handle: Any,
        split_role: str,
        metric_sample_keys: tuple[str, ...] | None = None,
        metric_id_columns: tuple[str, ...] = (),
        stability_block_count: int | None = None,
        max_batches: int | None = None,
        selected_data_fingerprint: str | None = None,
        reference_metrics: Mapping[str, float] | None = None,
    ) -> str:
        engine_files = sorted(path for path in engine_dir.rglob("*") if path.is_file())
        engine_hash = canonical_hash(
            [(str(path.relative_to(engine_dir)), _fingerprint_file(path)) for path in engine_files]
        )
        return canonical_hash(
            {
                "protocol": EVALUATION_PROTOCOL,
                "candidate_id": candidate_id,
                "engine_hash": engine_hash,
                "data_fingerprint": _handle_fingerprint(handle),
                "split_role": split_role,
                "backend": self.backend,
                "objective_protocol_hash": self.objective_protocol_hash,
                "runner_fingerprint": self.runner.execution_fingerprint,
                "column_policy_fingerprint": COLUMN_POLICY_FINGERPRINT,
                "evaluator_implementation_fingerprint": evaluator_implementation_fingerprint(),
                "metric_sample_keys": list(metric_sample_keys or ()),
                "metric_id_columns": list(metric_id_columns),
                "stability_block_count": stability_block_count,
                "max_batches": max_batches,
                "selected_data_fingerprint": selected_data_fingerprint,
                "reference_metrics": dict(sorted((reference_metrics or {}).items())),
                "evaluator_code": evaluator_implementation_document()["source_sha256"],
            }
        )

    def evaluate_engine(
        self,
        *,
        candidate_id: str,
        engine_dir: Path,
        data_handle: Any,
        split_role: str,
        reference_metrics: Mapping[str, float] | None = None,
        trace: bool = True,
        output_dir: Path | None = None,
        max_batches: int | None = None,
        objective_contract: Any | None = None,
        e0_metrics: Mapping[str, float] | None = None,
        incumbent_metrics: Mapping[str, float] | None = None,
        scenario_support: Mapping[str, Any] | None = None,
        stability: Mapping[str, Any] | None = None,
        metric_sample_keys: tuple[str, ...] | None = None,
        metric_id_columns: tuple[str, ...] = (),
        stability_block_count: int | None = None,
    ) -> dict[str, Any]:
        if split_role not in {"evolution", "validation", "test"}:
            raise ValueError("split_role must be evolution, validation, or test")
        if split_role == "test":
            raise ValueError("test is sealed; use evaluate_final_once after candidate freeze")
        engine_path = Path(engine_dir).expanduser().resolve()
        frame = self.load_data_frame(data_handle, max_batches=max_batches)
        key = self._key(
            candidate_id=candidate_id,
            engine_dir=engine_path,
            handle=data_handle,
            split_role=split_role,
            metric_sample_keys=metric_sample_keys,
            metric_id_columns=metric_id_columns,
            stability_block_count=stability_block_count,
            max_batches=max_batches,
            selected_data_fingerprint=_handle_fingerprint(frame),
            reference_metrics=reference_metrics,
        )
        cache_path = self.cache_root / split_role / key / "metrics.json"
        if cache_path.is_file():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            cached["cache_hit"] = True
            metrics = cached["metrics"]
            cached["scenario_support"] = dict(scenario_support or {})
            cached["stability"] = dict(stability or {})
            if objective_contract is not None:
                cached["oriented_objectives"] = {
                    item.name: item.oriented(metrics[item.name])
                    for item in objective_contract.metrics
                }
                if e0_metrics is not None:
                    feasible, deltas = objective_contract.feasible(metrics, e0_metrics)
                    cached["deltas_from_e0"] = deltas
                    cached["globally_feasible"] = feasible
                    cached["guardrail_margins"] = {
                        item.name: deltas[item.name] + float(item.rho_global)
                        for item in objective_contract.guardrails
                    }
                if incumbent_metrics is not None:
                    cached["deltas_from_incumbent"] = objective_contract.deltas(metrics, incumbent_metrics)
            return cached

        result = evaluate_engine_candidate_only(
            engine_path,
            batches=frame,
            config={
                "match_backend": self.backend,
                "product_id": "all",
                "engine_batch_mode": "per_batch",
                "metric_sample_keys": list(metric_sample_keys or ()),
                "metric_id_columns": list(metric_id_columns),
                "stability_block_count": stability_block_count,
            },
            trace=trace,
            output_dir=output_dir,
            candidate_runner=self.runner,
            max_rows=max(len(frame), 1),
        )
        metrics = {str(key): float(value) for key, value in result["metrics"].items()}
        deltas = (
            {
                name: metrics[name] - float(reference_metrics[name])
                for name in reference_metrics
                if name in metrics
            }
            if reference_metrics is not None
            else {}
        )
        document = {
            **result,
            "protocol": EVALUATION_PROTOCOL,
            "candidate_id": candidate_id,
            "split_role": split_role,
            "data_fingerprint": _handle_fingerprint(data_handle),
            "objective_protocol_hash": self.objective_protocol_hash,
            "metrics": metrics,
            "reference_metrics": dict(reference_metrics or {}),
            "raw_deltas": deltas,
            "cache_hit": False,
            "evaluator_provenance": {
                "protocol": EVALUATION_PROTOCOL,
                "backend": self.backend,
                "runner_fingerprint": self.runner.execution_fingerprint,
                "objective_protocol_hash": self.objective_protocol_hash,
            },
            "resource_statistics": {
                "input_rows": result.get("input_rows"),
                "eligible_rows": result.get("eligible_rows"),
                "filtered_rows": result.get("filtered_rows"),
                "batches": result.get("batches"),
                "timings": result.get("timings", {}),
            },
            "scenario_support": dict(scenario_support or {}),
            "stability": dict(stability or {}),
        }
        if objective_contract is not None:
            document["oriented_objectives"] = {
                item.name: item.oriented(metrics[item.name])
                for item in objective_contract.metrics
            }
            if e0_metrics is not None:
                feasible, deltas = objective_contract.feasible(metrics, e0_metrics)
                document["deltas_from_e0"] = deltas
                document["globally_feasible"] = feasible
                document["guardrail_margins"] = {
                    item.name: deltas[item.name] + float(item.rho_global)
                    for item in objective_contract.guardrails
                }
            if incumbent_metrics is not None:
                document["deltas_from_incumbent"] = objective_contract.deltas(
                    metrics, incumbent_metrics
                )
        atomic_write_json(cache_path, document)
        return document

    def _evaluate_loaded_frame(
        self,
        *,
        candidate_id: str,
        engine_dir: Path,
        frame: pd.DataFrame,
        output_dir: Path | None,
    ) -> dict[str, Any]:
        metrics_path = Path(output_dir) / "metrics.json" if output_dir is not None else None
        if metrics_path is not None and metrics_path.is_file():
            result = json.loads(metrics_path.read_text(encoding="utf-8"))
            return {
                **result,
                "protocol": EVALUATION_PROTOCOL,
                "candidate_id": candidate_id,
                "split_role": "test",
                "metrics": {
                    str(key): float(value)
                    for key, value in result["metrics"].items()
                },
            }
        result = evaluate_engine_candidate_only(
            Path(engine_dir).expanduser().resolve(),
            batches=frame,
            config={
                "match_backend": self.backend,
                "product_id": "all",
                "engine_batch_mode": "per_batch",
            },
            trace=True,
            output_dir=output_dir,
            candidate_runner=self.runner,
            max_rows=max(len(frame), 1),
        )
        return {
            **result,
            "protocol": EVALUATION_PROTOCOL,
            "candidate_id": candidate_id,
            "split_role": "test",
            "metrics": {str(key): float(value) for key, value in result["metrics"].items()},
        }

    def evaluate_final_once(
        self,
        *,
        frozen_candidate_id: str,
        frozen_engine_dir: Path,
        e0_candidate_id: str,
        e0_engine_dir: Path,
        test_handle: Any,
        sentinel_path: Path,
        output_dir: Path,
    ) -> dict[str, Any]:
        """Submit one sealed test request containing E0 and the frozen candidate."""

        sentinel = Path(sentinel_path).expanduser().resolve()
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        output = Path(output_dir)
        report_path = output / "heldout_report.json"
        e0_checkpoint = output / "e0_result.json"
        candidate_checkpoint = output / "candidate_result.json"
        if sentinel.is_file():
            status = json.loads(sentinel.read_text(encoding="utf-8"))
            if status.get("candidate_id") != frozen_candidate_id:
                raise RuntimeError("final-test sentinel is bound to a different candidate")
            if status.get("status") == "completed" and report_path.is_file():
                return json.loads(report_path.read_text(encoding="utf-8"))
            if status.get("status") != "attempted":
                raise RuntimeError("final test has already terminated without a resumable attempt")
        else:
            descriptor = os.open(sentinel, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"status": "attempted", "candidate_id": frozen_candidate_id}))
                handle.write("\n")

        try:
            frame = load_partition_frame(test_handle, chunksize=self.csv_chunk_size)
            completed: dict[str, dict[str, Any]] = {}
            checkpoints = {
                "e0": e0_checkpoint,
                "candidate": candidate_checkpoint,
            }
            requests = {
                "e0": (e0_candidate_id, e0_engine_dir, output / "e0"),
                "candidate": (
                    frozen_candidate_id,
                    frozen_engine_dir,
                    output / "candidate",
                ),
            }
            for name, checkpoint in checkpoints.items():
                if checkpoint.is_file():
                    completed[name] = json.loads(
                        checkpoint.read_text(encoding="utf-8")
                    )
            missing = [name for name in ("e0", "candidate") if name not in completed]

            def evaluate(name: str) -> tuple[str, dict[str, Any]]:
                candidate_id, engine_dir, phase_output = requests[name]
                return name, self._evaluate_loaded_frame(
                    candidate_id=candidate_id,
                    engine_dir=engine_dir,
                    frame=frame,
                    output_dir=phase_output,
                )

            if self.parallel_final_evaluations and len(missing) == 2:
                with ThreadPoolExecutor(
                    max_workers=2,
                    thread_name_prefix="full-dispatch-final",
                ) as executor:
                    futures = [executor.submit(evaluate, name) for name in missing]
                    for future in as_completed(futures):
                        name, result = future.result()
                        completed[name] = result
                        atomic_write_json(checkpoints[name], result)
            else:
                for name in missing:
                    _, result = evaluate(name)
                    completed[name] = result
                    atomic_write_json(checkpoints[name], result)
            e0 = completed["e0"]
            candidate = completed["candidate"]
            e0_metrics = e0["metrics"]
            candidate_metrics = candidate["metrics"]
            document = {
                "status": "ok",
                "submission_count": 1,
                "protocol": EVALUATION_PROTOCOL,
                "data_fingerprint": _handle_fingerprint(test_handle),
                "e0": e0,
                "candidate": candidate,
                "raw_deltas_from_e0": {
                    name: candidate_metrics[name] - e0_metrics[name]
                    for name in e0_metrics
                    if name in candidate_metrics
                },
            }
            atomic_write_json(report_path, document)
            atomic_write_json(sentinel, {"status": "completed", "candidate_id": frozen_candidate_id})
            return document
        except Exception as exc:
            atomic_write_json(
                sentinel,
                {
                    "status": "failed",
                    "candidate_id": frozen_candidate_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            raise
