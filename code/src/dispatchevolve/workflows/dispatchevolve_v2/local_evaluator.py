"""Offline local matching evaluator used by V2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import hashlib
import json

import pandas as pd

from dispatchevolve.baselines.candidates import RepositoryGenomeCodec

from dispatchevolve.tasks.full_dispatch.candidate_executor import LocalProcessCandidateRunner
from dispatchevolve.tasks.full_dispatch.evaluator import evaluate_engine_candidate_only

from .contracts import (GUARDRAIL_KEYS, METRIC_DIRECTIONS, METRIC_KEYS,
                        OBJECTIVE_KEYS)
from .protocol import atomic_gzip_json, atomic_json, file_hash


# Evaluator-owned business semantics for the frozen V2 objective vector. Numeric
# directions remain single-sourced by OBJECTIVE_DIRECTIONS.
OBJECTIVE_SEMANTICS: Mapping[str, Mapping[str, str]] = {
    "order_ar": {
        "meaning": "Average response probability based on DAR for dispatch events with one or two matched drivers.",
        "statistical_unit": "dispatch event",
        "unit": "probability, usually 0 to 1",
    },
    "mean_gmv": {
        "meaning": "Average gross merchandise value over matched order-driver pairs.",
        "statistical_unit": "matched order-driver pair",
        "unit": "the same monetary unit as the input GMV field",
    },
    "mean_eta": {
        "meaning": "Average estimated driver travel time to pickup over matched order-driver pairs.",
        "statistical_unit": "matched order-driver pair",
        "unit": "seconds",
    },
    "mean_pcaa": {
        "meaning": "Average passenger-side cancellation or abandonment risk over matched order-driver pairs.",
        "statistical_unit": "matched order-driver pair",
        "unit": "probability, usually 0 to 1",
    },
    "mean_dcaa": {
        "meaning": "Average driver-side cancellation or abandonment risk over matched order-driver pairs.",
        "statistical_unit": "matched order-driver pair",
        "unit": "probability, usually 0 to 1",
    },
    "mean_fqs": {
        "meaning": "CR in paper reports: average product of DAR, passenger non-cancellation probability, and driver non-cancellation probability over matched order-driver pairs.",
        "statistical_unit": "matched order-driver pair",
        "unit": "composite score, 0 to 1",
    },
    "order_br": {
        "meaning": "Share of distinct source orders that appear in the final local matches; decreases are protected but increases are not rewarded.",
        "statistical_unit": "distinct source order",
        "unit": "rate, 0 to 1",
    },
}


def frame_content_hash(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(list(map(str, frame.columns)), separators=(",", ":")).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return digest.hexdigest()


def metric_values(result: Mapping[str, Any]) -> dict[str, float]:
    metrics = result["metrics"]
    values = {key: metrics.get(key) for key in OBJECTIVE_KEYS}
    values["order_br"] = metrics.get("order_br")
    missing = [key for key, value in values.items() if value is None]
    if missing:
        raise ValueError(f"evaluator omitted V2 metrics: {missing}")
    return {key: float(value) for key, value in values.items()}


def oriented_delta(
    metrics: Mapping[str, float], baseline: Mapping[str, float], scales: Mapping[str, float]
) -> dict[str, float]:
    return {
        key: METRIC_DIRECTIONS[key] * (float(metrics[key]) - float(baseline[key])) / float(scales[key])
        for key in METRIC_KEYS
    }


def local_metric_scales(
    baseline: Mapping[str, float], global_scales: Mapping[str, float], *, epsilon: float,
) -> dict[str, float]:
    """Use scene-local guardrail scales and support legacy global checkpoints."""
    scales: dict[str, float] = {}
    for key in METRIC_KEYS:
        if key in GUARDRAIL_KEYS:
            scales[key] = max(abs(float(baseline[key])), float(epsilon))
        else:
            scales[key] = float(global_scales[key])
    return scales


def passes_global_acceptance(
    delta: Mapping[str, float], *, rho: float, tolerance: float,
    objective_keys: tuple[str, ...] = OBJECTIVE_KEYS,
    guardrail_keys: tuple[str, ...] = GUARDRAIL_KEYS,
) -> bool:
    """Check global feasibility and strict improvement before archive admission."""
    return all(delta[key] >= -tolerance for key in objective_keys) and any(
        delta[key] > tolerance for key in objective_keys
    ) and all(delta[key] >= -rho - tolerance for key in guardrail_keys)


def feasible(
    delta: Mapping[str, float],
    *,
    target: str | None,
    rho: float,
    tolerance: float,
    metric_keys: tuple[str, ...] = METRIC_KEYS,
    objective_keys: tuple[str, ...] = OBJECTIVE_KEYS,
    guardrail_keys: tuple[str, ...] = GUARDRAIL_KEYS,
) -> bool:
    if target is None:
        return passes_global_acceptance(
            delta,
            rho=rho,
            tolerance=tolerance,
            objective_keys=objective_keys,
            guardrail_keys=guardrail_keys,
        )
    if target not in objective_keys:
        raise ValueError(f"local target must be a benefit objective, got {target!r}")
    return delta[target] > tolerance and all(
        delta[key] >= -rho - tolerance for key in metric_keys if key != target
    )


@dataclass(frozen=True, slots=True)
class V2EvaluationIdentity:
    cache_path: Path
    engine_id: str
    frame_hash: str
    selector_hash: str
    metric_columns: tuple[str, ...]
    metric_keys: tuple[str, ...]

    def ledger_identity(self, protocol_hash: str) -> dict[str, str]:
        return {
            "protocol_hash": protocol_hash,
            "engine_id": self.engine_id,
            "frame_hash": self.frame_hash,
            "metric_selector_hash": self.selector_hash,
        }


class V2Evaluator:
    def __init__(self, *, runner_root: Path, cache_root: Path, protocol_hash: str,
                 backend: str, wall_timeout_seconds: float = 3600,
                 trace_transport_max_bytes: int = 268_435_456,
                 candidate_process_workers: int = 1, replay_budget_path: Path | None = None):
        self.replay_budget_path = replay_budget_path
        if backend != "local":
            raise ValueError("backend must be local")
        self.backend = backend
        self.cache_root = Path(cache_root).expanduser().resolve()
        self.protocol_hash = protocol_hash
        self.runner = LocalProcessCandidateRunner(
            workspace=runner_root, wall_timeout_seconds=wall_timeout_seconds,
            candidate_process_workers=candidate_process_workers,
            parallel_bundle_batches=False,
            max_trace_bytes=trace_transport_max_bytes,
        )

    @staticmethod
    def _metric_selector(metric_rows: pd.DataFrame | None) -> tuple[list[str], list[str]]:
        if metric_rows is None:
            return [], []
        if "__v2_metric_row_id" not in metric_rows:
            raise ValueError("scenario metric rows require __v2_metric_row_id")
        columns = ["__v2_metric_row_id", *[
            name for name in ("batch_id", "uuid", "order_id", "driver_id", "product_id")
            if name in metric_rows
        ]]
        keys = []
        for values in metric_rows.loc[:, columns].itertuples(index=False, name=None):
            normalized = ["<NA>" if pd.isna(value) else str(value) for value in values]
            keys.append(json.dumps(normalized, ensure_ascii=True, separators=(",", ":")))
        if not keys:
            raise ValueError("scenario metric selector cannot be empty")
        return columns, keys

    def evaluation_identity(
        self,
        engine_dir: Path,
        frame: pd.DataFrame,
        metric_rows: pd.DataFrame | None = None,
        *,
        trace: bool = True,
        metrics_only: bool = False,
    ) -> V2EvaluationIdentity:
        frame_hash = frame_content_hash(frame)
        metric_columns, metric_keys = self._metric_selector(metric_rows)
        selector_hash = hashlib.sha256(json.dumps(
            {"columns": metric_columns, "keys": sorted(metric_keys), "trace": bool(trace),
             "metrics_only": bool(metrics_only),
            }, sort_keys=True,
            separators=(",", ":"), ensure_ascii=True,
        ).encode()).hexdigest()
        engine_id = RepositoryGenomeCodec().candidate_id(engine_dir)
        key = hashlib.sha256(
            f"{self.protocol_hash}\0{self.backend}\0{engine_id}\0{frame_hash}\0{selector_hash}".encode()
        ).hexdigest()
        return V2EvaluationIdentity(
            cache_path=self.cache_root / key / "metrics.json",
            engine_id=engine_id,
            frame_hash=frame_hash,
            selector_hash=selector_hash,
            metric_columns=tuple(metric_columns),
            metric_keys=tuple(metric_keys),
        )

    def cache_path(self, engine_dir: Path, frame: pd.DataFrame,
                   metric_rows: pd.DataFrame | None = None, *, trace: bool = True,
                   metrics_only: bool = False) -> Path:
        return self.evaluation_identity(
            engine_dir, frame, metric_rows, trace=trace, metrics_only=metrics_only,
        ).cache_path

    @staticmethod
    def _recover_cached_path(path: Path) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        cached = json.loads(path.read_text(encoding="utf-8"))
        artifact = Path(str(cached.get("batch_traces_artifact", "")))
        if not artifact.is_file() or file_hash(artifact) != cached.get("batch_traces_sha256"):
            return None
        return cached

    def recover_cached(self, engine_dir: Path, frame: pd.DataFrame,
                       metric_rows: pd.DataFrame | None = None, *,
                       identity: V2EvaluationIdentity | None = None,
                       trace: bool = True,
                       metrics_only: bool = False) -> dict[str, Any] | None:
        resolved = identity or self.evaluation_identity(
            engine_dir, frame, metric_rows, trace=trace, metrics_only=metrics_only,
        )
        return self._recover_cached_path(resolved.cache_path)

    def evaluate(self, engine_dir: Path, frame: pd.DataFrame, *, output_dir: Path,
                 metric_rows: pd.DataFrame | None = None,
                 identity: V2EvaluationIdentity | None = None,
                 trace: bool = True,
                 metrics_only: bool = False, charge_replay: bool = True) -> dict[str, Any]:
        resolved = identity or self.evaluation_identity(
            engine_dir, frame, metric_rows, trace=trace, metrics_only=metrics_only,
        )
        cache_path = resolved.cache_path
        metric_columns = list(resolved.metric_columns)
        metric_keys = list(resolved.metric_keys)
        key = cache_path.parent.name
        cached = self._recover_cached_path(cache_path)
        if cached is not None:
            cached["cache_hit"] = True
            atomic_json(output_dir / "metrics.json", cached)
            return cached
        evaluator_config = {"match_backend": self.backend, "product_id": 1,
                            "engine_batch_mode": "per_batch"}
        if metric_keys:
            evaluator_config.update(metric_sample_keys=metric_keys, metric_id_columns=metric_columns)
        if metrics_only:
            evaluator_config["evaluation_detail"] = "metrics_only"
        if charge_replay and self.replay_budget_path is not None:
            from .replay_budget import ReplayBudget
            ReplayBudget(self.replay_budget_path).reserve(len(frame), key)
        result = evaluate_engine_candidate_only(
            engine_dir, frame, config=evaluator_config, trace=trace, output_dir=output_dir,
            candidate_runner=self.runner, max_rows=max(len(frame), 1),
        )
        result["cache_hit"] = False
        result["v2_evaluation_key"] = key
        batch_traces = result.pop("batch_traces", [])
        trace_path = cache_path.parent / "batch_traces.json.gz"
        result["batch_traces_artifact"] = str(trace_path)
        result["batch_traces_sha256"] = atomic_gzip_json(trace_path, batch_traces)
        result["batch_trace_count"] = len(batch_traces)
        atomic_json(output_dir / "metrics.json", result)
        atomic_json(cache_path, result)
        return result
