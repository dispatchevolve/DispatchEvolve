"""Evaluator for full_dispatch engine codebases."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import numpy as np

from dispatchevolve.repo_paths import find_repo_root, find_shared_root, remap_shared_path

from .column_policy import ColumnPolicy
from .feature_schema import canonicalize_feature_columns
from .utils import (
    build_breakdowns,
    ensure_metric_columns,
    has_match_columns,
    jsonable,
    local_stage_major_match,
    metric_deltas,
    safe_float_series,
    summarize_matches,
    validate_engine_codebase,
)


ROOT_DIR = find_repo_root(Path(__file__).resolve())
SHARED_ROOT_DIR = find_shared_root(ROOT_DIR)
DEFAULT_DATA_PATH = SHARED_ROOT_DIR / "data" / "Full_Dispatch" / "input.csv"
DEFAULT_OUTPUT_ROOT = SHARED_ROOT_DIR / "outputs" / "full_dispatch"
DEFAULT_PRODUCT_ID = 1
CANDIDATE_ONLY_DEFAULT_MAX_ROWS = 200_000
SUPPORTED_BACKENDS = ("local",)
ProgressCallback = Callable[[str, int, int], Any]
CandidateRunner = Callable[
    [Path, pd.DataFrame, bool, ProgressCallback | None],
    tuple[Any, Any],
]


def _objective_prompt_evidence(
    matches: pd.DataFrame,
    source: pd.DataFrame,
) -> dict[str, Any]:
    """Return deterministic counts and complete-Batch objective quantiles."""
    matched_rows = int(len(matches))
    source_orders = int(source["order_id"].nunique()) if "order_id" in source else 0
    matched_orders = int(matches["order_id"].nunique()) if "order_id" in matches else 0
    event_columns = [name for name in ("order_id", "batch_id") if name in matches.columns]
    if "order_id" not in event_columns:
        event_columns = []
    eligible_ar_events = 0
    if event_columns and "driver_id" in matches:
        event_driver_rows = matches.loc[:, [*event_columns, "driver_id"]].drop_duplicates()
        driver_counts = event_driver_rows.groupby(
            event_columns, dropna=False, sort=False,
        )["driver_id"].nunique()
        eligible_ar_events = int(driver_counts.isin((1, 2)).sum())
    counts = {
        "order_ar": {"unit": "dispatch events with one or two matched drivers", "contributing": eligible_ar_events},
        "mean_gmv": {"unit": "matched OD pairs", "contributing": matched_rows},
        "mean_eta": {"unit": "matched OD pairs", "contributing": matched_rows},
        "mean_pcaa": {"unit": "matched OD pairs", "contributing": matched_rows},
        "mean_dcaa": {"unit": "matched OD pairs", "contributing": matched_rows},
        "order_br": {"unit": "distinct orders", "matched": matched_orders, "source": source_orders},
        "mean_fqs": {"unit": "matched OD pairs", "contributing": matched_rows},
    }
    objective_keys = (
        "order_ar", "mean_gmv", "mean_eta", "mean_pcaa", "mean_dcaa", "order_br", "mean_fqs",
    )
    per_batch: dict[str, list[float]] = {key: [] for key in objective_keys}
    if "batch_id" in source:
        matches_by_batch = {
            str(batch_id): batch
            for batch_id, batch in matches.groupby("batch_id", sort=False, dropna=False)
        } if "batch_id" in matches else {}
        for batch_id, source_batch in source.groupby("batch_id", sort=False, dropna=False):
            match_batch = matches_by_batch.get(str(batch_id), matches.iloc[0:0])
            metrics = summarize_matches(match_batch, source_df=source_batch)
            for key in per_batch:
                value = metrics.get(key)
                if value is not None and np.isfinite(float(value)):
                    per_batch[key].append(float(value))
    quantiles = {
        key: ({
            "p10": float(np.quantile(values, 0.10)),
            "p50": float(np.quantile(values, 0.50)),
            "p90": float(np.quantile(values, 0.90)),
        } if values else {"p10": None, "p50": None, "p90": None})
        for key, values in per_batch.items()
    }
    return {"counts": counts, "batch_quantiles": quantiles}


def _candidate_behavior_signature(
    source: pd.DataFrame,
    eligible: pd.DataFrame,
    matches: pd.DataFrame,
) -> str:
    """Hash stable per-sample eligibility, scoring, and matching decisions."""

    identity = [
        name for name in ("batch_id", "uuid", "order_id", "driver_id", "product_id")
        if name in source.columns
    ]
    decision = [
        name for name in (
            "weight", "stage", "driver_lock_time_s", "order_lock_time_s",
            "is_filtered", "filter_rule", "filter_policy",
        )
        if name in eligible.columns
    ]

    def document(frame: pd.DataFrame, columns: list[str]) -> str:
        # Remote match post-processing can preserve duplicate-named columns
        # from the source and response frames. A behavior signature needs one
        # stable value per field, and pandas records JSON requires unique names.
        unique_frame = frame.loc[:, ~frame.columns.duplicated(keep="first")]
        requested = list(dict.fromkeys(columns))
        selected = unique_frame.loc[
            :, [name for name in requested if name in unique_frame.columns]
        ].copy()
        order = [name for name in identity if name in selected.columns]
        if order:
            selected = selected.sort_values(order, kind="stable", na_position="first")
        return selected.to_json(
            orient="records", date_format="iso", double_precision=15, force_ascii=True
        )

    payload = json.dumps(
        {
            "source_ids": document(source, identity),
            "eligible_decisions": document(eligible, [*identity, *decision]),
            "match_decisions": document(matches, [*identity, "weight", "stage"]),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _final_match_references(matches: pd.DataFrame) -> dict[str, list[str]]:
    """Return stable, privacy-preserving references to actual per-batch matches."""
    identity = [
        name for name in ("batch_id", "uuid", "order_id", "driver_id", "product_id", "weight", "stage")
        if name in matches.columns
    ]
    if "batch_id" not in identity:
        return {}
    unique = matches.loc[:, ~matches.columns.duplicated(keep="first")]
    references: dict[str, list[str]] = {}
    for batch_id, batch in unique.groupby("batch_id", sort=False, dropna=False):
        values = batch.loc[:, identity].astype("string").fillna("<NA>")
        rows = []
        for record in values.to_dict(orient="records"):
            payload = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            rows.append(hashlib.sha256(payload.encode()).hexdigest())
        references[str(batch_id)] = sorted(rows)
    return references


def _final_match_row_evidence(
    source: pd.DataFrame,
    matches: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Map final matches to exact per-batch source-row bitmaps without exposing IDs."""
    if "batch_id" not in source.columns:
        return {}
    identity = [
        name for name in ("order_id", "driver_id", "product_id", "uuid")
        if name in source.columns and name in matches.columns
    ]
    if not {"order_id", "driver_id"}.issubset(identity):
        raise ValueError("final match row evidence requires order_id and driver_id")
    unique_source = source.loc[:, ~source.columns.duplicated(keep="first")]
    unique_matches = matches.loc[:, ~matches.columns.duplicated(keep="first")]
    match_groups = {
        str(batch_id): batch
        for batch_id, batch in unique_matches.groupby("batch_id", sort=False, dropna=False)
    }
    result: dict[str, dict[str, Any]] = {}
    for batch_id, batch in unique_source.groupby("batch_id", sort=False, dropna=False):
        key = str(batch_id)
        source_keys = batch.loc[:, identity].astype("string").fillna("<NA>")
        if bool(source_keys.duplicated(keep=False).any()):
            raise ValueError(f"source OD-pair identity is not unique in batch {key}")
        positions = {
            tuple(row): position
            for position, row in enumerate(source_keys.itertuples(index=False, name=None))
        }
        selected_positions: list[int] = []
        matched_batch = match_groups.get(key)
        if matched_batch is not None and len(matched_batch):
            matched_keys = matched_batch.loc[:, identity].astype("string").fillna("<NA>")
            if bool(matched_keys.duplicated(keep=False).any()):
                raise ValueError(f"final match identity is not unique in batch {key}")
            for row in matched_keys.itertuples(index=False, name=None):
                identity_key = tuple(row)
                if identity_key not in positions:
                    raise ValueError(f"final match cannot be joined to source batch {key}")
                selected_positions.append(positions[identity_key])
        bitmap = bytearray((len(batch) + 7) // 8)
        for position in selected_positions:
            bitmap[position // 8] |= 1 << (position % 8)
        result[key] = {
            "count": len(selected_positions),
            "bitmap_hex": bytes(bitmap).hex(),
        }
    return result


def _metric_selector_mask(
    frame: pd.DataFrame,
    *,
    columns: list[str],
    selected_keys: set[str],
) -> pd.Series:
    if not selected_keys:
        return pd.Series(True, index=frame.index, dtype=bool)
    keys = _metric_selector_keys(frame, columns=columns)
    return pd.Series((key in selected_keys for key in keys), index=frame.index, dtype=bool)


def _filter_metric_frames_by_sample_keys(
    metric_source: pd.DataFrame,
    eligible_source: pd.DataFrame,
    matches: pd.DataFrame,
    *,
    columns: list[str],
    selected_keys: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply a source-row selector when matching output omits some stable-id fields."""

    selected_source = metric_source.loc[
        _metric_selector_mask(
            metric_source,
            columns=columns,
            selected_keys=selected_keys,
        )
    ].copy()
    selected_eligible = eligible_source.loc[
        _metric_selector_mask(
            eligible_source,
            columns=columns,
            selected_keys=selected_keys,
        )
    ].copy()

    match_columns = [
        name
        for name in columns
        if name in selected_source.columns and name in matches.columns
    ]
    if not match_columns:
        raise ValueError(
            "scenario metric selector has no stable id columns shared with matches"
        )
    match_keys = set(
        _metric_selector_keys(selected_source, columns=match_columns)
    )
    selected_matches = matches.loc[
        _metric_selector_mask(
            matches,
            columns=match_columns,
            selected_keys=match_keys,
        )
    ].copy()
    return selected_source, selected_eligible, selected_matches


def _metric_selector_keys(frame: pd.DataFrame, *, columns: list[str]) -> list[str]:
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"metric selector columns are missing from evaluator frame: {missing}")
    keys: list[str] = []
    for values in frame.loc[:, columns].itertuples(index=False, name=None):
        normalized = ["<NA>" if pd.isna(value) else str(value) for value in values]
        keys.append(json.dumps(normalized, ensure_ascii=True, separators=(",", ":")))
    return keys


STREAM_METRIC_COLUMNS = (
    "city_id",
    "product_id",
    "order_id",
    "driver_id",
    "group_id",
    "timestamp",
    "batch_id",
    "eta",
    "cr",
    "dar",
    "pcaa",
    "dcaa",
    "gmv",
    "pre_total_fee",
    "driver_dynamic_times",
    "if_broadcast",
    "is_broadcasted",
)


def _shared_data_path(path: str | Path) -> Path:
    return remap_shared_path(
        path,
        code_root=ROOT_DIR,
        shared_root=SHARED_ROOT_DIR,
        shared_dir_names=("data",),
    )


def _shared_output_path(path: str | Path) -> Path:
    return remap_shared_path(
        path,
        code_root=ROOT_DIR,
        shared_root=SHARED_ROOT_DIR,
        shared_dir_names=("outputs",),
    )


def _load_engine_module(engine_path: Path):
    engine_root = str(engine_path)
    while engine_root in sys.path:
        sys.path.remove(engine_root)
    sys.path.insert(0, engine_root)
    for name in list(sys.modules):
        if name == "policies" or name.startswith("policies."):
            sys.modules.pop(name, None)
    module_name = f"_fd_engine_{abs(hash(str(engine_path)))}"
    spec = importlib.util.spec_from_file_location(module_name, engine_path / "engine.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load engine.py from {engine_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "run_batch"):
        raise AttributeError(f"Engine module missing run_batch: {engine_path / 'engine.py'}")
    return module


def _frame_from_batches(batches: Any) -> pd.DataFrame:
    if isinstance(batches, pd.DataFrame):
        return pd.DataFrame(batches)
    return pd.DataFrame(batches)


def _normalize_product_id(value: Any) -> int | str:
    if value is None:
        return DEFAULT_PRODUCT_ID
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "all":
            return "all"
        value = normalized
    if isinstance(value, bool):
        raise ValueError("product_id must be an integer product ID or 'all'")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("product_id must be an integer product ID or 'all'") from exc


def _scope_product_rows(df: pd.DataFrame, product_id: int | str) -> pd.DataFrame:
    if product_id == "all":
        return df.copy()
    if "product_id" not in df.columns:
        raise ValueError("full_dispatch evaluation requires product_id unless product_id='all'")
    values = pd.to_numeric(df["product_id"], errors="coerce")
    return df.loc[values == int(product_id)].copy()


def _batch_count(df: pd.DataFrame, raw_batches: Any = None) -> int:
    if "batch_id" in df.columns:
        return int(df["batch_id"].nunique(dropna=False))
    if isinstance(raw_batches, list):
        return len(raw_batches)
    return 1 if len(df) else 0


def _read_debug_dataset(
    data_path: str | Path,
    *,
    max_batches: int | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    path = _shared_data_path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Full dispatch debug dataset does not exist: {path}")
    row_limit = _validate_candidate_max_rows(max_rows) if max_rows is not None else None

    if max_batches is None:
        if row_limit is not None:
            return canonicalize_feature_columns(
                _read_candidate_bounded_dataset(path, max_rows=row_limit)
            )
        try:
            return canonicalize_feature_columns(pd.read_csv(path, engine="pyarrow"))
        except Exception:
            return canonicalize_feature_columns(pd.read_csv(path))

    selected_batches: list[Any] = []
    chunks: list[pd.DataFrame] = []
    seen: set[str] = set()
    selected_rows = 0
    for chunk in pd.read_csv(path, chunksize=50_000):
        if "batch_id" not in chunk.columns:
            selected_rows += int(len(chunk))
            if row_limit is not None:
                _enforce_candidate_row_limit(selected_rows, max_rows=row_limit)
            chunks.append(chunk)
            break
        if len(selected_batches) < max_batches:
            for batch_id in chunk["batch_id"].drop_duplicates().tolist():
                key = str(batch_id)
                if key in seen:
                    continue
                seen.add(key)
                selected_batches.append(batch_id)
                if len(selected_batches) >= max_batches:
                    break
        if seen:
            batch_keys = chunk["batch_id"].astype(str)
            selected = chunk.loc[batch_keys.isin(seen)].copy()
            selected_rows += int(len(selected))
            if row_limit is not None:
                _enforce_candidate_row_limit(selected_rows, max_rows=row_limit)
            chunks.append(selected)
    if not chunks:
        return pd.DataFrame()
    data = pd.concat(chunks, axis=0, ignore_index=True)
    if "batch_id" in data.columns:
        data = data.loc[data["batch_id"].astype(str).isin(seen)].copy()
    if row_limit is not None:
        _enforce_candidate_row_limit(len(data), max_rows=row_limit)
    return canonicalize_feature_columns(data)


def _candidate_row_limit_error(max_rows: int) -> ValueError:
    return ValueError(
        f"candidate-only in-memory evaluation is limited to {max_rows} rows; "
        "use the partitioned adapter for full-data evaluation"
    )


def _validate_candidate_max_rows(max_rows: int) -> int:
    if isinstance(max_rows, bool) or not isinstance(max_rows, (int, np.integer)):
        raise ValueError("candidate-only max_rows must be a positive integer")
    normalized = int(max_rows)
    if normalized <= 0:
        raise ValueError("candidate-only max_rows must be a positive integer")
    return normalized


def _read_candidate_bounded_dataset(
    data_path: str | Path,
    *,
    max_rows: int,
) -> pd.DataFrame:
    path = _shared_data_path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Full dispatch debug dataset does not exist: {path}")
    data = pd.read_csv(path, nrows=max_rows + 1, low_memory=False)
    if len(data) > max_rows:
        raise _candidate_row_limit_error(max_rows)
    return data


def _enforce_candidate_row_limit(rows: int, *, max_rows: int) -> None:
    if rows > max_rows:
        raise _candidate_row_limit_error(max_rows)


def _trim_metric_frame(df: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in STREAM_METRIC_COLUMNS if column in df.columns]
    return df.loc[:, columns].copy()


def _iter_batch_aligned_csv(path: Path, *, chunksize: int):
    carry: pd.DataFrame | None = None
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        chunk = canonicalize_feature_columns(chunk)
        if carry is not None and len(carry):
            chunk = pd.concat([carry, chunk], axis=0, ignore_index=True)
            carry = None
        if "batch_id" not in chunk.columns or chunk.empty:
            yield chunk
            continue
        last_batch_id = chunk["batch_id"].iloc[-1]
        complete_mask = chunk["batch_id"] != last_batch_id
        complete = chunk.loc[complete_mask].copy()
        carry = chunk.loc[~complete_mask].copy()
        if len(complete):
            yield complete
    if carry is not None and len(carry):
        yield carry


def _partition_csv_by_batch(
    path: Path,
    *,
    partition_dir: Path,
    chunksize: int,
    partitions: int,
) -> list[Path]:
    partition_dir.mkdir(parents=True, exist_ok=True)
    written = [False] * partitions
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        chunk = canonicalize_feature_columns(chunk)
        if "batch_id" not in chunk.columns:
            part_path = partition_dir / "part_000.csv"
            chunk.to_csv(part_path, mode="a", header=not written[0], index=False)
            written[0] = True
            continue
        batch_hash = pd.util.hash_pandas_object(chunk["batch_id"].astype("string"), index=False).to_numpy(
            dtype=np.uint64,
            copy=False,
        )
        partition_ids = np.mod(batch_hash, partitions)
        for partition_id in np.unique(partition_ids):
            index = int(partition_id)
            part_path = partition_dir / f"part_{index:03d}.csv"
            part = chunk.loc[partition_ids == partition_id]
            part.to_csv(part_path, mode="a", header=not written[index], index=False)
            written[index] = True
    return [partition_dir / f"part_{index:03d}.csv" for index, exists in enumerate(written) if exists]


def _merge_count_dict(target: dict[str, int], values: dict[str, Any]) -> None:
    for key, value in values.items():
        target[str(key)] = target.get(str(key), 0) + int(value)


def _prepare_engine_config(engine_module: Any, config: dict[str, Any]) -> Any:
    if hasattr(engine_module, "load_active_config"):
        return engine_module.load_active_config(config.get("active_config_dir"))
    return config


def _notify_progress(
    callback: Callable[[str, int, int], Any] | None,
    stage: str,
    current: int,
    total: int,
) -> None:
    if callback is None:
        return
    try:
        callback(stage, int(current), int(total))
    except Exception:
        pass


def _run_engine_batches(
    engine_module: Any,
    df: pd.DataFrame,
    engine_config: Any,
    *,
    trace: bool,
    show_progress: bool = False,
    batch_mode: str = "per_batch",
    progress_callback: Callable[[str, int, int], Any] | None = None,
    normalize_adjusted_stage: bool = True,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if df.empty:
        _notify_progress(progress_callback, "engine_start", 0, 0)
        _notify_progress(progress_callback, "engine_end", 0, 0)
        return df.copy(), []

    if batch_mode == "whole_file":
        _notify_progress(progress_callback, "engine_start", 0, 1)
        accepts_config = "config" in inspect.signature(engine_module.run_batch).parameters
        if accepts_config:
            result = engine_module.run_batch(df.copy(), config=engine_config, trace=trace)
        else:
            result = engine_module.run_batch(df.copy(), trace=trace)
        if isinstance(result, tuple):
            engine_df, batch_trace = result
        else:
            engine_df, batch_trace = result, None
        traces = [dict(batch_trace)] if isinstance(batch_trace, dict) else []
        _notify_progress(progress_callback, "engine_batch", 1, 1)
        _notify_progress(progress_callback, "engine_end", 1, 1)
        return pd.DataFrame(engine_df).copy(), traces
    if batch_mode != "per_batch":
        raise ValueError("engine_batch_mode must be 'per_batch' or 'whole_file'")

    grouped = df.groupby("batch_id", sort=False, dropna=False) if "batch_id" in df.columns else [(None, df)]
    total_batches = grouped.ngroups if hasattr(grouped, "ngroups") else len(grouped)
    _notify_progress(progress_callback, "engine_start", 0, total_batches)
    grouped_iter = grouped
    if show_progress:
        try:
            from tqdm.auto import tqdm

            total = grouped.ngroups if hasattr(grouped, "ngroups") else None
            grouped_iter = tqdm(
                grouped,
                total=total,
                desc="dispatch_engine",
                unit="batch",
                mininterval=5.0,
                dynamic_ncols=True,
            )
        except ImportError:
            grouped_iter = grouped
    frames: list[pd.DataFrame] = []
    traces: list[dict[str, Any]] = []
    accepts_config = "config" in inspect.signature(engine_module.run_batch).parameters
    for current_batch, (batch_id, batch_df) in enumerate(grouped_iter, start=1):
        if accepts_config:
            result = engine_module.run_batch(batch_df.copy(), config=engine_config, trace=trace)
        else:
            result = engine_module.run_batch(batch_df.copy(), trace=trace)
        if isinstance(result, tuple):
            engine_df, batch_trace = result
        else:
            engine_df, batch_trace = result, None
        engine_df = pd.DataFrame(engine_df).copy()
        if (
            normalize_adjusted_stage
            and "adjusted_stage" not in engine_df.columns
            and "stage" in engine_df.columns
        ):
            engine_df["adjusted_stage"] = engine_df["stage"]
        frames.append(engine_df)
        if isinstance(batch_trace, dict):
            trace_payload = dict(batch_trace)
            if "batch_id" not in trace_payload:
                trace_payload["batch_id"] = batch_id
            traces.append(trace_payload)
        _notify_progress(progress_callback, "engine_batch", current_batch, total_batches)
    _notify_progress(progress_callback, "engine_end", total_batches, total_batches)
    if not frames:
        return df.iloc[0:0].copy(), traces
    return pd.concat(frames, axis=0, ignore_index=True), traces


def _eligible_rows(engine_df: pd.DataFrame) -> pd.DataFrame:
    if "is_filtered" not in engine_df.columns:
        return engine_df.copy()
    mask = ~engine_df["is_filtered"].astype(bool)
    return engine_df.loc[mask].copy()


def _filter_counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in df.columns:
        return {}
    return dict(Counter(df[column].dropna().astype(str).tolist()))


def _trace_counts(traces: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for trace in traces:
        values = trace.get(key, {})
        if isinstance(values, dict):
            counts.update({str(name): int(count) for name, count in values.items()})
    return dict(counts)


def _engine_diagnostics(
    source_df: pd.DataFrame,
    engine_df: pd.DataFrame,
    traces: list[dict[str, Any]],
) -> dict[str, Any]:
    eligible = _eligible_rows(engine_df)
    trace_filtered_rows = sum(int(item.get("filtered_count", 0)) for item in traces)
    filtered_rows = trace_filtered_rows if traces else int(len(engine_df) - len(eligible))
    filter_rule_counts = _filter_counts(engine_df, "filter_rule") or _trace_counts(traces, "filter_rule_counts")
    filter_policy_counts = _filter_counts(engine_df, "filter_policy") or _trace_counts(
        traces, "filter_policy_counts"
    )
    diagnostics: dict[str, Any] = {
        "input_rows": int(len(source_df)),
        "engine_rows": int(len(engine_df)),
        "eligible_rows": int(len(eligible)),
        "filtered_rows": int(filtered_rows),
        "filter_rule_counts": filter_rule_counts,
        "filter_policy_counts": filter_policy_counts,
        "trace_batches": int(len(traces)),
        "pre_matching_boundary": True,
    }
    if traces:
        if traces[0].get("active_config_count") is not None:
            diagnostics["active_config_count"] = traces[0]["active_config_count"]
        if traces[0].get("policy_count") is not None:
            diagnostics["policy_count"] = traces[0]["policy_count"]
        diagnostics["policy_version"] = traces[0].get("policy_version")
        diagnostics["lock_config_versions"] = traces[0].get("lock_config_versions", {})
        diagnostics["missing_feature_rows"] = int(
            sum(int(item.get("missing_feature_rows", 0)) for item in traces)
        )
    if "weight" in engine_df.columns and len(engine_df):
        weight = safe_float_series(engine_df, "weight")
        diagnostics["weight"] = {
            "min": float(weight.min()),
            "max": float(weight.max()),
            "mean": float(weight.mean()),
        }
    if "adjusted_stage" in engine_df.columns and len(engine_df):
        stage = safe_float_series(engine_df, "adjusted_stage")
        diagnostics["adjusted_stage"] = {
            "min": float(stage.min()),
            "max": float(stage.max()),
            "mean": float(stage.mean()),
        }
    return diagnostics


def _match_with_backend(
    df: pd.DataFrame,
    *,
    backend: str,
    config: dict[str, Any],
) -> pd.DataFrame:
    if backend == "local":
        return local_stage_major_match(df, stage_col="adjusted_stage", weight_col="weight")
    raise ValueError(f"Unsupported match_backend: {backend}")




def _validation_only_result(
    *,
    engine_path: Path,
    df: pd.DataFrame,
    raw_batches: Any,
    config: dict[str, Any],
    trace: bool,
) -> dict[str, Any]:
    return {
        "engine_dir": str(engine_path),
        "status": "validated",
        "batches": _batch_count(df, raw_batches),
        "rows": int(len(df)),
        "trace": trace,
        "config_keys": sorted(config.keys()),
    }


def _write_outputs(
    result: dict[str, Any],
    *,
    output_dir: str | Path,
    matches: pd.DataFrame,
    baseline_matches: pd.DataFrame,
    unfiltered_baseline_matches: pd.DataFrame,
    write_matches: bool,
) -> dict[str, Any]:
    path = _shared_output_path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    result = dict(result)
    result["output_dir"] = str(path)
    metrics_path = path / "metrics.json"
    result["metrics_path"] = str(metrics_path)
    metrics_path.write_text(json.dumps(jsonable(result), indent=2, ensure_ascii=False), encoding="utf-8")

    if write_matches:
        match_path = path / "candidate_matches.csv"
        baseline_path = path / "baseline_matches.csv"
        unfiltered_baseline_path = path / "unfiltered_baseline_matches.csv"
        matches.to_csv(match_path, index=False)
        baseline_matches.to_csv(baseline_path, index=False)
        unfiltered_baseline_matches.to_csv(unfiltered_baseline_path, index=False)
        result["matches_path"] = str(match_path)
        result["baseline_matches_path"] = str(baseline_path)
        result["unfiltered_baseline_matches_path"] = str(unfiltered_baseline_path)
        metrics_path.write_text(json.dumps(jsonable(result), indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _evaluate_engine_codebase_streaming_csv(
    engine_path: Path,
    *,
    data_path: str | Path,
    config: dict[str, Any],
    trace: bool,
    output_dir: str | Path | None,
    write_matches: bool,
    chunksize: int,
) -> dict[str, Any]:
    total_started = time.perf_counter()
    backend = str(config.get("match_backend", "local")).lower()
    path = _shared_data_path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Full dispatch debug dataset does not exist: {path}")

    engine_module = _load_engine_module(engine_path)
    engine_config = _prepare_engine_config(engine_module, config)
    engine_batch_mode = str(config.get("engine_batch_mode", "whole_file")).replace("-", "_")
    partition_count = max(int(config.get("streaming_csv_partitions", 64)), 1)
    if output_dir is not None:
        partition_dir = _shared_output_path(output_dir) / "_stream_partitions"
    else:
        partition_dir = Path(tempfile.mkdtemp(prefix="full_dispatch_stream_"))

    source_parts: list[pd.DataFrame] = []
    eligible_source_parts: list[pd.DataFrame] = []
    baseline_parts: list[pd.DataFrame] = []
    unfiltered_baseline_parts: list[pd.DataFrame] = []
    candidate_parts: list[pd.DataFrame] = []

    input_rows = 0
    matching_rows = 0
    batches = 0
    eligible_rows = 0
    filtered_rows = 0
    trace_batches = 0
    filter_rule_counts: dict[str, int] = {}
    filter_policy_counts: dict[str, int] = {}
    missing_feature_rows = 0
    data_read_seconds = 0.0
    engine_seconds = 0.0
    baseline_match_seconds = 0.0
    unfiltered_baseline_match_seconds = 0.0
    candidate_match_seconds = 0.0

    partition_started = time.perf_counter()
    partition_paths = _partition_csv_by_batch(
        path,
        partition_dir=partition_dir,
        chunksize=chunksize,
        partitions=partition_count,
    )
    data_read_seconds += float(time.perf_counter() - partition_started)

    try:
        partition_iter = enumerate(partition_paths, start=1)
        for chunk_index, partition_path in partition_iter:
            raw_chunk = canonicalize_feature_columns(pd.read_csv(partition_path, low_memory=False))
            if raw_chunk.empty:
                continue
            chunk_started = time.perf_counter()
            input_df = ensure_metric_columns(raw_chunk)
            data_read_seconds += float(time.perf_counter() - chunk_started)
            if not has_match_columns(input_df):
                return _validation_only_result(
                    engine_path=engine_path,
                    df=input_df,
                    raw_batches=None,
                    config=config,
                    trace=trace,
                )

            chunk_input_rows = int(len(input_df))
            input_rows += chunk_input_rows
            df = _scope_product_rows(input_df, "all")
            df["online_weight"] = safe_float_series(df, "weight")
            chunk_batches = _batch_count(df, None)
            batches += chunk_batches
            matching_rows += int(len(df))

            engine_started = time.perf_counter()
            engine_df, traces = _run_engine_batches(
                engine_module,
                df,
                engine_config,
                trace=trace,
                show_progress=False,
                batch_mode=engine_batch_mode,
            )
            engine_seconds += float(time.perf_counter() - engine_started)
            engine_df = ensure_metric_columns(engine_df)
            eligible_df = _eligible_rows(engine_df)
            eligible_rows += int(len(eligible_df))

            if traces:
                trace_batches += int(len(traces))
                for trace_payload in traces:
                    filtered_rows += int(trace_payload.get("filtered_count", 0))
                    missing_feature_rows += int(trace_payload.get("missing_feature_rows", 0))
                    _merge_count_dict(filter_rule_counts, trace_payload.get("filter_rule_counts", {}))
                    _merge_count_dict(filter_policy_counts, trace_payload.get("filter_policy_counts", {}))
            else:
                filtered_rows += int(len(engine_df) - len(eligible_df))

            online_baseline_df = eligible_df.copy()
            online_baseline_df["weight"] = safe_float_series(online_baseline_df, "online_weight")
            baseline_started = time.perf_counter()
            all_baseline_matches = _match_with_backend(online_baseline_df, backend=backend, config=config)
            baseline_match_seconds += float(time.perf_counter() - baseline_started)

            unfiltered_baseline_df = df.copy()
            unfiltered_baseline_df["weight"] = safe_float_series(unfiltered_baseline_df, "online_weight")
            unfiltered_baseline_started = time.perf_counter()
            all_unfiltered_baseline_matches = _match_with_backend(unfiltered_baseline_df, backend=backend, config=config)
            unfiltered_baseline_match_seconds += float(time.perf_counter() - unfiltered_baseline_started)

            candidate_started = time.perf_counter()
            all_candidate_matches = _match_with_backend(eligible_df, backend=backend, config=config)
            candidate_match_seconds += float(time.perf_counter() - candidate_started)

            metric_source_df = _scope_product_rows(df, DEFAULT_PRODUCT_ID)
            eligible_metric_source_df = _scope_product_rows(eligible_df, DEFAULT_PRODUCT_ID)
            source_parts.append(_trim_metric_frame(metric_source_df))
            eligible_source_parts.append(_trim_metric_frame(eligible_metric_source_df))
            baseline_parts.append(_trim_metric_frame(_scope_product_rows(all_baseline_matches, DEFAULT_PRODUCT_ID)))
            unfiltered_baseline_parts.append(
                _trim_metric_frame(_scope_product_rows(all_unfiltered_baseline_matches, DEFAULT_PRODUCT_ID))
            )
            candidate_parts.append(_trim_metric_frame(_scope_product_rows(all_candidate_matches, DEFAULT_PRODUCT_ID)))

            if config.get("show_progress"):
                print(
                    f"[stream partition {chunk_index}/{len(partition_paths)}] "
                    f"rows={chunk_input_rows} batches={chunk_batches} "
                    f"eligible={len(eligible_df)} filtered_total={filtered_rows}"
                )
    finally:
        if partition_dir.exists():
            shutil.rmtree(partition_dir, ignore_errors=True)

    def _concat(parts: list[pd.DataFrame]) -> pd.DataFrame:
        if not parts:
            return pd.DataFrame(columns=list(STREAM_METRIC_COLUMNS))
        return pd.concat(parts, axis=0, ignore_index=True)

    metric_source_df = _concat(source_parts)
    eligible_metric_source_df = _concat(eligible_source_parts)
    baseline_matches = _concat(baseline_parts)
    unfiltered_baseline_matches = _concat(unfiltered_baseline_parts)
    candidate_matches = _concat(candidate_parts)

    baseline_metrics = summarize_matches(baseline_matches, source_df=metric_source_df)
    unfiltered_baseline_metrics = summarize_matches(unfiltered_baseline_matches, source_df=metric_source_df)
    metrics = summarize_matches(candidate_matches, source_df=metric_source_df)
    objective_prompt_evidence = _objective_prompt_evidence(
        candidate_matches, metric_source_df,
    )
    eligible_metrics = summarize_matches(candidate_matches, source_df=eligible_metric_source_df)

    candidate_total_seconds = engine_seconds + candidate_match_seconds
    elapsed_seconds = float(time.perf_counter() - total_started)
    result: dict[str, Any] = {
        "engine_dir": str(engine_path),
        "status": "ok",
        "match_backend": backend,
        "product_id": "all",
        "matching_product_id": "all",
        "metric_product_id": DEFAULT_PRODUCT_ID,
        "input_rows": input_rows,
        "rows": matching_rows,
        "matching_rows": matching_rows,
        "metric_source_rows": int(len(metric_source_df)),
        "excluded_product_rows": 0,
        "batches": batches,
        "eligible_rows": eligible_rows,
        "filtered_rows": filtered_rows,
        "config_keys": sorted(config.keys()),
        "engine_batch_mode": engine_batch_mode,
        "streaming_csv_chunksize": chunksize,
        "streaming_csv_partitions": partition_count,
        "metrics": metrics,
        "objective_prompt_evidence": objective_prompt_evidence,
        "eligible_metrics": eligible_metrics,
        "baseline_metrics": baseline_metrics,
        "unfiltered_baseline_metrics": unfiltered_baseline_metrics,
        "deltas": metric_deltas(metrics, baseline_metrics),
        "policy_deltas": metric_deltas(baseline_metrics, unfiltered_baseline_metrics),
        "breakdowns": build_breakdowns(candidate_matches, metric_source_df),
        "eligible_breakdowns": build_breakdowns(candidate_matches, eligible_metric_source_df),
        "baseline_breakdowns": build_breakdowns(baseline_matches, metric_source_df),
        "unfiltered_baseline_breakdowns": build_breakdowns(unfiltered_baseline_matches, metric_source_df),
        "engine_diagnostics": {
            "input_rows": input_rows,
            "engine_rows": eligible_rows,
            "eligible_rows": eligible_rows,
            "filtered_rows": filtered_rows,
            "filter_rule_counts": filter_rule_counts,
            "filter_policy_counts": filter_policy_counts,
            "trace_batches": trace_batches,
            "missing_feature_rows": missing_feature_rows,
            "pre_matching_boundary": True,
        },
        "timings": {
            "elapsed_seconds": elapsed_seconds,
            "data_read_seconds": data_read_seconds,
            "engine_seconds": engine_seconds,
            "online_weight_baseline_match_seconds": baseline_match_seconds,
            "unfiltered_online_weight_baseline_match_seconds": unfiltered_baseline_match_seconds,
            "candidate_match_seconds": candidate_match_seconds,
            "online_weight_baseline_total_seconds": baseline_match_seconds,
            "dispatch_engines_current_total_seconds": candidate_total_seconds,
        },
    }

    if output_dir is not None:
        result = _write_outputs(
            result,
            output_dir=output_dir,
            matches=candidate_matches,
            baseline_matches=baseline_matches,
            unfiltered_baseline_matches=unfiltered_baseline_matches,
            write_matches=write_matches,
        )
    return jsonable(result)


def _candidate_engine_diagnostics(
    source_df: pd.DataFrame,
    rebuilt_df: pd.DataFrame,
    *,
    candidate_traces: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize only evaluator-trusted source and rebuilt candidate values."""
    batch_count = _batch_count(source_df)
    diagnostics: dict[str, Any] = {
        "input_rows": int(len(source_df)),
        "engine_rows": int(len(rebuilt_df)),
        "eligible_rows": int(len(rebuilt_df)),
        "filtered_rows": int(len(source_df) - len(rebuilt_df)),
        "filter_rule_counts": {},
        "filter_policy_counts": {},
        "evaluated_batches": batch_count,
        "trace_batches": int(len(candidate_traces)),
        "pre_matching_boundary": True,
    }
    if "weight" in rebuilt_df.columns and len(rebuilt_df):
        weight = safe_float_series(rebuilt_df, "weight")
        diagnostics["weight"] = {
            "min": float(weight.min()),
            "max": float(weight.max()),
            "mean": float(weight.mean()),
        }
    if "adjusted_stage" in rebuilt_df.columns and len(rebuilt_df):
        stage = safe_float_series(rebuilt_df, "adjusted_stage")
        diagnostics["adjusted_stage"] = {
            "min": float(stage.min()),
            "max": float(stage.max()),
            "mean": float(stage.mean()),
        }
    return diagnostics


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_candidate_metrics(
    result: dict[str, Any],
    *,
    output_dir: str | Path,
    total_started: float | None = None,
) -> dict[str, Any]:
    path = _shared_output_path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    written = dict(result)
    written["output_dir"] = str(path)
    metrics_path = path / "metrics.json"
    written["metrics_path"] = str(metrics_path)
    serializable = jsonable(written)
    write_started = time.perf_counter()
    _atomic_write_json(metrics_path, serializable)
    write_finished = time.perf_counter()
    timings = serializable.get("timings")
    if total_started is not None and isinstance(timings, dict):
        timings["artifact_write_seconds"] = float(write_finished - write_started)
        timings["elapsed_seconds"] = float(write_finished - total_started)
        _atomic_write_json(metrics_path, serializable)
    return serializable


def evaluate_engine_candidate_only(
    engine_dir: str | Path,
    batches: Any = None,
    *,
    config: dict[str, Any] | None = None,
    trace: bool = True,
    output_dir: str | Path | None = None,
    progress_callback: ProgressCallback | None = None,
    candidate_runner: CandidateRunner | None = None,
    trusted_in_process: bool = False,
    max_rows: int = CANDIDATE_ONLY_DEFAULT_MAX_ROWS,
) -> dict[str, Any]:
    """Evaluate one bounded candidate partition without exposing trusted columns."""
    total_started = time.perf_counter()
    engine_path = validate_engine_codebase(engine_dir)
    config = dict(config or {})
    backend = str(config.get("match_backend", "local")).lower()
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"match_backend must be one of {SUPPORTED_BACKENDS}, got {backend!r}")

    engine_batch_mode = str(config.get("engine_batch_mode", "per_batch"))
    if engine_batch_mode != "per_batch":
        raise ValueError("candidate-only engine_batch_mode must be 'per_batch'")
    evaluation_detail = str(config.get("evaluation_detail", "full"))
    if evaluation_detail not in {"full", "metrics_only"}:
        raise ValueError("candidate-only evaluation_detail must be 'full' or 'metrics_only'")
    if config.get("streaming_csv_chunksize") is not None:
        raise ValueError(
            "streaming_csv_chunksize is not implemented for candidate-only evaluation"
        )
    if candidate_runner is not None and trusted_in_process:
        raise ValueError("candidate_runner and trusted_in_process are mutually exclusive")
    if candidate_runner is None and not trusted_in_process:
        raise ValueError(
            "candidate_runner is required unless trusted_in_process=True is explicitly "
            "selected for smoke-only evaluation"
        )
    if candidate_runner is not None and not callable(candidate_runner):
        raise TypeError("candidate_runner must be callable")
    execution_isolation = (
        str(getattr(candidate_runner, "execution_isolation", "candidate_runner"))
        if candidate_runner is not None
        else "trusted_in_process_smoke_only"
    )
    candidate_execution: dict[str, Any] | None = None
    row_limit = _validate_candidate_max_rows(max_rows)

    read_started = time.perf_counter()
    if batches is None:
        data_path = config.get("data_path", DEFAULT_DATA_PATH)
        max_batches = config.get("max_batches")
        if max_batches:
            raw_df = _read_debug_dataset(
                data_path,
                max_batches=int(max_batches),
                max_rows=row_limit,
            )
        else:
            raw_df = _read_candidate_bounded_dataset(data_path, max_rows=row_limit)
        raw_batches = None
    else:
        if isinstance(batches, (pd.DataFrame, list, tuple)):
            _enforce_candidate_row_limit(len(batches), max_rows=row_limit)
        raw_df = _frame_from_batches(batches)
        raw_batches = batches
    _enforce_candidate_row_limit(len(raw_df), max_rows=row_limit)
    canonical_df = canonicalize_feature_columns(raw_df)
    # canonicalize_feature_columns already returns an evaluator-owned frame.
    # Add missing metric aliases in place instead of copying all source columns
    # a second time.
    input_df = ensure_metric_columns(canonical_df, copy=False)
    data_read_seconds = float(time.perf_counter() - read_started)
    if not has_match_columns(input_df):
        raise ValueError("candidate-only evaluation requires full match metric columns")

    matching_product_id = _normalize_product_id(
        config.get("product_id", DEFAULT_PRODUCT_ID)
    )
    if matching_product_id not in (DEFAULT_PRODUCT_ID, "all"):
        raise ValueError("local KM product_id must be 1 or 'all'")
    metric_product_id = DEFAULT_PRODUCT_ID
    input_rows = int(len(input_df))
    # ColumnPolicy.prepare creates the protected shadow before candidate execution.
    scoped_df = (
        input_df
        if matching_product_id == "all"
        else _scope_product_rows(input_df, matching_product_id)
    )

    column_policy = ColumnPolicy()
    prepared = column_policy.prepare(scoped_df)
    engine_started = time.perf_counter()
    if candidate_runner is not None:
        runner_result = candidate_runner(
            engine_path,
            prepared.visible,
            trace,
            progress_callback,
        )
        if not isinstance(runner_result, tuple) or len(runner_result) != 2:
            raise TypeError("candidate_runner must return (candidate_output, traces)")
        raw_candidate_output, raw_candidate_traces = runner_result
        candidate_output = pd.DataFrame(raw_candidate_output)
        candidate_traces = [
            dict(item)
            for item in (raw_candidate_traces or [])
            if isinstance(item, dict)
        ]
        raw_execution = getattr(candidate_runner, "last_run_metadata", None)
        if isinstance(raw_execution, dict):
            allowed_execution_keys = {
                "execution_isolation",
                "candidate_runner_fingerprint",
                "candidate_runner_protocol",
                "manifest_path",
                "run_dir",
                "candidate_child_profile",
                "status",
                "elapsed_seconds",
                "returncode",
                "input_sha256",
                "output_sha256",
                "input_rows",
                "output_rows",
                "trace_count",
                "callback_errors",
                "log_overflow",
                "wall_timeout_seconds",
                "batch_timeout_seconds",
                "max_message_bytes",
                "max_trace_bytes",
                "max_log_bytes",
            }
            candidate_execution = {
                str(key): value
                for key, value in raw_execution.items()
                if key in allowed_execution_keys
            }
    else:
        engine_module = _load_engine_module(engine_path)
        candidate_output, candidate_traces = _run_engine_batches(
            engine_module,
            prepared.visible,
            {},
            trace=trace,
            show_progress=False,
            batch_mode=engine_batch_mode,
            progress_callback=progress_callback,
            normalize_adjusted_stage=False,
        )
    rebuilt_df = column_policy.validate_and_rebuild(prepared, candidate_output)
    engine_seconds = float(time.perf_counter() - engine_started)

    candidate_started = time.perf_counter()
    match_config = dict(config)
    match_config["show_progress"] = False
    all_candidate_matches = _match_with_backend(
        rebuilt_df,
        backend=backend,
        config=match_config,
    )
    candidate_match_seconds = float(time.perf_counter() - candidate_started)

    metric_source_df = _scope_product_rows(scoped_df, metric_product_id)
    eligible_metric_source_df = _scope_product_rows(rebuilt_df, metric_product_id)
    candidate_matches = _scope_product_rows(all_candidate_matches, metric_product_id)
    if candidate_traces:
        match_references = _final_match_references(candidate_matches)
        match_row_evidence = _final_match_row_evidence(scoped_df, candidate_matches)
        for trace_item in candidate_traces:
            batch_key = str(trace_item.get("batch_id"))
            trace_item["final_match_references"] = match_references.get(batch_key, [])
            trace_item["final_match_count"] = len(trace_item["final_match_references"])
            trace_item["final_matched_target_rows"] = match_row_evidence.get(
                batch_key, {"count": 0, "bitmap_hex": ""}
            )
    selected_metric_keys = {
        str(item) for item in config.get("metric_sample_keys", ())
    }
    metric_id_columns = [str(item) for item in config.get("metric_id_columns", ())]
    if selected_metric_keys:
        if not metric_id_columns:
            raise ValueError("scenario metric selector requires stable id columns")
        (
            metric_source_df,
            eligible_metric_source_df,
            candidate_matches,
        ) = _filter_metric_frames_by_sample_keys(
            metric_source_df,
            eligible_metric_source_df,
            candidate_matches,
            columns=metric_id_columns,
            selected_keys=selected_metric_keys,
        )
    metrics = summarize_matches(candidate_matches, source_df=metric_source_df)
    if evaluation_detail == "metrics_only":
        elapsed_seconds = float(time.perf_counter() - total_started)
        result = {
            "engine_dir": str(engine_path),
            "status": "ok",
            "execution_isolation": execution_isolation,
            "match_backend": backend,
            "input_rows": input_rows,
            "rows": int(len(scoped_df)),
            "eligible_rows": int(len(rebuilt_df)),
            "metrics": metrics,
            "batch_traces": candidate_traces,
            "evaluation_detail": evaluation_detail,
            "timings": {
                "elapsed_seconds": elapsed_seconds,
                "data_read_seconds": data_read_seconds,
                "engine_seconds": engine_seconds,
                "candidate_match_seconds": candidate_match_seconds,
                "candidate_total_seconds": engine_seconds + candidate_match_seconds,
                "breakdown_seconds": 0.0,
                "artifact_write_seconds": 0.0,
            },
        }
        if candidate_execution is not None:
            result["candidate_execution"] = candidate_execution
        if output_dir is not None:
            return _write_candidate_metrics(
                result,
                output_dir=output_dir,
                total_started=total_started,
            )
        return jsonable(result)
    eligible_metrics = summarize_matches(
        candidate_matches,
        source_df=eligible_metric_source_df,
    )
    stability_blocks: list[dict[str, Any]] = []
    block_count = int(config.get("stability_block_count") or 0)
    if block_count:
        if "batch_id" not in metric_source_df or "order_create_timestamp" not in metric_source_df:
            raise ValueError("stability blocks require batch_id and order_create_timestamp")
        batch_order = (
            metric_source_df.assign(__batch_key=metric_source_df["batch_id"].astype(str))
            .groupby("__batch_key", sort=False)["order_create_timestamp"]
            .min()
            .sort_values(kind="stable")
            .index.to_numpy()
        )
        for block_index, block_ids in enumerate(np.array_split(batch_order, block_count)):
            selected_batches = set(map(str, block_ids.tolist()))
            def in_block(frame: pd.DataFrame) -> pd.Series:
                return frame["batch_id"].astype(str).isin(selected_batches)
            block_source = metric_source_df.loc[in_block(metric_source_df)]
            block_eligible = eligible_metric_source_df.loc[in_block(eligible_metric_source_df)]
            block_matches = candidate_matches.loc[in_block(candidate_matches)]
            stability_blocks.append({
                "block": block_index,
                "batch_count": len(selected_batches),
                "sample_count": len(block_source),
                "metrics": summarize_matches(block_matches, source_df=block_source),
                "eligible_metrics": summarize_matches(block_matches, source_df=block_eligible),
            })
    engine_diagnostics = _candidate_engine_diagnostics(
        scoped_df,
        rebuilt_df,
        candidate_traces=candidate_traces,
    )
    behavior_signature = _candidate_behavior_signature(
        metric_source_df,
        eligible_metric_source_df,
        candidate_matches,
    )

    breakdown_started = time.perf_counter()
    breakdowns = build_breakdowns(candidate_matches, metric_source_df)
    eligible_breakdowns = build_breakdowns(
        candidate_matches,
        eligible_metric_source_df,
    )
    breakdown_seconds = float(time.perf_counter() - breakdown_started)
    elapsed_seconds = float(time.perf_counter() - total_started)
    result: dict[str, Any] = {
        "engine_dir": str(engine_path),
        "status": "ok",
        "execution_isolation": execution_isolation,
        "full_data_capable": False,
        "max_rows": row_limit,
        "match_backend": backend,
        "product_id": matching_product_id,
        "matching_product_id": matching_product_id,
        "metric_product_id": metric_product_id,
        "input_rows": input_rows,
        "rows": int(len(scoped_df)),
        "matching_rows": int(len(scoped_df)),
        "metric_source_rows": int(len(metric_source_df)),
        "excluded_product_rows": int(input_rows - len(scoped_df)),
        "batches": _batch_count(scoped_df, raw_batches),
        "eligible_rows": int(len(rebuilt_df)),
        "filtered_rows": int(engine_diagnostics["filtered_rows"]),
        "config_keys": sorted(config.keys()),
        "engine_batch_mode": engine_batch_mode,
        "column_policy_fingerprint": column_policy.fingerprint,
        "metrics": metrics,
        "eligible_metrics": eligible_metrics,
        "breakdowns": breakdowns,
        "eligible_breakdowns": eligible_breakdowns,
        "engine_diagnostics": engine_diagnostics,
        "batch_traces": candidate_traces,
        "behavior_signature": behavior_signature,
        "stability_blocks": stability_blocks,
        "timings": {
            "elapsed_seconds": elapsed_seconds,
            "data_read_seconds": data_read_seconds,
            "engine_seconds": engine_seconds,
            "candidate_match_seconds": candidate_match_seconds,
            "candidate_total_seconds": engine_seconds + candidate_match_seconds,
            "breakdown_seconds": breakdown_seconds,
            "artifact_write_seconds": 0.0,
        },
    }
    if candidate_execution is not None:
        result["candidate_execution"] = candidate_execution

    if output_dir is not None:
        return _write_candidate_metrics(
            result,
            output_dir=output_dir,
            total_started=total_started,
        )
    return jsonable(result)


def evaluate_engine_codebase(
    engine_dir: str | Path,
    batches: Any = None,
    *,
    config: dict[str, Any] | None = None,
    trace: bool = True,
    output_dir: str | Path | None = None,
    write_matches: bool = False,
) -> dict[str, Any]:
    """Run a full_dispatch engine through the selected evaluator backend."""
    total_started = time.perf_counter()
    engine_path = validate_engine_codebase(engine_dir)
    config = dict(config or {})
    backend = str(config.get("match_backend", "local")).lower()
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"match_backend must be one of {SUPPORTED_BACKENDS}, got {backend!r}")

    read_started = time.perf_counter()
    if batches is None:
        data_path = config.get("data_path", DEFAULT_DATA_PATH)
        max_batches = config.get("max_batches")
        streaming_chunksize = config.get("streaming_csv_chunksize")
        if streaming_chunksize and max_batches is None:
            return _evaluate_engine_codebase_streaming_csv(
                engine_path,
                data_path=data_path,
                config=config,
                trace=trace,
                output_dir=output_dir,
                write_matches=write_matches,
                chunksize=int(streaming_chunksize),
            )
        df = _read_debug_dataset(data_path, max_batches=int(max_batches) if max_batches else None)
        raw_batches = None
    else:
        df = _frame_from_batches(batches)
        raw_batches = batches
    data_read_seconds = float(time.perf_counter() - read_started)

    df = canonicalize_feature_columns(df)
    input_df = ensure_metric_columns(df)
    if not has_match_columns(input_df):
        return _validation_only_result(
            engine_path=engine_path,
            df=input_df,
            raw_batches=raw_batches,
            config=config,
            trace=trace,
        )

    matching_product_id = _normalize_product_id(config.get("product_id", DEFAULT_PRODUCT_ID))
    if matching_product_id not in (DEFAULT_PRODUCT_ID, "all"):
        raise ValueError("local KM product_id must be 1 or 'all'")
    metric_product_id = DEFAULT_PRODUCT_ID
    input_rows = int(len(input_df))
    df = _scope_product_rows(input_df, matching_product_id)
    df["online_weight"] = safe_float_series(df, "weight")

    engine_module = _load_engine_module(engine_path)
    engine_config = _prepare_engine_config(engine_module, config)
    engine_batch_mode = str(config.get("engine_batch_mode", "per_batch")).replace("-", "_")
    engine_started = time.perf_counter()
    engine_df, traces = _run_engine_batches(
        engine_module,
        df,
        engine_config,
        trace=trace,
        show_progress=bool(config.get("show_progress", False)),
        batch_mode=engine_batch_mode,
    )
    engine_seconds = float(time.perf_counter() - engine_started)
    engine_df = ensure_metric_columns(engine_df)
    eligible_df = _eligible_rows(engine_df)

    online_baseline_df = eligible_df.copy()
    online_baseline_df["weight"] = safe_float_series(online_baseline_df, "online_weight")
    baseline_started = time.perf_counter()
    all_baseline_matches = _match_with_backend(online_baseline_df, backend=backend, config=config)
    baseline_match_seconds = float(time.perf_counter() - baseline_started)

    unfiltered_baseline_df = df.copy()
    unfiltered_baseline_df["weight"] = safe_float_series(unfiltered_baseline_df, "online_weight")
    unfiltered_baseline_started = time.perf_counter()
    all_unfiltered_baseline_matches = _match_with_backend(unfiltered_baseline_df, backend=backend, config=config)
    unfiltered_baseline_match_seconds = float(time.perf_counter() - unfiltered_baseline_started)

    candidate_started = time.perf_counter()
    all_candidate_matches = _match_with_backend(eligible_df, backend=backend, config=config)
    candidate_match_seconds = float(time.perf_counter() - candidate_started)

    metric_source_df = _scope_product_rows(df, metric_product_id)
    eligible_metric_source_df = _scope_product_rows(eligible_df, metric_product_id)
    baseline_matches = _scope_product_rows(all_baseline_matches, metric_product_id)
    unfiltered_baseline_matches = _scope_product_rows(all_unfiltered_baseline_matches, metric_product_id)
    candidate_matches = _scope_product_rows(all_candidate_matches, metric_product_id)

    baseline_metrics = summarize_matches(baseline_matches, source_df=metric_source_df)
    unfiltered_baseline_metrics = summarize_matches(
        unfiltered_baseline_matches,
        source_df=metric_source_df,
    )
    metrics = summarize_matches(candidate_matches, source_df=metric_source_df)
    eligible_metrics = summarize_matches(candidate_matches, source_df=eligible_metric_source_df)
    engine_diagnostics = _engine_diagnostics(df, engine_df, traces)

    candidate_total_seconds = engine_seconds + candidate_match_seconds
    elapsed_seconds = float(time.perf_counter() - total_started)
    result: dict[str, Any] = {
        "engine_dir": str(engine_path),
        "status": "ok",
        "match_backend": backend,
        "product_id": matching_product_id,
        "matching_product_id": matching_product_id,
        "metric_product_id": metric_product_id,
        "input_rows": input_rows,
        "rows": int(len(df)),
        "matching_rows": int(len(df)),
        "metric_source_rows": int(len(metric_source_df)),
        "excluded_product_rows": int(input_rows - len(df)),
        "batches": _batch_count(df, raw_batches),
        "eligible_rows": int(len(eligible_df)),
        "filtered_rows": int(engine_diagnostics["filtered_rows"]),
        "config_keys": sorted(config.keys()),
        "engine_batch_mode": engine_batch_mode,
        "metrics": metrics,
        "eligible_metrics": eligible_metrics,
        "baseline_metrics": baseline_metrics,
        "unfiltered_baseline_metrics": unfiltered_baseline_metrics,
        "deltas": metric_deltas(metrics, baseline_metrics),
        "policy_deltas": metric_deltas(baseline_metrics, unfiltered_baseline_metrics),
        "breakdowns": build_breakdowns(candidate_matches, metric_source_df),
        "eligible_breakdowns": build_breakdowns(candidate_matches, eligible_metric_source_df),
        "baseline_breakdowns": build_breakdowns(baseline_matches, metric_source_df),
        "unfiltered_baseline_breakdowns": build_breakdowns(
            unfiltered_baseline_matches,
            metric_source_df,
        ),
        "engine_diagnostics": engine_diagnostics,
        "timings": {
            "elapsed_seconds": elapsed_seconds,
            "data_read_seconds": data_read_seconds,
            "engine_seconds": engine_seconds,
            "online_weight_baseline_match_seconds": baseline_match_seconds,
            "unfiltered_online_weight_baseline_match_seconds": unfiltered_baseline_match_seconds,
            "candidate_match_seconds": candidate_match_seconds,
            "online_weight_baseline_total_seconds": baseline_match_seconds,
            "dispatch_engines_current_total_seconds": candidate_total_seconds,
        },
    }

    if output_dir is not None:
        result = _write_outputs(
            result,
            output_dir=output_dir,
            matches=candidate_matches,
            baseline_matches=baseline_matches,
            unfiltered_baseline_matches=unfiltered_baseline_matches,
            write_matches=write_matches,
        )
    return jsonable(result)


def _default_output_dir(backend: str) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return DEFAULT_OUTPUT_ROOT / f"{backend}_{timestamp}"


def cli_main(
    *,
    default_backend: str = "local",
    default_engine_dir: str | Path | None = None,
) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a full_dispatch engine codebase.")
    parser.add_argument("--engine-dir", default=str(default_engine_dir or ROOT_DIR / "dispatch_engines" / "current"))
    parser.add_argument("--match-backend", choices=SUPPORTED_BACKENDS, default=default_backend)
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument(
        "--product-id",
        default=str(DEFAULT_PRODUCT_ID),
        help="Local KM input scope: 1 or 'all' (default: 1)",
    )
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--write-matches", action="store_true")
    parser.add_argument("--no-trace", action="store_true")
    parser.add_argument("--show-progress", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir or _default_output_dir(args.match_backend)
    result = evaluate_engine_codebase(
        args.engine_dir,
        config={
            "match_backend": args.match_backend,
            "data_path": args.data_path,
            "product_id": args.product_id,
            "max_batches": args.max_batches,
            "show_progress": args.show_progress,
        },
        trace=not args.no_trace,
        output_dir=output_dir,
        write_matches=args.write_matches,
    )
    summary = {
        "status": result["status"],
        "match_backend": result.get("match_backend"),
        "product_id": result.get("product_id"),
        "matching_product_id": result.get("matching_product_id"),
        "metric_product_id": result.get("metric_product_id"),
        "input_rows": result.get("input_rows"),
        "rows": result.get("rows"),
        "matching_rows": result.get("matching_rows"),
        "metric_source_rows": result.get("metric_source_rows"),
        "excluded_product_rows": result.get("excluded_product_rows"),
        "batches": result.get("batches"),
        "metrics_path": result.get("metrics_path"),
        "metrics": result.get("metrics"),
        "baseline_metrics": result.get("baseline_metrics"),
        "unfiltered_baseline_metrics": result.get("unfiltered_baseline_metrics"),
        "deltas": result.get("deltas"),
        "engine_diagnostics": result.get("engine_diagnostics"),
    }
    print(json.dumps(jsonable(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    cli_main()
