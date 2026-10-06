"""Utilities for full_dispatch engine evaluation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


REQUIRED_ENGINE_FILES = ("engine.py", "README.md")
REQUIRED_ENGINE_DIRS = ("policies",)

MATCH_REQUIRED_COLUMNS = ("order_id", "driver_id", "weight", "eta", "cr")

CORE_METRIC_KEYS = (
    "mean_eta",
    "mean_cr",
    "mean_gmv",
    "mean_dar",
    "mean_pcaa",
    "mean_dcaa",
    "broadcast_count",
    "order_br",
    "br_ot",
    "order_ar",
    "1v1_ratio",
    "mean_fqs",
    "broadcast_match_rows",
)


def validate_engine_codebase(engine_dir: str | Path) -> Path:
    """Validate the minimum engine codebase layout."""
    path = Path(engine_dir).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Engine directory does not exist: {path}")

    missing_files = [name for name in REQUIRED_ENGINE_FILES if not (path / name).is_file()]
    missing_dirs = [name for name in REQUIRED_ENGINE_DIRS if not (path / name).is_dir()]
    if missing_files or missing_dirs:
        missing = ", ".join(missing_files + missing_dirs)
        raise ValueError(f"Engine codebase is missing required entries: {missing}")
    return path


def has_match_columns(df: pd.DataFrame) -> bool:
    """Return whether a frame can be evaluated by the local full dispatch matcher."""
    return set(MATCH_REQUIRED_COLUMNS).issubset(df.columns)


def safe_float_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    """Read a numeric column as finite floats, or synthesize a default series."""
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def ensure_metric_columns(
    df: pd.DataFrame,
    *,
    copy: bool = True,
) -> pd.DataFrame:
    """Add evaluator metric aliases that are useful across source and engine frames."""
    out = df.copy() if copy else df
    if "gmv" not in out.columns:
        out["gmv"] = np.maximum(
            safe_float_series(out, "pre_total_fee"),
            0.01,
        )
    if "cr" not in out.columns and {"dar", "pcaa", "dcaa"}.issubset(out.columns):
        out["cr"] = safe_float_series(out, "dar") * (1.0 - safe_float_series(out, "pcaa")) * (
            1.0 - safe_float_series(out, "dcaa")
        )
    return out


def _stage_major_scores(
    batch_df: pd.DataFrame,
    *,
    stage_col: str,
    weight_col: str,
) -> np.ndarray:
    stage = safe_float_series(batch_df, stage_col).to_numpy(dtype=float)
    weight = safe_float_series(batch_df, weight_col).to_numpy(dtype=float)
    finite_stage = np.where(np.isfinite(stage), stage, 0.0)
    finite_weight = np.where(np.isfinite(weight), weight, 0.0)
    min_stage = float(finite_stage.min()) if len(finite_stage) else 0.0
    max_abs_weight = float(np.max(np.abs(finite_weight))) if len(finite_weight) else 0.0
    scale = max_abs_weight * (len(batch_df) + 1) + 1.0
    return (finite_stage - min_stage) * scale + finite_weight


def _match_one_batch(
    batch_df: pd.DataFrame,
    *,
    stage_col: str,
    weight_col: str,
) -> pd.DataFrame:
    if batch_df.empty:
        return batch_df.iloc[0:0].copy()
    if not {"order_id", "driver_id", weight_col}.issubset(batch_df.columns):
        raise ValueError("Local KM requires order_id, driver_id, and weight columns")

    if batch_df[["order_id", "driver_id"]].isna().any().any():
        raise ValueError("Local KM requires non-missing request and resource identifiers")

    order_codes, _order_values = pd.factorize(batch_df["order_id"], sort=False)
    driver_codes, _driver_values = pd.factorize(batch_df["driver_id"], sort=False)
    order_count = int(order_codes.max() + 1) if len(order_codes) else 0
    driver_count = int(driver_codes.max() + 1) if len(driver_codes) else 0
    if order_count <= 0 or driver_count <= 0:
        return batch_df.iloc[0:0].copy()

    weight = safe_float_series(batch_df, weight_col).to_numpy(dtype=float)
    scores = _stage_major_scores(batch_df, stage_col=stage_col, weight_col=weight_col)
    valid = np.isfinite(scores) & np.isfinite(weight) & (weight > 0.0)
    if not bool(valid.any()):
        return batch_df.iloc[0:0].copy()

    cost = np.zeros((order_count, driver_count), dtype=float)
    chosen_local_rows = np.full((order_count, driver_count), -1, dtype=np.int64)
    best_scores = np.full((order_count, driver_count), -np.inf, dtype=float)
    for local_pos, (order_code, driver_code, score, is_valid) in enumerate(
        zip(order_codes, driver_codes, scores, valid, strict=False)
    ):
        if not is_valid or score <= best_scores[order_code, driver_code]:
            continue
        best_scores[order_code, driver_code] = score
        chosen_local_rows[order_code, driver_code] = local_pos
        cost[order_code, driver_code] = -score

    row_ind, col_ind = linear_sum_assignment(cost)
    matched_local_rows = chosen_local_rows[row_ind, col_ind]
    matched_local_rows = matched_local_rows[matched_local_rows >= 0]
    if not len(matched_local_rows):
        return batch_df.iloc[0:0].copy()

    result = batch_df.iloc[matched_local_rows].copy()
    result["is_matched"] = True
    result["stage_major_score"] = scores[matched_local_rows]
    return result


def local_stage_major_match(
    df: pd.DataFrame,
    *,
    stage_col: str = "adjusted_stage",
    weight_col: str = "weight",
) -> pd.DataFrame:
    """Run local scipy KM with stage-major score ordering within each batch."""
    if df.empty:
        result = df.iloc[0:0].copy()
        result["is_matched"] = True
        result["stage_major_score"] = []
        return result
    if stage_col not in df.columns:
        stage_col = "stage" if "stage" in df.columns else stage_col
    if stage_col not in df.columns:
        raise ValueError("Local stage-major KM requires adjusted_stage or stage")
    if weight_col not in df.columns:
        raise ValueError(f"Local stage-major KM requires score column: {weight_col}")

    work = df.copy()
    group_col = "batch_id"
    if group_col not in work.columns:
        group_col = "__full_dispatch_single_batch"
        work[group_col] = "batch_0"

    matched_batches = [
        _match_one_batch(batch, stage_col=stage_col, weight_col=weight_col)
        for _, batch in work.groupby(group_col, sort=False, dropna=False)
    ]
    matched_batches = [batch for batch in matched_batches if len(batch)]
    if not matched_batches:
        empty = work.iloc[0:0].copy()
        empty["is_matched"] = True
        empty["stage_major_score"] = []
        if group_col == "__full_dispatch_single_batch":
            empty = empty.drop(columns=[group_col])
        return empty
    result = pd.concat(matched_batches, axis=0).sort_index(kind="mergesort")
    if group_col == "__full_dispatch_single_batch":
        result = result.drop(columns=[group_col])
    return result


def _broadcast_mask(df: pd.DataFrame) -> pd.Series:
    if "if_broadcast" in df.columns:
        return safe_float_series(df, "if_broadcast") > 0.5
    if "is_broadcasted" in df.columns:
        return safe_float_series(df, "is_broadcasted") > 0.5
    return pd.Series(True, index=df.index, dtype=bool)


def _event_key_columns(df: pd.DataFrame) -> list[str]:
    if {"order_id", "group_id", "timestamp"}.issubset(df.columns):
        return ["order_id", "group_id", "timestamp"]
    if {"order_id", "batch_id"}.issubset(df.columns):
        return ["order_id", "batch_id"]
    if "order_id" in df.columns:
        return ["order_id"]
    return []


def _broadcast_event_key_columns(df: pd.DataFrame) -> list[str]:
    if {"order_id", "batch_id"}.issubset(df.columns):
        return ["order_id", "batch_id"]
    if {"order_id", "group_id", "timestamp"}.issubset(df.columns):
        return ["order_id", "group_id", "timestamp"]
    if "order_id" in df.columns:
        return ["order_id"]
    return []


def compute_order_broadcast_metrics(
    matched_df: pd.DataFrame,
    source_df: pd.DataFrame | None = None,
) -> dict[str, float]:
    metrics = {
        "order_br": 0.0,
        "br_ot": 0.0,
        "order_ar": 0.0,
        "1v1_ratio": 0.0,
    }
    if "order_id" not in matched_df.columns:
        return metrics

    matched_order_count = int(matched_df["order_id"].nunique())
    total_order_count = matched_order_count
    if source_df is not None and "order_id" in source_df.columns:
        total_order_count = int(source_df["order_id"].nunique())
    if total_order_count > 0:
        metrics["order_br"] = float(matched_order_count / total_order_count)

    broadcast_event_key = _broadcast_event_key_columns(matched_df)
    if broadcast_event_key:
        matched_event_count = int(matched_df.loc[:, broadcast_event_key].drop_duplicates().shape[0])
        total_event_count = matched_event_count
        if source_df is not None and set(broadcast_event_key).issubset(source_df.columns):
            total_event_count = int(source_df.loc[:, broadcast_event_key].drop_duplicates().shape[0])
        if total_event_count > 0:
            metrics["br_ot"] = float(matched_event_count / total_event_count)

    event_key = _event_key_columns(matched_df)
    if not event_key or "driver_id" not in matched_df.columns or matched_df.empty:
        return metrics

    event_driver_rows = matched_df.loc[:, event_key + ["driver_id"]].drop_duplicates()
    driver_counts = event_driver_rows.groupby(event_key, dropna=False, sort=False)["driver_id"].nunique()
    broadcast_event_count = int(len(driver_counts))
    if broadcast_event_count <= 0:
        return metrics

    one_to_one_event_count = int((driver_counts == 1).sum())
    metrics["1v1_ratio"] = float(one_to_one_event_count / broadcast_event_count)

    if "dar" not in matched_df.columns:
        return metrics

    dar_frame = matched_df.loc[:, event_key + ["driver_id", "dar"]].drop_duplicates(event_key + ["driver_id"])
    dar_frame["dar"] = pd.to_numeric(dar_frame["dar"], errors="coerce").fillna(0.0)
    ar_components = (
        dar_frame.assign(_reject_prob=1.0 - dar_frame["dar"])
        .groupby(event_key, dropna=False, sort=False)
        .agg(first_dar=("dar", "first"), reject_prob=("_reject_prob", "prod"))
    )
    ar_driver_counts = driver_counts.reindex(ar_components.index)
    one_to_one_mask = ar_driver_counts == 1
    one_to_two_mask = ar_driver_counts == 2
    if bool(one_to_one_mask.any() or one_to_two_mask.any()):
        ar_values = np.concatenate(
            [
                ar_components.loc[one_to_one_mask, "first_dar"].to_numpy(dtype=float),
                (1.0 - ar_components.loc[one_to_two_mask, "reject_prob"]).to_numpy(dtype=float),
            ]
        )
        metrics["order_ar"] = float(np.mean(ar_values)) if len(ar_values) else 0.0
    return metrics


def summarize_matches(
    matched_df: pd.DataFrame,
    *,
    source_df: pd.DataFrame | None = None,
) -> dict[str, float | None]:
    """Compute full_dispatch core match and broadcast metrics."""
    df = ensure_metric_columns(matched_df)
    source = ensure_metric_columns(source_df) if source_df is not None else None
    metrics: dict[str, float | None] = {key: 0.0 for key in CORE_METRIC_KEYS}

    if len(df):
        metrics["mean_eta"] = float(safe_float_series(df, "eta").mean())
        metrics["mean_cr"] = float(safe_float_series(df, "cr").mean())
        gmv = safe_float_series(df, "gmv")
        metrics["mean_gmv"] = float(gmv.mean())
        for column, metric_name in (
            ("dar", "mean_dar"),
            ("pcaa", "mean_pcaa"),
            ("dcaa", "mean_dcaa"),
        ):
            metrics[metric_name] = float(safe_float_series(df, column).mean()) if column in df.columns else None

        dar = safe_float_series(df, "dar")
        pcaa = safe_float_series(df, "pcaa")
        dcaa = safe_float_series(df, "dcaa")
        metrics["mean_fqs"] = float((dar * (1.0 - dcaa) * (1.0 - pcaa)).mean())
        broadcast_mask = _broadcast_mask(df)
        metrics["broadcast_match_rows"] = int(broadcast_mask.sum())

    metrics["broadcast_count"] = int(len(df))
    metrics.update(compute_order_broadcast_metrics(df, source))
    return metrics


def metric_deltas(
    candidate_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
) -> dict[str, float | None]:
    """Return absolute and pct deltas for numeric metric keys."""
    deltas: dict[str, float | None] = {}
    for key in sorted(set(candidate_metrics) | set(baseline_metrics)):
        candidate = _numeric_or_none(candidate_metrics.get(key))
        baseline = _numeric_or_none(baseline_metrics.get(key))
        if candidate is None or baseline is None:
            continue
        delta = candidate - baseline
        deltas[f"{key}_delta"] = float(delta)
        deltas[f"{key}_delta_pct"] = float(delta / abs(baseline)) if abs(baseline) > 1e-12 else None
    return deltas


def build_breakdowns(
    matches: pd.DataFrame,
    source_df: pd.DataFrame,
    *,
    group_columns: tuple[str, ...] = ("city_id", "product_id"),
) -> dict[str, dict[str, Any]]:
    """Build per-city/per-product summaries with the same metric schema."""
    breakdowns: dict[str, dict[str, Any]] = {}
    for column in group_columns:
        name = f"by_{column}"
        if column not in source_df.columns and column not in matches.columns:
            breakdowns[name] = {}
            continue
        source_values = source_df[column].dropna().astype(str).unique().tolist() if column in source_df.columns else []
        match_values = matches[column].dropna().astype(str).unique().tolist() if column in matches.columns else []
        groups = sorted(set(source_values) | set(match_values))
        breakdowns[name] = {}
        for value in groups:
            source_group = source_df.loc[source_df[column].astype(str) == value] if column in source_df.columns else source_df.iloc[0:0]
            match_group = matches.loc[matches[column].astype(str) == value] if column in matches.columns else matches.iloc[0:0]
            breakdowns[name][value] = {
                "source_rows": int(len(source_group)),
                "matched_rows": int(len(match_group)),
                "source_orders": int(source_group["order_id"].nunique()) if "order_id" in source_group.columns else 0,
                "matched_orders": int(match_group["order_id"].nunique()) if "order_id" in match_group.columns else 0,
                "metrics": summarize_matches(match_group, source_df=source_group),
            }
    return breakdowns


def _numeric_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def jsonable(value: Any) -> Any:
    """Convert numpy/pandas scalar containers to JSON-safe Python objects."""
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NA:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
