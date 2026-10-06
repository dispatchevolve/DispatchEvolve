"""Central registry of feature columns hidden from LLM-facing prompt surfaces.

A hidden feature stays present in the underlying data so the evaluation stage can
still use it (e.g. ``cr`` drives dar/gmv metric aggregation), but it is filtered
out of every surface the LLM reads during scene partitioning and formula
computation: dataset summaries, raw-column catalogs, batch feature frames, scene
rule schemas, and the genetic analysis column profile. The model therefore never
sees the feature and cannot route on it or write it into a scoring formula.

No feature is hidden by default. Hide specific raw columns per run with the
environment variable ``DISPATCHEVOLVE_HIDDEN_FEATURES`` (comma-separated raw
column names).
"""
from __future__ import annotations

import os

# No feature is hidden by default (see module docstring).
_DEFAULT_HIDDEN_FEATURES = frozenset()

# Aggregation prefixes produced by compute_batch_feature_frame; a derived column
# such as ``batch_mean_cr`` must be hidden whenever its raw column is hidden.
_BATCH_AGG_PREFIXES = (
    "batch_mean_",
    "batch_min_",
    "batch_max_",
    "batch_sum_",
    "batch_nunique_",
)


def hidden_features() -> frozenset[str]:
    """Return the set of raw feature names hidden from the LLM."""
    extra = os.environ.get("DISPATCHEVOLVE_HIDDEN_FEATURES", "")
    names = {token.strip() for token in extra.split(",") if token.strip()}
    return _DEFAULT_HIDDEN_FEATURES | frozenset(names)


def is_hidden_column(column: str) -> bool:
    """True if ``column`` is a hidden raw feature or a derived batch aggregate of one."""
    hidden = hidden_features()
    name = str(column)
    if name in hidden:
        return True
    for prefix in _BATCH_AGG_PREFIXES:
        if name.startswith(prefix) and name[len(prefix):] in hidden:
            return True
    return False


def visible_columns(columns) -> list[str]:
    """Filter an iterable of column names down to the LLM-visible ones."""
    return [str(column) for column in columns if not is_hidden_column(column)]
