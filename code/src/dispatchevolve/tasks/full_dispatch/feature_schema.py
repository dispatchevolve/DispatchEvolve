"""Canonical feature names shared by full-dispatch data consumers."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


DUPLICATE_FEATURE_ALIASES: Mapping[str, str] = {
    "request_id": "order_id",
    "resource_id": "driver_id",
    "travel_time": "eta",
    "value": "gmv",
}


def _equivalent_values(left: pd.Series, right: pd.Series) -> pd.Series:
    both_missing = left.isna() & right.isna()
    one_missing = left.isna() ^ right.isna()
    populated = ~(both_missing | one_missing)
    equivalent = both_missing.copy()
    if not bool(populated.any()):
        return equivalent

    left_values = left.loc[populated]
    right_values = right.loc[populated]
    left_numeric = pd.to_numeric(left_values, errors="coerce")
    right_numeric = pd.to_numeric(right_values, errors="coerce")
    numeric = left_numeric.notna() & right_numeric.notna()
    populated_equal = pd.Series(False, index=left_values.index, dtype=bool)
    if bool(numeric.any()):
        populated_equal.loc[numeric] = np.isclose(
            left_numeric.loc[numeric].astype(float),
            right_numeric.loc[numeric].astype(float),
            rtol=0.0,
            atol=1e-12,
        )
    textual = ~numeric
    if bool(textual.any()):
        populated_equal.loc[textual] = (
            left_values.loc[textual].astype(str).str.strip()
            == right_values.loc[textual].astype(str).str.strip()
        )
    equivalent.loc[populated] = populated_equal
    return equivalent


def canonicalize_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a frame containing at most one column per canonical feature."""
    result = df.copy()
    for alias, canonical in DUPLICATE_FEATURE_ALIASES.items():
        if alias not in result.columns:
            continue
        if canonical not in result.columns:
            result = result.rename(columns={alias: canonical})
            continue
        equivalent = _equivalent_values(result[alias], result[canonical])
        conflict_count = int((~equivalent).sum())
        if conflict_count:
            conflict_indices = result.index[~equivalent].tolist()[:5]
            noun = "row" if conflict_count == 1 else "rows"
            raise ValueError(
                f"duplicate feature alias {alias!r} conflicts with canonical "
                f"{canonical!r} on {conflict_count} conflicting {noun}; "
                f"sample indices={conflict_indices}"
            )
        result = result.drop(columns=[alias])
    return result
