"""Fail-closed candidate column boundary for full-dispatch evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib import resources
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import pandas as pd


COLUMN_POLICY_VERSION = "full-dispatch-column-policy-v2"
OPAQUE_ROW_ID_COLUMN = "__dispatch_evolve_row_id__"
_OPAQUE_ROW_ID_DOMAIN = b"dispatchevolve-full-dispatch-row-id-v1\0"

_OPTIONAL_ENGINE_FEATURE_ORDER = ("order_wait_time", "available_drivers", "pending_orders", "pickup_distance", "trip_distance")
OPTIONAL_ENGINE_FEATURES = frozenset(_OPTIONAL_ENGINE_FEATURE_ORDER)

_ONLINE_PREDICTIVE_FEATURE_ORDER = ("eta", "cr", "dar", "pcaa", "dcaa")
ONLINE_PREDICTIVE_FEATURES = frozenset(_ONLINE_PREDICTIVE_FEATURE_ORDER)

_MUTABLE_OUTPUT_ORDER = (
    "weight",
    "stage",
    "driver_lock_time_s",
    "order_lock_time_s",
)
MUTABLE_OUTPUT_COLUMNS = frozenset(_MUTABLE_OUTPUT_ORDER)

_EXPECTED_ENGINE_READ_COLUMN_ORDER = ("product_id", "eta", "dar", "pcaa", "dcaa", "stage", "weight")
EXPECTED_ENGINE_READ_COLUMNS = frozenset(_EXPECTED_ENGINE_READ_COLUMN_ORDER)


class ColumnPolicyError(ValueError):
    """Raised when a candidate frame violates the column boundary."""


def _load_resource_mapping(name: str, *, expected_count: int) -> dict[str, str]:
    resource = resources.files(__package__).joinpath(name)
    payload = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in payload.items()
    ):
        raise RuntimeError(f"{name} must contain a JSON object of string descriptions")
    if len(payload) != expected_count:
        raise RuntimeError(
            f"{name} contains {len(payload)} entries; expected {expected_count}"
        )
    return payload


_FEATURE_INFO_DICT = _load_resource_mapping("feature_info.json", expected_count=22)
_LABEL_INFO_DICT = _load_resource_mapping("label_info.json", expected_count=3)
_FEATURE_INFO: Mapping[str, str] = MappingProxyType(_FEATURE_INFO_DICT)
_LABEL_INFO: Mapping[str, str] = MappingProxyType(_LABEL_INFO_DICT)
_CANDIDATE_ALLOWLIST = (
    frozenset(_FEATURE_INFO)
    | OPTIONAL_ENGINE_FEATURES
    | ONLINE_PREDICTIVE_FEATURES
)

if len(EXPECTED_ENGINE_READ_COLUMNS) != 7:
    raise RuntimeError("EXPECTED_ENGINE_READ_COLUMNS must contain exactly 7 columns")
_uncovered_engine_columns = EXPECTED_ENGINE_READ_COLUMNS - _CANDIDATE_ALLOWLIST
if _uncovered_engine_columns:
    raise RuntimeError(
        "engine read columns are not candidate-visible: "
        f"{sorted(_uncovered_engine_columns)}"
    )

_FINGERPRINT_PAYLOAD = {
    "candidate_allowlist": sorted(_CANDIDATE_ALLOWLIST),
    "expected_engine_read_columns": sorted(EXPECTED_ENGINE_READ_COLUMNS),
    "feature_info": _FEATURE_INFO_DICT,
    "label_info": _LABEL_INFO_DICT,
    "complex_target_category_outputs": "rejected before numeric conversion",
    "mutable_output_columns": sorted(MUTABLE_OUTPUT_COLUMNS),
    "mutable_output_scope": "product_id == 1 only",
    "non_target_category_mutable_validation": (
        "strict numeric normalization followed by NaN-aware exact shadow equality"
    ),
    "target_category_mutable_validation": "finite real values; weight and locks nonnegative",
    "online_predictive_features": sorted(ONLINE_PREDICTIVE_FEATURES),
    "opaque_row_id_column": OPAQUE_ROW_ID_COLUMN,
    "opaque_row_id_strategy": "sha256(domain-v1 + policy fingerprint + ordinal)",
    "optional_engine_features": sorted(OPTIONAL_ENGINE_FEATURES),
    "version": COLUMN_POLICY_VERSION,
}
COLUMN_POLICY_FINGERPRINT = hashlib.sha256(
    json.dumps(
        _FINGERPRINT_PAYLOAD,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


def _opaque_row_id(ordinal: int) -> str:
    payload = (
        _OPAQUE_ROW_ID_DOMAIN
        + COLUMN_POLICY_FINGERPRINT.encode("ascii")
        + b"\0"
        + str(ordinal).encode("ascii")
    )
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class PreparedCandidateFrame:
    """Trusted shadow data paired with the isolated candidate-visible frame."""

    shadow: pd.DataFrame
    visible: pd.DataFrame
    version: str
    fingerprint: str

    @property
    def policy_version(self) -> str:
        return self.version

    @property
    def policy_fingerprint(self) -> str:
        return self.fingerprint


def _duplicate_columns(frame: pd.DataFrame) -> list[Any]:
    return frame.columns[frame.columns.duplicated()].tolist()


def _series_values_equal(left: pd.Series, right: pd.Series) -> pd.Series:
    left = left.reset_index(drop=True)
    right = right.reset_index(drop=True)
    both_missing = left.isna() & right.isna()
    try:
        equal = left.eq(right).fillna(False)
    except (TypeError, ValueError):
        equal = pd.Series(False, index=left.index, dtype=bool)
    return both_missing | equal


def _different_names(left: set[Any], right: set[Any]) -> str:
    missing = sorted((repr(value) for value in right - left))
    extra = sorted((repr(value) for value in left - right))
    return f"missing={missing}, extra={extra}"


def _contains_complex_values(values: pd.Series) -> bool:
    try:
        if np.issubdtype(values.dtype, np.complexfloating):
            return True
    except (TypeError, ValueError):
        pass
    return any(
        isinstance(value, (complex, np.complexfloating))
        for value in values.array
    )


def _validate_target_category_mutable_values(
    column: str,
    values: pd.Series,
) -> pd.Series:
    if values.empty:
        return pd.Series(index=values.index, dtype=float, name=column)
    if _contains_complex_values(values):
        raise ColumnPolicyError(
            f"complex values are forbidden for target-category mutable output {column!r}"
        )
    try:
        numeric = pd.to_numeric(values, errors="coerce")
        if _contains_complex_values(numeric):
            raise ColumnPolicyError(
                f"complex values are forbidden for target-category mutable output {column!r}"
            )
        converted = numeric.to_numpy(dtype=float, na_value=np.nan)
    except ColumnPolicyError:
        raise
    except Exception as exc:
        raise ColumnPolicyError(
            f"numeric conversion failed for target-category mutable output {column!r}"
        ) from exc
    if not bool(np.isfinite(converted).all()):
        raise ColumnPolicyError(
            f"candidate output {column!r} must contain finite numeric values"
        )
    if column != "stage" and bool((converted < 0.0).any()):
        raise ColumnPolicyError(f"candidate output {column!r} must be non-negative")
    return pd.Series(converted, index=values.index, dtype=float, name=column)


def _normalize_non_product_mutable_values(
    column: str,
    values: pd.Series,
) -> pd.Series:
    """Normalize protocol-numeric passthrough values without relaxing equality."""
    if values.empty:
        return pd.Series(index=values.index, dtype=float, name=column)
    if _contains_complex_values(values):
        raise ColumnPolicyError(
            f"complex values are forbidden for other-category mutable output {column!r}"
        )
    try:
        numeric = pd.to_numeric(values, errors="raise")
        converted = numeric.to_numpy(dtype=float, na_value=np.nan)
    except Exception as exc:
        raise ColumnPolicyError(
            f"other-category mutable output {column!r} must contain numeric passthrough values"
        ) from exc
    if bool(np.isinf(converted).any()):
        raise ColumnPolicyError(
            f"other-category mutable output {column!r} must not contain infinite values"
        )
    return pd.Series(converted, index=values.index, dtype=float, name=column)


class ColumnPolicy:
    """Prepare isolated candidate inputs and rebuild only validated outputs."""

    version = COLUMN_POLICY_VERSION
    fingerprint = COLUMN_POLICY_FINGERPRINT
    feature_info = _FEATURE_INFO
    label_info = _LABEL_INFO
    candidate_allowlist = _CANDIDATE_ALLOWLIST

    @property
    def policy_version(self) -> str:
        return self.version

    @property
    def policy_fingerprint(self) -> str:
        return self.fingerprint

    def prepare(self, source_df: pd.DataFrame) -> PreparedCandidateFrame:
        """Copy source data into a trusted shadow and allowlisted visible frame."""

        if not isinstance(source_df, pd.DataFrame):
            raise TypeError("ColumnPolicy.prepare requires a pandas DataFrame")
        duplicates = _duplicate_columns(source_df)
        if duplicates:
            raise ColumnPolicyError(f"source DataFrame has duplicate columns: {duplicates!r}")
        if OPAQUE_ROW_ID_COLUMN in source_df.columns:
            raise ColumnPolicyError(
                f"source DataFrame contains reserved column {OPAQUE_ROW_ID_COLUMN!r}"
            )

        shadow = source_df.copy(deep=True)
        row_ids = [_opaque_row_id(ordinal) for ordinal in range(len(shadow))]
        shadow[OPAQUE_ROW_ID_COLUMN] = pd.Series(
            row_ids,
            index=shadow.index,
            dtype="string",
        )

        visible_columns = [
            column for column in source_df.columns if column in self.candidate_allowlist
        ]
        visible = shadow[[*visible_columns, OPAQUE_ROW_ID_COLUMN]].copy(deep=True)
        return PreparedCandidateFrame(
            shadow=shadow,
            visible=visible,
            version=self.version,
            fingerprint=self.fingerprint,
        )

    def _validate_prepared(self, prepared: PreparedCandidateFrame) -> None:
        if not isinstance(prepared, PreparedCandidateFrame):
            raise TypeError(
                "ColumnPolicy.validate_and_rebuild requires PreparedCandidateFrame"
            )
        if prepared.version != self.version:
            raise ColumnPolicyError(
                f"prepared frame version {prepared.version!r} does not match {self.version!r}"
            )
        if prepared.fingerprint != self.fingerprint:
            raise ColumnPolicyError("prepared frame fingerprint does not match column policy")
        if not isinstance(prepared.shadow, pd.DataFrame) or not isinstance(
            prepared.visible, pd.DataFrame
        ):
            raise ColumnPolicyError("prepared shadow and visible values must be DataFrames")
        for name, frame in (("shadow", prepared.shadow), ("visible", prepared.visible)):
            duplicates = _duplicate_columns(frame)
            if duplicates:
                raise ColumnPolicyError(
                    f"prepared {name} has duplicate columns: {duplicates!r}"
                )
            if OPAQUE_ROW_ID_COLUMN not in frame.columns:
                raise ColumnPolicyError(
                    f"prepared {name} is missing reserved row identity column"
                )
            row_ids = frame[OPAQUE_ROW_ID_COLUMN]
            if row_ids.isna().any() or not row_ids.is_unique:
                raise ColumnPolicyError(f"prepared {name} row identities must be unique")

        expected_visible_columns = [
            column
            for column in prepared.shadow.columns
            if column != OPAQUE_ROW_ID_COLUMN and column in self.candidate_allowlist
        ]
        expected_visible_columns.append(OPAQUE_ROW_ID_COLUMN)
        if list(prepared.visible.columns) != expected_visible_columns:
            raise ColumnPolicyError("prepared visible columns do not match the column policy")
        if len(prepared.visible) != len(prepared.shadow):
            raise ColumnPolicyError("prepared visible row count does not match shadow")
        for column in expected_visible_columns:
            equal = _series_values_equal(
                prepared.visible[column], prepared.shadow[column]
            )
            if not bool(equal.all()):
                raise ColumnPolicyError(
                    f"prepared visible column {column!r} does not match shadow"
                )

    def validate_and_rebuild(
        self,
        prepared: PreparedCandidateFrame,
        candidate_output: pd.DataFrame,
    ) -> pd.DataFrame:
        """Validate candidate output and restore protected values from shadow."""

        self._validate_prepared(prepared)
        if not isinstance(candidate_output, pd.DataFrame):
            raise TypeError("candidate output must be a pandas DataFrame")
        duplicates = _duplicate_columns(candidate_output)
        if duplicates:
            raise ColumnPolicyError(
                f"candidate output has duplicate columns: {duplicates!r}"
            )

        expected_columns = set(prepared.visible.columns) | MUTABLE_OUTPUT_COLUMNS
        actual_columns = set(candidate_output.columns)
        if actual_columns != expected_columns or len(candidate_output.columns) != len(
            expected_columns
        ):
            raise ColumnPolicyError(
                "candidate output columns must exactly match visible columns plus mutable "
                f"outputs; {_different_names(actual_columns, expected_columns)}"
            )

        candidate = candidate_output.copy(deep=True)
        row_ids = candidate[OPAQUE_ROW_ID_COLUMN]
        if row_ids.isna().any() or not row_ids.is_unique:
            raise ColumnPolicyError("candidate row identities must be known and unique")
        if not all(isinstance(value, str) for value in row_ids.tolist()):
            raise ColumnPolicyError("candidate row identities must be known opaque strings")

        shadow_ids = prepared.shadow[OPAQUE_ROW_ID_COLUMN]
        known_ids = set(shadow_ids.tolist())
        candidate_ids = set(row_ids.tolist())
        unknown_ids = candidate_ids - known_ids
        if unknown_ids:
            raise ColumnPolicyError(
                f"candidate output contains unknown row identities: {len(unknown_ids)}"
            )

        if "product_id" in prepared.shadow.columns:
            product_ids = pd.to_numeric(
                prepared.shadow["product_id"], errors="coerce"
            )
            target_category_mask = product_ids.eq(1)
        else:
            target_category_mask = pd.Series(
                False,
                index=prepared.shadow.index,
                dtype=bool,
            )
        required_mask = ~target_category_mask
        required_ids = set(shadow_ids.loc[required_mask].tolist())
        target_category_ids = set(shadow_ids.loc[target_category_mask].tolist())
        missing_required = required_ids - candidate_ids
        if missing_required:
            raise ColumnPolicyError(
                "candidate output omitted other-category rows: "
                f"{len(missing_required)} required row(s) missing"
            )

        shadow_by_id = prepared.shadow.set_index(OPAQUE_ROW_ID_COLUMN, drop=False)
        candidate_by_id = candidate.set_index(OPAQUE_ROW_ID_COLUMN, drop=False)
        ordered_candidate = candidate_by_id.loc[row_ids.tolist()]
        immutable_columns = [
            column
            for column in prepared.visible.columns
            if column != OPAQUE_ROW_ID_COLUMN and column not in MUTABLE_OUTPUT_COLUMNS
        ]
        for column in immutable_columns:
            expected = shadow_by_id.loc[row_ids.tolist(), column]
            actual = ordered_candidate[column]
            equal = _series_values_equal(actual, expected)
            if not bool(equal.all()):
                raise ColumnPolicyError(
                    f"candidate changed non-mutable visible column {column!r}"
                )

        returned_non_target_category_ids = [
            row_id for row_id in row_ids.tolist() if row_id in required_ids
        ]
        for column in _MUTABLE_OUTPUT_ORDER:
            if column not in prepared.shadow.columns:
                continue
            expected = shadow_by_id.loc[returned_non_target_category_ids, column]
            actual = candidate_by_id.loc[returned_non_target_category_ids, column]
            normalized_expected = _normalize_non_product_mutable_values(
                column, expected
            )
            normalized_actual = _normalize_non_product_mutable_values(
                column, actual
            )
            equal = _series_values_equal(normalized_actual, normalized_expected)
            if not bool(equal.all()):
                raise ColumnPolicyError(
                    f"candidate changed other-category mutable output {column!r}"
                )

        returned_target_category_ids = [
            row_id for row_id in row_ids.tolist() if row_id in target_category_ids
        ]
        validated_target_category_values: dict[str, pd.Series] = {}
        for column in _MUTABLE_OUTPUT_ORDER:
            validated_target_category_values[column] = (
                _validate_target_category_mutable_values(
                    column,
                    candidate_by_id.loc[returned_target_category_ids, column],
                )
            )

        selected_mask = shadow_ids.isin(candidate_ids)
        rebuilt = prepared.shadow.loc[selected_mask].copy(deep=True)
        candidate_by_id = candidate.set_index(OPAQUE_ROW_ID_COLUMN, drop=False)
        rebuilt_target_category = rebuilt[OPAQUE_ROW_ID_COLUMN].isin(target_category_ids)
        for column in _MUTABLE_OUTPUT_ORDER:
            if column in rebuilt.columns:
                trusted_values = rebuilt[column].copy(deep=True)
            else:
                trusted_values = pd.Series(
                    np.nan,
                    index=rebuilt.index,
                    dtype=float,
                )
            candidate_values = rebuilt[OPAQUE_ROW_ID_COLUMN].map(
                validated_target_category_values[column]
            )
            rebuilt[column] = trusted_values.where(
                ~rebuilt_target_category,
                candidate_values,
            )
        rebuilt["adjusted_stage"] = rebuilt["stage"]
        return rebuilt.drop(columns=[OPAQUE_ROW_ID_COLUMN])


__all__ = [
    "COLUMN_POLICY_FINGERPRINT",
    "COLUMN_POLICY_VERSION",
    "EXPECTED_ENGINE_READ_COLUMNS",
    "MUTABLE_OUTPUT_COLUMNS",
    "OPAQUE_ROW_ID_COLUMN",
    "ONLINE_PREDICTIVE_FEATURES",
    "OPTIONAL_ENGINE_FEATURES",
    "ColumnPolicy",
    "ColumnPolicyError",
    "PreparedCandidateFrame",
]
