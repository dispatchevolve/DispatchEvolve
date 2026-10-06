"""Typed, canonical, non-executable scenario predicate DSL."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

import pandas as pd
import yaml

from .contracts import ScenarioPredicate, canonical_json, content_hash


_SIMPLE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s+(LT|LE|GT|GE|EQ|NE)\s+(.+)$")
_SYMBOLIC_SIMPLE = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_]*)\s*(<=|>=|==|!=|<|>)\s*(.+)$"
)
_BETWEEN = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s+BETWEEN\s+([^,]+),([^,]+)$")
_LEAF_OPS = {"lt", "le", "gt", "ge", "eq", "ne", "in", "not_in", "between", "is_null", "not_null"}


def _scalar(value: Any) -> float | str | bool | None:
    if value is None or isinstance(value, (bool, int, float)):
        return float(value) if isinstance(value, int) and not isinstance(value, bool) else value
    stripped = str(value).strip()
    try:
        return float(stripped)
    except ValueError:
        if not re.fullmatch(r"[A-Za-z0-9_.:/-]+", stripped):
            raise ValueError("DSL string literal contains unsupported characters")
        return stripped


def _legacy(text: str) -> Mapping[str, Any] | None:
    compact = " ".join(text.strip().split())
    between = _BETWEEN.fullmatch(compact)
    simple = _SIMPLE.fullmatch(compact)
    symbolic = _SYMBOLIC_SIMPLE.fullmatch(compact)
    if between:
        return {"field": between.group(1), "op": "between", "lower": float(between.group(2)), "upper": float(between.group(3))}
    if simple:
        return {"field": simple.group(1), "op": simple.group(2).lower(), "value": _scalar(simple.group(3))}
    if symbolic:
        operator = {"<": "lt", "<=": "le", ">": "gt", ">=": "ge", "==": "eq", "!=": "ne"}
        return {"field": symbolic.group(1), "op": operator[symbolic.group(2)],
                "value": _scalar(symbolic.group(3))}
    return None


def _normalize(node: Any, schema: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(node, Mapping):
        raise ValueError("predicate node must be a mapping")
    if "field" not in node and "op" not in node and len(node) == 1:
        candidate_field, specification = next(iter(node.items()))
        if candidate_field in schema and isinstance(specification, Mapping) and len(specification) == 1:
            operator, operand = next(iter(specification.items()))
            operator = str(operator).lower()
            if operator in {"lt", "le", "gt", "ge", "eq", "ne"}:
                node = {"field": candidate_field, "op": operator, "value": operand}
            elif operator in {"in", "not_in"}:
                node = {"field": candidate_field, "op": operator, "values": operand}
            elif operator == "between" and isinstance(operand, (list, tuple)) and len(operand) == 2:
                node = {"field": candidate_field, "op": operator,
                        "lower": operand[0], "upper": operand[1]}
            elif operator in {"is_null", "not_null"}:
                node = {"field": candidate_field, "op": operator}
    logical = [key for key in ("all", "any", "not") if key in node]
    if logical:
        if len(logical) != 1 or len(node) != 1:
            raise ValueError("logical predicate node must contain exactly one operator")
        key = logical[0]
        if key == "not":
            return {"not": _normalize(node[key], schema)}
        children = node[key]
        if not isinstance(children, list) or len(children) < 1:
            raise ValueError(f"{key} requires a nonempty list")
        normalized = [_normalize(item, schema) for item in children]
        return {key: sorted(normalized, key=canonical_json)}
    field = str(node.get("field", ""))
    op = str(node.get("op", "")).lower()
    if field not in schema:
        raise ValueError(f"predicate field is not in runtime schema: {field}")
    if op not in _LEAF_OPS:
        raise ValueError(f"unsupported predicate operator: {op}")
    result: dict[str, Any] = {"field": field, "op": op}
    if op == "between":
        lower, upper = float(node["lower"]), float(node["upper"])
        if lower > upper:
            raise ValueError("BETWEEN lower bound exceeds upper bound")
        result.update(lower=lower, upper=upper)
    elif op in {"in", "not_in"}:
        values = node.get("values")
        if not isinstance(values, list) or not values:
            raise ValueError(f"{op} requires nonempty values")
        result["values"] = sorted({_scalar(value) for value in values}, key=lambda value: str(value))
    elif op not in {"is_null", "not_null"}:
        result["value"] = _scalar(node.get("value"))
    return result


def parse_predicate(text: str, schema: Mapping[str, Any]) -> ScenarioPredicate:
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:yaml|yml|text)?\s*\n([\s\S]*?)\n```", stripped, re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    legacy = _legacy(text)
    if legacy is not None:
        raw: Any = legacy
    else:
        try:
            raw = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ValueError("predicate is neither legacy DSL nor YAML DSL") from exc
    expression = _normalize(raw, schema)
    canonical = json.dumps(expression, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return ScenarioPredicate(expression, canonical, content_hash(expression))


def _evaluate(node: Mapping[str, Any], frame: pd.DataFrame) -> pd.Series:
    if "all" in node:
        result = pd.Series(True, index=frame.index)
        for child in node["all"]:
            result &= _evaluate(child, frame)
        return result
    if "any" in node:
        result = pd.Series(False, index=frame.index)
        for child in node["any"]:
            result |= _evaluate(child, frame)
        return result
    if "not" in node:
        return ~_evaluate(node["not"], frame)
    field, op = str(node["field"]), str(node["op"])
    if field not in frame:
        raise ValueError(f"predicate field is absent: {field}")
    series = frame[field]
    if op in {"lt", "le", "gt", "ge", "between"}:
        series = pd.to_numeric(series, errors="coerce")
    if op == "between":
        return series.ge(node["lower"]) & series.le(node["upper"])
    operations = {
        "lt": lambda: series.lt(node["value"]), "le": lambda: series.le(node["value"]),
        "gt": lambda: series.gt(node["value"]), "ge": lambda: series.ge(node["value"]),
        "eq": lambda: series.eq(node["value"]), "ne": lambda: series.ne(node["value"]),
        "in": lambda: series.isin(node["values"]), "not_in": lambda: ~series.isin(node["values"]),
        "is_null": series.isna, "not_null": series.notna,
    }
    return operations[op]().fillna(False)


def evaluate_predicate(predicate: ScenarioPredicate, frame: pd.DataFrame) -> pd.Series:
    if predicate.expression.get("kind") == "batch_query":
        from .batch_query import evaluate_batch_query
        matched = evaluate_batch_query(predicate, frame)
        keys = frame["batch_id"].astype("string").fillna("<NA>").astype(str)
        return keys.isin(matched)
    return _evaluate(predicate.expression, frame)


def scenario_target_mask(
    predicate: ScenarioPredicate, frame: pd.DataFrame, *, product_id: int = 1,
) -> pd.Series:
    """Expand a row predicate into batch-level scene membership for target rows."""
    if "batch_id" not in frame:
        raise ValueError("batch-level scenario membership requires batch_id")
    target = pd.to_numeric(frame["product_id"], errors="coerce").eq(product_id)
    batch_keys = frame["batch_id"].astype("string").fillna("<NA>")
    if predicate.expression.get("kind") == "batch_query":
        from .batch_query import evaluate_batch_query
        matched_batches = evaluate_batch_query(predicate, frame, product_id=product_id)
    else:
        raw_hits = evaluate_predicate(predicate, frame) & target
        matched_batches = set(batch_keys.loc[raw_hits])
    return target & batch_keys.isin(matched_batches)


def scenario_batch_ids(
    predicate: ScenarioPredicate, frame: pd.DataFrame, *, product_id: int = 1,
) -> set[str]:
    membership = scenario_target_mask(predicate, frame, product_id=product_id)
    keys = frame["batch_id"].astype("string").fillna("<NA>")
    return set(keys.loc[membership].astype(str))


def coverage_ratio(predicate: ScenarioPredicate, frame: pd.DataFrame, *, product_id: int = 1) -> float:
    target = pd.to_numeric(frame["product_id"], errors="coerce").eq(product_id)
    if not bool(target.any()):
        raise ValueError("coverage denominator has no target-category batches")
    keys = frame["batch_id"].astype("string").fillna("<NA>")
    denominator = int(keys.loc[target].nunique(dropna=False))
    return float(len(scenario_batch_ids(predicate, frame, product_id=product_id)) / denominator)


def scenario_context_frame(predicate: ScenarioPredicate, frame: pd.DataFrame, *, product_id: int = 1) -> pd.DataFrame:
    """Keep complete engine batches selected by batch-level scene membership."""
    batches = scenario_batch_ids(predicate, frame, product_id=product_id)
    keys = frame["batch_id"].astype("string").fillna("<NA>").astype(str)
    return frame.loc[keys.isin(batches)].copy().reset_index(drop=True)


def scenario_evaluation_scope(
    predicate: ScenarioPredicate, frame: pd.DataFrame, *, product_id: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return matched full batches and all target rows in those batches for metrics."""
    membership = scenario_target_mask(predicate, frame, product_id=product_id)
    if not bool(membership.any()):
        raise ValueError("scenario has no matching target-category batches")
    batches = scenario_batch_ids(predicate, frame, product_id=product_id)
    keys = frame["batch_id"].astype("string").fillna("<NA>").astype(str)
    selected = frame.loc[keys.isin(batches)]
    context = selected.copy()
    context["__v2_metric_row_id"] = [str(index) for index in selected.index]
    metric_rows = context.loc[[str(index) in set(map(str, frame.index[membership]))
                               for index in selected.index]].copy()
    return context.reset_index(drop=True), metric_rows.reset_index(drop=True)


def predicate_overlap(left: ScenarioPredicate, right: ScenarioPredicate, frame: pd.DataFrame, *, product_id: int = 1) -> dict[str, Any]:
    left_batches = scenario_batch_ids(left, frame, product_id=product_id)
    right_batches = scenario_batch_ids(right, frame, product_id=product_id)
    intersection = len(left_batches & right_batches)
    union = len(left_batches | right_batches)
    return {
        "unit": "batch", "left_batches": len(left_batches), "right_batches": len(right_batches),
        "intersection_batches": intersection, "union_batches": union,
        "jaccard": float(intersection / union) if union else 0.0,
        "relation": "equal" if left.predicate_hash == right.predicate_hash else (
            "disjoint" if intersection == 0 else
            "left_contains_right" if right_batches <= left_batches else
            "right_contains_left" if left_batches <= right_batches else "partial_overlap"
        ),
    }


def predicates_provably_disjoint(left: ScenarioPredicate, right: ScenarioPredicate) -> bool:
    """Conservatively prove disjointness without treating absent observations as proof."""
    # General aggregate Queries require a symbolic arithmetic solver to prove
    # unsatisfiability.  Until such a proof exists they remain unresolved; an
    # observed empty intersection is intentionally insufficient.
    if left.expression.get("kind") == "batch_query" or right.expression.get("kind") == "batch_query":
        return False

    def leaf_interval(node: Mapping[str, Any]) -> tuple[str, float | None, bool, float | None, bool] | None:
        if "field" not in node or node.get("op") not in {"lt", "le", "gt", "ge", "eq", "between"}:
            return None
        field, op = str(node["field"]), str(node["op"])
        if op == "lt": return field, None, False, float(node["value"]), False
        if op == "le": return field, None, False, float(node["value"]), True
        if op == "gt": return field, float(node["value"]), False, None, False
        if op == "ge": return field, float(node["value"]), True, None, False
        if op == "eq":
            try: value = float(node["value"])
            except (TypeError, ValueError): return None
            return field, value, True, value, True
        return field, float(node["lower"]), True, float(node["upper"]), True

    def disjoint(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
        if "any" in a:
            return all(disjoint(item, b) for item in a["any"])
        if "any" in b:
            return all(disjoint(a, item) for item in b["any"])
        if "all" in a:
            return any(disjoint(item, b) for item in a["all"])
        if "all" in b:
            return any(disjoint(a, item) for item in b["all"])
        left_interval, right_interval = leaf_interval(a), leaf_interval(b)
        if left_interval and right_interval and left_interval[0] == right_interval[0]:
            _, left_low, left_low_closed, left_high, left_high_closed = left_interval
            _, right_low, right_low_closed, right_high, right_high_closed = right_interval
            if left_high is not None and right_low is not None:
                if left_high < right_low or (left_high == right_low and not (left_high_closed and right_low_closed)):
                    return True
            if right_high is not None and left_low is not None:
                if right_high < left_low or (right_high == left_low and not (right_high_closed and left_low_closed)):
                    return True
        if (a.get("field") == b.get("field") and a.get("op") == b.get("op") == "eq" and
                a.get("value") != b.get("value")):
            return True
        return False

    return disjoint(left.expression, right.expression)
