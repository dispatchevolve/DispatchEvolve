"""Safe, canonical batch-level Scene Query parsing and execution."""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from .contracts import ScenarioPredicate, canonical_json, content_hash


QUERY_DSL_VERSION = "batch-scene-query-v1"
FEATURE_CATALOG_VERSION = "full-dispatch-feature-catalog-v2"
AGGREGATES = {"count", "mean", "min", "max", "sum", "nunique"}
_PURE_IDENTIFIER_FEATURES = {
    "batch_id",
    "driver_id",
    "order_id",
    "passenger_id",
    "uuid",
}


class BatchQueryError(ValueError):
    """A stable, user-correctable Query validation failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class FeatureRecord:
    name: str
    type: str
    description: str
    stats: str


def _load_descriptions():
    from importlib import resources
    import json
    return json.loads(resources.files("dispatchevolve.tasks.full_dispatch")
                      .joinpath("feature_info.json").read_text())


_DESCRIPTIONS = _load_descriptions()


def _feature_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series.dtype):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series.dtype):
        return "number"
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return "datetime"
    return "category_or_string"


def _display(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NA"
    if isinstance(value, float):
        return f"{value:.12g}" if math.isfinite(value) else "NA"
    return str(value).replace("|", "/").replace("\n", " ")


def build_feature_catalog(frame: pd.DataFrame, *, excluded: set[str] | None = None) -> tuple[FeatureRecord, ...]:
    """Build the deterministic Prompt-facing catalog over the pre-scoped Search D."""
    excluded = set(excluded or ())
    records: list[FeatureRecord] = []
    for name in sorted(str(item) for item in frame.columns if str(item) not in excluded and not str(item).startswith("__")):
        series = frame[name]
        kind = _feature_type(series)
        missing = int(series.isna().sum())
        if name in _PURE_IDENTIFIER_FEATURES:
            # Raw identifier values and numeric moments do not help Scene
            # reasoning. Distinctness remains useful with nunique(FIELD).
            values = series.dropna()
            stats = f"missing={missing};distinct={int(values.nunique())}"
        elif kind == "number":
            values = pd.to_numeric(series, errors="coerce").dropna()
            stats = (
                f"missing={missing};mean={_display(values.mean() if len(values) else None)};"
                f"min={_display(values.min() if len(values) else None)};"
                f"max={_display(values.max() if len(values) else None)}"
            )
        elif kind == "boolean":
            values = series.dropna().astype(bool)
            stats = f"missing={missing};true_rate={_display(values.mean() if len(values) else None)}"
        elif kind == "datetime":
            values = pd.to_datetime(series, errors="coerce").dropna()
            stats = (
                f"missing={missing};min={_display(values.min() if len(values) else None)};"
                f"max={_display(values.max() if len(values) else None)}"
            )
        else:
            values = series.dropna().astype(str)
            examples = sorted(set(values))[:3]
            stats = f"missing={missing};distinct={int(values.nunique())};examples=[{','.join(_display(item) for item in examples)}]"
        description = _DESCRIPTIONS.get(name, f"Input feature {name.replace('_', ' ')}.")
        records.append(FeatureRecord(name, kind, description, stats))
    if not records:
        raise ValueError("Feature Catalog is empty")
    return tuple(records)


def render_feature_catalog(records: tuple[FeatureRecord, ...]) -> str:
    lines = ["name | type | description | stats"]
    lines.extend(
        " | ".join(
            str(value).replace("|", "/").replace("\n", " ")
            for value in (item.name, item.type, item.description, item.stats)
        )
        for item in records
    )
    return "\n".join(lines)


def feature_catalog_identity(records: tuple[FeatureRecord, ...]) -> Mapping[str, Any]:
    document = {
        "dsl_version": QUERY_DSL_VERSION,
        "catalog_version": FEATURE_CATALOG_VERSION,
        "features": [item.__dict__ for item in records],
        "missing_value_rule": "aggregate functions ignore nulls; empty aggregate is NA except count/nunique/sum",
        "division_by_zero": 0.0,
    }
    return {**document, "catalog_hash": content_hash(document)}


def _field_argument(node: ast.AST, schema: Mapping[str, Any]) -> str:
    if not isinstance(node, ast.Name):
        raise BatchQueryError("INVALID_QUERY_SYNTAX", "aggregate arguments must be bare Feature Catalog names")
    if node.id not in schema:
        raise BatchQueryError("UNKNOWN_FEATURE", f"query references unknown feature: {node.id}")
    return node.id


def _normalize(node: ast.AST, schema: Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(node, ast.Expression):
        return _normalize(node.body, schema)
    if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
        op = "and" if isinstance(node.op, ast.And) else "or"
        return {op: [_normalize(item, schema) for item in node.values]}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return {"not": _normalize(node.operand, schema)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return {"unary": "+" if isinstance(node.op, ast.UAdd) else "-", "value": _normalize(node.operand, schema)}
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        op = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}[type(node.op)]
        return {"binary": op, "left": _normalize(node.left, schema), "right": _normalize(node.right, schema)}
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise BatchQueryError("INVALID_QUERY_SYNTAX", "chained comparisons are not supported")
        operator = node.ops[0]
        allowed = {
            ast.Gt: ">", ast.GtE: ">=", ast.Lt: "<", ast.LtE: "<=",
            ast.Eq: "==", ast.NotEq: "!=", ast.In: "in", ast.NotIn: "not in",
        }
        if type(operator) not in allowed:
            raise BatchQueryError("INVALID_QUERY_SYNTAX", "unsupported comparison operator")
        return {"compare": allowed[type(operator)], "left": _normalize(node.left, schema),
                "right": _normalize(node.comparators[0], schema)}
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in AGGREGATES or node.keywords:
            raise BatchQueryError("INVALID_QUERY_SYNTAX", "only documented aggregate functions are allowed")
        if node.func.id == "count":
            if node.args:
                raise BatchQueryError("INVALID_QUERY_SYNTAX", "count() takes no arguments")
            return {"aggregate": "count"}
        if len(node.args) != 1:
            raise BatchQueryError("INVALID_QUERY_SYNTAX", f"{node.func.id}() takes one feature argument")
        field = _field_argument(node.args[0], schema)
        if node.func.id in {"mean", "min", "max", "sum"} and str(schema[field]) not in {"number", "boolean"}:
            raise BatchQueryError(
                "INVALID_QUERY_TYPE", f"{node.func.id}() requires a numeric or Boolean feature: {field}"
            )
        return {"aggregate": node.func.id, "field": field}
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, float, bool)):
        return {"literal": float(node.value) if isinstance(node.value, int) and not isinstance(node.value, bool) else node.value}
    if isinstance(node, (ast.List, ast.Tuple)):
        values = [_normalize(item, schema) for item in node.elts]
        if any(set(item) != {"literal"} for item in values):
            raise BatchQueryError("INVALID_QUERY_SYNTAX", "membership collections must contain literals only")
        return {"collection": [item["literal"] for item in values]}
    if isinstance(node, ast.Name):
        raise BatchQueryError("INVALID_QUERY_SYNTAX", "raw feature names may appear only inside aggregate functions")
    raise BatchQueryError("INVALID_QUERY_SYNTAX", f"unsupported query syntax: {type(node).__name__}")


def parse_batch_query(text: str, schema: Mapping[str, Any]) -> ScenarioPredicate:
    source = " ".join(str(text).strip().split())
    if not source:
        raise BatchQueryError("INVALID_QUERY_SYNTAX", "query is empty")
    try:
        parsed = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise BatchQueryError("INVALID_QUERY_SYNTAX", f"query syntax error: {exc.msg}") from exc
    tree = _normalize(parsed, schema)
    expression = {"kind": "batch_query", "version": QUERY_DSL_VERSION, "tree": tree}
    # AST unparse removes formatting-only identity differences while the normalized
    # tree remains the authoritative executable representation.
    canonical = ast.unparse(parsed.body)
    return ScenarioPredicate(expression, canonical, content_hash(expression))


def is_batch_query(predicate: ScenarioPredicate) -> bool:
    return predicate.expression.get("kind") == "batch_query"


def _aggregate(operation: str, field: str | None, batch: pd.DataFrame) -> Any:
    if operation == "count":
        return int(len(batch))
    series = batch[str(field)]
    if operation in {"mean", "min", "max", "sum"}:
        values = pd.to_numeric(series, errors="coerce").dropna()
        if operation == "sum":
            return float(values.sum()) if len(values) else 0.0
        if not len(values):
            return float("nan")
        return float(getattr(values, operation)())
    if operation == "nunique":
        return int(series.nunique(dropna=True))
    raise AssertionError(operation)


def _execute(node: Mapping[str, Any], batch: pd.DataFrame) -> Any:
    if "literal" in node:
        return node["literal"]
    if "collection" in node:
        return tuple(node["collection"])
    if "aggregate" in node:
        return _aggregate(str(node["aggregate"]), node.get("field"), batch)
    if "and" in node:
        return all(bool(_execute(item, batch)) for item in node["and"])
    if "or" in node:
        return any(bool(_execute(item, batch)) for item in node["or"])
    if "not" in node:
        return not bool(_execute(node["not"], batch))
    if "unary" in node:
        value = _execute(node["value"], batch)
        return +value if node["unary"] == "+" else -value
    if "binary" in node:
        left, right = _execute(node["left"], batch), _execute(node["right"], batch)
        if node["binary"] == "/" and (right == 0 or pd.isna(right)):
            return 0.0
        return {"+": lambda: left + right, "-": lambda: left - right,
                "*": lambda: left * right, "/": lambda: left / right}[node["binary"]]()
    if "compare" in node:
        left, right = _execute(node["left"], batch), _execute(node["right"], batch)
        if pd.isna(left) or (not isinstance(right, tuple) and pd.isna(right)):
            return False
        return {
            ">": lambda: left > right, ">=": lambda: left >= right,
            "<": lambda: left < right, "<=": lambda: left <= right,
            "==": lambda: left == right, "!=": lambda: left != right,
            "in": lambda: left in right, "not in": lambda: left not in right,
        }[node["compare"]]()
    raise ValueError(f"invalid normalized batch Query node: {canonical_json(node)}")


def evaluate_batch_query(predicate: ScenarioPredicate, frame: pd.DataFrame, *, product_id: int = 1) -> set[str]:
    if not is_batch_query(predicate):
        raise TypeError("predicate is not a batch Query")
    if "batch_id" not in frame or "product_id" not in frame:
        raise ValueError("batch Query execution requires batch_id and product_id")
    target = frame.loc[pd.to_numeric(frame["product_id"], errors="coerce").eq(product_id)]
    hits: set[str] = set()
    for batch_id, batch in target.groupby("batch_id", sort=False, dropna=False):
        result = _execute(predicate.expression["tree"], batch)
        if not isinstance(result, (bool, type(pd.NA))) and not isinstance(result, bool):
            # numpy.bool_ is accepted below through bool conversion; numeric roots are not.
            if type(result).__name__ != "bool_":
                raise BatchQueryError("INVALID_QUERY_TYPE", "Query root must return true or false")
        if not pd.isna(result) and bool(result):
            hits.add("<NA>" if pd.isna(batch_id) else str(batch_id))
    return hits
