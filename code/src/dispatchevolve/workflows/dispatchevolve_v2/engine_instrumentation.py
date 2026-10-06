"""Inject V2-only runtime tracing into a copied Full Dispatch engine snapshot."""

from __future__ import annotations

import ast
from pathlib import Path


_VERSION = "v2-runtime-trace-v3"
_HELPERS = r'''
V2_TRACE_INSTRUMENTATION_VERSION = "v2-runtime-trace-v3"

def _v2_row_evidence(frame, mask):
    values = sorted(int(value) for value in frame.loc[mask, "__engine_row_order"].tolist())
    universe_size = int(frame["__engine_row_order"].max()) + 1 if len(frame) else 0
    bitmap = bytearray((universe_size + 7) // 8)
    for value in values:
        bitmap[value // 8] |= 1 << (value % 8)
    return {"count": len(values), "bitmap_hex": bytes(bitmap).hex()}

def _v2_trace_before(target, policy_module, policy_symbol):
    before_filtered = target["is_filtered"].fillna(False).astype(bool).copy()
    before_values = {
        name: target[name].copy()
        for name in RESULT_FIELDS
        if name in target.columns
    }
    provider = getattr(policy_module, "trace_applicable_mask", None)
    if provider is None:
        raise ValueError(
            f"Policy {policy_module.__name__}.{policy_symbol} omits trace_applicable_mask"
        )
    applicable = provider(target, policy_symbol)
    if not isinstance(applicable, pd.Series) or not applicable.index.equals(target.index):
        raise ValueError(
            f"Policy {policy_module.__name__}.{policy_symbol} returned an invalid applicability mask"
        )
    applicable = applicable.fillna(False).astype(bool) & ~before_filtered
    return before_filtered, before_values, applicable

def _v2_trace_after(target, policy_call, policy_file, before, events):
    if events is None or before is None:
        return
    before_filtered, before_values, applicable = before
    after_filtered = target["is_filtered"].fillna(False).astype(bool)
    newly_filtered = after_filtered & ~before_filtered
    modified = {}
    for name, previous in before_values.items():
        if name not in target.columns:
            continue
        after = target[name]
        equal = previous.eq(after) | (previous.isna() & after.isna())
        # Attribute mutations only to rows that reached this Policy while still
        # eligible. Some legacy Policy functions also write values on rows that
        # an earlier Filter already removed; those writes cannot affect Matching.
        changed = ~equal.fillna(False) & applicable
        if bool(changed.any()):
            modified[name] = _v2_row_evidence(target, changed)
    events.append({
        "order": len(events),
        "policy_call": policy_call,
        "policy_file": policy_file,
        "input_row_count": int(len(target)),
        "eligible_before": int((~before_filtered).sum()),
        "eligible_after": int((~after_filtered).sum()),
        "eligible_before_rows": _v2_row_evidence(target, ~before_filtered),
        "applicable_before_rows": _v2_row_evidence(target, applicable),
        "newly_filtered_rows": _v2_row_evidence(target, newly_filtered),
        "modified_rows_by_field": modified,
        "filter_rule_counts": dict(sorted(Counter(target.loc[newly_filtered, "filter_rule"].astype(str)).items())),
        "filter_policy_counts": dict(sorted(Counter(target.loc[newly_filtered, "filter_policy"].astype(str)).items())),
    })
'''


def _policy_modules(tree: ast.Module) -> set[str]:
    modules: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "policies":
            modules.update(alias.asname or alias.name for alias in node.names)
    return modules


def _policy_call(node: ast.stmt, modules: set[str]) -> tuple[str, str, str, str] | None:
    call = None
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        call = node.value
    elif isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.value, ast.Call):
        call = node.value
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
        return None
    if not isinstance(call.func.value, ast.Name) or call.func.value.id not in modules:
        return None
    module, symbol = call.func.value.id, call.func.attr
    return f"{module}.{symbol}", f"policies/{module}.py", module, symbol


def _instrument_policy_function(function: ast.FunctionDef, modules: set[str]) -> None:
    if "policy_events" not in {argument.arg for argument in function.args.args}:
        function.args.args.append(ast.arg(arg="policy_events"))
        function.args.defaults.append(ast.Constant(None))
    rewritten: list[ast.stmt] = []
    for node in function.body:
        identity = _policy_call(node, modules)
        if identity is None:
            rewritten.append(node)
            continue
        policy_call, policy_file, policy_module, policy_symbol = identity
        before = ast.parse(
            f"_v2_before = _v2_trace_before(target, {policy_module}, {policy_symbol!r}) if policy_events is not None else None"
        ).body[0]
        after = ast.Expr(ast.Call(
            func=ast.Name("_v2_trace_after", ast.Load()),
            args=[ast.Name("target", ast.Load()), ast.Constant(policy_call), ast.Constant(policy_file),
                  ast.Name("_v2_before", ast.Load()), ast.Name("policy_events", ast.Load())],
            keywords=[],
        ))
        rewritten.extend((before, node, after))
    function.body = rewritten


class _RunBatchCalls(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "_run_target_policies" and len(node.args) == 1:
            node.args.append(ast.Name("policy_events", ast.Load()))
        return node


def _instrument_run_batch(function: ast.FunctionDef) -> None:
    _RunBatchCalls().visit(function)
    insert_at = next(
        (index for index, node in enumerate(function.body)
         if isinstance(node, ast.If) and isinstance(node.test, ast.Call) and
         isinstance(node.test.func, ast.Name) and node.test.func.id == "len"),
        None,
    )
    if insert_at is None:
        raise ValueError("cannot locate target-policy branch in run_batch")
    assignment = ast.parse("policy_events = [] if trace else None").body[0]
    function.body.insert(insert_at, assignment)
    trace_dict = None
    for node in ast.walk(function):
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "trace_payload" for target in node.targets):
            if isinstance(node.value, ast.Dict):
                trace_dict = node.value
    if trace_dict is None:
        raise ValueError("cannot locate trace_payload mapping in run_batch")
    additions = {
        "policy_events": ast.Name("policy_events", ast.Load()),
        "input_target_rows": ast.parse("_v2_row_evidence(target, pd.Series(True, index=target.index))", mode="eval").body,
        "final_eligible_target_rows": ast.parse("_v2_row_evidence(eligible_target, pd.Series(True, index=eligible_target.index))", mode="eval").body,
    }
    for key, value in additions.items():
        trace_dict.keys.append(ast.Constant(key))
        trace_dict.values.append(value)


def instrument_engine(engine_dir: Path) -> Path:
    """Instrument only the copied V2 snapshot; never mutate the shared seed."""
    engine_path = Path(engine_dir) / "engine.py"
    source = engine_path.read_text(encoding="utf-8")
    if _VERSION in source:
        return engine_path
    if "V2_TRACE_INSTRUMENTATION_VERSION" in source:
        raise ValueError("V2 trace instrumentation upgrades require a fresh engine snapshot")
    tree = ast.parse(source, str(engine_path))
    modules = _policy_modules(tree)
    policy_functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_run_target_policies"]
    run_functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_batch"]
    if len(policy_functions) != 1 or len(run_functions) != 1:
        raise ValueError("V2 trace instrumentation requires one _run_target_policies and one run_batch")
    _instrument_policy_function(policy_functions[0], modules)
    _instrument_run_batch(run_functions[0])
    helper_nodes = ast.parse(_HELPERS).body
    insertion = tree.body.index(policy_functions[0])
    tree.body[insertion:insertion] = helper_nodes
    ast.fix_missing_locations(tree)
    instrumented = ast.unparse(tree).rstrip() + "\n"
    compile(instrumented, str(engine_path), "exec")
    engine_path.write_text(instrumented, encoding="utf-8")
    return engine_path
