"""Deterministic compilation for a frozen Candidate set and precedence plan."""

from __future__ import annotations

import ast
import re
import shutil
from pathlib import Path
from typing import Callable

from dispatchevolve.baselines.candidates import RepositoryGenomeCodec

from .contracts import CombinationProposal, LocalCandidate, RelationEdge
from .llm import V2LLM
from .prompt_store import PromptStore
from .protocol import atomic_json


_ENGINE = re.compile(r"\A## Rationale\n+([\s\S]+?)\n+## Engine Source\n+```python\n([\s\S]+?)\n```\Z")


def copy_engine(source: Path, destination: Path) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return destination


def variant_policy_path(candidate_id: str, relative: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_]", "_", candidate_id[:12])
    return f"policies/v2_{token}_{Path(relative).name}"


def instrument_variant_call_probe(
    source: Path, destination: Path, expected_callables: dict[str, dict[str, str]],
    result_fields: tuple[str, ...],
) -> Path:
    """Create a system-owned copy that proves selected variant calls executed."""
    copy_engine(source, destination)
    engine_path = destination / "engine.py"
    source_text = engine_path.read_text(encoding="utf-8")
    tree = ast.parse(source_text, str(engine_path))
    module_aliases: dict[str, str] = {}
    direct_aliases: dict[str, tuple[str, str]] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in expected_callables:
                    module_aliases[alias.asname or alias.name.split(".")[0]] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                module = f"{node.module}.{alias.name}"
                if module in expected_callables:
                    module_aliases[alias.asname or alias.name] = module
                elif node.module in expected_callables and alias.name in expected_callables[node.module]:
                    direct_aliases[alias.asname or alias.name] = (node.module, alias.name)
    called_attributes: set[tuple[str, str]] = set()
    called_direct: set[str] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and
                isinstance(node.func.value, ast.Name) and node.func.value.id in module_aliases and
                node.func.attr in expected_callables[module_aliases[node.func.value.id]]):
            called_attributes.add((node.func.value.id, node.func.attr))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in direct_aliases:
            called_direct.add(node.func.id)
    observed = {
        (module_aliases[alias], symbol) for alias, symbol in called_attributes
    } | {direct_aliases[alias] for alias in called_direct}
    required = {(module, symbol) for module, symbols in expected_callables.items() for symbol in symbols}
    missing = sorted(required - observed)
    deterministic_integration = "V2_DETERMINISTIC_INTEGRATION_VERSION" in source_text
    if missing and not deterministic_integration:
        raise ValueError(f"system call probe found no callable use for variants: {missing}")
    wrappers = [
        "import pandas as pd",
        "_V2_SYSTEM_CALL_COUNTS = {}",
        "_V2_SYSTEM_OBSERVED_EVENTS = []",
        f"_V2_SYSTEM_RESULT_FIELDS = {result_fields!r}",
        "def _v2_system_row_evidence(frame, mask):",
        "    if '__engine_row_order' not in frame:",
        "        raise ValueError('variant probe requires __engine_row_order')",
        "    values = sorted(int(value) for value in frame.loc[mask, '__engine_row_order'].tolist())",
        "    universe = int(frame['__engine_row_order'].max()) + 1 if len(frame) else 0",
        "    bitmap = bytearray((universe + 7) // 8)",
        "    for value in values:",
        "        bitmap[value // 8] |= 1 << (value % 8)",
        "    return {'count': len(values), 'bitmap_hex': bytes(bitmap).hex()}",
        "def _v2_system_wrap(function, key):",
        "    def wrapped(*args, **kwargs):",
        "        _V2_SYSTEM_CALL_COUNTS[key] = _V2_SYSTEM_CALL_COUNTS.get(key, 0) + 1",
        "        frame = args[0] if args else None",
        "        if frame is None or not hasattr(frame, 'copy'):",
        "            raise ValueError('variant probe requires a DataFrame first argument')",
        "        before = frame.copy(deep=True)",
        "        before_filtered = (before['is_filtered'].fillna(False).astype(bool) if 'is_filtered' in before else pd.Series(False, index=before.index))",
        "        module = __import__(function.__module__, fromlist=['trace_applicable_mask'])",
        "        provider = getattr(module, 'trace_applicable_mask', None)",
        "        applicable = (provider(before, function.__name__) if provider is not None else pd.Series(True, index=before.index))",
        "        applicable = applicable.fillna(False).astype(bool) & ~before_filtered",
        "        result = function(*args, **kwargs)",
        # Policy return values are not required to be the mutated dispatch
        # frame (for example, scoring returns a component table).  Probe the
        # first DataFrame argument, which is the engine's in-place state.
        "        after = frame",
        "        after_filtered = after['is_filtered'].reindex(before.index).fillna(False).astype(bool)",
        "        newly_filtered = after_filtered & ~before_filtered",
        "        modified = {}",
        "        for name in _V2_SYSTEM_RESULT_FIELDS:",
        "            if name not in before or name not in after:",
        "                continue",
        "            previous = before[name]",
        "            current = after[name].reindex(before.index)",
        "            equal = previous.eq(current) | (previous.isna() & current.isna())",
        "            changed = ~equal.fillna(False) & applicable",
        "            if bool(changed.any()):",
        "                modified[name] = _v2_system_row_evidence(before, changed)",
        "        _V2_SYSTEM_OBSERVED_EVENTS.append({'key': key, 'newly_filtered_rows': _v2_system_row_evidence(before, newly_filtered), 'modified_rows_by_field': modified})",
        "        return result",
        "    return wrapped",
    ]
    if deterministic_integration:
        # The deterministic compiler replaces the incumbent's module aliases with
        # routing proxies.  Assigning a wrapper to such a proxy would shadow
        # ``__getattr__`` and force every route back to the incumbent callable.
        # Instrument the concrete original/variant modules instead so the probe
        # observes execution without changing routing semantics.
        for module, symbol in sorted(required):
            key = expected_callables[module][symbol]
            alias = f"_v2_system_module_{len(wrappers)}"
            wrappers.append(f"import {module} as {alias}")
            wrappers.append(
                f"{alias}.{symbol} = _v2_system_wrap({alias}.{symbol}, {key!r})"
            )
    else:
        for alias, symbol in sorted(called_attributes):
            key = expected_callables[module_aliases[alias]][symbol]
            wrappers.append(f"{alias}.{symbol} = _v2_system_wrap({alias}.{symbol}, {key!r})")
        for alias in sorted(called_direct):
            module, symbol = direct_aliases[alias]
            key = expected_callables[module][symbol]
            wrappers.append(f"{alias} = _v2_system_wrap({alias}, {key!r})")
    wrappers.extend([
        "_v2_system_original_run_batch = run_batch",
        "def run_batch(batch_data, *, trace=True):",
        "    _V2_SYSTEM_CALL_COUNTS.clear()",
        "    _V2_SYSTEM_OBSERVED_EVENTS.clear()",
        "    result = _v2_system_original_run_batch(batch_data, trace=trace)",
        "    if trace and isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):",
        "        payload = dict(result[1])",
        "        payload['_v2_system_call_counts'] = dict(_V2_SYSTEM_CALL_COUNTS)",
        "        payload['_v2_system_observed_events'] = list(_V2_SYSTEM_OBSERVED_EVENTS)",
        "        return result[0], payload",
        "    return result",
    ])
    probed = source_text.rstrip() + "\n\n" + "\n".join(wrappers) + "\n"
    compile(probed, str(engine_path), "exec")
    engine_path.write_text(probed, encoding="utf-8")
    return destination


def policy_call_symbols(engine_dir: Path, relative: str) -> tuple[str, ...]:
    """Resolve the exact callable symbols used for a policy file by its engine."""
    module = Path(relative).with_suffix("").as_posix().replace("/", ".")
    tree = ast.parse((Path(engine_dir) / "engine.py").read_text(encoding="utf-8"))
    module_aliases: set[str] = set()
    direct_aliases: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == module:
                    module_aliases.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            package, _, member = module.rpartition(".")
            for alias in node.names:
                if node.module == package and alias.name == member:
                    module_aliases.add(alias.asname or alias.name)
                elif node.module == module:
                    direct_aliases[alias.asname or alias.name] = alias.name
    symbols: set[str] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and
                isinstance(node.func.value, ast.Name) and node.func.value.id in module_aliases):
            symbols.add(node.func.attr)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in direct_aliases:
            symbols.add(direct_aliases[node.func.id])
    if not symbols:
        raise ValueError(f"candidate engine has no executable call for policy file: {relative}")
    return tuple(sorted(symbols))


def engine_policy_callables(engine_dir: Path) -> dict[str, tuple[str, ...]]:
    """Return every policy module and callable symbol used by an engine."""
    tree = ast.parse((Path(engine_dir) / "engine.py").read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names if alias.name.startswith("policies."))
        elif isinstance(node, ast.ImportFrom) and node.module == "policies":
            modules.update(f"policies.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("policies."):
            modules.add(node.module)
    result = {}
    for module in sorted(modules):
        relative = module.replace(".", "/") + ".py"
        try:
            result[module] = policy_call_symbols(engine_dir, relative)
        except ValueError:
            continue
    if not result:
        raise ValueError("incumbent engine exposes no executable policy callables")
    return result


def engine_result_fields(engine_dir: Path) -> tuple[str, ...]:
    """Read the incumbent's frozen result fields used for exact effect checks."""
    tree = ast.parse((Path(engine_dir) / "engine.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            target = node.targets[0] if isinstance(node, ast.Assign) else node.target
            if isinstance(target, ast.Name) and target.id == "RESULT_FIELDS" and node.value is not None:
                values = tuple(str(item) for item in ast.literal_eval(node.value))
                if values:
                    return values
    raise ValueError("incumbent engine must define a literal nonempty RESULT_FIELDS sequence")


def engine_frozen_scenarios(engine_dir: Path) -> dict[str, str]:
    """Read prior-round frozen scenario predicates, if this is an integrated engine."""
    tree = ast.parse((Path(engine_dir) / "engine.py").read_text(encoding="utf-8"))
    constants: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            target = node.targets[0] if isinstance(node, ast.Assign) else node.target
            if (isinstance(target, ast.Name) and
                    target.id in {"V2_ACTIVE_SCENARIO_PREDICATES", "V2_SCENARIO_PREDICATES"} and
                    node.value is not None):
                constants[target.id] = ast.literal_eval(node.value)
    value = constants.get("V2_ACTIVE_SCENARIO_PREDICATES", constants.get("V2_SCENARIO_PREDICATES", {}))
    if not isinstance(value, dict):
        raise ValueError("V2 active scenario registry must be a literal mapping")
    if value:
        return {str(key): str(predicate) for key, predicate in value.items()}
    return {}


def integrate(
    incumbent: Path, candidates: dict[str, LocalCandidate], proposal: CombinationProposal,
    edges: tuple[RelationEdge, ...], destination: Path, *, llm: V2LLM, prompts: PromptStore,
    llm_call: Callable[[str, object], str] | None = None,
    protocol_hash: str | None = None,
) -> tuple[Path, str]:
    """Deterministically compile a frozen Composition from independent artifacts."""
    del llm, prompts, llm_call
    copy_engine(incumbent, destination)
    variants: list[dict[str, str]] = []
    variant_modules: dict[str, dict[str, str]] = {}
    variant_files: dict[str, dict[str, str]] = {}
    for candidate_id in proposal.priority_plan.ordered_candidate_ids:
        candidate = candidates[candidate_id]
        for relative in candidate.policy_files:
            source = candidate.engine_dir / relative
            variant_file = variant_policy_path(candidate_id, relative)
            variant_name = Path(variant_file).name
            target = destination / variant_file
            shutil.copy2(source, target)
            logical_module = Path(relative).with_suffix("").as_posix().replace("/", ".")
            variant_module = f"policies.{Path(variant_name).stem}"
            variant_modules.setdefault(candidate_id, {})[logical_module] = variant_module
            variant_files.setdefault(candidate_id, {})[relative] = variant_file
            variants.append({
                "candidate_id": candidate_id, "scenario": candidate.scenario.canonical,
                "objective": candidate.objective, "original_policy": relative,
                "variant_module": variant_module, "variant_source": source.read_text(encoding="utf-8"),
            })
    selected_edges = [vars(edge) for edge in edges if edge.left in proposal.candidate_ids and edge.right in proposal.candidate_ids]
    current_predicates = {identifier: candidates[identifier].scenario.canonical for identifier in proposal.candidate_ids}
    current_expressions = {identifier: dict(candidates[identifier].scenario.expression) for identifier in proposal.candidate_ids}
    conflict_plan = tuple(dict(item) for item in proposal.priority_plan.precedence_edges)
    source = (incumbent / "engine.py").read_text(encoding="utf-8")
    tree = ast.parse(source, "engine.py")
    module_aliases: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("policies."):
                    module_aliases[alias.name] = alias.asname or alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "policies":
                for alias in node.names:
                    module_aliases[f"policies.{alias.name}"] = alias.asname or alias.name
    missing_originals = sorted({module for mapping in variant_modules.values() for module in mapping} - set(module_aliases))
    if missing_originals:
        raise ValueError(f"selected Policy is not imported by fixed E0: {missing_originals}")
    originals_literal = ",\n".join(
        f"    {module!r}: {module_aliases[module]}" for module in sorted({module for mapping in variant_modules.values() for module in mapping})
    )
    variant_aliases: dict[tuple[str, str], str] = {}
    variant_imports: list[str] = []
    for candidate_id, modules in variant_modules.items():
        for logical_module, module in modules.items():
            alias = f"_v2_variant_{len(variant_aliases)}"
            variant_aliases[(candidate_id, logical_module)] = alias
            variant_imports.append(f"import {module} as {alias}")
    loaded_variants_literal = "\n".join(
        f"    {candidate_id!r}: {{{', '.join(f'{logical!r}: {variant_aliases[(candidate_id, logical)]}' for logical in modules)}}},"
        for candidate_id, modules in variant_modules.items()
    )
    compiler = f'''

V2_DETERMINISTIC_INTEGRATION_VERSION = "v2-fixed-e0-precedence-graph-v2"
V2_SELECTED_CANDIDATE_IDS = {tuple(proposal.candidate_ids)!r}
V2_TOPOLOGICAL_EXECUTION_ORDER = {tuple(proposal.priority_plan.ordered_candidate_ids)!r}
V2_CONFLICT_RESOLUTION_PLAN = {conflict_plan!r}
V2_SCENARIO_PREDICATES = {current_predicates!r}
V2_SCENARIO_EXPRESSIONS = {current_expressions!r}
V2_VARIANT_MODULES = {variant_modules!r}
V2_VARIANT_POLICY_FILES = {variant_files!r}

{chr(10).join(variant_imports)}

_V2_ORIGINAL_POLICY_MODULES = {{
{originals_literal}
}}
_V2_LOADED_VARIANTS = {{
{loaded_variants_literal}
}}
_V2_ACTIVE_CANDIDATE_IDS = ()

def _v2_target_mask(frame):
    if "product_id" not in frame:
        raise ValueError("V2 composition requires product_id")
    return pd.to_numeric(frame["product_id"], errors="coerce").eq(1)

def _v2_candidate_for_policy(logical_policy_file):
    matches = [
        candidate_id for candidate_id in _V2_ACTIVE_CANDIDATE_IDS
        if logical_policy_file in V2_VARIANT_POLICY_FILES.get(candidate_id, {{}})
    ]
    if len(matches) > 1:
        raise ValueError(f"multiple active Local Candidates compete for {{logical_policy_file}}")
    return matches[0] if matches else None

class _V2PolicyProxy:
    def __init__(self, logical_module):
        self.logical_module = logical_module
    def __getattr__(self, name):
        logical_file = self.logical_module.replace(".", "/") + ".py"
        candidate_id = _v2_candidate_for_policy(logical_file)
        module = (_V2_LOADED_VARIANTS[candidate_id][self.logical_module]
                  if candidate_id is not None else _V2_ORIGINAL_POLICY_MODULES[self.logical_module])
        return getattr(module, name)

for _v2_module_name, _v2_alias in { {module: module_aliases[module] for module in sorted({module for mapping in variant_modules.values() for module in mapping})}!r}.items():
    globals()[_v2_alias] = _V2PolicyProxy(_v2_module_name)

_v2_original_trace_after = _v2_trace_after
def _v2_trace_after(target, policy_call, policy_file, before, events):
    _v2_original_trace_after(target, policy_call, policy_file, before, events)
    candidate_id = _v2_candidate_for_policy(policy_file)
    if events is not None and candidate_id is not None:
        variant_file = V2_VARIANT_POLICY_FILES[candidate_id].get(policy_file)
        if variant_file and events:
            events[-1]["candidate_id"] = candidate_id
            events[-1]["policy_file"] = variant_file

def _v2_row_expression(node, target):
    if "all" in node:
        result = pd.Series(True, index=target.index)
        for child in node["all"]: result &= _v2_row_expression(child, target)
        return result
    if "any" in node:
        result = pd.Series(False, index=target.index)
        for child in node["any"]: result |= _v2_row_expression(child, target)
        return result
    if "not" in node: return ~_v2_row_expression(node["not"], target)
    series, operation = target[node["field"]], node["op"]
    if operation in {{"lt", "le", "gt", "ge", "between"}}:
        series = pd.to_numeric(series, errors="coerce")
    if operation == "lt": result = series.lt(node["value"])
    elif operation == "le": result = series.le(node["value"])
    elif operation == "gt": result = series.gt(node["value"])
    elif operation == "ge": result = series.ge(node["value"])
    elif operation == "eq": result = series.eq(node["value"])
    elif operation == "ne": result = series.ne(node["value"])
    elif operation == "between": result = series.ge(node["lower"]) & series.le(node["upper"])
    elif operation == "in": result = series.isin(node["values"])
    elif operation == "not_in": result = ~series.isin(node["values"])
    elif operation == "is_null": result = series.isna()
    elif operation == "not_null": result = series.notna()
    else: raise ValueError(f"unsupported frozen V2 row predicate operation: {{operation}}")
    return result.fillna(False)

def _v2_batch_expression(node, target):
    if "literal" in node: return node["literal"]
    if "collection" in node: return tuple(node["collection"])
    if "aggregate" in node:
        operation = node["aggregate"]
        if operation == "count": return int(len(target))
        series = target[node["field"]]
        if operation in {{"mean", "min", "max", "sum"}}:
            values = pd.to_numeric(series, errors="coerce").dropna()
            if operation == "sum": return float(values.sum()) if len(values) else 0.0
            return float(getattr(values, operation)()) if len(values) else float("nan")
        if operation == "nunique": return int(series.nunique(dropna=True))
        raise ValueError(f"unsupported batch aggregate: {{operation}}")
    if "and" in node: return all(bool(_v2_batch_expression(item, target)) for item in node["and"])
    if "or" in node: return any(bool(_v2_batch_expression(item, target)) for item in node["or"])
    if "not" in node: return not bool(_v2_batch_expression(node["not"], target))
    if "unary" in node:
        value = _v2_batch_expression(node["value"], target)
        return +value if node["unary"] == "+" else -value
    if "binary" in node:
        left, right = _v2_batch_expression(node["left"], target), _v2_batch_expression(node["right"], target)
        if node["binary"] == "/" and (right == 0 or pd.isna(right)): return 0.0
        return {{"+": lambda: left + right, "-": lambda: left - right,
                "*": lambda: left * right, "/": lambda: left / right}}[node["binary"]]()
    if "compare" in node:
        left, right = _v2_batch_expression(node["left"], target), _v2_batch_expression(node["right"], target)
        if pd.isna(left) or (not isinstance(right, tuple) and pd.isna(right)): return False
        return {{">": lambda: left > right, ">=": lambda: left >= right,
                "<": lambda: left < right, "<=": lambda: left <= right,
                "==": lambda: left == right, "!=": lambda: left != right,
                "in": lambda: left in right, "not in": lambda: left not in right}}[node["compare"]]()
    raise ValueError("invalid frozen V2 batch Query node")

def _v2_scene_matches(candidate_id, target):
    expression = V2_SCENARIO_EXPRESSIONS[candidate_id]
    if expression.get("kind") == "batch_query":
        result = _v2_batch_expression(expression["tree"], target)
        if type(result).__name__ not in {{"bool", "bool_"}}:
            raise ValueError("frozen V2 batch Query root is not Boolean")
        return bool(result)
    return bool(_v2_row_expression(expression, target).any())

def _v2_select_active_candidates(frame):
    target = frame.loc[_v2_target_mask(frame)]
    if target.empty: return (), (), ()
    matched = tuple(candidate_id for candidate_id in V2_SELECTED_CANDIDATE_IDS
                    if _v2_scene_matches(candidate_id, target))
    active = []
    for candidate_id in V2_TOPOLOGICAL_EXECUTION_ORDER:
        if candidate_id not in matched: continue
        blocked = any(edge["lower"] == candidate_id and edge["higher"] in active
                      for edge in V2_CONFLICT_RESOLUTION_PLAN)
        if not blocked: active.append(candidate_id)
    blocked = tuple(item for item in matched if item not in active)
    return matched, tuple(active), blocked

_v2_fixed_e0_run_batch = run_batch
def run_batch(batch_data, *, trace=True):
    global _V2_ACTIVE_CANDIDATE_IDS
    matched, active, blocked = _v2_select_active_candidates(batch_data)
    _V2_ACTIVE_CANDIDATE_IDS = active
    try:
        result = _v2_fixed_e0_run_batch(batch_data, trace=trace)
    finally:
        _V2_ACTIVE_CANDIDATE_IDS = ()
    if trace and isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        payload = dict(result[1]); target_mask = _v2_target_mask(batch_data)
        payload["v2_routing"] = [
            {{"input_row_order": int(index),
              "matched_candidate_ids": list(matched) if bool(target_mask.iloc[index]) else [],
              "active_candidate_ids": list(active) if bool(target_mask.iloc[index]) else [],
              "blocked_candidate_ids": list(blocked) if bool(target_mask.iloc[index]) else []}}
            for index in range(len(batch_data))
        ]
        payload["v2_precedence_decisions"] = [
            {{"higher": edge["higher"], "lower": edge["lower"], "policies": list(edge["policies"]),
              "higher_active": edge["higher"] in active,
              "lower_blocked": edge["lower"] in blocked}}
            for edge in V2_CONFLICT_RESOLUTION_PLAN if edge["higher"] in matched and edge["lower"] in matched
        ]
        return result[0], payload
    return result
'''
    source = source.rstrip() + "\n" + compiler
    compile(source, str(destination / "engine.py"), "exec")
    (destination / "engine.py").write_text(source, encoding="utf-8")
    for path in (destination / "policies").glob("*.py"):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    engine_id = RepositoryGenomeCodec().candidate_id(destination)
    atomic_json(destination.parent / "integration_manifest.json", {
        "engine_id": engine_id, "proposal_id": proposal.proposal_id,
        "fixed_e0_id": RepositoryGenomeCodec().candidate_id(incumbent),
        "protocol_hash": protocol_hash,
        "selected_candidate_ids": proposal.candidate_ids,
        "conflict_resolution_plan": conflict_plan,
        "priority_plan": proposal.priority_plan, "variants": variants,
        "relation_edges": selected_edges,
        "producer": "deterministic_fixed_e0_precedence_graph_compiler_v2",
        "llm_called": False,
    })
    return destination, engine_id
