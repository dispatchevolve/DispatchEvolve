"""Configuration loaders and deterministic baseline matrix expansion."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from .contracts import (
    DatasetSpec,
    E0FinalizationKey,
    E0Key,
    MatrixPlan,
    ModelProfile,
    RunKey,
    SelectionKey,
    SuiteConfig,
)

_ENV_REFERENCE = re.compile(r"^\$\{[A-Za-z_][A-Za-z0-9_]*\}$")
_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9]+")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAIN_DATASET_FIELDS = (
    "heldout_path",
    "search_period",
    "heldout_period",
    "dataset_version",
    "sample_rate",
    "expected_search_sha256",
    "expected_heldout_sha256",
)


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that treats duplicate mapping keys as invalid input."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ValueError(f"unhashable YAML mapping key: {key!r}") from exc
        if duplicate:
            raise ValueError(f"duplicate YAML key {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _canonical_value(value: Any) -> Any:
    if is_dataclass(value):
        return _canonical_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_canonical_value(item) for item in value), key=repr)
    if isinstance(value, Path):
        return value.name
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported fingerprint value: {type(value).__name__}")


def canonical_fingerprint(value: Any) -> str:
    """Hash semantic data using stable JSON ordering and no timestamps."""

    payload = json.dumps(
        _canonical_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_document(path: Path) -> Any:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"configuration file does not exist: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        if resolved.suffix.lower() == ".json":
            return json.load(handle)
        return yaml.load(handle, Loader=_UniqueKeyLoader)


def _slug(value: str) -> str:
    slug = _SAFE_COMPONENT.sub("_", value).strip("_").lower()
    if not slug:
        raise ValueError(f"key {value!r} has no filesystem-safe characters")
    return slug


def _reject_slug_collisions(keys: Iterable[str], *, kind: str) -> None:
    seen: dict[str, str] = {}
    for key in keys:
        slug = _slug(key)
        if slug in seen and seen[slug] != key:
            raise ValueError(
                f"{kind} keys {seen[slug]!r} and {key!r} collide after slugging"
            )
        seen[slug] = key


def _resolve_data_path(
    raw: object, *, base: Path, label: str, require_file: bool = True
) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{label} must be a non-empty path")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if require_file and not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    return path


def _optional_data_path(
    raw: object, *, base: Path, label: str, require_file: bool = True
) -> Path | None:
    if raw is None:
        return None
    return _resolve_data_path(
        raw, base=base, label=label, require_file=require_file
    )


def _optional_text(entry: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = entry.get(key)
        if value is not None:
            text = str(value).strip()
            return text or None
    return None


def load_dataset_registry(
    path: Path, *, require_files: bool = True
) -> dict[str, DatasetSpec]:
    """Load dataset entries and resolve their paths relative to the registry."""

    resolved = path.expanduser().resolve()
    document = _load_document(resolved)
    raw_entries = document.get("datasets") if isinstance(document, Mapping) else None
    if not isinstance(raw_entries, list):
        raise ValueError("dataset registry must contain a 'datasets' list")

    registry: dict[str, DatasetSpec] = {}
    for raw in raw_entries:
        if not isinstance(raw, Mapping):
            raise ValueError("each dataset entry must be a mapping")
        key = str(raw.get("name", raw.get("key", ""))).strip()
        if not key:
            raise ValueError("dataset entry is missing name")
        if key == "all":
            raise ValueError("dataset key 'all' is reserved for selectors")
        if key in registry:
            raise ValueError(f"duplicate dataset key: {key!r}")
        search_path = _resolve_data_path(
            raw.get("train_path", raw.get("search_path")),
            base=resolved.parent,
            label=f"dataset {key!r} search_path",
            require_file=require_files,
        )
        heldout_path = _optional_data_path(
            raw.get("test_path", raw.get("heldout_path")),
            base=resolved.parent,
            label=f"dataset {key!r} heldout_path",
            require_file=require_files,
        )
        sample_rate_raw = raw.get("sample_rate")
        sample_rate = None if sample_rate_raw is None else float(sample_rate_raw)
        if sample_rate is not None and not 0.0 < sample_rate <= 1.0:
            raise ValueError(f"dataset {key!r} sample_rate must be in (0, 1]")
        weight_raw = raw.get("aggregation_weight")
        aggregation_weight = None if weight_raw is None else float(weight_raw)
        if aggregation_weight is not None and aggregation_weight <= 0:
            raise ValueError(f"dataset {key!r} aggregation_weight must be positive")
        registry[key] = DatasetSpec(
            key=key,
            search_path=search_path,
            heldout_path=heldout_path,
            search_period=_optional_text(raw, "search_period", "train_period"),
            heldout_period=_optional_text(raw, "heldout_period", "test_period"),
            dataset_version=_optional_text(raw, "dataset_version", "version"),
            sample_rate=sample_rate,
            expected_search_sha256=_optional_text(
                raw, "expected_search_sha256", "search_sha256", "train_sha256"
            ),
            expected_heldout_sha256=_optional_text(
                raw, "expected_heldout_sha256", "heldout_sha256", "test_sha256"
            ),
            aggregation_weight=aggregation_weight,
        )
    _reject_slug_collisions(registry, kind="dataset")
    return registry


def _model_entries(document: Any) -> list[tuple[str, Mapping[str, Any]]]:
    raw_profiles = document.get("models") if isinstance(document, Mapping) else None
    if raw_profiles is None and isinstance(document, Mapping):
        raw_profiles = document.get("profiles")
    if isinstance(raw_profiles, Mapping):
        result: list[tuple[str, Mapping[str, Any]]] = []
        for key, value in raw_profiles.items():
            if not isinstance(value, Mapping):
                raise ValueError(f"model profile {key!r} must be a mapping")
            result.append((str(key), value))
        return result
    if isinstance(raw_profiles, list):
        result: list[tuple[str, Mapping[str, Any]]] = []
        for value in raw_profiles:
            if not isinstance(value, Mapping):
                raise ValueError("each model profile must be a mapping")
            key = str(value.get("key", value.get("name", "")))
            result.append((key, value))
        return result
    raise ValueError("model profile file must contain a 'models' mapping or list")


def load_model_profiles(path: Path) -> dict[str, ModelProfile]:
    """Load model profiles while preserving, never resolving, secret references."""

    registry: dict[str, ModelProfile] = {}
    for raw_key, raw in _model_entries(_load_document(path)):
        key = raw_key.strip()
        if not key:
            raise ValueError("model profile is missing key")
        if key == "all":
            raise ValueError("model profile key 'all' is reserved for selectors")
        if key in registry:
            raise ValueError(f"duplicate model profile key: {key!r}")
        forbidden = sorted({"temperature", "max_tokens"}.intersection(raw))
        if forbidden:
            raise ValueError(
                f"model profile {key!r} must not set {', '.join(forbidden)}"
            )
        provider = str(raw.get("provider", "")).strip()
        model = str(raw.get("model", raw.get("name", ""))).strip()
        if not provider or not model:
            raise ValueError(f"model profile {key!r} requires provider and model")
        reference = str(
            raw.get("api_key_reference", raw.get("api_key", ""))
        ).strip()
        if reference and not _ENV_REFERENCE.fullmatch(reference):
            raise ValueError(
                f"model profile {key!r} api_key must be an environment reference"
            )
        main_enabled = raw.get("main_enabled", False)
        if not isinstance(main_enabled, bool):
            raise ValueError(
                f"model profile {key!r} main_enabled must be a YAML boolean"
            )
        registry[key] = ModelProfile(
            key=key,
            provider=provider,
            model=model,
            api_base=_optional_text(raw, "api_base"),
            api_key_reference=reference,
            main_enabled=main_enabled,
        )
    _reject_slug_collisions(registry, kind="model")
    return registry


def _select(
    selectors: Sequence[str], available: Sequence[str], *, kind: str
) -> tuple[str, ...]:
    if not selectors:
        raise ValueError(f"{kind} selector must not be empty")
    if len(set(selectors)) != len(selectors):
        raise ValueError(f"duplicate {kind} selector")
    if "all" in selectors:
        if len(selectors) != 1:
            raise ValueError(f"{kind} selector 'all' cannot be combined with explicit keys")
        if not available:
            raise ValueError(f"{kind} selector 'all' resolved to an empty registry")
        return tuple(available)
    unknown = tuple(item for item in selectors if item not in available)
    if unknown:
        raise ValueError(f"unknown {kind}: {', '.join(unknown)}")
    return tuple(selectors)


def _dataset_semantics(
    dataset: DatasetSpec, *, split_manifest_sha256: str
) -> dict[str, Any]:
    return {
        "key": dataset.key,
        "split_manifest_sha256": split_manifest_sha256,
    }


def _validate_main_dataset(dataset: DatasetSpec) -> None:
    missing = [field for field in _MAIN_DATASET_FIELDS if getattr(dataset, field) is None]
    if missing:
        raise ValueError(
            f"dataset {dataset.key!r} is missing main-mode metadata: {', '.join(missing)}"
        )
    _require_sha256(
        dataset.expected_search_sha256,
        field=f"dataset {dataset.key!r} expected_search_sha256",
    )
    _require_sha256(
        dataset.expected_heldout_sha256,
        field=f"dataset {dataset.key!r} expected_heldout_sha256",
    )


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a resolved 64-character lowercase SHA-256")
    return value


def _validate_main_runtime(config: SuiteConfig) -> None:
    if config.evaluator_backend != "local":
        raise ValueError("main mode requires evaluator_backend='local'")
    if config.candidate_sandbox.backend != "local_process":
        raise ValueError("main mode requires candidate backend='local_process'")


def expand_run_matrix(
    config: SuiteConfig,
    *,
    dataset_registry: Mapping[str, DatasetSpec],
    model_profiles: Mapping[str, ModelProfile],
    available_methods: Sequence[str],
    split_manifest_fingerprints: Mapping[str, str],
    method_fingerprints: Mapping[str, str],
) -> MatrixPlan:
    """Expand cells from explicit fingerprints emitted by trusted preflight."""

    if config.mode not in ("smoke", "full", "main"):
        raise ValueError(f"unknown suite mode: {config.mode!r}")
    if config.engine_batch_mode != "per_batch":
        raise ValueError("all baseline suite modes require engine_batch_mode='per_batch'")
    _require_sha256(config.seed_candidate_sha256, field="seed_candidate_sha256")
    _require_sha256(config.evaluator_fingerprint, field="evaluator_fingerprint")
    _require_sha256(
        config.column_policy_fingerprint, field="column_policy_fingerprint"
    )
    _require_sha256(config.objective_fingerprint, field="objective_fingerprint")
    if not config.suite_config_hash:
        raise ValueError("suite_config_hash must not be empty")
    dataset_keys = _select(config.datasets, tuple(dataset_registry), kind="datasets")

    method_selectors = config.methods
    if "all" in method_selectors:
        method_keys = _select(method_selectors, tuple(available_methods), kind="methods")
        finalize_e0 = False
    else:
        finalize_e0 = "e0" in method_selectors
        search_selectors = tuple(item for item in method_selectors if item != "e0")
        if len(set(method_selectors)) != len(method_selectors):
            raise ValueError("duplicate methods selector")
        method_keys = (
            _select(search_selectors, tuple(available_methods), kind="methods")
            if search_selectors
            else ()
        )
    if not method_keys and not finalize_e0:
        raise ValueError("methods selector must choose e0 or at least one search method")
    selected_method_set = set(method_keys)
    supplied_method_set = set(method_fingerprints)
    if supplied_method_set != selected_method_set:
        missing = sorted(selected_method_set - supplied_method_set)
        extra = sorted(supplied_method_set - selected_method_set)
        raise ValueError(
            "method_fingerprints must exactly cover selected search methods; "
            f"missing={missing}, extra={extra}"
        )
    resolved_method_fingerprints = {
        method: _require_sha256(
            method_fingerprints[method], field=f"method_fingerprints[{method!r}]"
        )
        for method in method_keys
    }
    if method_keys:
        if not config.seeds or len(set(config.seeds)) != len(config.seeds):
            raise ValueError("search seeds must be non-empty and unique")
        model_keys = _select(config.models, tuple(model_profiles), kind="models")
    else:
        model_keys = ()

    selected_dataset_set = set(dataset_keys)
    supplied_dataset_set = set(split_manifest_fingerprints)
    if supplied_dataset_set != selected_dataset_set:
        missing = sorted(selected_dataset_set - supplied_dataset_set)
        extra = sorted(supplied_dataset_set - selected_dataset_set)
        raise ValueError(
            "split_manifest_fingerprints must exactly cover selected datasets; "
            f"missing={missing}, extra={extra}"
        )
    resolved_split_fingerprints: dict[str, str] = {}
    for dataset_key in dataset_keys:
        resolved_split_fingerprints[dataset_key] = _require_sha256(
            split_manifest_fingerprints[dataset_key],
            field=f"split_manifest_fingerprints[{dataset_key!r}]",
        )
        if config.mode == "main":
            _validate_main_dataset(dataset_registry[dataset_key])
    if config.mode == "main":
        _validate_main_runtime(config)
    for model_key in model_keys:
        if config.mode == "main" and not model_profiles[model_key].main_enabled:
            raise ValueError(f"model profile {model_key!r} has main_enabled=false")

    content_semantics = {
        "backend": config.evaluator_backend,
        "seed_candidate_sha256": config.seed_candidate_sha256,
        "engine_batch_mode": config.engine_batch_mode,
        "evaluator_fingerprint": config.evaluator_fingerprint,
        "column_policy_fingerprint": config.column_policy_fingerprint,
    }
    runtime_semantics = {
        **content_semantics,
        "objective_fingerprint": config.objective_fingerprint,
        "budget": config.budget_profile,
        "candidate_backend": config.candidate_sandbox.backend,
    }
    e0_fixtures: list[E0Key] = []
    search_cells: list[RunKey] = []
    selection_groups: list[SelectionKey] = []
    e0_finalizations: list[E0FinalizationKey] = []

    for dataset_key in dataset_keys:
        dataset_semantics = _dataset_semantics(
            dataset_registry[dataset_key],
            split_manifest_sha256=resolved_split_fingerprints[dataset_key],
        )
        e0_fixtures.append(
            E0Key(
                dataset=dataset_key,
                evaluator_backend=config.evaluator_backend,
                seed_candidate_sha256=config.seed_candidate_sha256,
                engine_batch_mode=config.engine_batch_mode,
                evaluator_fingerprint=config.evaluator_fingerprint,
                column_policy_fingerprint=config.column_policy_fingerprint,
                split_manifest_sha256=resolved_split_fingerprints[dataset_key],
            )
        )
        if finalize_e0:
            e0_finalizations.append(
                E0FinalizationKey(
                    dataset=dataset_key,
                    evaluator_backend=config.evaluator_backend,
                    finalization_fingerprint=canonical_fingerprint(
                        {
                            **content_semantics,
                            "dataset": dataset_semantics,
                            "kind": "e0_final",
                        }
                    ),
                )
            )
        for method in method_keys:
            method_semantics = {
                "method": method,
                "method_fingerprint": resolved_method_fingerprints[method],
            }
            for model_key in model_keys:
                profile = model_profiles[model_key]
                model_semantics = {
                    "key": profile.key,
                    "provider": profile.provider,
                    "model": profile.model,
                    "api_base": profile.api_base,
                }
                selection_groups.append(
                    SelectionKey(
                        dataset=dataset_key,
                        method=method,
                        model=model_key,
                        evaluator_backend=config.evaluator_backend,
                        selection_fingerprint=canonical_fingerprint(
                            {
                                **runtime_semantics,
                                "dataset": dataset_semantics,
                                **method_semantics,
                                "model": model_semantics,
                                "kind": "selection",
                            }
                        ),
                    )
                )
                for seed in config.seeds:
                    search_cells.append(
                        RunKey(
                            dataset=dataset_key,
                            method=method,
                            model=model_key,
                            seed=seed,
                            evaluator_backend=config.evaluator_backend,
                            budget_profile=config.budget_profile,
                            semantic_fingerprint=canonical_fingerprint(
                                {
                                    **runtime_semantics,
                                    "dataset": dataset_semantics,
                                    **method_semantics,
                                    "model": model_semantics,
                                    "seed": seed,
                                    "kind": "search",
                                }
                            ),
                        )
                    )
    return MatrixPlan(
        e0_fixtures=tuple(e0_fixtures),
        search_cells=tuple(search_cells),
        selection_groups=tuple(selection_groups),
        e0_finalizations=tuple(e0_finalizations),
    )


__all__ = [
    "canonical_fingerprint",
    "expand_run_matrix",
    "load_dataset_registry",
    "load_model_profiles",
]
