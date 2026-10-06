"""Persistent active-reward archive and exact reference selection."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Sequence

from dispatchevolve.hypervolume import exact_hypervolume

from .contracts import ArchiveEntry, GUARDRAIL_KEYS, OBJECTIVE_KEYS


def dominates(
    left: ArchiveEntry,
    right: ArchiveEntry,
    tolerance: float,
    objective_keys: tuple[str, ...] = OBJECTIVE_KEYS,
) -> bool:
    return all(
        left.oriented_delta[key] >= right.oriented_delta[key] - tolerance
        for key in objective_keys
    ) and any(
        left.oriented_delta[key] > right.oriented_delta[key] + tolerance
        for key in objective_keys
    )


def equivalent(
    left: ArchiveEntry,
    right: ArchiveEntry,
    tolerance: float,
    objective_keys: tuple[str, ...] = OBJECTIVE_KEYS,
) -> bool:
    return all(
        abs(left.oriented_delta[key] - right.oriented_delta[key]) <= tolerance
        for key in objective_keys
    )


def update_archive(
    entries: list[ArchiveEntry],
    candidate: ArchiveEntry,
    *,
    tolerance: float,
    objective_keys: tuple[str, ...] = OBJECTIVE_KEYS,
    guardrail_keys: tuple[str, ...] = GUARDRAIL_KEYS,
    rho: float = 0.005,
) -> tuple[list[ArchiveEntry], dict]:
    # Feasibility precedes Pareto comparison: rejected engines cannot become
    # references or evict a feasible archive member. E0 is the initial anchor.
    from .local_evaluator import passes_global_acceptance
    is_anchor = (candidate.round_index == 0 and not candidate.candidate_ids
                 and all(abs(value) <= tolerance for value in candidate.oriented_delta.values()))
    if not is_anchor and not passes_global_acceptance(
        candidate.oriented_delta, rho=rho, tolerance=tolerance,
        objective_keys=objective_keys, guardrail_keys=guardrail_keys,
    ):
        return entries, {"candidate_id": candidate.engine_id, "status": "infeasible"}

    if any(item.engine_id == candidate.engine_id for item in entries):
        return entries, {"candidate_id": candidate.engine_id, "status": "duplicate_engine"}
    equal = next((
        item for item in entries
        if equivalent(item, candidate, tolerance, objective_keys)
    ), None)
    if equal is not None:
        preferred = min((equal, candidate), key=lambda item: (item.change_size, len(item.candidate_ids), item.engine_id))
        if preferred.engine_id == equal.engine_id:
            return entries, {"candidate_id": candidate.engine_id, "status": "equal_vector_not_retained", "representative": equal.engine_id}
        entries = [item for item in entries if item.engine_id != equal.engine_id]
    dominator = next((
        item for item in entries
        if dominates(item, candidate, tolerance, objective_keys)
    ), None)
    if dominator is not None:
        return entries, {"candidate_id": candidate.engine_id, "status": "dominated", "by": dominator.engine_id}
    removed = [
        item.engine_id for item in entries
        if dominates(candidate, item, tolerance, objective_keys)
    ]
    retained = sorted([item for item in entries if item.engine_id not in removed] + [candidate], key=lambda item: item.engine_id)
    return retained, {"candidate_id": candidate.engine_id, "status": "retained", "removed": removed}


def _point(
    entry: ArchiveEntry,
    objective_keys: tuple[str, ...] = OBJECTIVE_KEYS,
) -> tuple[float, ...]:
    return tuple(float(entry.oriented_delta[key]) for key in objective_keys)


@lru_cache(maxsize=256)
def _cached_hypervolume(points: tuple[tuple[float, ...], ...], reference: tuple[float, ...]) -> float:
    return exact_hypervolume(points, reference)


def _reference_vector(
    points: tuple[tuple[float, ...], ...], reference: float | Sequence[float],
    dimensions: int,
) -> tuple[float, ...]:
    if not isinstance(reference, (int, float)):
        vector = tuple(float(value) for value in reference)
        if points and len(vector) != len(points[0]):
            raise ValueError("point and reference dimensions differ")
        return vector
    preferred = float(reference)
    if not points:
        return (preferred,) * dimensions
    # Threshold-free Pareto maintenance can retain trade-offs below rho.  HV
    # therefore uses the preferred reference where valid and extends each
    # coordinate just below the current frontier nadir where necessary.
    return tuple(
        min(preferred, minimum - max(1e-12, abs(minimum) * 1e-9))
        for minimum in (
            min(point[index] for point in points) for index in range(dimensions)
        )
    )


def hypervolume(
    entries: Iterable[ArchiveEntry],
    reference: float | Sequence[float] = -0.005,
    objective_keys: tuple[str, ...] = OBJECTIVE_KEYS,
) -> float:
    selected = tuple(entries)
    points = tuple(sorted({_point(item, objective_keys) for item in selected}))
    return (
        _cached_hypervolume(
            points, _reference_vector(points, reference, len(objective_keys))
        )
        if selected else 0.0
    )


def hypervolume_contributions(
    entries: list[ArchiveEntry],
    reference: float = -0.005,
    objective_keys: tuple[str, ...] = OBJECTIVE_KEYS,
) -> dict[str, float]:
    points = tuple(sorted({_point(item, objective_keys) for item in entries}))
    shared_reference = _reference_vector(
        points, reference, len(objective_keys)
    )
    total = hypervolume(entries, shared_reference, objective_keys)
    return {item.engine_id: total - hypervolume(
        (other for other in entries if other.engine_id != item.engine_id),
        shared_reference,
        objective_keys,
    )
            for item in entries}


def select_incumbent(
    entries: list[ArchiveEntry], *, rho: float, tolerance: float, current_engine_id: str,
    objective_keys: tuple[str, ...] = OBJECTIVE_KEYS,
    guardrail_keys: tuple[str, ...] = GUARDRAIL_KEYS,
) -> tuple[ArchiveEntry | None, dict[str, float]]:
    """Select the largest-HV engine only among globally non-degrading entries.

    The archive already enforces feasibility. Offline fallback promotion
    rechecks eligibility: every reward objective must be nonnegative relative to
    E0, at least one must improve, and every guardrail must stay within rho.
    Each eligible engine is scored by its own active-reward dominated
    hypervolume from the E0 reference, rather than by its contribution to a
    mixed frontier.
    """
    eligible = [
        item for item in entries
        if item.online_eligibility != "NO"
        and all(float(item.oriented_delta[key]) >= -tolerance for key in objective_keys)
        and any(float(item.oriented_delta[key]) > tolerance for key in objective_keys)
        and all(float(item.oriented_delta[key]) >= -rho - tolerance for key in guardrail_keys)
    ]
    scores = {
        item.engine_id: hypervolume((item,), 0.0, objective_keys)
        for item in eligible
    }
    if not eligible:
        return next((item for item in entries if item.engine_id == current_engine_id), None), scores
    selected = min(
        eligible,
        key=lambda item: (-scores[item.engine_id], item.change_size, item.engine_id),
    )
    return selected, scores


def lineage_distance(entry: ArchiveEntry, selected: list[ArchiveEntry]) -> float:
    if not selected:
        return 1.0
    left = set(entry.candidate_ids)
    distances = []
    for other in selected:
        right = set(other.candidate_ids); union = left | right
        distances.append(1.0 - len(left & right) / len(union) if union else 0.0)
    return min(distances)


def select_references(
    entries: list[ArchiveEntry],
    limit: int,
    reference: float = -0.005,
    objective_keys: tuple[str, ...] = OBJECTIVE_KEYS,
) -> list[ArchiveEntry]:
    if len(entries) <= limit:
        return sorted(entries, key=lambda item: item.engine_id)
    selected: list[ArchiveEntry] = []
    for key in objective_keys:
        extreme = min(entries, key=lambda item: (-item.oriented_delta[key], item.change_size, item.engine_id))
        if extreme not in selected:
            selected.append(extreme)
    contributions = hypervolume_contributions(entries, reference, objective_keys)
    while len(selected) < limit:
        remaining = [item for item in entries if item not in selected]
        if not remaining:
            break
        remaining.sort(key=lambda item: (-contributions[item.engine_id], -lineage_distance(item, selected), item.change_size, item.engine_id))
        selected.append(remaining[0])
    return selected[:limit]


def reference_reasons(
    entries: list[ArchiveEntry],
    selected: list[ArchiveEntry],
    *,
    reference: float = -0.005,
    objective_keys: tuple[str, ...] = OBJECTIVE_KEYS,
) -> dict[str, str]:
    """Explain the deterministic evidence that caused each reference selection."""
    if len(entries) <= len(selected):
        return {
            item.engine_id: "complete current Pareto frontier; no reference truncation"
            for item in selected
        }
    extremes: dict[str, list[str]] = {}
    for key in objective_keys:
        item = min(
            entries,
            key=lambda value: (
                -value.oriented_delta[key],
                value.change_size,
                value.engine_id,
            ),
        )
        extremes.setdefault(item.engine_id, []).append(key)
    contributions = hypervolume_contributions(
        entries, reference, objective_keys
    )
    reasons: dict[str, str] = {}
    prefix: list[ArchiveEntry] = []
    for item in selected:
        keys = extremes.get(item.engine_id)
        if keys:
            reasons[item.engine_id] = (
                "active-reward frontier extreme for " + ",".join(keys)
            )
        else:
            distance = lineage_distance(item, prefix)
            reasons[item.engine_id] = (
                "post-extreme deterministic selection; "
                f"hypervolume_contribution={contributions[item.engine_id]:.12g}; "
                f"candidate_set_lineage_distance={distance:.12g}; "
                f"change_size={item.change_size}"
            )
        prefix.append(item)
    return reasons
