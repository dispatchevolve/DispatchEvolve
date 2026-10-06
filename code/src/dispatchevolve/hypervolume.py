"""Exact deterministic hypervolume utilities shared by workflow implementations."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import math


def exact_hypervolume(
    points: Iterable[Sequence[float]],
    reference: Sequence[float],
) -> float:
    """Return the exact maximize-space volume dominated above ``reference``."""

    ref = tuple(float(value) for value in reference)
    clean = sorted({tuple(float(value) for value in point) for point in points})
    if any(len(point) != len(ref) for point in clean):
        raise ValueError("point and reference dimensions differ")
    if any(
        any(not math.isfinite(value) for value in point) for point in clean
    ) or any(not math.isfinite(value) for value in ref):
        raise ValueError("hypervolume coordinates must be finite")
    if any(
        any(value < lower for value, lower in zip(point, ref))
        for point in clean
    ):
        raise ValueError("the frozen reference must be dominated by every point")
    if not clean:
        return 0.0

    def sliced(
        current: list[tuple[float, ...]],
        lower: tuple[float, ...],
    ) -> float:
        if len(lower) == 1:
            return max(point[0] for point in current) - lower[0]
        cuts = sorted({lower[0], *(point[0] for point in current)})
        total = 0.0
        for start, end in zip(cuts, cuts[1:]):
            active = [point[1:] for point in current if point[0] >= end]
            if active:
                total += (end - start) * sliced(active, lower[1:])
        return total

    return sliced(clean, ref)


__all__ = ["exact_hypervolume"]
