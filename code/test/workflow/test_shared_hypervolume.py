"""Long-term tests for the workflow-independent exact hypervolume utility.

Related files: dispatchevolve.hypervolume and the V2 Pareto archive.
Covered behavior: frozen exact values, duplicate handling, and invalid inputs.
"""

from __future__ import annotations

import math

import pytest

from dispatchevolve.hypervolume import exact_hypervolume


@pytest.mark.parametrize(
    ("points", "reference", "expected"),
    [
        ([], (0.0, 0.0), 0.0),
        ([(2.0, 1.0)], (0.0, 0.0), 2.0),
        ([(2.0, 1.0), (1.0, 2.0)], (0.0, 0.0), 3.0),
        ([(2.0, 1.0), (2.0, 1.0)], (0.0, 0.0), 2.0),
        ([(2.0, 2.0, 2.0)], (1.0, 1.0, 1.0), 1.0),
    ],
)
def test_exact_hypervolume_preserves_frozen_results(
    points: list[tuple[float, ...]],
    reference: tuple[float, ...],
    expected: float,
) -> None:
    assert exact_hypervolume(points, reference) == expected


@pytest.mark.parametrize(
    ("points", "reference", "message"),
    [
        ([(1.0,)], (0.0, 0.0), "dimensions differ"),
        ([(0.0, -1.0)], (0.0, 0.0), "must be dominated"),
        ([(math.inf, 1.0)], (0.0, 0.0), "must be finite"),
    ],
)
def test_exact_hypervolume_rejects_invalid_geometry(
    points: list[tuple[float, ...]],
    reference: tuple[float, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        exact_hypervolume(points, reference)
