"""Paper display names and held-out feasible average improvement (FAI)."""
from __future__ import annotations

import math
from .contracts import METRIC_DIRECTIONS

PAPER_METRICS = {
    'CR': 'mean_fqs', 'AR': 'order_ar', 'GMV': 'mean_gmv',
    'ETA': 'mean_eta', 'PCAA': 'mean_pcaa', 'DCAA': 'mean_dcaa',
}


def paper_report(metrics, baseline, *, rho=0.005, epsilon=1e-12, tolerance=1e-12):
    """Report signed percentage gains on the same split; BR is excluded from FAI."""
    if not math.isfinite(rho) or not 0 <= rho < 1:
        raise ValueError('rho must be finite and in [0, 1)')
    if not math.isfinite(epsilon) or epsilon <= 0 or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('epsilon must be positive and tolerance nonnegative')
    mapping = {**PAPER_METRICS, 'BR': 'order_br'}
    gains = {}
    for name, key in mapping.items():
        value, reference = float(metrics[key]), float(baseline[key])
        if not math.isfinite(value) or not math.isfinite(reference):
            raise ValueError(f'non-finite metric: {key}')
        gains[name] = METRIC_DIRECTIONS[key] * (value - reference) / max(abs(reference), epsilon)
    objectives_improved = all(gains[name] > tolerance for name in PAPER_METRICS)
    # The paper's reporting criterion uses BR improvement strictly above -0.5%.
    br_passed = gains['BR'] > -rho + tolerance
    feasible = objectives_improved and br_passed
    return {
        'metric_fields': mapping,
        'relative_improvement_percent': {name: 100 * gain for name, gain in gains.items()},
        'all_six_improved': objectives_improved,
        'br_passed': br_passed,
        'br_threshold_percent': -100 * rho,
        'br_comparison': 'strictly_greater',
        'fai_feasible': feasible,
        'fai_percent': 100 * sum(gains[name] for name in PAPER_METRICS) / 6 if feasible else 0.0,
    }
