# Adapted for DispatchEvolve; see THIRD_PARTY_NOTICES.md for origin and changes.
"""
Utilities module initialization
"""

from dispatchevolve.optimizer.genetic.utils.async_utils import (
    TaskPool,
    gather_with_concurrency,
    retry_async,
    run_in_executor,
)
from dispatchevolve.optimizer.genetic.utils.code_utils import (
    apply_diff,
    calculate_edit_distance,
    extract_code_language,
    extract_diffs,
    format_diff_summary,
    parse_evolve_blocks,
    parse_full_rewrite,
)
from dispatchevolve.optimizer.genetic.utils.format_utils import (
    format_metrics_safe,
    format_improvement_safe,
)
from dispatchevolve.optimizer.genetic.utils.metrics_utils import (
    safe_numeric_average,
    safe_numeric_sum,
)

__all__ = [
    "TaskPool",
    "gather_with_concurrency",
    "retry_async",
    "run_in_executor",
    "apply_diff",
    "calculate_edit_distance",
    "extract_code_language",
    "extract_diffs",
    "format_diff_summary",
    "parse_evolve_blocks",
    "parse_full_rewrite",
    "format_metrics_safe",
    "format_improvement_safe",
    "safe_numeric_average",
    "safe_numeric_sum",
]
