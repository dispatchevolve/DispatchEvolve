"""Example matching engine for generated candidate pairs."""
from collections import Counter
import pandas as pd
from policies import ranking

RESULT_FIELDS = ('is_filtered', 'filter_rule', 'filter_policy', 'weight', 'stage')


def _run_target_policies(target):
    ranking.apply(target)


def run_batch(batch_data, *, trace=True):
    result = batch_data.copy()
    result['__engine_row_order'] = range(len(result))
    result['is_filtered'] = False
    result['filter_rule'] = ''
    result['filter_policy'] = ''
    target = result.loc[result['product_id'].eq(1)].copy()
    if len(target):
        _run_target_policies(target)
    eligible_target = target.loc[~target['is_filtered']].copy()
    output = pd.concat([eligible_target, result.loc[~result['product_id'].eq(1)]])
    output = output.sort_values('__engine_row_order')
    trace_payload = None
    if trace:
        trace_payload = {'input_rows': len(result), 'output_rows': len(output),
                         'target_row_count': len(target), 'target_output_rows': len(eligible_target),
                         'passthrough_row_count': len(result) - len(target), 'filtered_count': 0,
                         'filter_rule_counts': {}, 'filter_policy_counts': {}}
    return output.drop(columns=['__engine_row_order', 'is_filtered', 'filter_rule', 'filter_policy']), trace_payload
