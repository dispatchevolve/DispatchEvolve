"""Long-term tests of paper_metrics: CR mapping, FAI and strict BR boundaries."""
import pytest
from dispatchevolve.workflows.dispatchevolve_v2.paper_metrics import paper_report
from dispatchevolve.workflows.dispatchevolve_v2.contracts import METRIC_KEYS, METRIC_DIRECTIONS


def values():
    baseline = dict.fromkeys(METRIC_KEYS, 100.)
    current = {key: 100. + METRIC_DIRECTIONS[key] for key in METRIC_KEYS}
    current['mean_cr'] = 0.  # This diagnostic input field is not the paper's CR.
    return baseline, current


def test_all_six_improvements_and_cr_mapping():
    baseline, current = values()
    current['order_br'] = 110.
    result = paper_report(current, baseline)
    assert result['fai_percent'] == pytest.approx(1.)
    assert result['relative_improvement_percent']['CR'] == pytest.approx(1.)
    assert result['metric_fields']['CR'] == 'mean_fqs'
    assert result['fai_feasible']


@pytest.mark.parametrize('br,passed', [(99.6, True), (99.5, False), (99.4, False)])
def test_br_reporting_threshold(br, passed):
    baseline, current = values()
    current['order_br'] = br
    result = paper_report(current, baseline)
    assert result['br_passed'] is passed
    assert result['fai_percent'] == pytest.approx(1. if passed else 0.)


@pytest.mark.parametrize('gain', [0., -.01])
def test_zero_or_negative_objective_sets_fai_zero(gain):
    baseline, current = values()
    current['mean_fqs'] = 100. * (1 + gain)
    assert paper_report(current, baseline)['fai_percent'] == 0.


def test_nonfinite_metric_rejected():
    baseline, current = values()
    current['mean_fqs'] = float('nan')
    with pytest.raises(ValueError, match='non-finite'):
        paper_report(current, baseline)


def test_main_config_requires_separate_test_and_baseline():
    from dataclasses import replace
    from pathlib import Path
    from dispatchevolve.workflows.dispatchevolve_v2.config import load_config
    config = load_config(Path(__file__).resolve().parents[2] / 'configs/paper.yaml')
    with pytest.raises(ValueError, match='test_path'):
        replace(config, test_path=None)
    with pytest.raises(ValueError, match='paths must differ'):
        replace(config, test_path=config.data_path)
    with pytest.raises(ValueError, match='evaluate_test_baseline'):
        replace(config, evaluate_test_baseline=False)
