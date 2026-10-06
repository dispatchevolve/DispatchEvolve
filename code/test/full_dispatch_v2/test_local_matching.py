"""Long-term offline matching tests for full_dispatch.utils.

Check optimality, sparse inputs, unique assignments, priority and invalid IDs.
"""
import random

import pandas as pd
import pytest

from dispatchevolve.tasks.full_dispatch.utils import local_stage_major_match


@pytest.mark.parametrize("seed", range(12))
def test_local_matching_agrees_with_exhaustive_search(seed):
    rng = random.Random(seed)
    rows = []
    for batch in range(2):
        for order in range(3):
            for driver in range(4):
                if rng.random() < 0.65:
                    rows.append(dict(batch_id=batch, order_id=order, driver_id=driver,
                                     stage=1, weight=rng.randrange(-2, 10)))
    frame = pd.DataFrame(rows)
    matched = local_stage_major_match(frame)
    assert not matched.duplicated(["batch_id", "order_id"]).any()
    assert not matched.duplicated(["batch_id", "driver_id"]).any()
    assert (matched.weight > 0).all()
    for batch, data in frame.groupby("batch_id"):
        edges = {(r.order_id, r.driver_id): r.weight for r in data.itertuples()
                 if r.weight > 0}
        def optimum(order, used):
            if order == 3:
                return 0
            choices = [optimum(order + 1, used)]
            for driver in range(4):
                if driver not in used and (order, driver) in edges:
                    choices.append(edges[order, driver] + optimum(order + 1, used | {driver}))
            return max(choices)
        assert matched.loc[matched.batch_id == batch, "weight"].sum() == optimum(0, set())


def test_duplicates_and_stage_priority():
    frame = pd.DataFrame(dict(batch_id=[0, 0, 0], order_id=[1, 1, 1],
                              driver_id=[1, 1, 2], stage=[1, 1, 2], weight=[2., 4., 1.]))
    assert local_stage_major_match(frame).driver_id.tolist() == [2]
    frame["stage"] = 1
    assert local_stage_major_match(frame).weight.tolist() == [4.]


def test_empty_and_nonpositive_edges():
    frame = pd.DataFrame(dict(batch_id=[0, 0], order_id=[1, 2], driver_id=[1, 2],
                              stage=[1, 1], weight=[0., -1.]))
    assert local_stage_major_match(frame).empty
    assert local_stage_major_match(frame.iloc[:0]).empty


def test_missing_identifiers_are_rejected():
    frame = pd.DataFrame(dict(batch_id=[0], order_id=[None], driver_id=[1],
                              stage=[1], weight=[2.]))
    with pytest.raises(ValueError, match="non-missing"):
        local_stage_major_match(frame)
