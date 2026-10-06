"""Initial travel-cost ranking rule for the evolution example."""
import pandas as pd


def trace_applicable_mask(frame, policy_symbol):
    return pd.Series(True, index=frame.index, dtype=bool)


def apply(frame):
    frame['weight'] = frame['eta']
    return frame
