"""Frozen-model pairwise ranking of the feasible archive, without A/B outcomes."""
from __future__ import annotations
from itertools import combinations
import hashlib
import json
from dispatchevolve.baselines.candidates import RepositoryGenomeCodec
from .contracts import OBJECTIVE_KEYS
from .pareto_archive import hypervolume_contributions
from .prompt_store import RenderedPrompt

SYSTEM = ('You are the pairwise online-uplift assessor. Compare two complete engines '
          'under the same pre-deployment context. Use their policy design and offline '
          'evidence. Choose the engine with better expected online performance. '
          'Return exactly A or B. No online outcomes are supplied.')


def rank_engines(entries, *, llm, context, objective_keys=OBJECTIVE_KEYS, reference=-0.005):
    ordered = sorted(entries, key=lambda item: item.engine_id)
    wins = {item.engine_id: 0 for item in ordered}
    contributions = hypervolume_contributions(ordered, reference, objective_keys)
    codec = RepositoryGenomeCodec()
    pairs = []
    for left, right in combinations(ordered, 2):
        document = {"pre_deployment_context": context,
                    "A": {"source": codec.encode(left.engine_dir), "offline_delta": left.oriented_delta},
                    "B": {"source": codec.encode(right.engine_dir), "offline_delta": right.oriented_delta}}
        user = json.dumps(document, sort_keys=True)
        digest = hashlib.sha256((SYSTEM + user).encode()).hexdigest()
        reply = llm.complete(RenderedPrompt('online_pairwise', SYSTEM, user, digest)).strip()
        if reply not in {'A', 'B'}:
            raise ValueError('pairwise assessor must return exactly A or B')
        winner = left if reply == 'A' else right
        wins[winner.engine_id] += 1
        pairs.append({"left": left.engine_id, "right": right.engine_id,
                      "winner": winner.engine_id, "prompt_hash": digest})
    ranked = sorted(ordered, key=lambda item: (-wins[item.engine_id],
                    -contributions[item.engine_id], item.engine_id))
    return ranked, {"rule": "pairwise_wins_then_hypervolume_contribution_then_engine_id",
                    "wins": wins, "hypervolume_contributions": contributions,
                    "pairs": pairs, "ranking": [item.engine_id for item in ranked]}
