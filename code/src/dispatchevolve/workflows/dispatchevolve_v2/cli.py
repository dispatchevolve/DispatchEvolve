"""CLI for the independent DispatchEvolve V2 workflow."""

from __future__ import annotations

import argparse
import json

from .config import load_config
from .orchestrator import DispatchEvolveV2Orchestrator


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-protocol-migration", action="store_true")
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--stop-after-candidate-admission-round", type=int)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.validate_only:
        print(json.dumps({"valid": True, "run_id": config.run_id, "mode": config.mode}))
        return 0
    orchestrator = DispatchEvolveV2Orchestrator(config)
    state = orchestrator.run(
        resume=args.resume, rounds=args.rounds,
        allow_protocol_migration=args.allow_protocol_migration,
        stop_after_candidate_admission_round=args.stop_after_candidate_admission_round,
    )
    print(json.dumps({"run_id": state.run_id, "stage": state.stage,
                      "completed_rounds": state.completed_rounds,
                      "incumbent_engine_id": state.incumbent_engine_id}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
