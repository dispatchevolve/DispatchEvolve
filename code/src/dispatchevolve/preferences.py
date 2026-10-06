"""Assemble DPO response pairs from explicitly labeled preferences.

Opportunity preferences come from observed local outcomes; online preferences
come from historical deployment decisions.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def assemble_pair(record: dict) -> dict:
    responses = record["responses"]
    preferred = record["preferred"]
    if len(responses) != 2 or preferred not in responses:
        raise ValueError("exactly two responses and an explicit preferred key are required")
    rejected = next(key for key in responses if key != preferred)
    values = [record["system"], record["user"], *responses.values()]
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("prompts and responses must be nonempty strings")
    if responses[preferred] == responses[rejected]:
        raise ValueError("chosen and rejected responses must differ")
    return {"system": record["system"],
            "conversations": [{"from": "human", "value": record["user"]}],
            "chosen": {"from": "gpt", "value": responses[preferred]},
            "rejected": {"from": "gpt", "value": responses[rejected]}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("workspaces/preferences"))
    args = parser.parse_args()
    records = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError("input must contain at least one preference record")
    rows = [assemble_pair(record) for record in records]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "preferences.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows))
    info = {"critic_preferences": {"file_name": "preferences.jsonl", "ranking": True,
        "formatting": "sharegpt", "columns": {"messages": "conversations",
        "system": "system", "chosen": "chosen", "rejected": "rejected"}}}
    (args.output_dir / "dataset_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"Wrote {len(rows)} preference pairs to {args.output_dir}")


if __name__ == "__main__":
    main()
