"""Stable, experiment-local Prompt ID contracts."""

from __future__ import annotations

import json

import pytest

from dispatchevolve.workflows.dispatchevolve_v2.prompt_ids import PromptIdRegistry


def test_prompt_ids_start_at_one_resume_stably_and_reset_per_experiment(
    tmp_path,
) -> None:
    first_path = tmp_path / "experiment-a" / "prompt_ids.json"
    first = PromptIdRegistry(first_path)
    assert first.get("candidate", "internal-candidate-a") == "00001"
    assert first.get("candidate", "internal-candidate-b") == "00002"
    assert first.get("reference", "internal-engine-a") == "R00001"
    assert first.get("attempt", "internal-composition-a") == "A00001"
    assert first.resolve("candidate", "00001") == "internal-candidate-a"

    resumed = PromptIdRegistry(first_path)
    assert resumed.get("candidate", "internal-candidate-a") == "00001"
    assert resumed.get("candidate", "internal-candidate-c") == "00003"
    assert resumed.get("reference", "internal-engine-a") == "R00001"
    assert resumed.get("attempt", "internal-composition-a") == "A00001"

    second = PromptIdRegistry(tmp_path / "experiment-b" / "prompt_ids.json")
    assert second.get("candidate", "internal-candidate-c") == "00001"
    assert second.get("reference", "internal-engine-a") == "R00001"
    assert second.get("attempt", "internal-composition-a") == "A00001"


def test_prompt_id_validation_and_exhaustion_are_explicit(tmp_path) -> None:
    path = tmp_path / "prompt_ids.json"
    registry = PromptIdRegistry(path)
    registry.validate("candidate", "00001")
    registry.validate("reference", "R00001")
    registry.validate("attempt", "A00001")
    with pytest.raises(ValueError, match="invalid candidate"):
        registry.validate("candidate", "1")
    with pytest.raises(ValueError, match="unknown or ambiguous"):
        registry.resolve("candidate", "00001")

    document = json.loads(path.read_text(encoding="utf-8"))
    document["namespaces"]["candidate"] = {
        "next_number": 100_000,
        "internal_to_prompt": {},
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="exhausted"):
        registry.get("candidate", "cannot-fit")
