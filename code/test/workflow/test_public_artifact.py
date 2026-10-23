"""Long-term review tests: real local matching, public transport, preferences."""
import asyncio
from types import SimpleNamespace
import pandas as pd
import pytest
from dispatchevolve.tasks.full_dispatch.utils import local_stage_major_match
from dispatchevolve.preferences import assemble_pair
from dispatchevolve.config import LLMProvider, ModelSpec
from dispatchevolve.provider import LiteLLMWrapper


def test_real_local_matching_maximizes_weight():
    frame = pd.DataFrame({"batch_id": [0]*4, "order_id": [0,0,1,1],
        "driver_id": [0,1,0,1], "stage": [1]*4, "weight": [9.,1.,2.,8.]})
    result = local_stage_major_match(frame)
    assert set(zip(result.order_id, result.driver_id)) == {(0,0),(1,1)}
    assert result.weight.sum() == 17


def test_public_provider_preserves_full_input_and_default_controls(monkeypatch):
    import litellm
    import dispatchevolve.call_logger as logger
    observed = {}
    def completion(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="OK"))])
    monkeypatch.setattr(litellm, "completion", completion)
    monkeypatch.setattr(logger, "track_success", lambda *a, **kw: None)
    client = LiteLLMWrapper(ModelSpec(name="review-model"),
        provider=LLMProvider.OPENAI_COMPATIBLE, api_base="https://example.org/v1",
        api_key="test-placeholder", transport_max_retries=0)
    message = "input " * 10000
    assert asyncio.run(client.generate(message)) == "OK"
    assert observed["messages"][0]["content"] == message
    assert "max_tokens" not in observed and "temperature" not in observed
    assert observed["api_base"] == "https://example.org/v1"


def test_preferences_require_explicit_observed_label():
    record = {"system": "Judge", "user": "Evidence", "preferred": "ACCEPT",
              "responses": {"ACCEPT": "accept response", "REJECT": "reject response"}}
    assert assemble_pair(record)["chosen"]["value"] == "accept response"
    with pytest.raises(ValueError):
        assemble_pair({**record, "preferred": "unknown"})


def test_public_schema_hides_observed_labels_and_unknown_fields():
    from dispatchevolve.tasks.full_dispatch.column_policy import ColumnPolicy
    policy=ColumnPolicy()
    assert len(policy.feature_info)==22 and len(policy.label_info)==3
    frame=pd.DataFrame({'order_id':[1], 'driver_id':[2], 'eta':[3.],
                        'observed_completion':[True], 'custom_private_field':[7]})
    visible=policy.prepare(frame).visible
    assert {'order_id','driver_id','eta'} <= set(visible)
    assert 'observed_completion' not in visible and 'custom_private_field' not in visible


def test_online_inference_has_no_history_prompt():
    from dispatchevolve.workflows.dispatchevolve_v2.config import DEFAULT_PROMPT_JSON
    from dispatchevolve.workflows.dispatchevolve_v2.prompt_store import PromptStore
    assert 'online_uplift' not in PromptStore(DEFAULT_PROMPT_JSON).roles


def test_budget_path_reaches_genetic_worker():
    from dispatchevolve.optimizer.genetic.config import Config
    from dispatchevolve.optimizer.genetic.process_parallel import ProcessParallelController
    config=Config()
    config.replay_budget_path='budget.sqlite'
    controller=ProcessParallelController.__new__(ProcessParallelController)
    controller.file_suffix='.py'
    assert controller._serialize_config(config)['replay_budget_path']=='budget.sqlite'
