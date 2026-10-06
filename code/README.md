# DispatchEvolve

Core implementation and a runnable synthetic example of trace-grounded local
policy evolution and Pareto-guided engine integration. The example generates
its inputs from arithmetic rules. Experiments on other datasets require an
engine, replay inputs and model endpoints supplied by the user.

## Run the complete example

Use Python 3.12 and uv, from the repository root:

```bash
uv sync --locked --extra dev
uv run python -m dispatchevolve.synthetic_demo
```

The command generates separate evolution and test inputs with 20 batches and
4 candidate pairs per batch in each split, instruments a toy
engine, discovers a scenario, reviews an opportunity, runs the shared genetic
optimizer, evaluates modified policies using real local matching, composes a
complete engine, checks global feasibility and selects from the Pareto archive.
The initial example policy ranks pairs by travel cost; the scripted
mutation replaces it with an inverse-cost rule.

The offline example supplies scripted LLM responses through a loopback HTTP
endpoint and executes the optimizer, source patches, isolated candidates,
matching, metrics and integration. It runs without an external account or API
key. Use the live mode below to evaluate model-generated decisions.

Inspect `workspaces/synthetic_demo/summary.json` for test-set baseline and selected
metrics, direction-aligned improvements and FAI. The selected engine and its
lineage are under `workspaces/synthetic_demo/runs/synthetic/final/`. The command fails if the scripted example produces no
measured feasible combination. With one non-dominated engine the online ranking
has no pair to compare; the tests exercise multi-engine pairwise ranking.

Outputs are ignored by Git. Existing output directories are never overwritten;
use `--output-dir workspaces/another_demo` for another run. The example uses 80
synthetic rows per split: 160 rows in total across 40 batches. The splits use
disjoint batch, order and driver identifiers. Search uses `synthetic.csv`; final
evaluation uses `test.csv`. Evolution metrics are recorded separately under
`evolution_metrics` in the summary.

## Use a real model

The same generated inputs and core workflow can use an external model:

```bash
export OPENAI_API_KEY=your_key
uv run python -m dispatchevolve.synthetic_demo --live \
  --model your-model --api-base https://api.openai.com/v1 \
  --output-dir workspaces/live_demo
```

Live mode uses your configured API account. To configure models separately,
first generate inputs without running:

```bash
uv run python -m dispatchevolve.synthetic_demo --prepare-only \
  --output-dir workspaces/synthetic_inputs
uv run dispatchevolve --config configs/example.yaml --validate-only
```

Edit `configs/example.yaml`, then run without `--validate-only`. `model` handles
proposal and evolution; optional `critic_model` handles the opportunity gate;
optional `online_model` handles pairwise online-uplift assessment. Each accepts
`provider`, `model`, `api_base` and `api_key_env`. Without separate settings the
base model supplies these roles. Configure the trained critic endpoints to use
DPO adapters. Supported transports include OpenAI-compatible, Gemini and Vertex
routes. LLM temperature and output-token limits remain unset by default.

The `--validate-only` option checks configuration structure. Use `backend: local`
for offline KM matching. Each batch is solved as a maximum-weight bipartite
assignment using SciPy, with at most one resource per request and one request
per resource. Matching requires no remote service.

## Relationship to the paper

The main implementation is under `src/dispatchevolve/workflows/dispatchevolve_v2/`:

1. `batch_query.py`, `trace_analysis.py`, `policy_trace.py` and `prompt_store.py`
   discover complete-batch scenarios and construct trace-grounded opportunities.
2. `shared_genetic_adapter.py` runs policy-scoped local evolution through the
   optimizer in `optimizer/genetic/`; `local_evaluator.py` checks measured local
   gains and protected metrics.
3. `relation_graph.py`, `combination_search.py` and `integration.py` construct
   and evaluate complete engines without summing local improvements.
4. `pareto_archive.py` admits globally feasible improvements before applying
   Pareto dominance. Infeasible candidates cannot become references or evict
   feasible engines. The main workflow stops after a round without progress.
5. `pairwise_ranking.py` compares archive engines under common pre-deployment
   context, ranks by wins, then hypervolume contribution and fixed engine ID.
   The frozen assessor receives engine code, offline metrics and shared context.
   Historical online outcomes are used during training only.
6. `critic_data.py`, training configurations and `scripts/train_critic_dpo.py`
   provide data preparation and a LLaMA-Factory LoRA DPO entrypoint.

The six implementation groups above cover the core workflow. Setting
`online_uplift_enabled: false` explicitly uses an offline hypervolume fallback.
The release implements the DispatchEvolve main offline experiment: evolution on
the training split, frozen engine selection, then held-out test reporting.
The synthetic example and Critic training utilities support this workflow.

The public evaluator uses a general matching schema and a local matcher.
See [public fields and metric formulas](docs/public-data-model.md)
for the 22 features, three observed labels and seven reported metrics. The paper experiments use the production replay simulator.

## Paper experiment settings

Use `configs/paper.yaml` as the main-experiment template. Supply your own engine,
disjoint evolution and test inputs, and model endpoints before running it. The
public toy engine was written for this example and contains one ranking rule.

The paper template specifies 10 outer rounds, at most 10 admitted opportunities
per round and 14 Pareto references. All evolution replay shares one cost budget:

```yaml
budget:
  full_replay_equivalents: 30
```

One full evolution-set replay costs 1. A local replay costs the number of input
rows replayed divided by the number of rows in the evolution set. For example,
a scene covering 10% of those rows costs 0.1 per replay; ten such replays cost 1.
Scenarios replay complete batches, and the charge uses every replayed row, even
when metrics are computed on a smaller selection.

Initial baseline, trace construction, scene evaluation, local genetic candidates,
candidate admission and full-engine composition share this budget. Valid cache
hits cost zero. Each uncached replay reserves its cost before execution; failed
or interrupted attempts retain that charge, and a new physical retry is charged
again. Parallel processes share an atomic SQLite ledger. Resume reuses the ledger.

Search stops when the budget is exhausted or the next replay cannot fit. It
selects from the already evaluated archive, then evaluates the frozen selection
and baseline on the test set. Test evaluation and offline critic-data collection
are outside the evolution budget. Round, opportunity and local-iteration limits
remain additional stopping conditions, so a run may use less than 30 equivalents.

The run directory contains `replay_budget.sqlite` with per-attempt row charges.
`final/replay_cost.json` reports total charged rows, full-replay equivalents used,
remaining budget and denied requests. The synthetic summary includes the same
report. To exercise budget termination:

```bash
uv run python -m dispatchevolve.synthetic_demo \
  --full-replay-equivalents 1 --output-dir workspaces/budget_demo
```

The default normalized regression tolerance is `rho: 0.005`: BR may decrease by
at most 0.5% relative to its reference, not 0.5 percentage points. Global archive
admission also requires no regression on any objective and improvement on at
least one. The current implementation shares `rho` with local guardrails, critic
label preparation and the negative hypervolume reference; their defaults agree.

`random_seed` is a freely configurable integer.
The paper uses Gemini 3 Flash for generation and separately frozen Qwen3-8B DPO
critics. Endpoint model names are deployment-specific placeholders in the
configuration. The scripted demo and `configs/example.yaml`
use reduced iteration limits for a quick execution check. Both the example and paper
configuration use `mode: main` and `evaluate_test_baseline: true`. Main mode
requires distinct evolution/test file paths and test baseline evaluation.
Prepare non-overlapping, temporally ordered replay splits before running.

The final `final/test_result.json` contains test metrics and `paper_report`.
CR maps to `mean_fqs`, computed as the matched-pair mean of
`DAR * (1 - DCAA) * (1 - PCAA)`. The input-column average `mean_cr` is a separate
diagnostic field. AR maps to `order_ar`; GMV, ETA, PCAA and DCAA map to
`mean_gmv`, `mean_eta`, `mean_pcaa` and `mean_dcaa`, respectively. BR maps to
`order_br`. These are six objectives and one guardrail.

`paper_report.relative_improvement_percent` uses paper display names and
`100 * direction * (selected - baseline) / max(abs(baseline), epsilon)`.
FAI is the mean of the six objective improvements when each is strictly positive
and the BR improvement is strictly greater than -0.5%; otherwise FAI is zero.
BR is excluded from that mean. Comparisons use the configured numerical tolerance.
The synthetic example leaves some objectives unchanged, so its FAI is zero.
Test results are written after search and are not supplied to candidate selection.

## Critic training

See [DPO data preparation and training](docs/critic-training.md).
The guide covers opportunity labels derived from trial outcomes, historical
online preferences, validation splits by group, token-length checks and separate
critic checkpoints. Prepare training records on a CPU and train the adapters on
a GPU.

## Tests

```bash
uv run pytest test/ -q
```

The tests include a real one-round synthetic run with only the external LLM
replaced, plus tests of archive rejection, pairwise ranking, trace attribution,
policy restrictions, DPO grouping and dataset integrity. Mocked orchestration tests cover failure and recovery paths.

Generated data, responses, logs and checkpoints are stored in Git-ignored
output directories such as `workspaces/`, `.cache/` and `logs/`.

## Licensing

Licensed under [Apache-2.0](LICENSE). Third-party components and attribution
are listed in [third-party notices](THIRD_PARTY_NOTICES.md).
