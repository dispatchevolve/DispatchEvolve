# Preparing and training the two critics

The opportunity critic predicts whether a trace-grounded local optimization task
is actionable. The online assessor compares two offline-feasible engines. Train
these as separate LoRA DPO adapters; freeze their parameters before the evaluated
search. This repository supplies data preparation, two training configurations
and a launcher. Supply labeled records to train the adapters.

The launcher follows LLaMA-Factory's documented
[ShareGPT preference format](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md#preference-dataset-1)
and [Qwen3 LoRA DPO configuration](https://github.com/hiyouga/LLaMA-Factory/blob/main/examples/train_lora/qwen3_lora_dpo.yaml).
Run data preparation and configuration checks on a CPU, then train the adapters
in a GPU environment with LLaMA-Factory installed. The launcher records the
installed package version.

## Opportunity supervision

Construct each prompt from information available before local optimization:
scenario definition and distribution, the primary objective, policy code or
trace evidence, baseline metrics and guardrails. Use the same prompt contract
as the runtime `opportunity_critic` role in `prompt_store.py`.

Run the corresponding local optimization with a fixed budget. Aggregate the
trial at the opportunity level: prefer ACCEPT if at least one evaluated candidate
has strictly positive oriented gain on the primary objective and all protected
metrics stay within the configured regression tolerance; otherwise prefer
REJECT. Use completed trials for supervision. The builder excludes records whose
`evaluation_status` is not `completed`, including failed or interrupted evaluations.

Supply one JSONL object per labeled opportunity, with these fields:

- `group_id`: the source scenario/run or a broader leakage group. Related traces,
  engine variants and repeated trials must share the same group.
- `system`, `user`: complete pre-optimization prompts, without outcome-derived
  explanations or labels leaking into the input.
- `objective`: one active reward metric, such as `order_ar`.
- `oriented_delta`: all seven normalized metric changes, calculated against the
  correct baseline and scale; larger is better after orientation. The fields are
  `order_ar`, `mean_gmv`, `mean_eta`, `mean_pcaa`, `mean_dcaa`, `mean_fqs` and `order_br`.
- `evaluation_status`: `completed` only for a completed, valid trial.
- `responses`: two full strings under `ACCEPT` and `REJECT`, following the runtime
  Decision/Confidence/Reason headings. The measured trial outcome determines
  which response is preferred.

The paper display name CR maps to `mean_fqs`, the matched-pair mean of
`DAR * (1 - DCAA) * (1 - PCAA)`.

There are seven delta fields: six reward coordinates and one protected matching
coverage coordinate. The builder derives the preferred answer from these deltas.
Use the best feasible candidate when one exists; otherwise use the best evaluated
unsuccessful candidate. Record the candidate selection rule with the trial results.

Write both responses from pre-optimization evidence, with each response expressing
its named decision. Keep trial outcomes in the labels, separate from model inputs.
To use records from `dpo_data_collection`, convert their frozen prompts, trial
outcomes and selected candidate deltas into the schema above.

```bash
uv run python -m dispatchevolve.critic_data --kind opportunity \
  --input data/opportunity_records.jsonl \
  --output-dir workspaces/opportunity_dpo --rho 0.005
```

Use the same `rho` and normalization as the source trials. Acceptance requires
positive gain; floating-point comparisons use a tolerance of `1e-12`.

## Online assessor supervision

Use historical decisions only to label training examples. Each input must contain
a common pre-deployment context and both engine designs, following `SYSTEM` and
the user-document layout in `pairwise_ranking.py`. Exclude realized outcomes,
significance results and deployment decisions from the model input.

The preparation schema uses `group_id`, `system`, `user`, `responses` with keys
`A` and `B`, and `preferred` equal to `A`, `B` or null. Prefer the engine supported
by an unambiguous historical production decision after considering the relevant
metrics and guardrails. Set `preferred` to null to exclude an ambiguous pair.
Use the complete responses `A` and `B` to match inference.

Randomize presentation order in real training data and remap the winner with it.
All pairs sharing an experiment or related engine family must stay in one group;
use connected groups when experiments share engines. Assign these group IDs
before data preparation. Evaluate position bias and accuracy on held-out engine
families.

```bash
uv run python -m dispatchevolve.critic_data --kind online \
  --input data/online_preference_records.jsonl \
  --output-dir workspaces/online_dpo
```

The inference code compares every unordered pair, counts wins, and breaks ties
using hypervolume contribution and then engine ID. No historical outcomes are
retrieved at inference. A singleton archive requires no comparison.

## Validate with synthetic data

```bash
uv run python -m dispatchevolve.critic_data --kind opportunity --synthetic \
  --output-dir workspaces/opportunity_dpo
uv run python scripts/train_critic_dpo.py --dry-run
uv run python -m dispatchevolve.critic_data --kind online --synthetic \
  --output-dir workspaces/online_dpo
uv run python scripts/train_critic_dpo.py --config configs/online_critic_dpo.yaml --dry-run
```

Each synthetic command creates ten artificial pairs from ten artificial groups:
eight training groups/pairs and two validation groups/pairs, with zero group
overlap. Use these examples to check the data format and training setup; use
labeled trial records to train the critics. Choose empty output directories.

For real inputs the builder writes `train.jsonl`, `eval.jsonl`,
`dataset_info.json` and `manifest.json`. Splitting is deterministic by group hash;
no group crosses the boundary. Duplicate prompts are rejected, and the manifest
reports input, retained, excluded, group and split counts plus dataset hashes.
Reserve separate final test groups for evaluation after training.

## Launch LoRA DPO training

On a suitable GPU machine, install LLaMA-Factory using its
[official installation instructions](https://github.com/hiyouga/LLaMA-Factory#installation).
Run the launcher in the LLaMA-Factory Python environment. Configure the base
model, compatible template, prepared dataset and available hardware.

```bash
python scripts/train_critic_dpo.py --config configs/critic_dpo.yaml
python scripts/train_critic_dpo.py --config configs/online_critic_dpo.yaml
```

The supplied configurations use sigmoid DPO, LoRA rank 8, alpha 16, dropout 0.05,
beta 0.1, learning rate `5e-6` and three epochs as starting values. Set the batch
accumulation and precision to suit the GPU. BF16 requires hardware support.

Before training, the launcher verifies dataset hashes, the registry and the
explicit train/eval split. It uses the installed LLaMA-Factory template and model
tokenizer to measure complete chosen/rejected sequences. It refuses to proceed
if `cutoff_len` would truncate any example. Increase the limit within the model's
context capacity and available memory, or redesign the example construction.
The dry-run checks dataset and configuration structure. Token-length validation
runs when launching training.

Training is delegated to `llamafactory-cli train`. Resolved configuration and
preflight metadata are written under `workspaces/training_preflight/`. Logs go to
`logs/scripts/train_critic_dpo/train_critic_dpo_<UTC timestamp>.log`. The two weight
output directories default to `workspaces/critic-dpo/` and
`workspaces/online-critic-dpo/`; nonempty outputs are rejected.

Serve each frozen adapter through a compatible endpoint. In the workflow YAML,
set `critic_model` to the opportunity adapter and `online_model` to the online
adapter, with `online_uplift_enabled: true`. Evaluate response format, held-out
label accuracy and presentation-order bias.

## Data and outputs

Example rows are generated by the synthetic commands. Keep supplied training
records in `data/` and generated datasets and checkpoints in `workspaces/`;
both directories are ignored by Git.
