# Public data model and metrics

Each row is a possible request-resource pair. A batch groups the pairs evaluated
in one matching step. The example calls requests `order_id` and resources
`driver_id`; these names describe the two sides of a bipartite matching problem.
Service category `product_id = 1` is the editable example category. Other
categories pass through unchanged.

## Features

| Field | Meaning |
| --- | --- |
| `batch_id` | Matching batch identifier |
| `order_id` | Request identifier |
| `driver_id` | Available resource identifier |
| `product_id` | Service category |
| `eta` | Estimated pickup travel time, seconds |
| `dar` | Predicted acceptance probability |
| `pcaa` | Predicted customer cancellation probability after acceptance |
| `dcaa` | Predicted resource-side cancellation probability after acceptance |
| `cr` | Optional predicted completion probability, a diagnostic input |
| `gmv` | Estimated matched-pair value |
| `pre_total_fee` | Estimated request value, fallback for `gmv` |
| `weight` | Matching score |
| `stage` | Matching priority |
| `driver_lock_time_s` | Resource reservation duration, seconds |
| `order_lock_time_s` | Request reservation duration, seconds |
| `if_broadcast` | Pair offered for assignment |
| `order_create_timestamp` | Request creation timestamp |
| `order_wait_time` | Elapsed request waiting time, seconds |
| `available_drivers` | Available resource count |
| `pending_orders` | Pending request count |
| `pickup_distance` | Estimated pickup distance, meters |
| `trip_distance` | Estimated service distance, meters |

The public feature dictionary contains 22 fields. They are general matching
inputs, predictions and policy outputs. The candidate column boundary allows
this dictionary and hides observed labels from candidate programs.

## Observed labels

| Field | Meaning |
| --- | --- |
| `observed_assignment` | Pair was assigned |
| `observed_acceptance` | Assignment was accepted |
| `observed_completion` | Service completed |

There are three observed-label fields. They are separate from the 22 feature
fields and are excluded from candidate-visible inputs.

## Reported metrics

Let $P$ be the matched pairs, $O$ the distinct requests in the replay input,
and $O_P$ the requests represented in $P$.

| Paper name | Code field | Calculation | Direction |
| --- | --- | --- | --- |
| CR | `mean_fqs` | Mean of `dar * (1 - pcaa) * (1 - dcaa)` over matched pairs | Higher |
| AR | `order_ar` | Mean event acceptance probability | Higher |
| GMV | `mean_gmv` | Mean `gmv` over matched pairs | Higher |
| ETA | `mean_eta` | Mean `eta` over matched pairs | Lower |
| PCAA | `mean_pcaa` | Mean `pcaa` over matched pairs | Lower |
| DCAA | `mean_dcaa` | Mean `dcaa` over matched pairs | Lower |
| BR | `order_br` | Distinct matched requests divided by distinct input requests | Guardrail |

There are seven reported metrics: six objectives and one guardrail. For one
assigned resource, event acceptance is `dar`; for two, it is
`1 - (1 - dar_1) * (1 - dar_2)`. The included local matcher assigns at most one
resource to each request. `mean_cr` separately averages the optional `cr` input;
the paper's CR display field is `mean_fqs`.

For each metric, the direction-aligned percentage improvement on the test set is

$$
g_j = 100 d_j \frac{m_j(E^*)-m_j(E_0)}{\max(|m_j(E_0)|,\epsilon)},
$$

where $d_j$ is 1 for higher-is-better metrics and -1 otherwise. Both engines use
the same test input. FAI is the mean of the six objective improvements when all
six are strictly positive and BR improvement is greater than -0.5%; otherwise
FAI is zero. BR is excluded from the mean. Comparisons use the configured
numerical tolerance.

## Offline matching

`backend: local` performs maximum-weight bipartite matching independently for
each batch using SciPy's linear assignment solver. Each request and resource
appears in at most one returned pair. Only positive-weight edges are eligible;
missing edges remain unmatched. Duplicate request-resource edges retain the
highest-scoring row. Request and resource identifiers must be present.

The matching score combines policy stage and weight. For a batch with $n$ rows,
let $W$ be its largest absolute finite weight. An edge receives the score

$$
s_{ij} = (\mathrm{stage}_{ij} - \min \mathrm{stage})\,[W(n+1)+1] + w_{ij}.
$$

The assignment maximizes the sum of these scores. Integer stages therefore
receive priority over within-stage weight differences. The matching solver runs
locally and requires no service credentials or business data.
