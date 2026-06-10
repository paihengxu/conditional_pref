# PRISM Global vs. Contextual Feature Selection

Date: 2026-06-10

Run: `outputs/reproduction/wimhf_exact/wimhf_exact_full_prism_gpt5mini_s42_tron_20260608_6985771`

Contextual selector: `conversation_type` demeaned-reweighted LASSO, with `length_delta` retained as the nuisance control and fixed budgets `top_k={5,10}`.

Artifacts:

- `outputs/analysis/prism_covariate_selector/wimhf_exact_full_prism_gpt5mini_s42_tron_20260608_6985771/covariate_predictive_metrics.json`
- `outputs/analysis/prism_covariate_selector/wimhf_exact_full_prism_gpt5mini_s42_tron_20260608_6985771/covariate_selection_summary.csv`
- `outputs/analysis/prism_covariate_selector/wimhf_exact_full_prism_gpt5mini_s42_tron_20260608_6985771/covariate_lasso_coefficients.json`
- `outputs/analysis/prism_covariate_selector/wimhf_exact_full_prism_gpt5mini_s42_tron_20260608_6985771/covariate_support.json`

## Support

`conversation_type` is well supported in the prediction split:

| Conversation type | Train n | Val n |
|---|---:|---:|
| controversy guided | 4,847 | 1,128 |
| unguided | 4,251 | 1,028 |
| values guided | 4,786 | 1,316 |

The equal-stratum weights are mild: min `0.955`, max `1.089`, mean `1.000`. This means the contextual result is not being driven by extreme upweighting of a tiny stratum.

## Held-Out AUC

At equal feature budgets, the contextual selector does not improve held-out AUC. It is slightly worse than global selection in every context.

| Budget | Selector | Overall val AUC | Controversy guided | Unguided | Values guided |
|---:|---|---:|---:|---:|---:|
| 5 | global | 0.6656 | 0.6738 | 0.6379 | 0.6803 |
| 5 | contextual | 0.6615 | 0.6700 | 0.6334 | 0.6767 |
| 10 | global | 0.6679 | 0.6802 | 0.6366 | 0.6820 |
| 10 | contextual | 0.6649 | 0.6771 | 0.6334 | 0.6793 |

The full `M=32` SAE baseline remains higher at val AUC `0.6712`, so neither fixed-budget selector captures all useful predictive signal.

## Selected-Feature Overlap

The contextual selector mostly re-ranks the global features rather than discovering a separate feature set.

| Budget | Global selected | Contextual selected | Overlap |
|---:|---|---|---:|
| 5 | 0, 1, 28, 13, 7 | 0, 5, 7, 1, 28 | 4/6 union |
| 10 | 0, 1, 28, 13, 7, 14, 4, 19, 11, 9 | 0, 5, 7, 1, 28, 13, 19, 14, 11, 9 | 9/11 union |

The only contextual-only feature at these budgets is neuron `5`: "provides a lengthy, multi-point explanation or list rather than a short, one-sentence reply." Global-only features are neuron `13` at top 5 and neuron `4` at top 10:

- `13`: "includes unrelated tangents and conversation/log artifacts instead of directly addressing the prompt"
- `4`: "avoids taking a definitive stance, framing the issue as complex with arguments on multiple sides"

## Coefficient Shifts

The contextual coefficients shrink the dominant global feature and raise the rank of feature `5`.

| Neuron | Interpretation | Global coef, top 10 | Contextual coef, top 10 |
|---:|---|---:|---:|
| 0 | detailed neutral explanation without personal opinions | 0.4047 | 0.0951 |
| 1 | no AI/person-opinion disclaimer | 0.0879 | 0.0174 |
| 28 | no word-count limit | 0.0768 | 0.0120 |
| 13 | tangents/log artifacts instead of answering | -0.0568 | -0.0067 |
| 7 | substantive sensitive/controversial response rather than refusal | 0.0411 | 0.0182 |
| 14 | irrelevant or meandering non-answer | -0.0290 | -0.0031 |
| 4 | avoids definitive stance; multiple sides | 0.0130 | not selected |
| 5 | lengthy multi-point explanation/list | not selected | 0.0419 |

The top-5 shift is similar: contextual selection replaces global neuron `13` with neuron `5`. At top 10 it replaces global neuron `4` with neuron `5`.

## Qualitative Novelty

The contextual selector's main novelty is not a sign-reversing or context-exclusive preference. It surfaces a length/format preference after demeaning by conversation type: PRISM preferences still reward detailed neutral content, but the contextual selection gives higher priority to "multi-point explanation/list" than the global selector does.

This is meaningful contextual re-ranking, but it is limited. The selected sets remain heavily overlapping, and per-context AUC does not improve. The current `M=32` SAE likely has too little spare capacity to expose qualitatively new context-specific concepts beyond the strongest global PRISM themes.

## Decision

Proceed to capacity sensitivity with PRISM `M=64` and `M=128`, but keep the interpretation scope narrow:

- compare only global/contextual selected neurons and their union;
- look for additional non-redundant contextual-only features beyond format/length;
- require either per-context AUC improvement, clear selected-feature novelty, or strong coefficient moderation before claiming a conditional preference regime.

If larger `M` still mostly re-ranks the same global concepts without per-context gains, the next diagnostic should be an interaction/covariate LASSO to test explicit context-feature moderation and possible sign reversals.
