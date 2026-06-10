# PRISM Global vs. Contextual Feature Selection

Date: 2026-06-10

Primary comparison: PRISM Qwen3.5-27B local-annotator runs at `M=32`, `M=64`, and `M=128`.

Contextual selector: `conversation_type` demeaned-reweighted LASSO, with `length_delta` retained as the nuisance control and fixed budgets `top_k={5,10}`.

Primary artifacts:

- `outputs/analysis/prism_covariate_selector/wimhf_local_full_prism_qwen35_27b_s42_clip_20260609_6986752/`
- `outputs/analysis/prism_covariate_selector/wimhf_local_full_prism_qwen35_27b_m64_s42_clip_20260610_6995780/`
- `outputs/analysis/prism_covariate_selector/wimhf_local_full_prism_qwen35_27b_m128_s42_clip_20260610_6995781/`

GPT-5-mini exact M32 artifacts are used only as an annotator check:

- `outputs/analysis/prism_covariate_selector/wimhf_exact_full_prism_gpt5mini_s42_tron_20260608_6985771/`

## Support

`conversation_type` is well supported and identical across the local capacity runs:

| Conversation type | Train n | Val n |
|---|---:|---:|
| controversy guided | 4,847 | 1,128 |
| unguided | 4,251 | 1,028 |
| values guided | 4,786 | 1,316 |

The equal-stratum weights are mild: min `0.955`, max `1.089`, mean `1.000`. This means the contextual result is not being driven by extreme upweighting of a tiny stratum.

## Local Capacity Sweep

For a fair capacity comparison, use the local Qwen3.5 M32 run as the baseline. Against that baseline, larger `M` increases the number of retained interpretable neurons and improves reconstruction, but it does not produce a clear contextual-selection win.

| Capacity | Kept neurons | Mean abs fidelity corr. | Recon. norm MSE | Full SAE val AUC | Global top-10 val AUC | Context top-10 val AUC |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 22 / 32 | 0.477 | 0.875 | 0.6650 | 0.6618 | 0.6600 |
| 64 | 46 / 64 | 0.480 | 0.859 | 0.6679 | 0.6594 | 0.6583 |
| 128 | 77 / 128 | 0.442 | 0.848 | 0.6647 | 0.6647 | 0.6639 |

M128 has the best fixed-budget global top-10 AUC and M64 has the best full-SAE AUC, but the contextual selector remains slightly below global selection at top 10 for every local capacity.

## Held-Out AUC

At equal feature budgets, contextual selection is mostly flat or slightly worse than global selection. The only overall contextual improvement is small: M64 top 5 improves by `+0.0010`, while M64 top 10 is still worse than global.

| Capacity | Budget | Global val AUC | Context val AUC | Delta |
|---:|---:|---:|---:|---:|
| 32 | 5 | 0.6609 | 0.6563 | -0.0046 |
| 32 | 10 | 0.6618 | 0.6600 | -0.0018 |
| 64 | 5 | 0.6538 | 0.6549 | +0.0010 |
| 64 | 10 | 0.6594 | 0.6583 | -0.0012 |
| 128 | 5 | 0.6640 | 0.6614 | -0.0026 |
| 128 | 10 | 0.6647 | 0.6639 | -0.0008 |

Per-context changes are also mixed:

| Capacity | Budget | Controversy delta | Unguided delta | Values delta |
|---:|---:|---:|---:|---:|
| 32 | 5 | -0.0110 | +0.0040 | -0.0056 |
| 32 | 10 | -0.0105 | +0.0031 | +0.0022 |
| 64 | 5 | +0.0009 | -0.0038 | +0.0055 |
| 64 | 10 | -0.0071 | -0.0012 | +0.0053 |
| 128 | 5 | +0.0007 | -0.0041 | -0.0041 |
| 128 | 10 | +0.0017 | -0.0057 | +0.0001 |

The most consistent contextual gain is in `values guided` for M64, but it is not large enough to improve top-10 overall AUC.

## Selected-Feature Overlap

The contextual selector mostly re-ranks global features. Overlap increases at larger budgets and capacities.

| Capacity | Budget | Global selected | Contextual selected | Overlap |
|---:|---:|---|---|---:|
| 32 | 5 | 0, 4, 25, 6, 13 | 0, 4, 7, 1, 6 | 3 / 7 union |
| 32 | 10 | 0, 4, 25, 6, 13, 14, 11, 8, 7, 1 | 0, 4, 1, 6, 7, 25, 10, 5, 2, 11 | 7 / 13 union |
| 64 | 5 | 2, 5, 4, 53, 26 | 2, 3, 4, 5, 53 | 4 / 6 union |
| 64 | 10 | 2, 5, 4, 53, 26, 23, 16, 6, 48, 32 | 2, 3, 4, 5, 53, 6, 63, 26, 23, 32 | 8 / 12 union |
| 128 | 5 | 3, 1, 35, 0, 6 | 3, 2, 0, 6, 1 | 4 / 6 union |
| 128 | 10 | 3, 1, 0, 35, 6, 5, 79, 21, 52, 23 | 3, 2, 0, 1, 6, 35, 52, 5, 23, 104 | 8 / 12 union |

Contextual-only features are mostly variants of existing PRISM themes: brevity, refusal/disclaimer behavior, word-count/format constraints, and assistant identity. They do not yet look like qualitatively new `conversation_type`-specific preference concepts.

| Capacity | Budget | Contextual-only features |
|---:|---:|---|
| 32 | 5 | `7`: deflects by apologizing or claiming inability; `1`: substantive response without no-opinion disclaimer |
| 32 | 10 | `10`: no bullets/numbered lists; `5`: says it is an AI with no personal opinions/emotions; `2`: does not ask follow-up questions |
| 64 | 5 | `3`: brief minimal response instead of detailed multi-point explanation |
| 64 | 10 | `3`: brief minimal response; `63`: does not limit response to about 50 words |
| 128 | 5 | `2`: brief single-point response |
| 128 | 10 | `2`: brief single-point response; `104`: says it is an Anthropic AI assistant |

## Interpretation

The local capacity sweep changes the feature inventory but not the main conclusion. Larger SAEs expose more non-redundant interpretable features, and M128 gives the strongest fixed-budget global selector, but the demeaned-reweighted contextual selector does not outperform global selection in a reliable way.

The strongest evidence for a conditional effect is a small reweighting toward `values guided` at M64 and a few context-only format/disclaimer features. That is useful as a diagnostic, but not enough to claim a robust conditional preference regime.

## GPT-5-Mini Annotator Check

The exact GPT-5-mini M32 run reaches higher AUC than the local Qwen3.5 M32 run, but it shows the same qualitative result: contextual selection is worse than global at equal budgets.

| Budget | Global val AUC | Context val AUC | Delta | Contextual-only feature |
|---:|---:|---:|---:|---|
| 5 | 0.6656 | 0.6615 | -0.0041 | `5`: long multi-point explanation/list |
| 10 | 0.6679 | 0.6649 | -0.0030 | `5`: long multi-point explanation/list |

The GPT-5-mini check therefore supports the local-run conclusion: the current contextual selector mostly re-ranks global PRISM features and does not deliver a predictive gain.

## Decision

Use the local M128 run as the richer PRISM feature inventory when discussing capacity, and keep M32 as the WIMHF-compatible baseline. Do not claim that `conversation_type` + demeaned-reweighted LASSO discovers a strong conditional preference regime.

Next diagnostic: run an interaction/covariate LASSO to test explicit context-feature moderation and possible sign reversals. The current residualized selector is better interpreted as a balancing/re-ranking check than as evidence of context-specific preference structure.
