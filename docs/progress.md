# Conditional Preference Discovery Progress

Last updated: 2026-06-08.

This file is the main standalone context file for continuing the project. Keep it concise and update it as decisions change. Put bulky inventories or detailed notes in separate docs and link them here.

## Project Goal

Discover and evaluate context-dependent preference structure in public LLM preference datasets. The core hypothesis is that global preference-feature analysis can hide, dilute, or reverse preference signals that become visible when conditioning on principled, pre-specified contexts.

The project should demonstrate more than generic preference-feature discovery: it must show that conditional preference regimes are qualitatively different, predictive, and potentially actionable for post-training data curation, reward modeling, DPO, SFT, or evaluation.

## Scope Principles

- Work on existing public preference datasets before considering synthetic benchmarks.
- Choose contexts before inspecting preference effects.
- Use contexts that are preference-relevant, pre-training/pre-curation available, actionable, sufficiently supported, and non-circular.
- Avoid contexts derived from the same response features used to explain preferences.
- Continue only if conditional analysis reveals credible signal beyond global WIMHF-style analysis.

Broader initial framing is in `docs/cond_pref.md`. Cached dataset details are in `docs/dataset_inventory.md`.

## Baseline And Differentiation

WIMHF is the baseline method: it trains SAEs on preference-pair embedding differences, interprets learned features, and estimates global expressed preferences.

This project extends that frame by testing whether WIMHF-style global features miss context-dependent regimes. Minimum useful outputs:

- dataset/context inventory;
- support table for candidate contexts;
- global vs. context-aware top features;
- examples of hidden, diluted, or sign-reversing preferences;
- held-out preference prediction by context.

## Current Dataset Plan

First reproduce WIMHF on datasets with strong context metadata:

1. CommunityAlign: conversation IDs, language, annotator demographics, feedback fields, turn metadata.
2. PRISM: user/conversation/type/turn/model metadata.

Backup/secondary candidates:

- ChatbotArena: model/source/language contexts.
- Reddit: domain/source contexts.
- HH-RLHF: useful as an optional baseline/comparison, but weak for non-circular context analysis because its extra fields are mostly response-derived LLM descriptions.

## Workspace State

- WIMHF repo: `repos/wimhf`.
- HypotheSAEs reference repo: `../text_diff/repos/HypotheSAEs`.
- Cached WIMHF-style data: `data/wimhf/raw/*.json`.
- Local configs: `configs/community_align_wimhf_local.json`, `configs/prism_wimhf_local.json`, `configs/hh_rlhf_wimhf_local.json`.
- CommunityAlign and PRISM local configs use `text-embedding-3-small`, SAE `M=32`, `K=4`, prefixes `[8, 32]`, and `length_delta` control.
- `repos/wimhf` is a nested git repo; local WIMHF patches are tracked there, not by the outer project repo.
- WIMHF runner: `scripts/run_wimhf.py`; Slurm wrappers following `docs/slurm.md`: `scripts/sbatch/run_wimhf_local_community_align_clip.sbatch`, `scripts/sbatch/run_wimhf_local_prism_tron.sbatch`, `scripts/sbatch/run_wimhf_exact_community_align_clip.sbatch`, `scripts/sbatch/run_wimhf_exact_prism_tron.sbatch`.

## Environment Decision

Use existing conda env `textdiff` for initial reproduction.

Known state:

- Python 3.10.18.
- Has `torch`, `vllm`, `skglm`, `datasets`, `hypothesaes`, `openai`, `pandas`, `scipy`.
- Does not appear to have `flash_attn`; this should not block OpenAI-API WIMHF reproduction.
- `conda activate textdiff` is not initialized in noninteractive Codex shells; use `conda run -n textdiff ...`.

Setup:

```bash
conda activate textdiff
cd /fs/clip-projects/clip-k12/paiheng/conditional_pref
pip install -e repos/wimhf --no-deps
export OAI_WIMHF="$OPENAI_API_KEY"
```

Use `--no-deps` initially to avoid disturbing the working Torch/vLLM stack. Create a separate env later only if results look promising and post-training experiments require it.

## WIMHF Reproduction Decision

Use `outputs/reproduction/wimhf_exact` as the trusted global WIMHF baseline for CommunityAlign and PRISM. This run uses the paper-style setup: `text-embedding-3-small`, SAE `M=32`, `K=4`, prefixes `[8, 32]`, `length_delta` control, `gpt-5` interpreter, and `gpt-5-mini` annotator/abbreviator.

Exact reproduction results:

- CommunityAlign: `31/32` features kept, mean fidelity `0.545`, full-SAE val AUC `0.686`. This matches the paper's `31` high-fidelity CA features and recovers paper-like themes: concrete/task-specific answers, structured lists/templates, omission of sustainability/environment, and omission of community/social/ethical framing.
- PRISM: `21/32` features kept, mean fidelity `0.376`, full-SAE val AUC `0.671`. This is close to the paper's `23` high-fidelity PRISM features and recovers themes including detailed neutral answers, avoiding AI disclaimers, substantive controversial-topic engagement, dispreference for "yes"-only answers, and off-topic/tangential content.
- Approximate API cost for exact CommunityAlign + PRISM reproduction: `$40`.

Do not use GPT-5-mini for every exploratory run by default. Use it for trusted baselines and final validation of selected conditional findings. For broad sweeps, first use cached exact global features plus cheap/statistical context analyses, and evaluate better open-weight annotators before scaling annotation-heavy workflows. The first local annotator tried, `Qwen/Qwen3-30B-A3B-Instruct-2507`, was acceptable for PRISM but weak for CommunityAlign fidelity retention (`13/32` kept vs. `31/32` with GPT-5-mini), so it should not be the default final judge without further calibration.

The nested WIMHF patch supports batched local-vLLM annotation and `WIMHF_VLLM_*` engine settings. Suggested local annotator environment for 2x A6000:

```bash
export CUDA_VISIBLE_DEVICES=0,1
export WIMHF_VLLM_TENSOR_PARALLEL_SIZE=2
export WIMHF_VLLM_MAX_MODEL_LEN=8192
export WIMHF_VLLM_GPU_MEMORY_UTILIZATION=0.85
```

## HypotheSAEs Decision

Use HypotheSAEs as a method reference, not as a direct dependency for first reproduction.

Do not copy the full HypotheSAEs selection module. If conditional selection is needed, copy or adapt only the chosen method with provenance and tests.

Likely relevant methods: interaction/covariate LASSO, demeaned/reweighted LASSO, group LASSO, conditional separation score.

## Next Execution Steps

1. Install WIMHF editable in `textdiff` with `--no-deps`. Human: Done. 
2. Verify import/config load for CommunityAlign and PRISM configs. Codex: Done with `conda run -n textdiff`; both cached dataset paths exist.
3. Create local-annotator config variants for CommunityAlign and PRISM using `Qwen/Qwen3-30B-A3B-Instruct-2507` as `annotator_model`.
4. Optional smoke sanity check with `RUN_MODE=smoke`. [Human: remove this]
5. Run full local-annotator WIMHF reproduction for CommunityAlign and PRISM. Codex: runner and sbatch wrapper written; not submitted. [Human: submitted]
6. Compare feature tables, fidelity metrics, and predictive metrics. Codex: Done; PRISM matches reasonably, CommunityAlign needs GPT-5-mini verification.
7. Run GPT-5-mini annotator reproduction for CommunityAlign and PRISM, prioritizing CommunityAlign. Human: Done; outputs in `outputs/reproduction/wimhf_exact`; use as trusted baseline.
8. Build context inventories and support tables for CommunityAlign and PRISM.
9. Calibrate a stronger open-weight/local annotator against the GPT-5-mini feature-retention and top-feature results before using local annotation for large conditional sweeps.
10. Decide the first conditional extension dataset and method.

## Open Questions

- Which conditional feature-selection approach should be tried first?
- After local-annotator reproduction, should the next pass keep OpenAI embeddings for WIMHF comparability or switch to local/newer embedders for stronger predictive signal?
- Which open-weight annotator best matches GPT-5-mini fidelity scoring on CommunityAlign and PRISM at materially lower cost?

## TODO

- Try adding the pre-specified context to the embedding prompt before changing the SAE objective. This follows the WIMHF README discussion that instruction-aware prompt-response embeddings can capture more preference-relevant information than response-only embeddings; context can be included in that prompt while keeping the SAE reconstruction-trained and pushing conditionality into downstream selection/evaluation.
