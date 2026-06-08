# Conditional Preference Discovery Progress

Last updated: 2026-06-07.

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
- WIMHF runner: `scripts/run_wimhf.py`; Slurm wrappers following `docs/slurm.md`: `scripts/sbatch/run_wimhf_local_community_align_clip.sbatch`, `scripts/sbatch/run_wimhf_local_prism_tron.sbatch`.

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

## Local Annotator Decision

Use `Qwen/Qwen3-30B-A3B-Instruct-2507` as the first local annotator model. The nested WIMHF patch supports batched local-vLLM annotation and `WIMHF_VLLM_*` engine settings.

Suggested local annotator environment for 2x A6000:

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
6. Compare feature tables, fidelity metrics, and predictive metrics; decide whether Qwen is sufficient or a second local annotator should be tested.
7. Build context inventories and support tables for CommunityAlign and PRISM.
8. Decide the first conditional extension dataset and method.

## Open Questions

- Which conditional feature-selection approach should be tried first?
- After local-annotator reproduction, should the next pass keep OpenAI embeddings for WIMHF comparability or switch to local/newer embedders for stronger predictive signal?
