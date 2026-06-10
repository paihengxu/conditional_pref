# Conditional Preference Discovery Progress

Last updated: 2026-06-10.

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
- Exact configs: `configs/community_align_wimhf_exact.json`, `configs/prism_wimhf_exact.json`, `configs/hh_rlhf_wimhf_exact.json`.
- CommunityAlign and PRISM WIMHF configs use `text-embedding-3-small`, SAE `M=32`, `K=4`, prefixes `[8, 32]`, and `length_delta` control.
- `repos/wimhf` is a nested git repo; local WIMHF patches are tracked there, not by the outer project repo.
- WIMHF runner: `scripts/run_wimhf.py`; Slurm wrappers following `docs/slurm.md`: `scripts/sbatch/run_wimhf_local_community_align_clip.sbatch`, `scripts/sbatch/run_wimhf_local_prism_tron.sbatch`, `scripts/sbatch/run_wimhf_exact_community_align_clip.sbatch`, `scripts/sbatch/run_wimhf_exact_prism_tron.sbatch`.

## Environment Decision

Use conda env `condpref` for project runs.

Known state:

- Python 3.10.18.
- Has `torch`, `vllm`, `skglm`, `datasets`, `hypothesaes`, `openai`, `pandas`, `scipy`.
- Does not appear to have `flash_attn`; this should not block OpenAI-API WIMHF reproduction.
- `conda activate condpref` may not be initialized in noninteractive Codex shells; use `conda run -n condpref ...`.

Setup:

```bash
conda activate condpref
cd /fs/clip-projects/clip-k12/paiheng/conditional_pref
pip install -e repos/wimhf --no-deps
export OAI_WIMHF="$OPENAI_API_KEY"
```

Use `--no-deps` initially to avoid disturbing the working Torch/vLLM stack.

## WIMHF Reproduction Decision

Use `outputs/reproduction/wimhf_exact` as the trusted global WIMHF baseline for CommunityAlign and PRISM. This run uses the paper-style setup: `text-embedding-3-small`, SAE `M=32`, `K=4`, prefixes `[8, 32]`, `length_delta` control, `gpt-5` interpreter, and `gpt-5-mini` annotator/abbreviator.

The WIMHF paper explicitly justifies the small SAE size. It argues that preference-pair response differences occupy a much smaller concept space than token-level LLM activation corpora, and that larger `M` produced more redundant, less interpretable features with little or no preference-prediction AUC gain. Appendix A reports an `M ∈ {16, 32, 64, 128}` sweep: the fraction of non-redundant high-fidelity features drops from `39.3%` at `M=32` to `29.2%` at `M=128`, while AUC does not improve. However, the paper also notes the tradeoff that larger `M` can yield more total usable features despite lower purity; for CommunityAlign, `M=32` gives `24` non-redundant features while `M=128` gives `72`. Therefore `M=32` is the faithful WIMHF-compatible baseline, not necessarily the final capacity for conditional discovery.

Exact reproduction results:

- CommunityAlign: `31/32` features kept, mean fidelity `0.545`, full-SAE val AUC `0.686`. This matches the paper's `31` high-fidelity CA features and recovers paper-like themes: concrete/task-specific answers, structured lists/templates, omission of sustainability/environment, and omission of community/social/ethical framing.
- PRISM: `21/32` features kept, mean fidelity `0.376`, full-SAE val AUC `0.671`. This is close to the paper's `23` high-fidelity PRISM features and recovers themes including detailed neutral answers, avoiding AI disclaimers, substantive controversial-topic engagement, dispreference for "yes"-only answers, and off-topic/tangential content.
- Approximate API cost for exact CommunityAlign + PRISM reproduction: `$40`.

Do not use GPT-5-mini for every exploratory run by default. Use it for trusted baselines and final validation of selected conditional findings. For broad sweeps and capacity discussion, base comparisons on the Qwen3.5-27B PRISM reproduction after calibrating it against GPT-5-mini; reserve GPT-5-mini for validating selected claims.

Local annotator calibration summary: `Qwen/Qwen3-30B-A3B-Instruct-2507` was close on some downstream AUCs but poor at reproducing high-fidelity feature retention (`13/32` kept on CommunityAlign and `8/32` on PRISM). `Qwen/Qwen3.5-27B` is much closer for feature-table reproduction (`31/32` kept on CommunityAlign and `22/32` on PRISM), though PRISM full-SAE val AUC remains below exact/paper (`0.665` vs. `0.671`). Detailed table and paper references: `docs/wimhf_local_calibration.md`.

Model selection should live in the JSON configs. `scripts/run_wimhf.py` should not rewrite the configured interpreter, annotator, or abbreviator model. Current local configs set `annotator_model` to `Qwen/Qwen3.5-27B`; exact configs set `annotator_model` and `abbreviator_model` to `gpt-5-mini`. The runner takes one required config path with `--config`; sbatch files should pass the exact config path directly rather than using dataset/profile shortcuts.

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

First conditional extension should modify SAE feature selection, not the SAE objective or embedding text. This follows the prior `text_diff` project: train the SAE normally, interpret/score features normally, then condition the statistical selection step over SAE activations on a pre-specified context. For PRISM, the first context should be `conversation_type` (`values guided`, `controversy guided`, `unguided`) because it is pre-existing, non-response-derived, balanced enough, and preference-relevant.

Preferred first method: `demeaned-reweighted-lasso` with `conversation_type` as the contextual stratum and `length_delta` retained as a nuisance control. This residualizes activations and labels against the context/control design, applies equal-stratum weighting, then uses the same fixed feature budgets as WIMHF. It is appropriate when the context is a nuisance or balancing factor and the within-stratum preference effect is expected to have consistent sign. Use interaction/covariate LASSO as a secondary diagnostic for sign reversals or strong context-specific moderation. Other possible methods: group LASSO and conditional separation score.

Hyperparameter discipline for conditional selection:

- Keep WIMHF's `M=32`, `K=4`, prefixes `[8, 32]` for the first contextual baseline.
- Keep fixed selection budgets, initially `top_k=[5, 10]`, so global and contextual selectors are compared at equal feature count.
- Choose L1 strength by binary search to hit the feature budget, not by validation AUC.
- Use held-out and per-context AUC as diagnostics, not as the selection criterion.
- After the `M=32` contextual baseline is working, run PRISM capacity sensitivity at `M=64` and `M=128`; increase capacity only if it yields more non-redundant, high-fidelity, context-relevant selected features.
- Keep `K=4` fixed for the first capacity sensitivity so the only intended capacity axis is `M`. Do not sweep `K` until the `M` sweep indicates either dead/redundant capacity or undercomplete feature discovery; if needed, treat `K` as a secondary SAE-quality sensitivity and evaluate reconstruction error, feature prevalence/dead-neuron rate, feature redundancy, fidelity retention, and stability of the global/contextual selected-feature union rather than selecting by validation AUC alone.

## Next Execution Steps

1. Install WIMHF editable in `condpref` with `--no-deps`. Human: Done. 
2. Verify import/config load for CommunityAlign and PRISM configs. Codex: Done with previous env; rerun with `conda run -n condpref` after migration if needed. Both cached dataset paths exist.
3. Create local-annotator config variants for CommunityAlign and PRISM using `Qwen/Qwen3-30B-A3B-Instruct-2507` as `annotator_model`. Done; this model was calibrated and found weak on CommunityAlign.
4. Run full local-annotator WIMHF reproduction for CommunityAlign and PRISM. Codex: runner and sbatch wrapper written; not submitted. [Human: submitted]
5. Compare feature tables, fidelity metrics, and predictive metrics. Codex: Done; see `docs/wimhf_local_calibration.md`.
6. Run GPT-5-mini annotator reproduction for CommunityAlign and PRISM, prioritizing CommunityAlign. Human: Done; outputs in `outputs/reproduction/wimhf_exact`; use as trusted baseline.
7. Build context inventories and support tables for CommunityAlign and PRISM.
8. Calibrate `Qwen/Qwen3.5-27B` as a stronger open-weight/local annotator against the GPT-5-mini feature-retention and top-feature results before using local annotation for large conditional sweeps. Codex: Done for CommunityAlign and PRISM; Qwen3.5 is the better local default for feature retention, with GPT-5-mini reserved for final validation.
9. Implement the first PRISM covariate/contextual selector: `conversation_type` + `demeaned-reweighted-lasso`, preserving WIMHF's `M=32` baseline and fixed `top_k` selection budgets. Codex: Done; reusable selector lives in `repos/wimhf/wimhf/feature_selection.py`, PRISM runner is `scripts/run_prism_covariate_selector.py`, GPT-5-mini M32 outputs are in `outputs/analysis/prism_covariate_selector/wimhf_exact_full_prism_gpt5mini_s42_tron_20260608_6985771`, and Qwen3.5 M32/M64/M128 selector sbatch wrappers are written.
10. Compare global vs. contextual selected features on PRISM using held-out AUC, per-context AUC, selected-feature overlap, coefficient shifts, and qualitative feature novelty. Codex: Done for exact PRISM M32; see `docs/prism_contextual_comparison.md`. Result: contextual selection mostly re-ranks global features, adds neuron `5` (long multi-point explanation/list), and is slightly worse than global on overall/per-context AUC.
11. If the `M=32` run shows meaningful contextual re-ranking but limited discovery capacity, run PRISM capacity sensitivity with `M=64` and `M=128`, interpreting only the union of global/context-selected neurons needed for comparison.

## Open Questions

- After the first PRISM contextual baseline, does `M=32` provide enough context-specific discovery capacity, or should PRISM move to `M=64`/`M=128`? Initial answer: run `M=64`/`M=128` capacity sensitivity, because `M=32` shows contextual re-ranking but limited novelty and no AUC gain.
- Should the next pass keep OpenAI embeddings for WIMHF comparability or switch to local/newer embedders for stronger predictive signal?
- Which open-weight annotator best matches GPT-5-mini fidelity scoring on CommunityAlign and PRISM at materially lower cost?

## TODO

- Run PRISM Qwen3.5 M32/M64/M128 covariate selectors and decide whether the capacity runs add useful non-redundant context-relevant features beyond the M32 re-ranking result.
- Revisit prompt/context-conditioned embeddings or conditional/gated SAE encoders only after the selection-stage contextual baseline is established. See `docs/sae_conditioning_options.md`.
