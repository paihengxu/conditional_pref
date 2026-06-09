# Conditional Preference Discovery for Post-Training Data

Brief research note for later continuation. This is for ideation, not a finalized project plan.

## Working Idea

Use context-aware discovery on existing preference datasets to find preference signals that are hidden, diluted, or reversed in global analysis.

The intended sequence is:

1. Reanalyze public preference datasets with pre-specified contexts.
2. Check whether context-aware features differ meaningfully from global WIMHF-style features.
3. Only if the reanalysis finds clear signal, study whether the conditional signals affect post-training decisions or outcomes.

Do not start with a synthetic preference benchmark unless a later reviewer-facing need becomes clear.

## What Is Agreed

- The target is preference datasets and LLM post-training.
- The central question is conditional preference structure, not generic preference-feature discovery.
- Contexts should be chosen in a principled way before looking at preference effects.
- The work is worth pursuing only if it has credible ICLR-level potential.

## TBD

- Exact datasets.
- Exact contexts.
- Exact post-training experiment.
- Whether any synthetic or controlled validation is needed later.

## Context Selection

A context is a good candidate only if it is:

- preference-relevant;
- available before post-training data curation or training;
- actionable for filtering, reweighting, stratified modeling, or evaluation;
- sufficiently supported by the data;
- not circularly derived from the same response features used to explain preferences.

Best first context families:

- prompt task category;
- safety or policy domain;
- dataset source or collection pipeline.

Other possible contexts:

- response generator, if metadata exists;
- annotator or user identity/group, if reliable metadata exists;
- time, batch, or collection wave, if collection conditions changed.

### Context candidate: Values-in-the-Wild Taxonomy

Paper: "Values in the Wild: Discovering and Analyzing Values in Real-World Language Model Interactions" (`https://arxiv.org/html/2504.15236v1`).

Current judgment: the values taxonomy is a promising candidate source for value-related contexts, but it should not be treated as the default primary context yet.

Use cautiously because:

- the labels are LLM-extracted, not original preference-dataset metadata;
- response-derived value labels may be circular with preference-feature discovery;
- fine-grained values may be sparse;
- top-level values may be too coarse;
- the taxonomy comes from Claude.ai interactions, not necessarily public preference-pair datasets.

Safer current position:

> Values-in-the-Wild may help define secondary value-related contexts, but it must first pass non-circularity, support, and actionability checks on the datasets we analyze.

## Relevant papers

WIMHF: What’s In My Human Feedback? Learning Interpretable Descriptions of Preference Data (https://openreview.net/forum?id=sC6A1bFDUt)

### Differentiation From WIMHF

WIMHF already uses SAEs to discover interpretable preference features and estimate global expressed preferences.

This idea must instead show that global preference analysis misses context-dependent preference regimes, and that those regimes matter for post-training data curation, reward modeling, DPO, SFT, or evaluation.

## First Feasibility Check

Minimum useful outputs:

- dataset/context inventory;
- support table for candidate contexts;
- global vs. context-aware top features;
- examples of hidden, diluted, or sign-reversing preferences;
- held-out preference prediction by context.

Continue only if context-aware analysis reveals qualitatively different and predictive or actionable signals.

## Other Ideas

These are promising extensions, but should stay secondary until the basic conditional-preference check shows clear signal.

### Distributional WIMHF

Extend WIMHF from global expressed preferences to distributions of preference weights over interpretable SAE features.

Core question:

> For each interpretable preference feature, is it a consensus preference, a context-dependent preference, or a genuinely distributional preference across annotators or groups?

Useful model:

```text
Pr(y_i = 1) = sigmoid(alpha + beta_j z_ij + beta_j,g z_ij * group_i + controls)
```

or:

```text
beta_j,g ~ Normal(beta_j, tau_j)
```

where `tau_j` measures disagreement over feature `j`.

Three possible group types:

- metadata groups: country, language, politics, age, gender, platform, subreddit, dataset source;
- task/context groups: prompt category, safety domain, value domain, subjective vs. objective task;
- latent annotator groups: cluster annotators by estimated feature-weight vectors, with strict train/test splitting.

Good first datasets: PRISM, Community Alignment, Reddit/SHP, and safety datasets with policy-domain labels.

Main risks: group sparsity, noisy effects, novelty beyond WIMHF personalization, and sensitivity around demographic comparisons.

Differentiation from WIMHF:

> We learn interpretable axes of disagreement from response-pair data, then estimate the full distribution of preferences over those axes.

### GRPO / On-Policy Extension

High-upside but likely more expensive and harder to validate than offline distributional analysis.

Concerns:

- full GRPO needs repeated rollouts, reward scoring, policy updates, feature extraction, checkpoint evaluation, and seeds;
- reward, advantage, sampling, or filtering interventions make attribution difficult.

Lower-cost path:

> Use conditional SAE features to audit what post-training objectives would reinforce, before committing to full on-policy intervention.

Staged path:

1. Offline/logged-rollout audit: correlate features with reward, normalized advantage, or selected updates.
2. Counterfactual intervention: rerank, filter, relabel, or reweight examples by context-feature interactions.
3. Small GRPO demo only after offline evidence is strong.

The strongest eventual claim would be:

> Global reward or advantage analysis misses context-specific features that on-policy optimization reinforces; conditioning on context reveals reward-hacking or minority-preference failures and enables targeted fixes.
