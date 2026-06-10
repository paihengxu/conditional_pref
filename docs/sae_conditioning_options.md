# SAE Conditioning Options

Short notes from the June 9 discussion.

## Current WIMHF Issue

Paper-style WIMHF trains the SAE on response-pair embedding differences:

```text
d = emb(response_A) - emb(response_B)
z = SAE(d)
```

The interpretation prompt includes the original user prompt as `CONTEXT`, but the SAE latent only sees prompt information indirectly through the responses unless the embedded text includes the prompt.

## Option 1: Instructed Prompt-Response Embeddings

Minimal change:

```text
d = emb(instruction, prompt, response_A) - emb(instruction, prompt, response_B)
z = SAE(d)
```

This follows the WIMHF README update. It is easy to run and keeps the WIMHF objective unchanged, but prompt-only information can still cancel in the difference; it mainly helps through nonlinear prompt-response interactions in the embedder.

## Option 2: Conditional Or Gated SAE

Change the representation learner itself. Let `c` be prompt/context representation.

Simple conditional encoder:

```text
z = TopK(W_d d + W_c c + b)
```

Cleaner gated encoder:

```text
a = W_d d
g = sigmoid(W_g c)
z = TopK(g * a + b)
```

The gated version is preferable as a first conditional-SAE design because context changes sensitivity to response-difference features without directly creating features from context alone. Keep the decoder global at first for interpretability.

## Option 3: Context-Dependent Feature Weights

Keep global SAE features, then model context-specific preference effects:

```text
Pr(A wins) = sigmoid(beta^T z + gamma^T (c * z) + controls)
```

This identifies hidden, diluted, or sign-reversing expressed preferences without changing the SAE feature basis. It is the closest fit to the current project goal.

## Relation To Prior `text_diff`

The relevant prior method is HypotheSAEs selection in:

```text
../text_diff/repos/HypotheSAEs/hypothesaes/select_neurons.py
```

`interaction-lasso` / `covariate-lasso` is option 3: select neurons using main effects plus context-feature interactions. `demeaned-reweighted-lasso` residualizes activations and targets within context strata, then reweights strata; this targets nuisance-confound adjustment when feature effects have consistent sign across strata.

See also:

```text
../text_diff/docs/outlines.md
```

Practical next step: start with option 3 on trusted WIMHF features, using HypotheSAEs methods as references. Move to option 2 only if global features cannot express the conditional signal.
