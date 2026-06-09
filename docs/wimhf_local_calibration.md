# WIMHF Local Annotator Calibration

Last updated: 2026-06-09.

This note compares local Qwen annotator runs against the paper and the trusted
`outputs/reproduction/wimhf_exact` GPT-5-mini runs. The paper reference is
`references/Movva et al. - 2026 - What's In My Human Feedback Learning Interpretable Descriptions of Preference Data.txt`.

## Paper Reference

From the paper appendix tables:

- Community Alignment: `31` high-fidelity WIMHF features; `29/31` predict preference.
- PRISM: `23` high-fidelity WIMHF features; `13/23` predict preference.

From Figure 4 OCR/text:

- CommunityAlign SAE AUC: approximately `0.687`.
- PRISM SAE AUC: approximately `0.671`.

## Run Comparison

| Dataset / run | Kept features | Mean fidelity | Full-SAE val AUC | Interpretation |
| --- | ---: | ---: | ---: | --- |
| CommunityAlign paper | `31` | n/a | `0.687` | Reference |
| CommunityAlign exact GPT-5-mini | `31/32` | `0.545` | `0.686` | Matches paper closely |
| CommunityAlign Qwen3-30B | `13/32` | `0.403` | `0.687` | AUC close, feature retention poor |
| CommunityAlign Qwen3.5-27B | `31/32` | `0.748` | `0.688` | Close on retention, themes, and AUC; likely over-generous fidelity calibration |
| PRISM paper | `23` | n/a | `0.671` | Reference |
| PRISM exact GPT-5-mini | `21/32` | `0.376` | `0.671` | Close to paper |
| PRISM Qwen3-30B | `8/32` | `0.361` | `0.665` | AUC close, feature retention poor |
| PRISM Qwen3.5-27B | `22/32` | `0.477` | `0.665` | Close on retention, not closer on full-SAE AUC |

## Takeaways

- The earlier statement that Qwen3-30B was "acceptable for PRISM" should be read narrowly: it was reasonably close on downstream AUC and some top-level themes, but it was not close on WIMHF's high-fidelity feature-retention criterion.
- Qwen3.5-27B is a clear improvement for local annotation when the goal is to reproduce the interpretable WIMHF feature table: CommunityAlign matches exact retention (`31/32`), and PRISM nearly matches exact/paper retention (`22/32` vs exact `21/32`, paper `23`).
- Qwen3.5-27B's mean fidelity is higher than exact on both datasets (`0.748` vs `0.545` for CommunityAlign, `0.477` vs `0.376` for PRISM). Treat this as a calibration difference in the local judge, not evidence that it is strictly better than GPT-5-mini.
- For PRISM, Qwen3.5-27B does not improve the full-SAE validation AUC over Qwen3-30B (`0.665` for both), and both are below exact/paper (`0.671`).
- For future conditional sweeps, Qwen3.5-27B is the better local default than Qwen3-30B if interpretable feature retention matters. Use GPT-5-mini for final validation of selected findings.

## Relevant Outputs

- Exact trusted baseline: `outputs/reproduction/wimhf_exact`.
- Qwen3-30B CommunityAlign: `outputs/reproduction/wimhf_local/wimhf_local_full_community_align_s42_clip_20260607_6984139`.
- Qwen3-30B PRISM: `outputs/reproduction/wimhf_local/wimhf_local_full_prism_s42_tron_20260607_6984140`.
- Qwen3.5-27B CommunityAlign: `outputs/reproduction/wimhf_local/wimhf_local_full_community_align_qwen35_27b_s42_clip_20260609_6986828`.
- Qwen3.5-27B PRISM: `outputs/reproduction/wimhf_local/wimhf_local_full_prism_qwen35_27b_s42_clip_20260609_6986752`.

Note: the Qwen3.5-27B PRISM directory includes failed earlier attempts in its logs, but the resumed child log `6986808.log` completed and produced the result files used above.
