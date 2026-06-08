# WIMHF Cached Dataset Inventory

Last updated: 2026-06-06.

Raw files are cached under `data/wimhf/raw/`.

## Summary

- `chatbot_arena.json`: 27,463 rows. Has model/source/language metadata.
- `community_alignment.json`: 204,390 rows. Has conversation, language, annotator demographics, and feedback metadata.
- `hh_rlhf.json`: 67,266 rows. Mostly prompt/response/label plus response-derived descriptors.
- `pku.json`: 31,698 rows. Similar to HH-RLHF; mostly prompt/response/label plus response-derived descriptors.
- `prism.json`: 25,918 rows. Has conversation/user/type/turn/model metadata.
- `reddit.json`: 31,425 rows. Has post/domain metadata.
- `tulu.json`: 34,720 rows. Minimal metadata.

## Columns

`chatbot_arena.json`: `prompt`, `response_A`, `response_B`, `label`, `model_A`, `model_B`, `original_source`, `language`, `winner`, `subjectivity_A`, `subjectivity_B`, `length_delta`.

`community_alignment.json`: `conversation_id`, `assigned_lang`, `annotator_id`, `first_turn_feedback`, `second_turn_feedback`, `third_turn_feedback`, `fourth_turn_feedback`, `annotator_age`, `annotator_gender`, `annotator_education_level`, `annotator_political`, `annotator_ethnicity`, `annotator_country`, `is_pregenerated_first_prompt`, `in_balanced_subset`, `in_balanced_subset_10`, `turn_id`, `prompt`, `response_A`, `response_B`, `label`, `length_delta`.

`hh_rlhf.json`: `prompt`, `response_A`, `response_B`, `label`, `length_delta`, `subjectivity_A`, `subjectivity_B`, `llm_description_A`, `llm_description_B`, `llm_pairwise_description`.

`pku.json`: `prompt`, `response_A`, `response_B`, `label`, `length_delta`, `subjectivity_A`, `subjectivity_B`, `llm_description_A`, `llm_description_B`, `llm_pairwise_description`.

`prism.json`: `prompt`, `response_A`, `response_B`, `label`, `score_delta`, `length_delta`, `conversation_id`, `user_id`, `conversation_type`, `turn`, `model_A`, `model_B`, `subjectivity_A`, `subjectivity_B`, `transcript_A`, `transcript_B`.

`reddit.json`: `prompt`, `response_A`, `response_B`, `label`, `length_delta`, `subjectivity_A`, `subjectivity_B`, `llm_description_A`, `llm_description_B`, `llm_pairwise_description`, `post_id`, `domain`.

`tulu.json`: `prompt`, `response_A`, `response_B`, `label`, `subjectivity_A`, `subjectivity_B`, `length_delta`.

## Current Read

HH-RLHF is the first replication target, not the strongest conditional-analysis target. PRISM, CommunityAlign, ChatbotArena, and Reddit expose more plausible non-circular context fields.
