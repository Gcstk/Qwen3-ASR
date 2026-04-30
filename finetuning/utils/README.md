## Finetuning Utilities

This folder contains utility scripts used to prepare or maintain manifests for
training and evaluation.

Current utilities:

- `convert_easy_turn_to_qwen_asr_jsonl.py`: convert Easy-Turn tar shards into Qwen3-ASR JSONL
- `convert_easy_turn_testset_to_qwen_asr_jsonl.py`: convert Easy-Turn test TSV metadata into Qwen3-ASR JSONL
- `convert_qwen_asr_jsonl_remove_language.py`: remove language prediction from already converted JSONL files and optionally rewrite the turn-state output schema / prompt template
- `rewrite_qwen_asr_jsonl_audio_paths.py`: rewrite `audio` path prefixes when moving datasets across servers
