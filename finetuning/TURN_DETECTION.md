## Turn Detection Finetuning with Qwen3-ASR

This guide fine-tunes a lightweight **turn detector** on top of Qwen3-ASR. It is intentionally a **binary classifier**:

- `complete`
- `incomplete`

It does **not** train a `waiting` label. In practice, `waiting` is usually better handled by an upper policy / LLM layer after ASR, because cases like "you wait first" are intent-level response-control decisions, not pure speech-boundary decisions.

### 1) Why this setup

This repository's existing training script is designed for generative ASR output such as:

```text
language English<asr_text>Hello world.
```

Turn detection is different:

- it is a boundary classification task
- it should be cheaper than text generation
- it should work after a VAD candidate endpoint

So this script reuses Qwen3-ASR as an audio-text backbone and adds a classifier head on top.

### 2) Install

```bash
pip install -U qwen-asr datasets
```

Optional but recommended:

```bash
pip install -U flash-attn --no-build-isolation
```

### 3) Input JSONL format

Each line must contain:

- `audio`: path to the source audio
- `label`: `complete` or `incomplete`

Optional fields:

- `cut_time_ms`: candidate end-point in milliseconds inside the source audio
- `left_context_ms`: left window around the candidate point
- `right_context_ms`: right window around the candidate point
- `prompt`: task instruction override

If `cut_time_ms` is omitted, the whole audio file is used as the classification clip.

Example:

```jsonl
{"audio":"/data/dialog/utt001.wav","cut_time_ms":2350,"left_context_ms":2000,"right_context_ms":600,"label":"complete"}
{"audio":"/data/dialog/utt002.wav","cut_time_ms":1810,"left_context_ms":2000,"right_context_ms":600,"label":"incomplete"}
{"audio":"/data/dialog/utt003.wav","label":"complete"}
```

### 4) Label definition

`complete`

- the user has finished this utterance at the candidate point
- the system may move on to the next stage
- whether the system should answer immediately is a separate policy decision

`incomplete`

- this pause is only a hesitation, breath, clause boundary, or unfinished continuation
- the system should keep listening

Important:

- phrases like "you wait first" or "don't answer yet" should still be labeled `complete`
- the later policy / LLM layer can map the final ASR text to `hold` / `reply_now`

### 5) Training

Single GPU:

```bash
python finetuning/qwen3_turn_detection.py \
  --model_path Qwen/Qwen3-ASR-0.6B \
  --train_file ./turn_train.jsonl \
  --eval_file ./turn_eval.jsonl \
  --output_dir ./qwen3-turn-detection-out \
  --batch_size 8 \
  --grad_acc 2 \
  --lr 1e-4 \
  --epochs 3 \
  --save_steps 200
```

Multi GPU:

```bash
export CUDA_VISIBLE_DEVICES=0,1

torchrun --nproc_per_node=2 finetuning/qwen3_turn_detection.py \
  --model_path Qwen/Qwen3-ASR-0.6B \
  --train_file ./turn_train.jsonl \
  --eval_file ./turn_eval.jsonl \
  --output_dir ./qwen3-turn-detection-out \
  --batch_size 8 \
  --grad_acc 2 \
  --lr 1e-4 \
  --epochs 3 \
  --save_steps 200
```

### 6) Freezing strategy

Default behavior:

- freeze the Qwen3-ASR audio tower
- freeze the text model
- only train the classifier head

This is the safest first pass when you want to reuse the pretrained audio knowledge without needing large task data.

If classifier-only training is not enough, you can try:

```bash
--freeze_text_model 1 --unfreeze_last_n_layers 4
```

That keeps most of the backbone frozen while allowing the top decoder layers to adapt.

### 7) Quick inference

```python
import torch
from qwen_asr.turn_detection import Qwen3TurnDetector

detector = Qwen3TurnDetector.from_pretrained(
    "./qwen3-turn-detection-out/checkpoint-200",
    dtype=torch.bfloat16,
    device_map="cuda:0",
)

result = detector.predict(
    audio="/data/dialog/utt001.wav",
    cut_time_ms=2350,
    left_context_ms=2000,
    right_context_ms=600,
)

print(result.label, result.complete_prob, result.incomplete_prob)
```

### 8) Recommended system design

Recommended serving stack:

1. VAD proposes a candidate endpoint
2. Turn detector predicts `complete` / `incomplete`
3. If `complete`, ASR text goes to a policy / LLM layer
4. The policy layer decides `reply_now` vs `hold`

This keeps speech-boundary classification and response-policy semantics separate.
