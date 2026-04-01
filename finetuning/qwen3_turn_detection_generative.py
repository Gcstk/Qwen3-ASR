# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import librosa
import torch
from datasets import load_dataset
from transformers import EarlyStoppingCallback, GenerationConfig, Trainer, TrainerCallback, TrainingArguments, set_seed

from qwen_asr.turn_detection import (
    DEFAULT_TURN_DETECTION_PROMPT,
    Qwen3GenerativeTurnDetector,
    TURN_LABELS,
)
from qwen_asr.turn_detection.qwen3_turn_detector import build_turn_detection_prompt_text


def patch_outer_forward(model):
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return

    if not hasattr(model, "thinker") or not hasattr(model.thinker, "forward"):
        raise RuntimeError(
            "Cannot patch forward: model has no `.thinker.forward`. "
            "Your qwen3_asr model may be incompatible."
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_features=None,
        feature_attention_mask=None,
        labels=None,
        **kwargs,
    ):
        return self.thinker.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels,
            **kwargs,
        )

    cls.forward = forward
    cls._forward_patched = True


_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    if not output_dir or not os.path.isdir(output_dir):
        return None
    best_step = None
    best_path = None
    for name in os.listdir(output_dir):
        m = _CKPT_RE.match(name)
        if not m:
            continue
        step = int(m.group(1))
        path = os.path.join(output_dir, name)
        if os.path.isdir(path) and (best_step is None or step > best_step):
            best_step = step
            best_path = path
    return best_path


def normalize_label(label: str) -> str:
    s = str(label).strip().lower()
    if s not in TURN_LABELS:
        raise ValueError(f"Unsupported label={label!r}. Supported labels: {TURN_LABELS}")
    return s


def load_audio(path: str, sr: int = 16000):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


def slice_candidate_window(
    wav,
    sr: int,
    cut_time_ms: Optional[float],
    left_context_ms: float,
    right_context_ms: float,
):
    if cut_time_ms is None:
        return wav
    cut_sample = int(round(float(cut_time_ms) * sr / 1000.0))
    left_samples = int(round(float(left_context_ms) * sr / 1000.0))
    right_samples = int(round(float(right_context_ms) * sr / 1000.0))
    start = max(0, cut_sample - max(0, left_samples))
    end = min(len(wav), cut_sample + max(0, right_samples))
    if end <= start:
        return wav
    return wav[start:end]


def make_preprocess_fn(detector: Qwen3GenerativeTurnDetector, default_prompt: str):
    def _preprocess(ex: Dict[str, Any]) -> Dict[str, Any]:
        prompt = ex.get("prompt", "") or default_prompt
        label = normalize_label(ex["label"])
        prompt_text = build_turn_detection_prompt_text(detector.processor, prompt)
        return {
            "audio": ex["audio"],
            "target_label": label,
            "prompt_text": prompt_text,
            "cut_time_ms": ex.get("cut_time_ms", None),
            "left_context_ms": ex.get("left_context_ms", None),
            "right_context_ms": ex.get("right_context_ms", None),
        }

    return _preprocess


@dataclass
class DataCollatorForQwen3TurnDetectionGenerative:
    processor: Any
    sampling_rate: int = 16000
    default_left_context_ms: float = 2000.0
    default_right_context_ms: float = 600.0

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        wavs = []
        prompt_texts = []
        targets = []
        for feature in features:
            wav = load_audio(feature["audio"], sr=self.sampling_rate)
            wav = slice_candidate_window(
                wav=wav,
                sr=self.sampling_rate,
                cut_time_ms=feature.get("cut_time_ms"),
                left_context_ms=self.default_left_context_ms
                if feature.get("left_context_ms") in (None, "")
                else float(feature["left_context_ms"]),
                right_context_ms=self.default_right_context_ms
                if feature.get("right_context_ms") in (None, "")
                else float(feature["right_context_ms"]),
            )
            wavs.append(wav)
            prompt_texts.append(feature["prompt_text"])
            targets.append(feature["target_label"])

        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [prompt + target + eos for prompt, target in zip(prompt_texts, targets)]

        full_inputs = self.processor(
            text=full_texts,
            audio=wavs,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        prefix_inputs = self.processor(
            text=prompt_texts,
            audio=wavs,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()
        labels = full_inputs["input_ids"].clone()
        for i, prefix_len in enumerate(prefix_lens):
            labels[i, : int(prefix_len)] = -100

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        full_inputs["labels"] = labels
        return full_inputs


class CastFloatInputsTrainer(Trainer):
    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is not None:
            for k, v in list(inputs.items()):
                if torch.is_tensor(v) and v.is_floating_point():
                    inputs[k] = v.to(dtype=model_dtype)
        return inputs


def copy_required_hf_files_for_qwen_asr(src_dir: str, dst_dir: str):
    os.makedirs(dst_dir, exist_ok=True)
    required = [
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "chat_template.json",
        "merges.txt",
        "vocab.json",
    ]
    for fn in required:
        src = os.path.join(src_dir, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, fn))


class MakeEveryCheckpointInferableCallback(TrainerCallback):
    def __init__(self, base_model_path: str, prompt: str):
        self.base_model_path = base_model_path
        self.prompt = prompt

    def _write_metadata(self, path: str):
        payload = {
            "base_model_path": self.base_model_path,
            "prompt": self.prompt,
            "labels": list(TURN_LABELS),
        }
        with open(os.path.join(path, "turn_detection_generative_config.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def on_save(self, args: TrainingArguments, state, control, **kwargs):
        if args.process_index != 0:
            return control
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.isdir(ckpt_dir):
            copy_required_hf_files_for_qwen_asr(self.base_model_path, ckpt_dir)
            self._write_metadata(ckpt_dir)
        return control

    def on_train_end(self, args: TrainingArguments, state, control, **kwargs):
        if args.process_index != 0:
            return control
        if os.path.isdir(args.output_dir):
            copy_required_hf_files_for_qwen_asr(self.base_model_path, args.output_dir)
            self._write_metadata(args.output_dir)
        return control


def parse_args():
    p = argparse.ArgumentParser("Qwen3-ASR Generative Turn Detection Finetuning")
    p.add_argument("--model_path", type=str, default="Qwen/Qwen3-ASR-0.6B")
    p.add_argument("--train_file", type=str, default="train_turn_detection.jsonl")
    p.add_argument("--eval_file", type=str, default="")
    p.add_argument("--output_dir", type=str, default="./qwen3-turn-detection-generative-out")
    p.add_argument("--default_prompt", type=str, default=DEFAULT_TURN_DETECTION_PROMPT)

    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--default_left_context_ms", type=float, default=2000.0)
    p.add_argument("--default_right_context_ms", type=float, default=600.0)

    p.add_argument("--freeze_audio_tower", type=int, default=1)
    p.add_argument("--freeze_text_model", type=int, default=1)
    p.add_argument("--unfreeze_last_n_layers", type=int, default=0)
    p.add_argument("--train_lm_head", type=int, default=1)

    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_acc", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--lr_scheduler_type", type=str, default="linear")
    p.add_argument("--warmup_ratio", type=float, default=0.05)

    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", type=int, default=1)
    p.add_argument("--persistent_workers", type=int, default=1)
    p.add_argument("--prefetch_factor", type=int, default=2)

    p.add_argument("--save_strategy", type=str, default="steps")
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=3)

    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--resume", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--load_best_model_at_end", type=int, default=1)
    p.add_argument("--metric_for_best_model", type=str, default="eval_loss")
    p.add_argument("--greater_is_better", type=int, default=0)
    p.add_argument("--early_stopping_patience", type=int, default=0)
    return p.parse_args()


def main():
    args_cli = parse_args()
    if not args_cli.train_file:
        raise ValueError("TRAIN_FILE is required. Expected fields: audio, label, optional prompt/cut_time_ms.")

    set_seed(int(args_cli.seed))
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8

    detector = Qwen3GenerativeTurnDetector.from_pretrained(
        args_cli.model_path,
        prompt=args_cli.default_prompt,
        max_new_tokens=4,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map=None,
    )
    detector.set_trainable_parts(
        freeze_audio_tower=(args_cli.freeze_audio_tower == 1),
        freeze_text_model=(args_cli.freeze_text_model == 1),
        unfreeze_last_n_layers=args_cli.unfreeze_last_n_layers,
        train_lm_head=(args_cli.train_lm_head == 1),
    )

    model = detector.model
    processor = detector.processor
    patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)

    raw_ds = load_dataset(
        "json",
        data_files={
            "train": args_cli.train_file,
            **({"validation": args_cli.eval_file} if args_cli.eval_file else {}),
        },
    )
    ds = raw_ds.map(make_preprocess_fn(detector, args_cli.default_prompt), num_proc=1)

    keep = {
        "audio",
        "target_label",
        "prompt_text",
        "cut_time_ms",
        "left_context_ms",
        "right_context_ms",
    }
    for split in ds.keys():
        drop = [c for c in ds[split].column_names if c not in keep]
        if drop:
            ds[split] = ds[split].remove_columns(drop)

    collator = DataCollatorForQwen3TurnDetectionGenerative(
        processor=processor,
        sampling_rate=args_cli.sr,
        default_left_context_ms=args_cli.default_left_context_ms,
        default_right_context_ms=args_cli.default_right_context_ms,
    )

    training_args = TrainingArguments(
        output_dir=args_cli.output_dir,
        per_device_train_batch_size=args_cli.batch_size,
        per_device_eval_batch_size=args_cli.batch_size,
        gradient_accumulation_steps=args_cli.grad_acc,
        learning_rate=args_cli.lr,
        num_train_epochs=args_cli.epochs,
        logging_steps=args_cli.log_steps,
        lr_scheduler_type=args_cli.lr_scheduler_type,
        warmup_ratio=args_cli.warmup_ratio,
        dataloader_num_workers=args_cli.num_workers,
        dataloader_pin_memory=(args_cli.pin_memory == 1),
        dataloader_persistent_workers=(args_cli.persistent_workers == 1),
        dataloader_prefetch_factor=args_cli.prefetch_factor if args_cli.num_workers > 0 else None,
        save_strategy=args_cli.save_strategy,
        save_steps=args_cli.save_steps,
        save_total_limit=args_cli.save_total_limit,
        save_safetensors=True,
        eval_strategy="steps" if args_cli.eval_file else "no",
        eval_steps=args_cli.save_steps if args_cli.eval_file else None,
        do_eval=bool(args_cli.eval_file),
        bf16=use_bf16,
        fp16=not use_bf16,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to="none",
        label_names=["labels"],
        seed=int(args_cli.seed),
        load_best_model_at_end=bool(args_cli.load_best_model_at_end) and bool(args_cli.eval_file),
        metric_for_best_model=args_cli.metric_for_best_model if args_cli.eval_file else None,
        greater_is_better=bool(args_cli.greater_is_better) if args_cli.eval_file else None,
    )

    callbacks: List[TrainerCallback] = [
        MakeEveryCheckpointInferableCallback(args_cli.model_path, args_cli.default_prompt)
    ]
    if args_cli.eval_file and int(args_cli.early_stopping_patience) > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=int(args_cli.early_stopping_patience)))

    trainer = CastFloatInputsTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation", None),
        data_collator=collator,
        tokenizer=processor.tokenizer,
        callbacks=callbacks,
    )

    resume_from = (args_cli.resume_from or "").strip()
    if not resume_from and args_cli.resume == 1:
        resume_from = find_latest_checkpoint(training_args.output_dir) or ""

    if resume_from:
        if trainer.args.process_index == 0:
            print(f"[resume] resume_from_checkpoint = {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()

    if trainer.args.process_index == 0:
        trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
