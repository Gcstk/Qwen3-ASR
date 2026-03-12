# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import librosa
import numpy as np
import torch
from datasets import load_dataset
from transformers import Trainer, TrainerCallback, TrainingArguments

from qwen_asr.turn_detection import DEFAULT_TURN_DETECTION_PROMPT, Qwen3TurnDetector
from qwen_asr.turn_detection.qwen3_turn_detector import (
    TURN_LABELS,
    build_turn_detection_prompt_text,
)

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


def load_audio(path: str, sr: int = 16000):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


def slice_candidate_window(
    wav: np.ndarray,
    sr: int,
    cut_time_ms: Optional[float],
    left_context_ms: float,
    right_context_ms: float,
) -> np.ndarray:
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


def normalize_label(label: str) -> str:
    s = str(label).strip().lower()
    if s not in TURN_LABELS:
        raise ValueError(f"Unsupported label={label!r}. Supported labels: {TURN_LABELS}")
    return s


def make_preprocess_fn(processor, default_prompt: str):
    def _preprocess(ex: Dict[str, Any]) -> Dict[str, Any]:
        prompt = ex.get("prompt", "") or default_prompt
        label = normalize_label(ex["label"])
        prompt_text = build_turn_detection_prompt_text(processor, prompt)
        return {
            "audio": ex["audio"],
            "label_id": TURN_LABELS.index(label),
            "prompt_text": prompt_text,
            "cut_time_ms": ex.get("cut_time_ms", None),
            "left_context_ms": ex.get("left_context_ms", None),
            "right_context_ms": ex.get("right_context_ms", None),
        }

    return _preprocess


@dataclass
class DataCollatorForQwen3TurnDetection:
    processor: Any
    sampling_rate: int = 16000
    default_left_context_ms: float = 2000.0
    default_right_context_ms: float = 600.0

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        wavs = []
        prompt_texts = []
        labels = []
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
            labels.append(int(feature["label_id"]))

        inputs = self.processor(
            text=prompt_texts,
            audio=wavs,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        inputs["labels"] = torch.tensor(labels, dtype=torch.long)
        return inputs


class CastFloatInputsTrainer(Trainer):
    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is not None:
            for k, v in list(inputs.items()):
                if torch.is_tensor(v) and v.is_floating_point():
                    inputs[k] = v.to(dtype=model_dtype)
        return inputs


class MakeEveryCheckpointLoadableCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if args.process_index != 0:
            return control
        model = kwargs.get("model", None)
        if model is None:
            return control
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.isdir(ckpt_dir):
            model.save_pretrained(ckpt_dir)
        return control


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    labels = labels.astype(np.int64)
    acc = float((preds == labels).mean())

    def _binary_metrics(pos_id: int):
        tp = int(((preds == pos_id) & (labels == pos_id)).sum())
        fp = int(((preds == pos_id) & (labels != pos_id)).sum())
        fn = int(((preds != pos_id) & (labels == pos_id)).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-8, precision + recall)
        return precision, recall, f1

    complete_precision, complete_recall, complete_f1 = _binary_metrics(TURN_LABELS.index("complete"))
    return {
        "accuracy": acc,
        "complete_precision": complete_precision,
        "complete_recall": complete_recall,
        "complete_f1": complete_f1,
    }


def parse_args():
    p = argparse.ArgumentParser("Qwen3-ASR Turn Detection Finetuning")
    p.add_argument("--model_path", type=str, default="Qwen/Qwen3-ASR-0.6B")
    p.add_argument("--train_file", type=str, default="train_turn_detection.jsonl")
    p.add_argument("--eval_file", type=str, default="")
    p.add_argument("--output_dir", type=str, default="./qwen3-turn-detection-out")
    p.add_argument("--default_prompt", type=str, default=DEFAULT_TURN_DETECTION_PROMPT)

    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--default_left_context_ms", type=float, default=2000.0)
    p.add_argument("--default_right_context_ms", type=float, default=600.0)

    p.add_argument("--freeze_audio_tower", type=int, default=1)
    p.add_argument("--freeze_text_model", type=int, default=1)
    p.add_argument("--unfreeze_last_n_layers", type=int, default=0)
    p.add_argument("--pooling", type=str, default="last_token")

    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_acc", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
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
    return p.parse_args()


def main():
    args_cli = parse_args()
    if not args_cli.train_file:
        raise ValueError("TRAIN_FILE is required. Expected fields: audio, label, optional prompt/cut_time_ms.")

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    model = Qwen3TurnDetector.from_qwen3_asr_pretrained(
        args_cli.model_path,
        prompt=args_cli.default_prompt,
        pooling=args_cli.pooling,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map=None,
    )
    model.set_trainable_parts(
        freeze_audio_tower=(args_cli.freeze_audio_tower == 1),
        freeze_text_model=(args_cli.freeze_text_model == 1),
        unfreeze_last_n_layers=args_cli.unfreeze_last_n_layers,
    )

    raw_ds = load_dataset(
        "json",
        data_files={
            "train": args_cli.train_file,
            **({"validation": args_cli.eval_file} if args_cli.eval_file else {}),
        },
    )
    ds = raw_ds.map(make_preprocess_fn(model.processor, args_cli.default_prompt), num_proc=1)

    keep = {
        "audio",
        "label_id",
        "prompt_text",
        "cut_time_ms",
        "left_context_ms",
        "right_context_ms",
    }
    for split in ds.keys():
        drop = [c for c in ds[split].column_names if c not in keep]
        if drop:
            ds[split] = ds[split].remove_columns(drop)

    collator = DataCollatorForQwen3TurnDetection(
        processor=model.processor,
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
        save_safetensors=False,
        eval_strategy="steps",
        eval_steps=args_cli.save_steps,
        do_eval=bool(args_cli.eval_file),
        bf16=use_bf16,
        fp16=not use_bf16,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to="none",
        label_names=["labels"],
    )

    trainer = CastFloatInputsTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation", None),
        data_collator=collator,
        compute_metrics=compute_metrics if args_cli.eval_file else None,
        callbacks=[MakeEveryCheckpointLoadableCallback()],
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
