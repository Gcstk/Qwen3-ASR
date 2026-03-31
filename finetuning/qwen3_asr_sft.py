# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Qwen3-ASR 的基础 SFT 训练脚本。

这份脚本的目标不是“从零训练一个 ASR”，而是：
1. 复用已经训练好的 Qwen3-ASR 底座；
2. 把 JSONL 里的音频-文本对整理成模型当前推理协议需要的格式；
3. 用 Hugging Face Trainer 做标准监督微调。

如果你是 Transformers 新手，建议先抓住下面这条主线：
    原始样本 -> 生成 prefix_text -> collator 里读音频并编码 -> 构造 labels -> Trainer 调用 model.forward 训练
"""
import argparse
import os
import re
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import librosa
import torch
from datasets import load_dataset
from qwen_asr import Qwen3ASRModel
from transformers import (GenerationConfig, Trainer, TrainerCallback,
                          TrainingArguments)


def patch_outer_forward(model):
    """
    给最外层 model 动态补一个 Trainer 能直接调用的 `forward`。

    背景：
    - 当前 Qwen3-ASR 的真实训练逻辑在 `model.thinker.forward(...)` 里。
    - 但 Hugging Face Trainer 默认希望传入的最外层对象本身就有一个兼容的
      `forward(input_ids, attention_mask, input_features, labels, ...)`。
    - 所以这里做一个轻量“适配层”，把外层的 forward 转发到 thinker.forward。

    为什么不直接改源码类定义：
    - 这样写对当前仓库侵入更小；
    - 训练脚本可以独立控制，不影响常规推理路径。
    """
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
    """扫描 output_dir，返回步数最大的 checkpoint 目录。"""
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
    """训练阶段统一把音频读成 16k 单声道波形。"""
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


def build_prefix_messages(prompt: str, audio_array):
    """
    按 Qwen3-ASR 当前 chat template 需要的格式拼消息。

    这里的结构和推理时一致：
    - system: 放任务说明 / prompt
    - user: 放音频
    """
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]


def make_preprocess_fn_prefix_only(processor):
    """
    只预先生成 prefix_text，不在 map 阶段读取音频。

    这样设计有两个好处：
    1. `datasets.map` 阶段只做轻量文本处理，避免提前把整份音频读进内存；
    2. 真正的音频读取放到 collator，能和 batch 对齐，逻辑更接近训练时的真实输入。

    prefix_text 的作用：
    - 它是“系统提示 + 音频占位 + assistant 起始位”的文本前缀；
    - 后面会再把 target 文本接上，形成完整监督目标。
    """
    def _preprocess(ex: Dict[str, Any]) -> Dict[str, Any]:
        prompt = ex.get("prompt", "")
        dummy_audio = None
        prefix_msgs = build_prefix_messages(prompt, dummy_audio)
        prefix_text = processor.apply_chat_template(
            [prefix_msgs], add_generation_prompt=True, tokenize=False
        )[0]
        return {
            "prompt": prompt,
            "audio": ex["audio"],
            "target": ex["text"],
            "prefix_text": prefix_text,
        }

    return _preprocess


@dataclass
class DataCollatorForQwen3ASRFinetuning:
    processor: Any
    sampling_rate: int = 16000

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        把一批原始样本整理成 Trainer 可直接喂给模型的 tensor。

        这是整个脚本最关键的部分，核心做了三件事：
        1. 读取一批音频；
        2. 构造 `full_text = prefix + target + eos`；
        3. 构造 labels，只监督 target，不监督 prefix。

        重点理解 label masking：
        - prefix 是输入条件，不是模型要学习“复述”的内容；
        - 所以前缀部分全部置为 -100，让 loss 只在 target 上计算；
        - pad token 同样置为 -100，避免 padding 位置参与 loss。
        """
        audio_paths = [f["audio"] for f in features]
        prefix_texts = [f["prefix_text"] for f in features]
        targets = [f["target"] for f in features]

        eos = self.processor.tokenizer.eos_token or ""
        # 完整监督目标：prompt 前缀 + 目标输出 + eos
        full_texts = [pfx + tgt + eos for pfx, tgt in zip(prefix_texts, targets)]
        audios = [load_audio(p, sr=self.sampling_rate) for p in audio_paths]

        # full_inputs 用来真正训练；prefix_inputs 只用来算前缀长度，
        # 进而决定哪些 token 应该在 labels 里被 mask 掉。
        full_inputs = self.processor(
            text=full_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        prefix_inputs = self.processor(
            text=prefix_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()
        labels = full_inputs["input_ids"].clone()
        for i, pl in enumerate(prefix_lens):
            # prefix 区域不参与 loss
            labels[i, :pl] = -100

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            # padding 区域同样不参与 loss
            labels[labels == pad_id] = -100

        full_inputs["labels"] = labels
        return full_inputs


class CastFloatInputsTrainer(Trainer):
    """
    让浮点输入张量自动对齐到模型 dtype。

    为什么要做这个小覆盖：
    - processor 输出的音频特征通常是 float32；
    - 但模型训练时可能跑在 fp16 / bf16；
    - 如果不手动对齐，有些环境下会产生额外 cast、显存浪费，甚至 dtype 不一致问题。
    """
    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is not None:
            for k, v in list(inputs.items()):
                if torch.is_tensor(v) and v.is_floating_point():
                    inputs[k] = v.to(dtype=model_dtype)
        return inputs


def copy_required_hf_files_for_qwen_asr(src_dir: str, dst_dir: str):
    """
    把 checkpoint 变成“可直接 from_pretrained 加载”的目录。

    Trainer 默认只保证权重能保存下来；
    但 Qwen3-ASR 的实际推理还依赖 tokenizer / processor / chat template 等文件。
    所以每次保存 checkpoint 后，这里会把必要文件也拷过去。
    """
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
    def __init__(self, base_model_path: str):
        self.base_model_path = base_model_path

    def on_save(self, args: TrainingArguments, state, control, **kwargs):
        """每次保存时，顺手把 checkpoint 补成可推理格式。"""
        if args.process_index != 0:
            return control

        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(ckpt_dir):
            ckpt_dir = kwargs.get("checkpoint", ckpt_dir)

        copy_required_hf_files_for_qwen_asr(self.base_model_path, ckpt_dir)
        return control


def parse_args():
    p = argparse.ArgumentParser("Qwen3-ASR Finetuning")

    # Paths
    p.add_argument("--model_path", type=str, default="Qwen/Qwen3-ASR-1.7B")
    p.add_argument("--train_file", type=str, default="train.jsonl")
    p.add_argument("--eval_file", type=str, default="")
    p.add_argument("--output_dir", type=str, default="./qwen3-asr-finetuning-out")

    # Audio
    p.add_argument("--sr", type=int, default=16000)

    # Train hyper-params
    # 这里直接暴露的是最常用的一组参数。
    # 当前脚本没有直接接入 DeepSpeed config，也没有用 HfArgumentParser，
    # 所以暂时不能像官方 Trainer CLI 那样直接传 `--deepspeed ds_config.json`。
    # 如果后续要接 DeepSpeed，推荐把这里改造成 dataclass + HfArgumentParser。
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_acc", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--lr_scheduler_type", type=str, default="linear")
    p.add_argument("--warmup_ratio", type=float, default=0.02)

    # DataLoader
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", type=int, default=1)
    p.add_argument("--persistent_workers", type=int, default=1)
    p.add_argument("--prefetch_factor", type=int, default=2)

    # Save
    p.add_argument("--save_strategy", type=str, default="steps")
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=5)

    # Resume
    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--resume", type=int, default=0)

    return p.parse_args()


def main():
    args_cli = parse_args()

    if not args_cli.train_file:
        raise ValueError("TRAIN_FILE is required (json/jsonl). Needs fields: audio, text, optional prompt")

    # bf16 通常在较新的 GPU 上更稳；如果硬件不支持，再退回 fp16。
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    asr_wrapper = Qwen3ASRModel.from_pretrained(
        args_cli.model_path,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map=None,
    )
    model = asr_wrapper.model
    processor = asr_wrapper.processor

    # 让 Trainer 可以直接调用 model.forward(...)
    patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)

    # 训练数据只要求 json/jsonl，字段最核心的是 audio 和 text。
    raw_ds = load_dataset(
        "json",
        data_files={
            "train": args_cli.train_file,
            **({"validation": args_cli.eval_file} if args_cli.eval_file else {}),
        },
    )
    ds = raw_ds.map(make_preprocess_fn_prefix_only(processor), num_proc=1)

    # 保留训练真正需要的列，其他字段都删掉，避免 Trainer / Dataset 携带无用信息。
    keep = {"prompt", "audio", "target", "prefix_text"}
    for split in ds.keys():
        drop = [c for c in ds[split].column_names if c not in keep]
        if drop:
            ds[split] = ds[split].remove_columns(drop)

    collator = DataCollatorForQwen3ASRFinetuning(processor=processor, sampling_rate=args_cli.sr)

    # TrainingArguments 是 Transformers Trainer 的“训练总开关”。
    # 你可以把它理解成：
    # - batch / lr / epoch 这些是优化器相关参数；
    # - dataloader_* 是数据读取相关参数；
    # - save_* / eval_* 是训练过程控制参数。
    training_args = TrainingArguments(
        output_dir=args_cli.output_dir,
        per_device_train_batch_size=args_cli.batch_size,
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
        eval_strategy="steps",
        eval_steps=args_cli.save_steps,
        do_eval=bool(args_cli.eval_file),
        bf16=use_bf16,
        fp16=not use_bf16,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to="none",
    )

    # tokenizer 传给 Trainer 主要是为了保存 / 日志等辅助能力，
    # 真正把样本变成 tensor 的工作是由上面的 collator 完成。
    trainer = CastFloatInputsTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation", None),
        data_collator=collator,
        tokenizer=processor.tokenizer,
        callbacks=[MakeEveryCheckpointInferableCallback(base_model_path=args_cli.model_path)],
    )

    resume_from = (args_cli.resume_from or "").strip()
    if not resume_from and args_cli.resume == 1:
        resume_from = find_latest_checkpoint(training_args.output_dir) or ""

    if resume_from:
        # 这里走的是 Trainer 自带的断点恢复逻辑，会继续使用 optimizer/scheduler 状态。
        if trainer.args.process_index == 0:
            print(f"[resume] resume_from_checkpoint = {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()


if __name__ == "__main__":
    main()
