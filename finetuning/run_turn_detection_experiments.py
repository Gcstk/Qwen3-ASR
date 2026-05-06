# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
import shlex
import subprocess
from typing import Dict, List


def parse_args():
    p = argparse.ArgumentParser("Run Qwen3-ASR turn detection experiment matrix")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--train_file", type=str, required=True)
    p.add_argument("--eval_file", type=str, required=True)
    p.add_argument("--test_file", type=str, default="")
    p.add_argument("--output_root", type=str, default="./turn_detection_experiments")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_acc", type=int, default=2)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr_classifier", type=float, default=1e-4)
    p.add_argument("--lr_generative", type=float, default=5e-5)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--torchrun_nproc", type=int, default=0)
    p.add_argument("--execute", type=int, default=0)
    return p.parse_args()


def build_train_commands(args) -> List[Dict[str, str]]:
    presets = [
        {
            "name": "A1_classifier_head_only",
            "script": "finetuning/qwen3_turn_detection.py",
            "extra": "--freeze_audio_tower 1 --freeze_text_model 1 --unfreeze_last_n_layers 0 --pooling last_token",
            "lr": args.lr_classifier,
        },
        {
            "name": "A2_classifier_top4",
            "script": "finetuning/qwen3_turn_detection.py",
            "extra": "--freeze_audio_tower 1 --freeze_text_model 1 --unfreeze_last_n_layers 4 --pooling last_token",
            "lr": args.lr_classifier,
        },
        {
            "name": "A3_classifier_top8",
            "script": "finetuning/qwen3_turn_detection.py",
            "extra": "--freeze_audio_tower 1 --freeze_text_model 1 --unfreeze_last_n_layers 8 --pooling last_token",
            "lr": args.lr_classifier,
        },
        {
            "name": "B1_generative_lm_head",
            "script": "finetuning/qwen3_turn_detection_generative.py",
            "extra": "--freeze_audio_tower 1 --freeze_text_model 1 --unfreeze_last_n_layers 0 --train_lm_head 1",
            "lr": args.lr_generative,
        },
        {
            "name": "B2_generative_top4",
            "script": "finetuning/qwen3_turn_detection_generative.py",
            "extra": "--freeze_audio_tower 1 --freeze_text_model 1 --unfreeze_last_n_layers 4 --train_lm_head 1",
            "lr": args.lr_generative,
        },
    ]

    commands: List[Dict[str, str]] = []
    launcher_prefix = ""
    use_torchrun = int(args.torchrun_nproc) > 0
    if use_torchrun:
        launcher_prefix = f"torchrun --nproc_per_node={int(args.torchrun_nproc)} "

    for preset in presets:
        for seed in args.seeds:
            out_dir = os.path.join(args.output_root, preset["name"], f"seed_{seed}")
            train_invocation = f"{launcher_prefix}{preset['script']}" if use_torchrun else f"python {preset['script']}"
            train_cmd = (
                f"{train_invocation} "
                f"--model_path {shlex.quote(args.model_path)} "
                f"--train_file {shlex.quote(args.train_file)} "
                f"--eval_file {shlex.quote(args.eval_file)} "
                f"--output_dir {shlex.quote(out_dir)} "
                f"--batch_size {int(args.batch_size)} "
                f"--grad_acc {int(args.grad_acc)} "
                f"--epochs {float(args.epochs)} "
                f"--lr {float(preset['lr'])} "
                f"--save_steps {int(args.save_steps)} "
                f"--early_stopping_patience 3 "
                f"--seed {int(seed)} "
                f"{preset['extra']}"
            )
            eval_cmd = (
                f"python finetuning/eval_qwen3_turn_detection.py "
                f"--mode {'classifier' if preset['name'].startswith('A') else 'generative'} "
                f"--model_path {shlex.quote(out_dir)} "
                f"--eval_file {shlex.quote(args.eval_file)} "
                f"{f'--test_file {shlex.quote(args.test_file)} ' if args.test_file else ''}"
                f"--output_json {shlex.quote(os.path.join(out_dir, 'eval_report.json'))} "
                f"--predictions_jsonl {shlex.quote(os.path.join(out_dir, 'predictions.jsonl'))}"
            )
            commands.append(
                {
                    "name": preset["name"],
                    "seed": str(seed),
                    "train": train_cmd,
                    "eval": eval_cmd,
                }
            )
    return commands


def main():
    args = parse_args()
    commands = build_train_commands(args)
    for item in commands:
        print(f"\n[{item['name']}][seed={item['seed']}]")
        print(item["train"])
        print(item["eval"])
        if int(args.execute) == 1:
            subprocess.run(item["train"], shell=True, check=True)
            subprocess.run(item["eval"], shell=True, check=True)


if __name__ == "__main__":
    main()
