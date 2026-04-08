# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Post-process already converted Qwen3-ASR JSONL data to remove language
# prediction from targets and prompts.
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

from tqdm import tqdm


DEFAULT_PROMPT_NO_LANGUAGE = (
    "请转录音频内容，并严格使用格式：<turn_state><标签><asr_text>转写文本。"
    "其中 <complete> 表示语义完整，<incomplete> 表示语义不完整，<backchannel> 表示简短附和，"
    "<wait> 表示请求暂停或终止对话。"
)


def parse_args():
    p = argparse.ArgumentParser(
        "Remove language prediction from existing Qwen3-ASR JSONL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input_jsonl", type=str, required=True, help="Input Qwen3-ASR JSONL path.")
    p.add_argument("--output_jsonl", type=str, required=True, help="Output JSONL path after normalization.")
    p.add_argument(
        "--prompt_strategy",
        type=str,
        choices=["rewrite", "preserve_if_clean", "empty"],
        default="rewrite",
        help=(
            "`rewrite` replaces every prompt with one stable language-free instruction. "
            "`preserve_if_clean` keeps prompts that already do not mention language, otherwise rewrites them. "
            "`empty` clears the prompt field."
        ),
    )
    p.add_argument(
        "--fixed_prompt",
        type=str,
        default=DEFAULT_PROMPT_NO_LANGUAGE,
        help="Prompt used when --prompt_strategy rewrite or when preserve_if_clean needs fallback rewriting.",
    )
    return p.parse_args()


def log(message: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[remove_language][{ts}] {message}", file=sys.stdout, flush=True)


def strip_language_prefix(text: str) -> Tuple[str, bool]:
    value = text or ""
    if "<turn_state>" in value:
        idx = value.index("<turn_state>")
        return value[idx:], idx > 0
    if value.startswith("language ") and "<asr_text>" in value:
        idx = value.index("<asr_text>")
        return value[idx:], True
    return value, False


def prompt_mentions_language(prompt: str) -> bool:
    lowered = (prompt or "").lower()
    return "language " in lowered or "语种" in prompt


def normalize_prompt(prompt: str, strategy: str, fixed_prompt: str) -> Tuple[str, bool]:
    if strategy == "empty":
        return "", prompt != ""
    if strategy == "rewrite":
        return fixed_prompt, prompt != fixed_prompt
    if not prompt_mentions_language(prompt):
        return prompt, False
    return fixed_prompt, True


def transform_record(record: Dict[str, object], prompt_strategy: str, fixed_prompt: str) -> Tuple[Dict[str, object], bool]:
    updated = dict(record)
    changed = False

    text = str(updated.get("text", ""))
    new_text, text_changed = strip_language_prefix(text)
    if text_changed:
        updated["text"] = new_text
        changed = True

    prompt = str(updated.get("prompt", ""))
    new_prompt, prompt_changed = normalize_prompt(prompt, prompt_strategy, fixed_prompt)
    if prompt_changed:
        updated["prompt"] = new_prompt
        changed = True

    updated["predict_language"] = 0
    if updated.get("easy_turn_lang_qwen") not in (None, ""):
        updated["easy_turn_lang_prediction_removed"] = True
    return updated, changed


def main():
    args = parse_args()
    input_jsonl = Path(args.input_jsonl).resolve()
    output_jsonl = Path(args.output_jsonl).resolve()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    if not input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_jsonl}")

    total = 0
    changed = 0
    start_time = time.time()
    log(
        f"Start normalization | input_jsonl={input_jsonl} output_jsonl={output_jsonl} "
        f"prompt_strategy={args.prompt_strategy}"
    )

    with open(input_jsonl, "r", encoding="utf-8") as fin:
        total_lines = sum(1 for line in fin if line.strip())

    with open(input_jsonl, "r", encoding="utf-8") as fin, open(output_jsonl, "w", encoding="utf-8") as fout:
        for line_no, line in enumerate(
            tqdm(fin, total=total_lines, desc="Removing language", unit="record"),
            start=1,
        ):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {input_jsonl}: {exc}") from exc
            total += 1
            new_record, record_changed = transform_record(record, args.prompt_strategy, args.fixed_prompt)
            if record_changed:
                changed += 1
            fout.write(json.dumps(new_record, ensure_ascii=False) + "\n")

    elapsed = time.time() - start_time
    log(f"Finished normalization | total={total} changed={changed} elapsed_seconds={elapsed:.3f}")
    print(
        json.dumps(
            {
                "input_jsonl": str(input_jsonl),
                "output_jsonl": str(output_jsonl),
                "total": total,
                "changed": changed,
                "prompt_strategy": args.prompt_strategy,
                "elapsed_seconds": round(elapsed, 3),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
