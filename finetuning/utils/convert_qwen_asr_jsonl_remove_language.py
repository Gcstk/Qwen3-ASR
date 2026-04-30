# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Post-process already converted Qwen3-ASR JSONL data to remove language
# prediction from targets and prompts, and optionally rewrite the turn-state
# schema for existing manifests.
import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm


LABEL_TOKENS = ("<complete>", "<incomplete>", "<backchannel>", "<wait>")

DEFAULT_PROMPT_NO_LANGUAGE = (
    "请转录音频内容，并严格使用格式：<turn_state><标签><asr_text>转写文本。"
    "其中 <complete> 表示语义完整，<incomplete> 表示语义不完整，<backchannel> 表示简短附和，"
    "<wait> 表示请求暂停或终止对话。"
)

DEFAULT_PROMPT_NO_LANGUAGE_NO_TURN_STATE = (
    "请转录音频内容，并输出轮次状态标签。输出格式固定为："
    "<complete|incomplete|backchannel|wait><asr_text>转写文本。"
    "其中 <complete> 表示语义完整，<incomplete> 表示语义不完整，<backchannel> 表示简短附和，"
    "<wait> 表示请求暂停或终止对话。"
)

PROMPT_POOL_NO_LANGUAGE_NO_TURN_STATE = [
    "请转录音频内容，并输出轮次状态标签。输出格式固定为：<complete|incomplete|backchannel|wait><asr_text>转写文本。其中 <complete> 表示语义完整，<incomplete> 表示语义不完整，<backchannel> 表示简短附和，<wait> 表示请求暂停或终止对话。",
    "请将音频转写为文本，并在开头输出轮次状态标签。输出格式固定为：<complete|incomplete|backchannel|wait><asr_text>转写文本。其中 <complete> 表示语义完整，<incomplete> 表示语义不完整，<backchannel> 表示简短附和，<wait> 表示请求暂停或终止对话。",
    "请先判断当前语音的轮次状态，再转录音频内容。输出格式固定为：<complete|incomplete|backchannel|wait><asr_text>转写文本。其中 <complete> 表示语义完整，<incomplete> 表示语义不完整，<backchannel> 表示简短附和，<wait> 表示请求暂停或终止对话。",
    "请将音频内容转换为文字，并在开头附加一个轮次状态标签。输出格式固定为：<complete|incomplete|backchannel|wait><asr_text>转写文本。其中 <complete> 表示语义完整，<incomplete> 表示语义不完整，<backchannel> 表示简短附和，<wait> 表示请求暂停或终止对话。",
    "请对音频进行文字转录，并同时给出轮次状态标签。输出格式固定为：<complete|incomplete|backchannel|wait><asr_text>转写文本。其中 <complete> 表示语义完整，<incomplete> 表示语义不完整，<backchannel> 表示简短附和，<wait> 表示请求暂停或终止对话。",
]


def parse_args():
    p = argparse.ArgumentParser(
        "Remove language prediction from existing Qwen3-ASR JSONL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input_jsonl", type=str, required=True, help="Input Qwen3-ASR JSONL path.")
    p.add_argument("--output_jsonl", type=str, required=True, help="Output JSONL path after normalization.")
    p.add_argument(
        "--output_format",
        type=str,
        choices=["label_first", "label_first_no_turn_state", "plain_asr"],
        default="label_first",
        help=(
            "Target schema written into `text` and `prompt`. "
            "`label_first` keeps `<turn_state><label><asr_text>...`; "
            "`label_first_no_turn_state` rewrites into `<label><asr_text>...`; "
            "`plain_asr` removes turn labels and keeps only `<asr_text>...`."
        ),
    )
    p.add_argument(
        "--prompt_strategy",
        type=str,
        choices=["rewrite", "rewrite_pool", "preserve_if_clean", "empty"],
        default="rewrite",
        help=(
            "`rewrite` replaces every prompt with one stable instruction. "
            "`rewrite_pool` samples prompts from a curated pool. "
            "`preserve_if_clean` keeps prompts that already match the target schema, otherwise rewrites them. "
            "`empty` clears the prompt field."
        ),
    )
    p.add_argument(
        "--fixed_prompt",
        type=str,
        default="",
        help="Prompt used when --prompt_strategy rewrite or when preserve_if_clean needs fallback rewriting.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --prompt_strategy rewrite_pool.",
    )
    return p.parse_args()


def log(message: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[remove_language][{ts}] {message}", file=sys.stdout, flush=True)


def get_default_prompt(output_format: str) -> str:
    if output_format == "label_first_no_turn_state":
        return DEFAULT_PROMPT_NO_LANGUAGE_NO_TURN_STATE
    return DEFAULT_PROMPT_NO_LANGUAGE


def get_prompt_pool(output_format: str) -> List[str]:
    if output_format == "label_first_no_turn_state":
        return list(PROMPT_POOL_NO_LANGUAGE_NO_TURN_STATE)
    return [get_default_prompt(output_format)]


def strip_language_prefix(text: str) -> Tuple[str, bool]:
    value = text or ""
    marker_positions = []
    for marker in ("<turn_state>", *LABEL_TOKENS, "<asr_text>"):
        idx = value.find(marker)
        if idx >= 0:
            marker_positions.append(idx)
    if marker_positions:
        idx = min(marker_positions)
        return value[idx:], idx > 0
    if value.startswith("language "):
        return value, False
    return value, False


def normalize_text_schema(text: str, output_format: str) -> Tuple[str, bool]:
    value, changed = strip_language_prefix(text)

    if output_format == "label_first":
        if value.startswith("<turn_state>"):
            return value, changed
        for token in LABEL_TOKENS:
            if value.startswith(token):
                return "<turn_state>" + value, True
        return value, changed

    if output_format == "label_first_no_turn_state":
        if value.startswith("<turn_state>"):
            return value[len("<turn_state>") :], True
        return value, changed

    if output_format == "plain_asr":
        if value.startswith("<turn_state>"):
            value = value[len("<turn_state>") :]
            changed = True
        for token in LABEL_TOKENS:
            prefix = token + "<asr_text>"
            if value.startswith(prefix):
                return "<asr_text>" + value[len(prefix) :], True
        return value, changed

    raise ValueError(f"Unsupported output_format: {output_format}")


def prompt_mentions_language(prompt: str) -> bool:
    lowered = (prompt or "").lower()
    return "language " in lowered or "语种" in (prompt or "")


def prompt_matches_output_format(prompt: str, output_format: str) -> bool:
    value = prompt or ""
    if prompt_mentions_language(value):
        return False
    if output_format == "label_first_no_turn_state":
        return "<turn_state>" not in value
    if output_format == "label_first":
        return "<turn_state>" in value
    return "<turn_state>" not in value


def normalize_prompt(
    prompt: str,
    strategy: str,
    fixed_prompt: str,
    prompt_pool: Sequence[str],
    output_format: str,
    rng: random.Random,
) -> Tuple[str, bool]:
    if strategy == "empty":
        return "", prompt != ""
    if strategy == "rewrite":
        return fixed_prompt, prompt != fixed_prompt
    if strategy == "rewrite_pool":
        sampled = rng.choice(list(prompt_pool))
        return sampled, prompt != sampled
    if prompt_matches_output_format(prompt, output_format):
        return prompt, False
    return fixed_prompt, True


def transform_record(
    record: Dict[str, object],
    prompt_strategy: str,
    fixed_prompt: str,
    output_format: str = "label_first",
    prompt_pool: Optional[Sequence[str]] = None,
    rng: Optional[random.Random] = None,
) -> Tuple[Dict[str, object], bool]:
    updated = dict(record)
    changed = False
    local_pool = list(prompt_pool) if prompt_pool is not None else get_prompt_pool(output_format)
    local_rng = rng if rng is not None else random.Random(42)

    text = str(updated.get("text", ""))
    new_text, text_changed = normalize_text_schema(text, output_format)
    if text_changed:
        updated["text"] = new_text
        changed = True

    prompt = str(updated.get("prompt", ""))
    new_prompt, prompt_changed = normalize_prompt(
        prompt=prompt,
        strategy=prompt_strategy,
        fixed_prompt=fixed_prompt,
        prompt_pool=local_pool,
        output_format=output_format,
        rng=local_rng,
    )
    if prompt_changed:
        updated["prompt"] = new_prompt
        changed = True

    if updated.get("output_format") != output_format:
        updated["output_format"] = output_format
        changed = True

    updated["predict_language"] = 0
    if updated.get("easy_turn_lang_qwen") not in (None, ""):
        updated["easy_turn_lang_prediction_removed"] = True
    if output_format == "label_first_no_turn_state":
        updated["easy_turn_turn_state_removed"] = True
    return updated, changed


def main():
    args = parse_args()
    input_jsonl = Path(args.input_jsonl).resolve()
    output_jsonl = Path(args.output_jsonl).resolve()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    if not input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_jsonl}")

    fixed_prompt = args.fixed_prompt if args.fixed_prompt else get_default_prompt(args.output_format)
    prompt_pool = get_prompt_pool(args.output_format)
    rng = random.Random(int(args.seed))

    total = 0
    changed = 0
    start_time = time.time()
    log(
        f"Start normalization | input_jsonl={input_jsonl} output_jsonl={output_jsonl} "
        f"prompt_strategy={args.prompt_strategy} output_format={args.output_format} seed={args.seed}"
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
            new_record, record_changed = transform_record(
                record=record,
                prompt_strategy=args.prompt_strategy,
                fixed_prompt=fixed_prompt,
                output_format=args.output_format,
                prompt_pool=prompt_pool,
                rng=rng,
            )
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
                "output_format": args.output_format,
                "seed": int(args.seed),
                "elapsed_seconds": round(elapsed, 3),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
