# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Convert Easy-Turn TSV testset metadata into Qwen3-ASR JSONL format.
#
# This script is designed for Easy-Turn test data organized around one
# `all_labels.tsv` file, for example:
#
#   key  subset  label  transcript  tagged_text  task  wav  source_list
#
# Typical usage:
#   python finetuning/convert_easy_turn_testset_to_qwen_asr_jsonl.py \
#       --input_tsv /path/to/Easy-Turn-Testset/testset/all_labels.tsv \
#       --output_jsonl /path/to/easy_turn_test_qwen_asr.jsonl \
#       --dataset_root /path/to/Easy-Turn-Testset/testset \
#       --prompt_mode fixed
#
# By default, the script writes label-first supervision:
#   language {LANG}<turn_state><LABEL><asr_text>{TRANSCRIPT}
#
# The generated JSONL can be used as an eval/test manifest for
# finetuning/qwen3_asr_sft.py or for later offline evaluation scripts.
import argparse
import csv
import json
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


_TASK_TOKEN_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_TAIL_LABEL_RE = re.compile(r"^(.*?)(<(?:COMPLETE|INCOMPLETE|BACKCHANNEL|WAIT)>)\s*$", re.IGNORECASE | re.DOTALL)

LABEL_MAP = {
    "COMPLETE": "complete",
    "INCOMPLETE": "incomplete",
    "BACKCHANNEL": "backchannel",
    "WAIT": "wait",
}

PROMPT_TEMPLATE_POOL = [
    "请转录音频内容，并严格使用格式：language 语种<turn_state><标签><asr_text>转写文本。其中 <complete> 表示语义完整，<incomplete> 表示语义不完整，<backchannel> 表示简短附和，<wait> 表示请求暂停或终止对话。",
    "请将音频转写为文本，并按固定格式输出：language 语种<turn_state><标签><asr_text>文本。标签含义为：<complete>（语义完整）、<incomplete>（语义不完整）、<backchannel>（附和语句）、<wait>（请求暂停或结束对话）。",
    "请先判断打断状态并转录音频内容，输出格式必须为：language 语种<turn_state><标签><asr_text>文本。其中 <complete> 表示语义完整，<incomplete> 表示语义不完整，<backchannel> 表示简单附和，<wait> 表示请求暂停或中止对话。",
    "请将音频内容转换为文字，并严格输出为：language 语种<turn_state><标签><asr_text>文本。标签说明：<complete>（语义完整）、<incomplete>（语义不完整）、<backchannel>（反馈信号）、<wait>（表示希望暂停或结束对话）。",
    "请对音频进行文字转录，并使用固定协议输出：language 语种<turn_state><标签><asr_text>文本。四种标签含义为：<complete> 代表语义完整，<incomplete> 代表语义不完整，<backchannel> 代表简短附和，<wait> 代表请求暂停或终止交流。",
]

PROMPT_TEMPLATE_POOL_NO_LANGUAGE = [
    "请转录音频内容，并严格使用格式：<turn_state><标签><asr_text>转写文本。其中 <complete> 表示语义完整，<incomplete> 表示语义不完整，<backchannel> 表示简短附和，<wait> 表示请求暂停或终止对话。",
    "请将音频转写为文本，并按固定格式输出：<turn_state><标签><asr_text>文本。标签含义为：<complete>（语义完整）、<incomplete>（语义不完整）、<backchannel>（附和语句）、<wait>（请求暂停或终止对话）。",
    "请先判断轮次状态并转录音频内容，输出格式必须为：<turn_state><标签><asr_text>文本。其中 <complete> 表示语义完整，<incomplete> 表示语义不完整，<backchannel> 表示简短附和，<wait> 表示请求暂停或终止对话。",
    "请将音频内容转换为文字，并严格输出为：<turn_state><标签><asr_text>文本。标签说明：<complete>（语义完整）、<incomplete>（语义不完整）、<backchannel>（反馈信号）、<wait>（表示希望暂停或结束对话）。",
    "请对音频进行文字转录，并使用固定协议输出：<turn_state><标签><asr_text>文本。四种标签含义为：<complete> 代表语义完整，<incomplete> 代表语义不完整，<backchannel> 代表简短附和，<wait> 代表请求暂停或终止交流。",
]

DEFAULT_FIXED_PROMPT_WITH_LANGUAGE = (
    "请转录音频内容，并严格使用格式：language 语种<turn_state><标签><asr_text>转写文本。"
    "其中 <complete> 表示语义完整，<incomplete> 表示语义不完整，<backchannel> 表示简短附和，"
    "<wait> 表示请求暂停或终止对话。"
)
DEFAULT_FIXED_PROMPT_NO_LANGUAGE = (
    "请转录音频内容，并严格使用格式：<turn_state><标签><asr_text>转写文本。"
    "其中 <complete> 表示语义完整，<incomplete> 表示语义不完整，<backchannel> 表示简短附和，"
    "<wait> 表示请求暂停或终止对话。"
)


def parse_args():
    p = argparse.ArgumentParser(
        "Convert Easy-Turn TSV testset to Qwen3-ASR JSONL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Read an Easy-Turn all_labels.tsv file, resolve the audio files referenced by "
            "each row, convert row-level labels into Qwen3-ASR supervision text, and "
            "write a final JSONL manifest suitable for eval/test usage.\n\n"
            "Recommended setting for deterministic test conversion:\n"
            "  --output_format label_first\n"
            "  --prompt_mode fixed"
        ),
    )
    p.add_argument(
        "--input_tsv",
        type=str,
        required=True,
        help=(
            "Path to the Easy-Turn test metadata TSV, usually the `all_labels.tsv` file. "
            "The TSV should contain columns such as key, subset, label, transcript, "
            "tagged_text, task, and wav."
        ),
    )
    p.add_argument(
        "--output_jsonl",
        type=str,
        required=True,
        help=(
            "Output JSONL manifest path in Qwen3-ASR format. Each line will contain at "
            "least `audio`, `text`, and `prompt`, plus preserved Easy-Turn metadata."
        ),
    )
    p.add_argument(
        "--dataset_root",
        type=str,
        default="",
        help=(
            "Root directory used to resolve relative `wav` paths in the TSV. If omitted, "
            "the parent directory of --input_tsv is used."
        ),
    )
    p.add_argument(
        "--default_lang",
        type=str,
        default="Chinese",
        help=(
            "Language name written into the final supervision text when the TSV does not "
            "contain an explicit language column. Easy-Turn test examples are often "
            "Chinese, so `Chinese` is the default."
        ),
    )
    p.add_argument(
        "--copy_audio",
        type=int,
        default=0,
        help=(
            "Whether to copy referenced wav files into --audio_dir. `0` keeps the original "
            "wav path in the final JSONL. `1` copies audio files into --audio_dir and makes "
            "the JSONL point to the copied files."
        ),
    )
    p.add_argument(
        "--audio_dir",
        type=str,
        default="",
        help=(
            "Destination directory used only when --copy_audio 1. Copied audio files will "
            "be materialized here so the eval set becomes self-contained."
        ),
    )
    p.add_argument(
        "--audio_path_mode",
        type=str,
        choices=["abs", "rel_to_jsonl"],
        default="abs",
        help=(
            "How to serialize the `audio` field into JSONL. `abs` writes absolute paths. "
            "`rel_to_jsonl` writes paths relative to the output JSONL."
        ),
    )
    p.add_argument(
        "--prompt_mode",
        type=str,
        choices=["template_pool", "task", "empty", "fixed"],
        default="fixed",
        help=(
            "How to build the `prompt` field. `fixed` is the recommended default for test "
            "conversion because it keeps eval prompts deterministic. `template_pool` samples "
            "from the curated prompt pool. `task` derives a prompt from the raw task field. "
            "`empty` leaves prompt blank."
        ),
    )
    p.add_argument(
        "--fixed_prompt",
        type=str,
        default="",
        help=(
            "Fixed prompt string used when --prompt_mode fixed. The default value is the "
            "recommended prompt template for the selected predict_language setting."
        ),
    )
    p.add_argument(
        "--predict_language",
        type=int,
        choices=[0, 1],
        default=0,
        help=(
            "Whether the generated target text and prompt should require explicit language "
            "prediction. `0` removes the `language {LANG}` prefix from targets and uses "
            "language-free prompt templates. `1` keeps the original language-prediction setup."
        ),
    )
    p.add_argument(
        "--strip_txt",
        type=int,
        default=1,
        help=(
            "Whether to strip leading and trailing whitespace from transcript text before "
            "constructing the final supervision target."
        ),
    )
    p.add_argument(
        "--keep_inline_tags",
        type=int,
        default=1,
        help=(
            "Whether to keep inline angle-bracket tags inside transcript text. In most test "
            "rows the clean transcript is already provided separately, so this mainly affects "
            "fallback parsing from `tagged_text`."
        ),
    )
    p.add_argument(
        "--write_bad_records",
        type=str,
        default="",
        help=(
            "Optional JSONL path used to record missing or malformed rows, such as rows with "
            "missing wav, unknown labels, or unresolved audio paths."
        ),
    )
    p.add_argument(
        "--output_format",
        type=str,
        choices=["label_first", "label_last", "plain_asr"],
        default="label_first",
        help=(
            "How to construct the final supervision text. `label_first` produces "
            "`language {LANG}<turn_state><LABEL><asr_text>{TEXT}` and is the recommended "
            "format for the current turn-taking plan. `label_last` produces "
            "`language {LANG}<asr_text>{TEXT}<LABEL>`. `plain_asr` drops the label."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when sampling prompts from the template pool.",
    )
    p.add_argument(
        "--log_every",
        type=int,
        default=500,
        help=(
            "How often to print row-level progress. `500` means one progress log every 500 "
            "rows plus the final row. Set to 0 to disable row-level progress logs."
        ),
    )
    return p.parse_args()


def log(message: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[convert_easy_turn_test][{ts}] {message}", file=sys.stdout, flush=True)


def format_seconds(seconds: float) -> str:
    seconds_int = max(0, int(round(seconds)))
    mins, sec = divmod(seconds_int, 60)
    hrs, mins = divmod(mins, 60)
    if hrs > 0:
        return f"{hrs}h{mins:02d}m{sec:02d}s"
    if mins > 0:
        return f"{mins}m{sec:02d}s"
    return f"{sec}s"


def maybe_write_bad_record(path: Optional[Path], payload: Dict[str, object]) -> None:
    if path is None:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_tsv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [dict(row) for row in reader]


def normalize_target_text(raw_text: str, keep_inline_tags: bool, strip_text: bool) -> str:
    text = raw_text or ""
    if strip_text:
        text = text.strip()
    if not keep_inline_tags:
        text = _TASK_TOKEN_RE.sub("", text)
        text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def parse_task_tokens(task_text: str) -> List[str]:
    return _TASK_TOKEN_RE.findall(task_text or "")


def parse_label_from_tagged_text(tagged_text: str) -> Tuple[str, Optional[str], Optional[str]]:
    text = tagged_text or ""
    m = _TAIL_LABEL_RE.match(text.strip())
    if not m:
        return text, None, None
    transcript = m.group(1).strip()
    label_token = m.group(2).upper()
    label_name = LABEL_MAP.get(label_token[1:-1], None)
    return transcript, label_name, label_token


def parse_label(
    raw_label: str,
    raw_subset: str,
    tagged_text: str,
    raw_task: str,
) -> Tuple[Optional[str], Optional[str]]:
    label_text = (raw_label or "").strip().upper()
    if label_text in LABEL_MAP:
        return LABEL_MAP[label_text], f"<{label_text}>"

    subset_text = (raw_subset or "").strip().upper()
    if subset_text in LABEL_MAP:
        return LABEL_MAP[subset_text], f"<{subset_text}>"

    _, label_from_tagged, label_token_from_tagged = parse_label_from_tagged_text(tagged_text)
    if label_from_tagged:
        return label_from_tagged, label_token_from_tagged

    task_tokens = [tok.upper() for tok in parse_task_tokens(raw_task)]
    matched = [tok for tok in task_tokens if tok.startswith("<") and tok.endswith(">") and tok[1:-1] in LABEL_MAP]
    if len(matched) == 1:
        token = matched[0]
        return LABEL_MAP[token[1:-1]], token

    return None, None


def build_qwen_asr_target(
    lang_name: str,
    transcript: str,
    label: Optional[str],
    output_format: str,
    predict_language: bool,
) -> str:
    if output_format == "plain_asr" or not label:
        prefix = f"language {lang_name}" if predict_language else ""
        return f"{prefix}<asr_text>{transcript}"

    label_token = f"<{label}>"
    if output_format == "label_last":
        prefix = f"language {lang_name}" if predict_language else ""
        return f"{prefix}<asr_text>{transcript}{label_token}"
    prefix = f"language {lang_name}" if predict_language else ""
    return f"{prefix}<turn_state>{label_token}<asr_text>{transcript}"


def build_prompt(task_text: str, prompt_mode: str, fixed_prompt: str, predict_language: bool) -> str:
    if prompt_mode == "empty":
        return ""
    if prompt_mode == "fixed":
        if fixed_prompt:
            return fixed_prompt
        return DEFAULT_FIXED_PROMPT_WITH_LANGUAGE if predict_language else DEFAULT_FIXED_PROMPT_NO_LANGUAGE
    if prompt_mode == "template_pool":
        pool = PROMPT_TEMPLATE_POOL if predict_language else PROMPT_TEMPLATE_POOL_NO_LANGUAGE
        return random.choice(pool)

    tokens = parse_task_tokens(task_text)
    if tokens:
        task_desc = " ".join(tokens)
        return (
            f"Easy-Turn task specification: {task_desc}. "
            "Transcribe the speech and produce the reference text exactly. "
            + (
                "Use the exact output format `language <lang><turn_state><label><asr_text><text>`."
                if predict_language
                else "Use the exact output format `<turn_state><label><asr_text><text>`."
            )
        )
    return (
        "Transcribe the speech and produce the reference text exactly. "
        + (
            "Use the exact output format `language <lang><turn_state><label><asr_text><text>`."
            if predict_language
            else "Use the exact output format `<turn_state><label><asr_text><text>`."
        )
    )


def resolve_audio_path(raw_wav: str, dataset_root: Path) -> Path:
    wav_path = Path((raw_wav or "").strip())
    if wav_path.is_absolute():
        return wav_path
    return (dataset_root / wav_path).resolve()


def ensure_unique_output_name(audio_dir: Path, sample_id: str, suffix: str, used_names: Dict[str, int]) -> Path:
    count = used_names[sample_id]
    used_names[sample_id] += 1
    if count == 0:
        name = f"{sample_id}.{suffix}"
    else:
        name = f"{sample_id}__dup{count}.{suffix}"
    return audio_dir / name


def audio_path_for_manifest(path: Path, output_jsonl: Path, mode: str) -> str:
    if mode == "rel_to_jsonl":
        return os.path.relpath(str(path), start=str(output_jsonl.parent))
    return str(path.resolve())


def infer_transcript(raw_transcript: str, tagged_text: str, keep_inline_tags: bool, strip_text: bool) -> str:
    transcript = (raw_transcript or "").strip()
    if transcript:
        return normalize_target_text(transcript, keep_inline_tags=keep_inline_tags, strip_text=strip_text)

    transcript_from_tagged, _, _ = parse_label_from_tagged_text(tagged_text)
    return normalize_target_text(
        transcript_from_tagged,
        keep_inline_tags=keep_inline_tags,
        strip_text=strip_text,
    )


def main():
    args = parse_args()
    if args.prompt_mode == "fixed" and args.fixed_prompt and not args.fixed_prompt.strip():
        raise ValueError("--prompt_mode fixed requires a non-empty --fixed_prompt.")
    if args.copy_audio == 1 and not args.audio_dir.strip():
        raise ValueError("--copy_audio 1 requires --audio_dir.")

    random.seed(int(args.seed))
    input_tsv = Path(args.input_tsv).resolve()
    if not input_tsv.exists():
        raise FileNotFoundError(f"Input TSV not found: {input_tsv}")

    output_jsonl = Path(args.output_jsonl).resolve()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    dataset_root = Path(args.dataset_root).resolve() if args.dataset_root else input_tsv.parent.resolve()
    audio_dir = Path(args.audio_dir).resolve() if args.audio_dir else None
    if audio_dir is not None:
        audio_dir.mkdir(parents=True, exist_ok=True)

    bad_records_path = Path(args.write_bad_records).resolve() if args.write_bad_records else None
    if bad_records_path is not None:
        bad_records_path.parent.mkdir(parents=True, exist_ok=True)
        if bad_records_path.exists():
            bad_records_path.unlink()

    rows = read_tsv_rows(str(input_tsv))
    total_rows = len(rows)
    if total_rows == 0:
        raise ValueError(f"No row found in TSV: {input_tsv}")

    used_names: Dict[str, int] = {}
    num_written = 0
    num_skipped = 0
    start_time = time.time()

    log(
        "Start conversion with "
        f"rows={total_rows}, input_tsv={input_tsv}, dataset_root={dataset_root}, "
        f"output_jsonl={output_jsonl}, copy_audio={args.copy_audio}, "
        f"output_format={args.output_format}, prompt_mode={args.prompt_mode}"
    )
    if bad_records_path is not None:
        log(f"Bad records will be written to: {bad_records_path}")

    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for idx, row in enumerate(rows, start=1):
            sample_id = (row.get("key") or "").strip()
            raw_subset = row.get("subset", "")
            raw_label = row.get("label", "")
            raw_transcript = row.get("transcript", "")
            raw_tagged_text = row.get("tagged_text", "")
            raw_task = row.get("task", "")
            raw_wav = row.get("wav", "")
            raw_source_list = row.get("source_list", "")

            if not sample_id:
                num_skipped += 1
                maybe_write_bad_record(
                    bad_records_path,
                    {"reason": "missing_key", "row_index": idx, "row": row},
                )
                continue

            audio_src_path = resolve_audio_path(raw_wav, dataset_root)
            if not audio_src_path.exists():
                num_skipped += 1
                maybe_write_bad_record(
                    bad_records_path,
                    {
                        "reason": "missing_wav",
                        "row_index": idx,
                        "sample_id": sample_id,
                        "raw_wav": raw_wav,
                        "resolved_wav": str(audio_src_path),
                    },
                )
                continue

            label, label_token = parse_label(raw_label, raw_subset, raw_tagged_text, raw_task)
            if args.output_format != "plain_asr" and not label:
                num_skipped += 1
                maybe_write_bad_record(
                    bad_records_path,
                    {
                        "reason": "missing_label",
                        "row_index": idx,
                        "sample_id": sample_id,
                        "raw_label": raw_label,
                        "raw_subset": raw_subset,
                        "raw_tagged_text": raw_tagged_text,
                        "raw_task": raw_task,
                    },
                )
                continue

            transcript = infer_transcript(
                raw_transcript,
                raw_tagged_text,
                keep_inline_tags=bool(args.keep_inline_tags),
                strip_text=bool(args.strip_txt),
            )
            if not transcript:
                num_skipped += 1
                maybe_write_bad_record(
                    bad_records_path,
                    {
                        "reason": "empty_transcript",
                        "row_index": idx,
                        "sample_id": sample_id,
                        "raw_transcript": raw_transcript,
                        "raw_tagged_text": raw_tagged_text,
                    },
                )
                continue

            final_audio_path = audio_src_path
            if args.copy_audio == 1:
                suffix = audio_src_path.suffix.lstrip(".") or "wav"
                copied_audio_path = ensure_unique_output_name(audio_dir, sample_id, suffix, used_names)
                shutil.copy2(str(audio_src_path), str(copied_audio_path))
                final_audio_path = copied_audio_path

            prompt = build_prompt(
                raw_task,
                args.prompt_mode,
                args.fixed_prompt,
                bool(args.predict_language),
            )
            target_text = build_qwen_asr_target(
                args.default_lang,
                transcript,
                label,
                args.output_format,
                bool(args.predict_language),
            )

            record = {
                "audio": audio_path_for_manifest(final_audio_path, output_jsonl, args.audio_path_mode),
                "text": target_text,
                "prompt": prompt,
                "source_dataset": "easy-turn-testset",
                "source_tsv": str(input_tsv),
                "source_sample_id": sample_id,
                "easy_turn_subset": raw_subset.strip(),
                "easy_turn_label_raw": raw_label.strip(),
                "easy_turn_label": label,
                "easy_turn_label_token": label_token,
                "easy_turn_transcript": transcript,
                "easy_turn_tagged_text": raw_tagged_text.strip(),
                "easy_turn_task_raw": raw_task.strip(),
                "easy_turn_wav_raw": raw_wav.strip(),
                "easy_turn_source_list": raw_source_list.strip(),
                "easy_turn_default_lang": args.default_lang,
                "output_format": args.output_format,
                "predict_language": int(args.predict_language),
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            num_written += 1

            if args.log_every > 0 and (idx % args.log_every == 0 or idx == total_rows):
                log(
                    f"Progress {idx}/{total_rows} | written={num_written} skipped={num_skipped} "
                    f"| elapsed={format_seconds(time.time() - start_time)}"
                )

    total_elapsed = time.time() - start_time
    log(
        f"Conversion finished | rows={total_rows} written={num_written} skipped={num_skipped} "
        f"| elapsed={format_seconds(total_elapsed)}"
    )
    print(
        json.dumps(
            {
                "input_tsv": str(input_tsv),
                "output_jsonl": str(output_jsonl),
                "dataset_root": str(dataset_root),
                "copy_audio": args.copy_audio,
                "audio_dir": str(audio_dir) if audio_dir else "",
                "num_written": num_written,
                "num_skipped": num_skipped,
                "prompt_mode": args.prompt_mode,
                "output_format": args.output_format,
                "audio_path_mode": args.audio_path_mode,
                "predict_language": int(args.predict_language),
                "elapsed_seconds": round(total_elapsed, 3),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
