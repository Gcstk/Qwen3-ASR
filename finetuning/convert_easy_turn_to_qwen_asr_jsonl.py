# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Convert Easy-Turn tar shards into Qwen3-ASR SFT JSONL format.
#
# This script is intended as the first data bridge from Easy-Turn shard data to
# the training format consumed by finetuning/qwen3_asr_sft.py.
#
# The source dataset is expected to be organized as:
#   1. One index file, for example shards_list.txt
#   2. Each non-empty line in that file points to one .tar shard
#   3. Each shard contains multiple files that share the same sample id prefix,
#      for example:
#        <sample_id>.wav
#        <sample_id>.txt
#        <sample_id>.lang
#        <sample_id>.task
#        <sample_id>.state
#
# The output manifest is intentionally aligned with Qwen3-ASR SFT:
#   - required fields: audio, text
#   - optional field: prompt
#
# We also preserve Easy-Turn metadata fields for traceability and for later
# turn-taking / control-token experiments.
#
# Typical usage:
#   python finetuning/convert_easy_turn_to_qwen_asr_jsonl.py \
#       --shards_list /path/to/shards_list.txt \
#       --output_jsonl /path/to/easy_turn_train.jsonl \
#       --audio_dir /path/to/extracted_audio \
#       --prompt_mode template_pool
#
# By default, the script writes label-first supervision:
#   language {LANG}<turn_state><LABEL><asr_text>{TRANSCRIPT}
import argparse
import json
import os
import re
import random
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


_TASK_TOKEN_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_TAIL_LABEL_RE = re.compile(r"^(.*?)(<(?:COMPLETE|INCOMPLETE|BACKCHANNEL|WAIT)>)\s*$", re.IGNORECASE | re.DOTALL)


LANGUAGE_TAG_MAP = {
    "CN": "Chinese",
    "ZH": "Chinese",
    "MANDARIN": "Chinese",
    "EN": "English",
    "ENG": "English",
    "YUE": "Cantonese",
    "CANTONESE": "Cantonese",
    "AR": "Arabic",
    "DE": "German",
    "FR": "French",
    "ES": "Spanish",
    "PT": "Portuguese",
    "ID": "Indonesian",
    "IT": "Italian",
    "KO": "Korean",
    "RU": "Russian",
    "TH": "Thai",
    "VI": "Vietnamese",
    "JA": "Japanese",
    "TR": "Turkish",
    "HI": "Hindi",
    "MS": "Malay",
    "NL": "Dutch",
    "SV": "Swedish",
    "DA": "Danish",
    "FI": "Finnish",
    "PL": "Polish",
    "CS": "Czech",
    "FIL": "Filipino",
    "FA": "Persian",
    "EL": "Greek",
    "RO": "Romanian",
    "HU": "Hungarian",
    "MK": "Macedonian",
}

LABEL_MAP = {
    "<COMPLETE>": "complete",
    "<INCOMPLETE>": "incomplete",
    "<BACKCHANNEL>": "backchannel",
    "<WAIT>": "wait",
}

PROMPT_TEMPLATE_POOL = [
    "请转录音频内容，并严格使用格式：language 语种<turn_state><标签><asr_text>转写文本。其中 <complete> 表示语义完整，<incomplete> 表示语义不完整，<backchannel> 表示简短附和，<wait> 表示请求暂停或终止对话。",
    "请将音频转写为文本，并按固定格式输出：language 语种<turn_state><标签><asr_text>文本。标签含义为：<complete>（语义完整）、<incomplete>（语义不完整）、<backchannel>（附和语句）、<wait>（请求暂停或结束对话）。",
    "请先判断打断状态并转录音频内容，输出格式必须为：language 语种<turn_state><标签><asr_text>文本。其中 <complete> 表示语义完整，<incomplete> 表示语义不完整，<backchannel> 表示简单附和，<wait> 表示请求暂停或中止对话。",
    "请将音频内容转换为文字，并严格输出为：language 语种<turn_state><标签><asr_text>文本。标签说明：<complete>（语义完整）、<incomplete>（语义不完整）、<backchannel>（反馈信号）、<wait>（表示希望暂停或结束对话）。",
    "请对音频进行文字转录，并使用固定协议输出：language 语种<turn_state><标签><asr_text>文本。四种标签含义为：<complete> 代表语义完整，<incomplete> 代表语义不完整，<backchannel> 代表简短附和，<wait> 代表请求暂停或终止交流。",
]


def parse_args():
    p = argparse.ArgumentParser(
        "Convert Easy-Turn tar shards to Qwen3-ASR JSONL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Read an Easy-Turn shard index file, extract samples from every tar shard, "
            "convert sample-level supervision into Qwen3-ASR SFT records, and write "
            "a final JSONL manifest plus extracted .wav files.\n\n"
            "Recommended v1 setting for turn-taking joint generation:\n"
            "  --output_format label_first\n"
            "  --prompt_mode template_pool\n\n"
            "The script keeps extra Easy-Turn metadata fields in each JSONL row so the "
            "converted dataset can be audited or re-used for later experiments."
        ),
    )
    p.add_argument(
        "--shards_list",
        type=str,
        required=True,
        help=(
            "Easy-Turn shard index file. This should be a plain text file where each "
            "non-empty line points to one .tar shard on disk. Lines beginning with '#'"
            " are ignored. This is the main input entry for the converter."
        ),
    )
    p.add_argument(
        "--output_jsonl",
        type=str,
        required=True,
        help=(
            "Output JSONL manifest path in Qwen3-ASR SFT format. Each line in the "
            "generated file will contain at least `audio`, `text`, and `prompt`, plus "
            "preserved Easy-Turn metadata fields."
        ),
    )
    p.add_argument(
        "--audio_dir",
        type=str,
        required=True,
        help=(
            "Directory used to write extracted .wav files. Every sample kept in the "
            "final manifest will have its audio payload materialized here so the "
            "generated JSONL can be consumed directly by qwen3_asr_sft.py."
        ),
    )
    p.add_argument(
        "--audio_path_mode",
        type=str,
        choices=["abs", "rel_to_jsonl"],
        default="abs",
        help=(
            "How to serialize the `audio` field into JSONL. `abs` writes absolute file "
            "paths and is the safest default for distributed training. `rel_to_jsonl` "
            "writes paths relative to the output JSONL location, which is convenient "
            "when the dataset directory is moved as a whole."
        ),
    )
    p.add_argument(
        "--prompt_mode",
        type=str,
        choices=["template_pool", "task", "empty", "fixed"],
        default="template_pool",
        help=(
            "How to build the `prompt` field for every sample. `template_pool` samples "
            "from the recommended prompt pool in doc/TURN_TAKING_LABEL_FIRST_PLAN_ZH.md; "
            "`task` derives a prompt from the raw Easy-Turn .task field; `empty` leaves "
            "prompt blank; `fixed` uses exactly the string passed through --fixed_prompt. "
            "For turn-taking joint generation, `template_pool` is the recommended mode."
        ),
    )
    p.add_argument(
        "--fixed_prompt",
        type=str,
        default="",
        help=(
            "Fixed prompt string used only when --prompt_mode fixed. This is useful when "
            "you want all samples in one split to share the exact same instruction."
        ),
    )
    p.add_argument(
        "--strip_txt",
        type=int,
        default=1,
        help=(
            "Whether to strip leading and trailing whitespace from the raw Easy-Turn .txt "
            "content before building the final target text. Use 1 in almost all cases."
        ),
    )
    p.add_argument(
        "--keep_inline_tags",
        type=int,
        default=1,
        help=(
            "Whether to keep inline control-like tags found inside the Easy-Turn .txt "
            "content. Set to 1 to preserve tags that are part of the transcript payload. "
            "Set to 0 to remove all angle-bracket tags from the transcript body before "
            "constructing the final supervision text."
        ),
    )
    p.add_argument(
        "--skip_missing_txt",
        type=int,
        default=1,
        help=(
            "How to handle samples that do not have a .txt file in the shard. `1` means "
            "skip the sample and optionally record it into --write_bad_records. `0` means "
            "raise an error immediately."
        ),
    )
    p.add_argument(
        "--skip_missing_wav",
        type=int,
        default=1,
        help=(
            "How to handle samples that do not have a .wav file in the shard. `1` means "
            "skip the sample and optionally record it into --write_bad_records. `0` means "
            "raise an error immediately."
        ),
    )
    p.add_argument(
        "--write_bad_records",
        type=str,
        default="",
        help=(
            "Optional JSONL path used to record skipped or malformed samples, for example "
            "missing shards, missing .wav, or missing .txt. Strongly recommended for "
            "large-scale conversion so bad source samples can be audited afterwards."
        ),
    )
    p.add_argument(
        "--output_format",
        type=str,
        choices=["label_first", "label_last", "plain_asr"],
        default="label_first",
        help=(
            "How to construct the final supervision text in the JSONL `text` field. "
            "`label_first` produces `language {LANG}<turn_state><LABEL><asr_text>{TEXT}` "
            "and is the recommended format for the current turn-taking plan. "
            "`label_last` produces `language {LANG}<asr_text>{TEXT}<LABEL>` for ablation "
            "experiments. `plain_asr` drops turn-taking labels entirely and outputs a pure "
            "ASR target."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed used when sampling prompts from the template pool. Keeping this "
            "fixed makes the conversion process reproducible."
        ),
    )
    p.add_argument(
        "--log_every",
        type=int,
        default=500,
        help=(
            "How often to print sample-level progress inside one shard. For example, "
            "`500` means the script prints one progress line every 500 processed samples "
            "plus the final sample of each shard. Set to 0 to disable sample-level "
            "progress logs and keep only shard-level logs."
        ),
    )
    return p.parse_args()


def read_lines(path: str) -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            yield s


def decode_text_bytes(data: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "gbk", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def normalize_lang(raw_lang: str) -> str:
    s = (raw_lang or "").strip()
    if not s:
        return "None"
    if s.startswith("<") and s.endswith(">"):
        s = s[1:-1].strip()
    key = s.upper()
    return LANGUAGE_TAG_MAP.get(key, "None")


def normalize_target_text(raw_text: str, keep_inline_tags: bool, strip_text: bool) -> str:
    text = raw_text or ""
    if strip_text:
        text = text.strip()
    if not keep_inline_tags:
        text = _TASK_TOKEN_RE.sub("", text)
        text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def parse_label_from_text_tail(raw_text: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Parse a turn-taking label appended at the end of Easy-Turn `.txt`.

    Expected examples:
      "我想查一下明天北京天气<COMPLETE>"
      "先停一下<WAIT>"

    Returns:
      1. transcript text with the trailing label removed
      2. normalized lowercase label, such as "complete"
      3. original label token, such as "<COMPLETE>"

    If the text does not end with one of the supported label tokens, the
    original text is returned and the label fields are None.
    """
    text = raw_text or ""
    m = _TAIL_LABEL_RE.match(text.strip())
    if not m:
        return text, None, None
    transcript = m.group(1).strip()
    label_token = m.group(2).upper()
    return transcript, LABEL_MAP.get(label_token), label_token


def parse_label_from_state(raw_state: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse a unique supervision label from the `.state` sidecar file.

    The `.state` file is treated as a fallback source because some Easy-Turn
    samples may store the turn-taking label there instead of appending it to
    `.txt`. Only a single unambiguous label is accepted. If multiple supported
    labels appear at once, the function returns None to avoid silently guessing.
    """
    tokens = [tok.upper() for tok in parse_task_tokens(raw_state)]
    matched = [tok for tok in tokens if tok in LABEL_MAP]
    if len(matched) == 1:
        return LABEL_MAP[matched[0]], matched[0]
    return None, None


def parse_label_from_task(raw_task: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse a unique supervision label from the `.task` sidecar file.

    This is the weakest fallback because `.task` often describes the capability
    set of the sample rather than a single ground-truth label. For that reason,
    the function only accepts `.task` when exactly one supported label token is
    present. Otherwise it returns None.
    """
    tokens = [tok.upper() for tok in parse_task_tokens(raw_task)]
    matched = [tok for tok in tokens if tok in LABEL_MAP]
    if len(matched) == 1:
        return LABEL_MAP[matched[0]], matched[0]
    return None, None


def build_qwen_asr_target(lang_name: str, transcript: str, label: Optional[str], output_format: str) -> str:
    """
    Build the final supervision text consumed by Qwen3-ASR SFT.

    Supported target formats:
      - plain_asr:
          language {LANG}<asr_text>{TRANSCRIPT}
      - label_first:
          language {LANG}<turn_state><LABEL><asr_text>{TRANSCRIPT}
      - label_last:
          language {LANG}<asr_text>{TRANSCRIPT}<LABEL>

    `label_first` is the recommended default for the current turn-taking plan
    because it keeps the turn-state protocol explicit and stable.
    """
    if output_format == "plain_asr" or not label:
        return f"language {lang_name}<asr_text>{transcript}"

    label_token = f"<{label}>"
    if output_format == "label_last":
        return f"language {lang_name}<asr_text>{transcript}{label_token}"
    return f"language {lang_name}<turn_state>{label_token}<asr_text>{transcript}"


def parse_task_tokens(task_text: str) -> List[str]:
    """Extract all angle-bracket tokens, such as `<COMPLETE>` or `<CN>`."""
    return _TASK_TOKEN_RE.findall(task_text or "")


def build_prompt(task_text: str, prompt_mode: str, fixed_prompt: str) -> str:
    """
    Build the training prompt for one sample.

    Modes:
      - empty:
          no prompt is written into the JSONL record
      - fixed:
          use the exact string from `--fixed_prompt`
      - template_pool:
          sample one prompt from the curated prompt pool in the turn-taking plan
      - task:
          derive a lightweight prompt from the raw Easy-Turn `.task` field

    In practice, `template_pool` is the recommended default because it matches
    the project documentation and keeps the prompt wording controlled.
    """
    if prompt_mode == "empty":
        return ""
    if prompt_mode == "fixed":
        return fixed_prompt or ""
    if prompt_mode == "template_pool":
        return random.choice(PROMPT_TEMPLATE_POOL)

    tokens = parse_task_tokens(task_text)
    if tokens:
        task_desc = " ".join(tokens)
        return (
            f"Easy-Turn task specification: {task_desc}. "
            "Transcribe the speech and produce the reference text exactly. "
            "Keep any inline control tags in the target text unchanged."
        )
    return (
        "Transcribe the speech and produce the reference text exactly. "
        "Keep any inline control tags in the target text unchanged."
    )


def member_sample_id_and_ext(member_name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Split one shard member path into `(sample_id, ext)`.

    Example:
      "abc123.wav" -> ("abc123", "wav")
      "nested/abc123.txt" -> ("abc123", "txt")

    Files without a valid basename or without an extension are ignored by
    returning `(None, None)`.
    """
    base = os.path.basename(member_name)
    if not base or base.endswith("/"):
        return None, None
    if "." not in base:
        return None, None
    sample_id, ext = base.rsplit(".", 1)
    if not sample_id or not ext:
        return None, None
    return sample_id, ext.lower()


def iter_shard_records(shard_path: str) -> Dict[str, Dict[str, object]]:
    """
    Read one Easy-Turn tar shard and group sidecar files by sample id.

    The returned structure looks like:
      {
        "<sample_id>": {
          "wav_bytes": ...,
          "txt": "...",
          "lang": "<CN>",
          "task": "<TRANSCRIBE> <COMPLETE>",
          ...
        }
      }

    This function is the key bridge between the raw shard layout and the later
    sample-level conversion logic in `main()`.
    """
    grouped: Dict[str, Dict[str, object]] = defaultdict(dict)
    with tarfile.open(shard_path, "r:*") as tf:
        for member in tf:
            if not member.isfile():
                continue
            sample_id, ext = member_sample_id_and_ext(member.name)
            if sample_id is None:
                continue
            extracted = tf.extractfile(member)
            if extracted is None:
                continue
            payload = extracted.read()
            if ext == "wav":
                grouped[sample_id]["wav_bytes"] = payload
                grouped[sample_id]["wav_ext"] = ext
            else:
                grouped[sample_id][ext] = decode_text_bytes(payload)
    return grouped


def ensure_unique_output_name(audio_dir: Path, sample_id: str, suffix: str, used_names: Dict[str, int]) -> Path:
    """
    Generate a collision-safe output audio path under `audio_dir`.

    If the same `sample_id` appears more than once across shards, later files are
    written as `<sample_id>__dupN.wav` so conversion can continue without
    overwriting previously extracted audio.
    """
    count = used_names[sample_id]
    used_names[sample_id] += 1
    if count == 0:
        name = f"{sample_id}.{suffix}"
    else:
        name = f"{sample_id}__dup{count}.{suffix}"
    return audio_dir / name


def audio_path_for_manifest(path: Path, output_jsonl: Path, mode: str) -> str:
    """Serialize the extracted audio path according to the requested manifest mode."""
    if mode == "rel_to_jsonl":
        return os.path.relpath(str(path), start=str(output_jsonl.parent))
    return str(path.resolve())


def maybe_write_bad_record(path: Optional[Path], payload: Dict[str, object]) -> None:
    """Append one skipped or malformed sample record into the optional audit JSONL."""
    if path is None:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def log(message: str) -> None:
    """Print one timestamped log line to stdout and flush immediately."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[convert_easy_turn][{ts}] {message}", file=sys.stdout, flush=True)


def format_seconds(seconds: float) -> str:
    """Format a duration into a short human-readable string."""
    seconds_int = max(0, int(round(seconds)))
    mins, sec = divmod(seconds_int, 60)
    hrs, mins = divmod(mins, 60)
    if hrs > 0:
        return f"{hrs}h{mins:02d}m{sec:02d}s"
    if mins > 0:
        return f"{mins}m{sec:02d}s"
    return f"{sec}s"


def main():
    args = parse_args()
    if args.prompt_mode == "fixed" and not args.fixed_prompt.strip():
        raise ValueError(
            "--prompt_mode fixed requires a non-empty --fixed_prompt. "
            "Either provide an explicit prompt string or switch to "
            "--prompt_mode template_pool / task / empty."
        )
    random.seed(int(args.seed))
    output_jsonl = Path(args.output_jsonl).resolve()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    audio_dir = Path(args.audio_dir).resolve()
    audio_dir.mkdir(parents=True, exist_ok=True)

    bad_records_path = Path(args.write_bad_records).resolve() if args.write_bad_records else None
    if bad_records_path is not None:
        bad_records_path.parent.mkdir(parents=True, exist_ok=True)
        if bad_records_path.exists():
            bad_records_path.unlink()

    shard_paths = list(read_lines(args.shards_list))
    total_shards = len(shard_paths)
    if total_shards == 0:
        raise ValueError(f"No valid shard path found in --shards_list: {args.shards_list}")

    used_names: Dict[str, int] = defaultdict(int)
    num_written = 0
    num_skipped = 0
    total_start_time = time.time()

    log(
        "Start conversion with "
        f"shards={total_shards}, output_jsonl={output_jsonl}, audio_dir={audio_dir}, "
        f"output_format={args.output_format}, prompt_mode={args.prompt_mode}, "
        f"log_every={args.log_every}"
    )
    if bad_records_path is not None:
        log(f"Bad records will be written to: {bad_records_path}")

    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for shard_idx, shard_path in enumerate(shard_paths, start=1):
            shard_start_time = time.time()
            shard_path_obj = Path(shard_path)
            shard_name = shard_path_obj.name
            log(f"[shard {shard_idx}/{total_shards}] Reading shard: {shard_path_obj}")
            if not shard_path_obj.exists():
                num_skipped += 1
                log(f"[shard {shard_idx}/{total_shards}] Missing shard, skipped: {shard_path_obj}")
                maybe_write_bad_record(
                    bad_records_path,
                    {
                        "reason": "missing_shard",
                        "shard_path": shard_path,
                    },
                )
                continue

            grouped = iter_shard_records(str(shard_path_obj))
            shard_total_samples = len(grouped)
            shard_written = 0
            shard_skipped = 0
            log(
                f"[shard {shard_idx}/{total_shards}] Loaded {shard_total_samples} grouped samples "
                f"from {shard_name}"
            )
            for sample_idx, (sample_id, sample) in enumerate(grouped.items(), start=1):
                wav_bytes = sample.get("wav_bytes")
                raw_txt = str(sample.get("txt", ""))
                raw_task = str(sample.get("task", ""))
                raw_lang = str(sample.get("lang", ""))
                raw_state = str(sample.get("state", ""))

                if wav_bytes is None and args.skip_missing_wav == 1:
                    num_skipped += 1
                    shard_skipped += 1
                    maybe_write_bad_record(
                        bad_records_path,
                        {
                            "reason": "missing_wav",
                            "shard_path": shard_path,
                            "sample_id": sample_id,
                            "available_fields": sorted(sample.keys()),
                        },
                    )
                    continue
                if wav_bytes is None:
                    raise ValueError(
                        f"Missing .wav for sample_id={sample_id!r} in shard={shard_path!r}. "
                        "Either fix the source shard or set --skip_missing_wav 1."
                    )

                if (not raw_txt.strip()) and args.skip_missing_txt == 1:
                    num_skipped += 1
                    shard_skipped += 1
                    maybe_write_bad_record(
                        bad_records_path,
                        {
                            "reason": "missing_txt",
                            "shard_path": shard_path,
                            "sample_id": sample_id,
                            "available_fields": sorted(sample.keys()),
                        },
                    )
                    continue
                if not raw_txt.strip():
                    raise ValueError(
                        f"Missing .txt for sample_id={sample_id!r} in shard={shard_path!r}. "
                        "Either fix the source shard or set --skip_missing_txt 1."
                    )

                lang_name = normalize_lang(raw_lang)
                transcript_from_txt, label_from_txt, label_token_from_txt = parse_label_from_text_tail(raw_txt)
                label_from_state, label_token_from_state = parse_label_from_state(raw_state)
                label_from_task, label_token_from_task = parse_label_from_task(raw_task)

                label = label_from_txt or label_from_state or label_from_task
                label_token = label_token_from_txt or label_token_from_state or label_token_from_task

                target_transcript = normalize_target_text(
                    transcript_from_txt,
                    keep_inline_tags=bool(args.keep_inline_tags),
                    strip_text=bool(args.strip_txt),
                )
                target_text = build_qwen_asr_target(
                    lang_name=lang_name,
                    transcript=target_transcript,
                    label=label,
                    output_format=args.output_format,
                )
                prompt = build_prompt(
                    raw_task,
                    prompt_mode=args.prompt_mode,
                    fixed_prompt=args.fixed_prompt,
                )

                audio_out_path = ensure_unique_output_name(audio_dir, sample_id, "wav", used_names)
                if wav_bytes is not None:
                    with open(audio_out_path, "wb") as fw:
                        fw.write(wav_bytes)

                record = {
                    "audio": audio_path_for_manifest(audio_out_path, output_jsonl, args.audio_path_mode),
                    "text": target_text,
                    "prompt": prompt,
                    "source_dataset": "easy-turn",
                    "source_shard": str(shard_path_obj.resolve()),
                    "source_sample_id": sample_id,
                    "easy_turn_text_raw": raw_txt.strip() if args.strip_txt == 1 else raw_txt,
                    "easy_turn_transcript": target_transcript,
                    "easy_turn_lang_raw": raw_lang.strip(),
                    "easy_turn_lang_qwen": lang_name,
                    "easy_turn_task_raw": raw_task.strip(),
                    "easy_turn_state_raw": raw_state.strip(),
                    "easy_turn_label": label,
                    "easy_turn_label_token": label_token,
                    "output_format": args.output_format,
                }

                for meta_key in ("duration", "emotion", "extra", "gender", "speaker", "state"):
                    if meta_key in sample:
                        record[f"easy_turn_{meta_key}"] = str(sample[meta_key]).strip()

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                num_written += 1
                shard_written += 1

                if args.log_every > 0 and (
                    sample_idx % args.log_every == 0 or sample_idx == shard_total_samples
                ):
                    log(
                        f"[shard {shard_idx}/{total_shards}] Progress {sample_idx}/{shard_total_samples} "
                        f"| shard_written={shard_written} shard_skipped={shard_skipped} "
                        f"| total_written={num_written} total_skipped={num_skipped}"
                    )

            log(
                f"[shard {shard_idx}/{total_shards}] Finished {shard_name} "
                f"| samples={shard_total_samples} written={shard_written} skipped={shard_skipped} "
                f"| elapsed={format_seconds(time.time() - shard_start_time)}"
            )

    total_elapsed = time.time() - total_start_time
    log(
        f"Conversion finished | shards={total_shards} written={num_written} "
        f"skipped={num_skipped} | elapsed={format_seconds(total_elapsed)}"
    )
    print(
        json.dumps(
            {
                "output_jsonl": str(output_jsonl),
                "audio_dir": str(audio_dir),
                "num_written": num_written,
                "num_skipped": num_skipped,
                "prompt_mode": args.prompt_mode,
                "audio_path_mode": args.audio_path_mode,
                "output_format": args.output_format,
                "log_every": args.log_every,
                "elapsed_seconds": round(total_elapsed, 3),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
