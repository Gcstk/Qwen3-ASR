# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Rewrite `audio` paths inside an existing Qwen3-ASR JSONL manifest by replacing
# one absolute path prefix with another.
import argparse
import json
import sys
import time
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Dict, Tuple

from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(
        "Rewrite Qwen3-ASR JSONL audio path prefixes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input_jsonl", type=str, required=True, help="Input JSONL file to rewrite.")
    p.add_argument("--output_jsonl", type=str, required=True, help="Output JSONL file after rewriting.")
    p.add_argument("--old_prefix", type=str, required=True, help="Old absolute audio root prefix.")
    p.add_argument("--new_prefix", type=str, required=True, help="New absolute audio root prefix.")
    p.add_argument(
        "--strict",
        type=int,
        choices=[0, 1],
        default=1,
        help="Whether to fail when an audio path does not start with --old_prefix.",
    )
    p.add_argument(
        "--must_exist",
        type=int,
        choices=[0, 1],
        default=0,
        help="Whether to require the rewritten target audio path to exist on the current machine.",
    )
    return p.parse_args()


def log(message: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[rewrite_audio_paths][{ts}] {message}", file=sys.stdout, flush=True)


def _looks_windows_path(path: str) -> bool:
    value = path or ""
    return len(value) >= 2 and value[1] == ":"


def normalize_prefix(path: str) -> str:
    value = (path or "").strip()
    if not value:
        raise ValueError("Path prefix cannot be empty.")
    if _looks_windows_path(value):
        normalized = str(PureWindowsPath(value))
        return normalized.rstrip("\\/")
    normalized = str(PurePosixPath(value.replace("\\", "/")))
    return normalized.rstrip("/")


def rewrite_audio_path(audio_path: str, old_prefix: str, new_prefix: str) -> Tuple[str, bool]:
    current = (audio_path or "").strip()
    if not current:
        return current, False

    old_norm = normalize_prefix(old_prefix)
    new_norm = normalize_prefix(new_prefix)

    if _looks_windows_path(current):
        current_norm = str(PureWindowsPath(current))
        prefix_with_sep = old_norm + "\\"
        if current_norm == old_norm:
            return new_norm, True
        if current_norm.startswith(prefix_with_sep):
            suffix = current_norm[len(prefix_with_sep):]
            return str(PureWindowsPath(new_norm) / PureWindowsPath(suffix)), True
        return current, False

    current_norm = current.replace("\\", "/")
    prefix_with_sep = old_norm + "/"
    if current_norm == old_norm:
        return new_norm, True
    if current_norm.startswith(prefix_with_sep):
        suffix = current_norm[len(prefix_with_sep):]
        return str(PurePosixPath(new_norm) / PurePosixPath(suffix)), True
    return current, False


def main():
    args = parse_args()
    input_jsonl = Path(args.input_jsonl).resolve()
    output_jsonl = Path(args.output_jsonl).resolve()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    if not input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_jsonl}")

    with open(input_jsonl, "r", encoding="utf-8") as fin:
        total_lines = sum(1 for line in fin if line.strip())

    rewritten = 0
    unchanged = 0
    start_time = time.time()
    log(
        f"Start rewriting | input_jsonl={input_jsonl} output_jsonl={output_jsonl} "
        f"old_prefix={args.old_prefix} new_prefix={args.new_prefix}"
    )

    with open(input_jsonl, "r", encoding="utf-8") as fin, open(output_jsonl, "w", encoding="utf-8") as fout:
        for line_no, line in enumerate(
            tqdm(fin, total=total_lines, desc="Rewriting audio paths", unit="record"),
            start=1,
        ):
            text = line.strip()
            if not text:
                continue
            try:
                record: Dict[str, object] = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {input_jsonl}: {exc}") from exc

            audio_path = str(record.get("audio", ""))
            new_audio_path, changed = rewrite_audio_path(audio_path, args.old_prefix, args.new_prefix)
            if not changed:
                if int(args.strict) == 1:
                    raise ValueError(
                        f"Line {line_no} audio path does not start with old prefix:\n"
                        f"  audio={audio_path}\n"
                        f"  old_prefix={args.old_prefix}"
                    )
                unchanged += 1
            else:
                if int(args.must_exist) == 1 and not Path(new_audio_path).exists():
                    raise FileNotFoundError(
                        f"Rewritten audio path does not exist on line {line_no}: {new_audio_path}"
                    )
                record["audio"] = new_audio_path
                rewritten += 1

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    elapsed = time.time() - start_time
    log(
        f"Finished rewriting | rewritten={rewritten} unchanged={unchanged} "
        f"elapsed_seconds={elapsed:.3f}"
    )
    print(
        json.dumps(
            {
                "input_jsonl": str(input_jsonl),
                "output_jsonl": str(output_jsonl),
                "rewritten": rewritten,
                "unchanged": unchanged,
                "old_prefix": args.old_prefix,
                "new_prefix": args.new_prefix,
                "elapsed_seconds": round(elapsed, 3),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
