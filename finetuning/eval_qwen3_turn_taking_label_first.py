# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""
Evaluate Qwen3-ASR turn-taking joint generation outputs.

This script is designed for the current "do not change model structure" route:
the model still follows the Qwen3-ASR generative path, but the target text is
aligned to a joint protocol such as:

    language Chinese<turn_state><complete><asr_text>你好

or the label-last ablation:

    language Chinese<asr_text>你好<complete>

Compared with `eval_qwen3_turn_detection.py`, this script evaluates:

- output schema validity
- turn label quality (4-way accuracy / macro F1 / per-class metrics)
- transcript quality (CER / WER / exact match)
- language field quality
- joint metrics across language + label + transcript
"""
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import load_dataset

from qwen_asr import Qwen3ASRModel
from qwen_asr.inference.utils import detect_and_fix_repetitions, normalize_audios, normalize_language_name
from qwen_asr.turn_detection.metrics import finite_or_none, latency_summary


TURN_LABELS = ("complete", "incomplete", "backchannel", "wait")
TURN_LABEL_TOKENS = {f"<{label}>": label for label in TURN_LABELS}
TURN_LABEL_TOKENS_UPPER = {token.upper(): label for token, label in TURN_LABEL_TOKENS.items()}
LABEL_FIRST_MODES = ("label_first", "label_last", "auto")

_ASR_TEXT_TAG = "<asr_text>"
_TURN_STATE_TAG = "<turn_state>"
_LANG_RE = re.compile(r"language\s+([^\n<]+)", re.IGNORECASE)
_TAIL_LABEL_RE = re.compile(r"^(.*?)(<(?:complete|incomplete|backchannel|wait)>)\s*$", re.IGNORECASE | re.DOTALL)


@dataclass
class ParsedJointOutput:
    raw_text: str
    language: str
    turn_label: Optional[str]
    transcript: str
    position_used: str
    has_language_prefix: bool
    has_asr_tag: bool
    has_turn_state_tag: bool
    has_label_token: bool
    strict_schema_valid: bool
    soft_parse_success: bool


def parse_args():
    p = argparse.ArgumentParser("Evaluate Qwen3-ASR label-first / label-last turn-taking outputs")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--eval_file", type=str, default="")
    p.add_argument("--test_file", type=str, default="")
    p.add_argument("--output_json", type=str, default="")
    p.add_argument("--predictions_jsonl", type=str, default="")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--label_position", type=str, choices=LABEL_FIRST_MODES, default="label_first")
    p.add_argument("--device_map", type=str, default="auto")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--case_insensitive", type=int, default=1)
    p.add_argument("--group_by_fields", nargs="*", default=["gold_turn_label", "gold_language"])
    p.add_argument("--measure_latency", type=int, default=1)
    p.add_argument("--measure_ttft", type=int, default=1)
    return p.parse_args()


def load_json_records(path: str) -> List[Dict[str, Any]]:
    ds = load_dataset("json", data_files={"data": path})["data"]
    return [dict(row) for row in ds]


def batched(items: Sequence[Dict[str, Any]], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def maybe_write_json(path: str, payload: Dict[str, Any]) -> None:
    if not path:
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def maybe_write_predictions(path: str, records: List[Dict[str, Any]]) -> None:
    if not path:
        return
    with open(path, "w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _normalize_turn_label(label: Any) -> Optional[str]:
    if label is None:
        return None
    s = str(label).strip().lower()
    if not s:
        return None
    if s in TURN_LABELS:
        return s
    if s in TURN_LABEL_TOKENS:
        return TURN_LABEL_TOKENS[s]
    if s.upper() in TURN_LABEL_TOKENS_UPPER:
        return TURN_LABEL_TOKENS_UPPER[s.upper()]
    return None


def _extract_language(meta_text: str) -> Tuple[str, bool]:
    m = _LANG_RE.search(meta_text or "")
    if not m:
        return "", False
    raw = (m.group(1) or "").strip()
    if not raw:
        return "", False
    try:
        return normalize_language_name(raw), True
    except Exception:
        return raw, True


def _find_label_token(text: str) -> Optional[str]:
    lower = str(text or "").lower()
    for token, label in TURN_LABEL_TOKENS.items():
        if token in lower:
            return label
    return None


def _parse_label_first(text: str) -> ParsedJointOutput:
    s = detect_and_fix_repetitions(str(text or "").strip())
    has_asr_tag = _ASR_TEXT_TAG in s.lower()
    meta_part = s
    transcript = ""
    if has_asr_tag:
        idx = s.lower().find(_ASR_TEXT_TAG)
        meta_part = s[:idx]
        transcript = s[idx + len(_ASR_TEXT_TAG) :].strip()

    language, has_language_prefix = _extract_language(meta_part)
    has_turn_state_tag = _TURN_STATE_TAG in meta_part.lower()
    turn_label = _find_label_token(meta_part)
    has_label_token = turn_label is not None
    strict_schema_valid = bool(has_language_prefix and has_asr_tag and has_turn_state_tag and has_label_token)
    soft_parse_success = bool(has_asr_tag and has_label_token)
    return ParsedJointOutput(
        raw_text=s,
        language=language,
        turn_label=turn_label,
        transcript=transcript,
        position_used="label_first",
        has_language_prefix=has_language_prefix,
        has_asr_tag=has_asr_tag,
        has_turn_state_tag=has_turn_state_tag,
        has_label_token=has_label_token,
        strict_schema_valid=strict_schema_valid,
        soft_parse_success=soft_parse_success,
    )


def _parse_label_last(text: str) -> ParsedJointOutput:
    s = detect_and_fix_repetitions(str(text or "").strip())
    has_asr_tag = _ASR_TEXT_TAG in s.lower()
    meta_part = s
    text_part = ""
    if has_asr_tag:
        idx = s.lower().find(_ASR_TEXT_TAG)
        meta_part = s[:idx]
        text_part = s[idx + len(_ASR_TEXT_TAG) :].strip()

    language, has_language_prefix = _extract_language(meta_part)
    has_turn_state_tag = _TURN_STATE_TAG in meta_part.lower()
    turn_label = None
    transcript = text_part
    m = _TAIL_LABEL_RE.match(text_part)
    if m:
        transcript = m.group(1).strip()
        turn_label = _normalize_turn_label(m.group(2))
    has_label_token = turn_label is not None
    strict_schema_valid = bool(has_language_prefix and has_asr_tag and has_label_token)
    soft_parse_success = bool(has_asr_tag and has_label_token)
    return ParsedJointOutput(
        raw_text=s,
        language=language,
        turn_label=turn_label,
        transcript=transcript,
        position_used="label_last",
        has_language_prefix=has_language_prefix,
        has_asr_tag=has_asr_tag,
        has_turn_state_tag=has_turn_state_tag,
        has_label_token=has_label_token,
        strict_schema_valid=strict_schema_valid,
        soft_parse_success=soft_parse_success,
    )


def parse_joint_output(text: Any, label_position: str) -> ParsedJointOutput:
    label_position = str(label_position).strip().lower()
    if label_position == "label_first":
        return _parse_label_first(text)
    if label_position == "label_last":
        return _parse_label_last(text)

    first = _parse_label_first(text)
    last = _parse_label_last(text)

    def _score(parsed: ParsedJointOutput) -> Tuple[int, int, int, int]:
        return (
            1 if parsed.strict_schema_valid else 0,
            1 if parsed.soft_parse_success else 0,
            1 if parsed.has_turn_state_tag else 0,
            len(parsed.transcript),
        )

    return first if _score(first) >= _score(last) else last


def parse_gold_row(row: Dict[str, Any], label_position: str) -> ParsedJointOutput:
    gold_text = row.get("text", "")
    parsed = parse_joint_output(gold_text, label_position=label_position)

    gold_label = _normalize_turn_label(row.get("turn_label"))
    if gold_label is None:
        gold_label = _normalize_turn_label(row.get("label"))
    if gold_label is not None:
        parsed.turn_label = gold_label

    gold_transcript = row.get("transcript")
    if gold_transcript not in (None, ""):
        parsed.transcript = str(gold_transcript).strip()

    gold_lang = row.get("language", row.get("lang"))
    if gold_lang not in (None, ""):
        try:
            parsed.language = normalize_language_name(str(gold_lang))
        except Exception:
            parsed.language = str(gold_lang).strip()

    return parsed


def normalize_text_for_compare(text: Any, case_insensitive: bool = True) -> str:
    s = str(text or "")
    s = " ".join(s.strip().split())
    if case_insensitive:
        s = s.lower()
    return s


def char_tokens(text: Any, case_insensitive: bool = True) -> List[str]:
    s = normalize_text_for_compare(text, case_insensitive=case_insensitive)
    s = "".join(ch for ch in s if not ch.isspace())
    return list(s)


def word_tokens(text: Any, case_insensitive: bool = True) -> List[str]:
    s = normalize_text_for_compare(text, case_insensitive=case_insensitive)
    if not s:
        return []
    tokens = s.split()
    if len(tokens) > 1:
        return tokens
    return char_tokens(s, case_insensitive=False)


def edit_distance(ref: Sequence[str], hyp: Sequence[str]) -> int:
    n = len(ref)
    m = len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,
                dp[j - 1] + 1,
                prev + cost,
            )
            prev = cur
    return int(dp[m])


def summarize_text_metric(records: List[Dict[str, Any]], ref_key: str, hyp_key: str, unit: str, case_insensitive: bool) -> Dict[str, Any]:
    if unit not in {"char", "word"}:
        raise ValueError(f"Unsupported unit={unit}")

    total_edits = 0
    total_ref_len = 0
    exact_match = 0
    nonempty = 0

    for row in records:
        ref = row.get(ref_key, "")
        hyp = row.get(hyp_key, "")
        ref_tokens = char_tokens(ref, case_insensitive) if unit == "char" else word_tokens(ref, case_insensitive)
        hyp_tokens = char_tokens(hyp, case_insensitive) if unit == "char" else word_tokens(hyp, case_insensitive)
        dist = edit_distance(ref_tokens, hyp_tokens)
        denom = max(1, len(ref_tokens))
        row[f"{unit}_distance"] = dist
        row[f"{unit}_ref_len"] = len(ref_tokens)
        row[f"{unit}_error_rate"] = float(dist / denom)
        total_edits += dist
        total_ref_len += len(ref_tokens)
        if len(ref_tokens) > 0:
            nonempty += 1
        if normalize_text_for_compare(ref, case_insensitive) == normalize_text_for_compare(hyp, case_insensitive):
            exact_match += 1

    return {
        f"{unit}_error_rate": float(total_edits / max(1, total_ref_len)),
        f"{unit}_edit_count": int(total_edits),
        f"{unit}_ref_total": int(total_ref_len),
        "exact_match_rate": float(exact_match / max(1, len(records))),
        "nonempty_ref_count": int(nonempty),
    }


def summarize_multiclass(gold_labels: Sequence[Optional[str]], pred_labels: Sequence[Optional[str]], labels: Sequence[str]) -> Dict[str, Any]:
    valid_total = 0
    accuracy_num = 0
    confusion: Dict[str, Dict[str, int]] = {
        gold: {pred: 0 for pred in list(labels) + ["<invalid>"]}
        for gold in labels
    }
    per_class: Dict[str, Dict[str, float]] = {}

    for gold, pred in zip(gold_labels, pred_labels):
        if gold not in labels:
            continue
        valid_total += 1
        pred_key = pred if pred in labels else "<invalid>"
        confusion[str(gold)][pred_key] += 1
        if gold == pred:
            accuracy_num += 1

    macro_p = 0.0
    macro_r = 0.0
    macro_f1 = 0.0
    valid_class_count = 0

    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[g][label] for g in labels if g != label)
        fn = sum(confusion[label][p] for p in confusion[label] if p != label)
        support = sum(confusion[label].values())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-8, precision + recall)
        per_class[label] = {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "support": int(support),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        macro_p += precision
        macro_r += recall
        macro_f1 += f1
        valid_class_count += 1

    return {
        "accuracy": float(accuracy_num / max(1, valid_total)),
        "macro_precision": float(macro_p / max(1, valid_class_count)),
        "macro_recall": float(macro_r / max(1, valid_class_count)),
        "macro_f1": float(macro_f1 / max(1, valid_class_count)),
        "evaluated_count": int(valid_total),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def maybe_cuda_sync(model: Qwen3ASRModel) -> None:
    if getattr(model, "backend", "") != "transformers":
        return
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


def _decode_generated_continuations(model: Qwen3ASRModel, sequences: torch.Tensor, input_len: int) -> List[str]:
    return list(
        model.processor.batch_decode(
            sequences[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    )


def infer_asr_with_metrics(
    model: Qwen3ASRModel,
    contexts: List[str],
    wavs: List[np.ndarray],
    measure_latency: bool,
    measure_ttft: bool,
) -> Tuple[List[str], Dict[str, Any]]:
    metrics: Dict[str, Any] = {
        "latency_ms": None,
        "ttft_ms": None,
        "full_inference_ms": None,
    }

    if getattr(model, "backend", "") != "transformers":
        total_start = time.perf_counter() if measure_latency else None
        if measure_latency:
            maybe_cuda_sync(model)
            start = time.perf_counter()
            raw_outputs = model._infer_asr(contexts, wavs, [None] * len(wavs))
            maybe_cuda_sync(model)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            per_item_ms = float(elapsed_ms / max(1, len(wavs)))
            metrics["latency_ms"] = per_item_ms
            metrics["full_inference_ms"] = per_item_ms
        else:
            raw_outputs = model._infer_asr(contexts, wavs, [None] * len(wavs))
        if measure_latency and total_start is not None:
            total_elapsed_ms = (time.perf_counter() - total_start) * 1000.0
            metrics["full_inference_ms"] = float(total_elapsed_ms / max(1, len(wavs)))
        return raw_outputs, metrics

    total_start = time.perf_counter() if measure_latency else None
    texts = [model._build_text_prompt(context=c, force_language=None) for c in contexts]
    inputs = model.processor(text=texts, audio=wavs, return_tensors="pt", padding=True)
    inputs = inputs.to(model.model.device).to(model.model.dtype)
    input_len = int(inputs["input_ids"].shape[1])

    if measure_ttft:
        maybe_cuda_sync(model)
        ttft_start = time.perf_counter()
        _ = model.model.generate(**inputs, max_new_tokens=1)
        maybe_cuda_sync(model)
        ttft_ms = (time.perf_counter() - ttft_start) * 1000.0
        metrics["ttft_ms"] = float(ttft_ms / max(1, len(wavs)))

    if measure_latency:
        maybe_cuda_sync(model)
        start = time.perf_counter()
        outputs = model.model.generate(**inputs, max_new_tokens=model.max_new_tokens)
        maybe_cuda_sync(model)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
    else:
        outputs = model.model.generate(**inputs, max_new_tokens=model.max_new_tokens)
        elapsed_ms = None

    raw_outputs = _decode_generated_continuations(model, outputs.sequences, input_len=input_len)

    if elapsed_ms is not None:
        metrics["latency_ms"] = float(elapsed_ms / max(1, len(wavs)))
    if measure_latency and total_start is not None:
        total_elapsed_ms = (time.perf_counter() - total_start) * 1000.0
        metrics["full_inference_ms"] = float(total_elapsed_ms / max(1, len(wavs)))

    return raw_outputs, metrics


def summarize_boolean_rate(records: List[Dict[str, Any]], key: str) -> float:
    vals = [1.0 if bool(row.get(key)) else 0.0 for row in records]
    return float(sum(vals) / max(1, len(vals)))


def build_group_summary(records: List[Dict[str, Any]], group_key: str, case_insensitive: bool) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in records:
        value = row.get(group_key, "")
        key = str(value).strip() if value not in (None, "") else "<empty>"
        grouped.setdefault(key, []).append(row)

    out: Dict[str, Any] = {}
    for key, items in grouped.items():
        gold_labels = [row.get("gold_turn_label") for row in items]
        pred_labels = [row.get("pred_turn_label") for row in items]
        label_summary = summarize_multiclass(gold_labels, pred_labels, TURN_LABELS)
        cer_summary = summarize_text_metric(items, "gold_transcript", "pred_transcript", unit="char", case_insensitive=case_insensitive)
        wer_summary = summarize_text_metric(items, "gold_transcript", "pred_transcript", unit="word", case_insensitive=case_insensitive)
        out[key] = {
            "count": int(len(items)),
            "schema_valid_rate": summarize_boolean_rate(items, "pred_strict_schema_valid"),
            "label_accuracy": float(label_summary["accuracy"]),
            "label_macro_f1": float(label_summary["macro_f1"]),
            "cer": float(cer_summary["char_error_rate"]),
            "wer": float(wer_summary["word_error_rate"]),
            "joint_label_text_exact_match_rate": summarize_boolean_rate(items, "joint_label_text_exact_match"),
        }
    return out


def evaluate_split(model: Qwen3ASRModel, records: List[Dict[str, Any]], args, split_name: str) -> Dict[str, Any]:
    enriched_records: List[Dict[str, Any]] = []
    latency_ms_values: List[float] = []
    ttft_ms_values: List[float] = []
    full_inference_ms_values: List[float] = []

    for batch in batched(records, max(1, int(args.batch_size))):
        wavs = normalize_audios([row["audio"] for row in batch])
        contexts = [str(row.get("prompt", "") or "") for row in batch]
        raw_outputs, timing = infer_asr_with_metrics(
            model,
            contexts=contexts,
            wavs=wavs,
            measure_latency=bool(args.measure_latency),
            measure_ttft=bool(args.measure_ttft),
        )
        latency_ms = finite_or_none(timing.get("latency_ms"))
        ttft_ms = finite_or_none(timing.get("ttft_ms"))
        full_inference_ms = finite_or_none(timing.get("full_inference_ms"))
        if latency_ms is not None:
            latency_ms_values.extend([latency_ms] * len(batch))
        if ttft_ms is not None:
            ttft_ms_values.extend([ttft_ms] * len(batch))
        if full_inference_ms is not None:
            full_inference_ms_values.extend([full_inference_ms] * len(batch))

        for row, raw_out in zip(batch, raw_outputs):
            gold = parse_gold_row(row, label_position=args.label_position)
            pred = parse_joint_output(raw_out, label_position=args.label_position)

            pred_turn_label = pred.turn_label
            gold_turn_label = _normalize_turn_label(gold.turn_label)

            gold_lang = gold.language or ""
            pred_lang = pred.language or ""
            gold_text = gold.transcript or ""
            pred_text = pred.transcript or ""

            label_match = gold_turn_label == pred_turn_label and gold_turn_label is not None
            language_match = normalize_text_for_compare(gold_lang, bool(args.case_insensitive)) == normalize_text_for_compare(
                pred_lang,
                bool(args.case_insensitive),
            )
            text_match = normalize_text_for_compare(gold_text, bool(args.case_insensitive)) == normalize_text_for_compare(
                pred_text,
                bool(args.case_insensitive),
            )

            enriched = dict(row)
            enriched.update(
                {
                    "split": split_name,
                    "raw_output": str(raw_out),
                    "gold_language": gold_lang,
                    "gold_turn_label": gold_turn_label,
                    "gold_transcript": gold_text,
                    "pred_language": pred_lang,
                    "pred_turn_label": pred_turn_label,
                    "pred_transcript": pred_text,
                    "pred_position_used": pred.position_used,
                    "pred_has_language_prefix": bool(pred.has_language_prefix),
                    "pred_has_asr_tag": bool(pred.has_asr_tag),
                    "pred_has_turn_state_tag": bool(pred.has_turn_state_tag),
                    "pred_has_label_token": bool(pred.has_label_token),
                    "pred_soft_parse_success": bool(pred.soft_parse_success),
                    "pred_strict_schema_valid": bool(pred.strict_schema_valid),
                    "latency_ms": latency_ms,
                    "ttft_ms": ttft_ms,
                    "full_inference_ms": full_inference_ms,
                    "label_match": bool(label_match),
                    "language_match": bool(language_match),
                    "transcript_exact_match": bool(text_match),
                    "joint_label_text_exact_match": bool(label_match and text_match),
                    "joint_language_label_text_exact_match": bool(language_match and label_match and text_match),
                }
            )
            enriched_records.append(enriched)

    gold_labels = [row.get("gold_turn_label") for row in enriched_records]
    pred_labels = [row.get("pred_turn_label") for row in enriched_records]
    label_summary = summarize_multiclass(gold_labels, pred_labels, TURN_LABELS)

    cer_summary = summarize_text_metric(
        enriched_records,
        ref_key="gold_transcript",
        hyp_key="pred_transcript",
        unit="char",
        case_insensitive=bool(args.case_insensitive),
    )
    wer_summary = summarize_text_metric(
        enriched_records,
        ref_key="gold_transcript",
        hyp_key="pred_transcript",
        unit="word",
        case_insensitive=bool(args.case_insensitive),
    )

    valid_records = [row for row in enriched_records if row.get("pred_strict_schema_valid")]
    label_correct_records = [row for row in enriched_records if row.get("label_match")]
    language_known_records = [row for row in enriched_records if str(row.get("gold_language", "")).strip() != ""]

    summary: Dict[str, Any] = {
        "count": int(len(enriched_records)),
        "label_position": args.label_position,
        "schema": {
            "strict_schema_valid_rate": summarize_boolean_rate(enriched_records, "pred_strict_schema_valid"),
            "soft_parse_success_rate": summarize_boolean_rate(enriched_records, "pred_soft_parse_success"),
            "language_prefix_rate": summarize_boolean_rate(enriched_records, "pred_has_language_prefix"),
            "asr_tag_rate": summarize_boolean_rate(enriched_records, "pred_has_asr_tag"),
            "turn_state_tag_rate": summarize_boolean_rate(enriched_records, "pred_has_turn_state_tag"),
            "label_token_rate": summarize_boolean_rate(enriched_records, "pred_has_label_token"),
        },
        "turn_label": label_summary,
        "transcript": {
            "cer": float(cer_summary["char_error_rate"]),
            "wer": float(wer_summary["word_error_rate"]),
            "transcript_exact_match_rate": float(cer_summary["exact_match_rate"]),
        },
        "language": {
            "language_accuracy": float(
                sum(1.0 if row.get("language_match") else 0.0 for row in language_known_records) / max(1, len(language_known_records))
            ),
            "evaluated_count": int(len(language_known_records)),
        },
        "joint": {
            "joint_label_text_exact_match_rate": summarize_boolean_rate(enriched_records, "joint_label_text_exact_match"),
            "joint_language_label_text_exact_match_rate": summarize_boolean_rate(
                enriched_records,
                "joint_language_label_text_exact_match",
            ),
        },
        "latency": latency_summary(latency_ms_values),
        "ttft": latency_summary(ttft_ms_values),
        "full_inference": latency_summary(full_inference_ms_values),
        "subsets": {
            "valid_schema_count": int(len(valid_records)),
            "label_correct_count": int(len(label_correct_records)),
        },
    }

    if valid_records:
        valid_cer = summarize_text_metric(valid_records, "gold_transcript", "pred_transcript", unit="char", case_insensitive=bool(args.case_insensitive))
        valid_wer = summarize_text_metric(valid_records, "gold_transcript", "pred_transcript", unit="word", case_insensitive=bool(args.case_insensitive))
        summary["transcript"]["cer_valid_schema_only"] = float(valid_cer["char_error_rate"])
        summary["transcript"]["wer_valid_schema_only"] = float(valid_wer["word_error_rate"])

    if label_correct_records:
        label_correct_cer = summarize_text_metric(
            label_correct_records,
            "gold_transcript",
            "pred_transcript",
            unit="char",
            case_insensitive=bool(args.case_insensitive),
        )
        label_correct_wer = summarize_text_metric(
            label_correct_records,
            "gold_transcript",
            "pred_transcript",
            unit="word",
            case_insensitive=bool(args.case_insensitive),
        )
        summary["transcript"]["cer_label_correct_only"] = float(label_correct_cer["char_error_rate"])
        summary["transcript"]["wer_label_correct_only"] = float(label_correct_wer["word_error_rate"])

    if args.group_by_fields:
        summary["group_metrics"] = {
            field: build_group_summary(enriched_records, field, case_insensitive=bool(args.case_insensitive))
            for field in args.group_by_fields
        }

    return {
        "summary": summary,
        "records": enriched_records,
    }


def load_model(args) -> Qwen3ASRModel:
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    return Qwen3ASRModel.from_pretrained(
        args.model_path,
        dtype=dtype,
        device_map=args.device_map,
        max_inference_batch_size=max(1, int(args.batch_size)),
        max_new_tokens=int(args.max_new_tokens),
    )


def main():
    args = parse_args()
    if not args.eval_file and not args.test_file:
        raise ValueError("At least one of --eval_file or --test_file must be provided.")

    model = load_model(args)
    report: Dict[str, Any] = {
        "model_path": args.model_path,
        "label_position": args.label_position,
        "case_insensitive": bool(args.case_insensitive),
    }
    all_records: List[Dict[str, Any]] = []

    if args.eval_file:
        eval_records = load_json_records(args.eval_file)
        eval_result = evaluate_split(model, eval_records, args, split_name="eval")
        report["eval"] = {
            "file": args.eval_file,
            "summary": eval_result["summary"],
        }
        all_records.extend(eval_result["records"])

    if args.test_file:
        test_records = load_json_records(args.test_file)
        test_result = evaluate_split(model, test_records, args, split_name="test")
        report["test"] = {
            "file": args.test_file,
            "summary": test_result["summary"],
        }
        all_records.extend(test_result["records"])

    maybe_write_json(args.output_json, report)
    maybe_write_predictions(args.predictions_jsonl, all_records)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
